import argparse
import io
import math
import sys
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest

import config
import trade_config
from strategy import Decision, MA_RSI_CROSSOVER, StrategySpec, WARMUP_SAFETY_BARS
from timeframes import parse_timeframe


@dataclass
class OpenPosition:
    entry_time: Any
    entry_price: float
    entry_market_price: float
    requested_notional: float
    gross_quantity: float
    quantity: float
    buy_fee_quantity: float
    entry_slippage_cost: float
    entry_reason: str


def required_warmup_bars(
    safety_margin: int = WARMUP_SAFETY_BARS,
    strategy: StrategySpec | None = None,
) -> int:
    selected_strategy = strategy or MA_RSI_CROSSOVER
    return selected_strategy.required_warmup_bars(safety_margin)


def fetch_history(
    lookback_days: int,
    timeframe: str,
    *,
    warmup_bars: Optional[int] = None,
    end_time: Optional[datetime] = None,
) -> pd.DataFrame:
    alpaca_timeframe, minutes_per_bar = parse_timeframe(timeframe)
    end = end_time or datetime.now(timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    test_start = end - timedelta(days=lookback_days)
    warmup_bars = required_warmup_bars() if warmup_bars is None else max(warmup_bars, 0)
    request = CryptoBarsRequest(
        symbol_or_symbols=[config.SYMBOL],
        timeframe=alpaca_timeframe,
        start=test_start - timedelta(minutes=minutes_per_bar * warmup_bars),
        end=end,
    )
    bars = CryptoHistoricalDataClient().get_crypto_bars(request).df
    if bars.empty:
        raise RuntimeError(f"Alpaca returned no candles for {config.SYMBOL} in this period")

    if isinstance(bars.index, pd.MultiIndex):
        try:
            bars = bars.xs(config.SYMBOL, level=0)
        except KeyError as error:
            raise RuntimeError(f"Alpaca returned no candles for {config.SYMBOL}") from error

    bars = bars.sort_index()
    bars = bars.loc[~bars.index.duplicated(keep="last")]
    completed_by = pd.Timestamp(end)
    candle_closes = bars.index + pd.to_timedelta(minutes_per_bar, unit="m")
    bars = bars.loc[candle_closes <= completed_by]
    required_columns = {"open", "high", "low", "close", "volume"}
    missing = required_columns.difference(bars.columns)
    if missing:
        raise RuntimeError(f"Historical candles are missing fields: {sorted(missing)}")
    bars.attrs["test_start"] = pd.Timestamp(test_start)
    bars.attrs["warmup_bars_requested"] = warmup_bars
    bars.attrs["requested_start_time"] = pd.Timestamp(
        test_start - timedelta(minutes=minutes_per_bar * warmup_bars)
    )
    bars.attrs["requested_end_time"] = pd.Timestamp(end)
    return bars


def data_quality_diagnostics(
    bars: pd.DataFrame,
    timeframe: str,
    *,
    start_time: Any = None,
    end_time: Any = None,
) -> dict[str, int | float]:
    """Count expected, missing, and zero-volume candles without filling gaps."""
    _, minutes_per_bar = parse_timeframe(timeframe)
    duration = pd.Timedelta(minutes=minutes_per_bar)
    if bars.empty:
        return {
            "expected_candle_count": 0,
            "actual_candle_count": 0,
            "missing_candle_count": 0,
            "missing_candle_percent": 0.0,
            "zero_volume_candle_count": 0,
            "zero_volume_candle_percent": 0.0,
        }
    index = pd.DatetimeIndex(pd.to_datetime(bars.index, utc=True))
    requested_start = start_time or bars.attrs.get("requested_start_time") or index.min()
    requested_end = end_time or bars.attrs.get("requested_end_time") or (index.max() + duration)
    requested_start = pd.Timestamp(requested_start)
    requested_end = pd.Timestamp(requested_end)
    requested_start = requested_start.tz_localize("UTC") if requested_start.tzinfo is None else requested_start.tz_convert("UTC")
    requested_end = requested_end.tz_localize("UTC") if requested_end.tzinfo is None else requested_end.tz_convert("UTC")
    duration_seconds = int(duration.total_seconds())
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    offset_seconds = int((requested_start - epoch).total_seconds())
    aligned_start = epoch + pd.Timedelta(
        seconds=((offset_seconds + duration_seconds - 1) // duration_seconds) * duration_seconds
    )
    expected = (
        pd.date_range(start=aligned_start, end=requested_end - duration, freq=duration)
        if requested_end > aligned_start
        else pd.DatetimeIndex([], tz="UTC")
    )
    actual_set = set(index).intersection(set(expected))
    expected_count = len(expected)
    actual_count = len(actual_set)
    missing_count = max(expected_count - actual_count, 0)
    in_span = (index >= aligned_start) & (index < requested_end)
    actual_positions = [i for i, included in enumerate(in_span) if included]
    volume = pd.to_numeric(bars["volume"], errors="coerce")
    zero_volume_count = int((volume.iloc[actual_positions] == 0).sum()) if actual_positions else 0
    return {
        "expected_candle_count": expected_count,
        "actual_candle_count": actual_count,
        "missing_candle_count": missing_count,
        "missing_candle_percent": missing_count / expected_count * 100 if expected_count else 0.0,
        "zero_volume_candle_count": zero_volume_count,
        "zero_volume_candle_percent": zero_volume_count / actual_count * 100 if actual_count else 0.0,
    }


def _trade_row(
    position: OpenPosition,
    exit_time: Any,
    exit_market_price: float,
    exit_price: float,
    exit_reason: str,
    exit_fee: float,
    exit_slippage_cost: float,
) -> dict[str, Any]:
    gross_pnl = (exit_market_price - position.entry_market_price) * position.quantity
    # BUY fees are withheld in BTC and valued at the entry market price; SELL fees are USD.
    entry_fee = position.buy_fee_quantity * position.entry_market_price
    fees = entry_fee + exit_fee
    slippage_cost = position.entry_slippage_cost + exit_slippage_cost
    net_pnl = gross_pnl - fees - slippage_cost
    cash_flow_net_pnl = exit_price * position.quantity - exit_fee - position.requested_notional
    if not math.isclose(net_pnl, cash_flow_net_pnl, rel_tol=1e-9, abs_tol=1e-9):
        raise AssertionError("Trade PnL does not reconcile with simulated cash flows")
    gross_return = (
        gross_pnl / (position.entry_market_price * position.quantity) * 100
        if position.quantity
        else None
    )
    invested = position.requested_notional
    duration_minutes = (exit_time - position.entry_time).total_seconds() / 60
    return {
        "entry_time": position.entry_time,
        "entry_price": position.entry_price,
        "entry_fill_price": position.entry_price,
        "entry_market_price": position.entry_market_price,
        "exit_time": exit_time,
        "exit_price": exit_price,
        "exit_fill_price": exit_price,
        "exit_market_price": exit_market_price,
        "requested_notional": position.requested_notional,
        "gross_quantity_before_fee": position.gross_quantity,
        "quantity_after_buy_fee": position.quantity,
        # quantity remains as a compatibility alias for historical consumers.
        "quantity": position.quantity,
        "entry_reason": position.entry_reason,
        "exit_reason": exit_reason,
        "gross_pnl": gross_pnl,
        "entry_fee": entry_fee,
        "exit_fee": exit_fee,
        "fees": fees,
        "entry_slippage_cost": position.entry_slippage_cost,
        "exit_slippage_cost": exit_slippage_cost,
        "slippage_cost": slippage_cost,
        "net_pnl": net_pnl,
        "same_path_zero_cost_quantity": (
            position.requested_notional / position.entry_market_price
            if position.entry_market_price
            else 0.0
        ),
        "same_path_zero_cost_pnl": (
            position.requested_notional
            / position.entry_market_price
            * (exit_market_price - position.entry_market_price)
            if position.entry_market_price
            else 0.0
        ),
        "gross_return_percent": gross_return,
        "return_percent": net_pnl / invested * 100 if invested else None,
        "holding_duration_minutes": duration_minutes,
    }


def run_backtest(
    bars: pd.DataFrame,
    starting_capital: Optional[float] = None,
    timeframe: Optional[str] = None,
    fee_rate: Optional[float] = None,
    slippage: Optional[float] = None,
    test_start: Optional[Any] = None,
    strategy: StrategySpec | None = None,
    liquidate_at_end: bool = True,
) -> dict[str, Any]:
    starting_capital = (
        config.BACKTEST_STARTING_CAPITAL
        if starting_capital is None
        else starting_capital
    )
    timeframe = timeframe or trade_config.LIVE_TIMEFRAME
    fee_rate = config.BACKTEST_FEE_PERCENT if fee_rate is None else fee_rate
    slippage = config.BACKTEST_SLIPPAGE_PERCENT if slippage is None else slippage
    selected_strategy = strategy or MA_RSI_CROSSOVER
    stop_loss_percent = selected_strategy.stop_loss_percent
    take_profit_percent = selected_strategy.take_profit_percent
    max_holding_minutes = selected_strategy.max_holding_minutes
    if starting_capital <= 0:
        raise ValueError("Starting capital must be greater than zero")
    if not 0 <= fee_rate < 1 or not 0 <= slippage < 1:
        raise ValueError("Fee and slippage rates must be in the range [0, 1)")
    if max_holding_minutes is not None and max_holding_minutes <= 0:
        raise ValueError("max_holding_minutes must be greater than zero")
    if len(bars) < 2:
        raise ValueError("At least two candles are required for a backtest")

    requested_start = test_start or bars.attrs.get("test_start") or bars.index[0]
    requested_start = pd.Timestamp(requested_start)
    test_start_index = int(bars.index.searchsorted(requested_start, side="left"))
    if test_start_index >= len(bars) - 1:
        raise ValueError("At least two candles must be available in the test period")
    test_bars = bars.iloc[test_start_index:]

    _, minutes_per_bar = parse_timeframe(timeframe)
    indicators = selected_strategy.prepare_indicators(bars)
    cash = float(starting_capital)
    position: Optional[OpenPosition] = None
    completed_trades: list[dict[str, Any]] = []
    entry_events: list[dict[str, Any]] = []
    bar_duration = pd.Timedelta(minutes=minutes_per_bar)
    equity_times = [test_bars.index[0]]
    equity_values = [cash]
    invested_bar_count = 0
    capital_exposures: list[float] = []
    exposure_samples: list[dict[str, Any]] = []

    def close_position(base_exit_price: float, timestamp: Any, reason: str) -> None:
        nonlocal cash, position
        if position is None:
            return
        exit_price = base_exit_price * (1 - slippage)
        exit_notional = exit_price * position.quantity
        exit_fee = exit_notional * fee_rate
        exit_slippage_cost = (base_exit_price - exit_price) * position.quantity
        cash += exit_notional - exit_fee
        if cash < -1e-9:
            raise AssertionError("Backtest cash became negative after closing a position")
        completed_trades.append(
            _trade_row(
                position,
                timestamp,
                base_exit_price,
                exit_price,
                reason,
                exit_fee,
                exit_slippage_cost,
            )
        )
        position = None

    # TODO: optimize this reference loop before large grids/walk-forward jobs;
    # regression-test output against this implementation before replacing it.
    for candle_index in range(test_start_index, len(bars)):
        candle = bars.iloc[candle_index]
        timestamp = bars.index[candle_index]
        open_price = float(candle["open"])
        decision = (
            Decision("HOLD", "test_period_starts_flat")
            if candle_index == test_start_index
            else selected_strategy.decide_at(indicators, candle_index - 1)
        )
        exited_at_open = False

        if position is not None:
            stop_at_open = (
                position.entry_price * (1 - stop_loss_percent)
                if stop_loss_percent is not None
                else None
            )
            take_profit_at_open = (
                position.entry_price * (1 + take_profit_percent)
                if take_profit_percent is not None
                else None
            )
            if stop_at_open is not None and open_price <= stop_at_open:
                close_position(open_price, timestamp, "stop_loss")
                exited_at_open = True
            elif take_profit_at_open is not None and open_price >= take_profit_at_open:
                close_position(open_price, timestamp, "take_profit")
                exited_at_open = True
            elif (
                max_holding_minutes is not None
                and (timestamp - position.entry_time).total_seconds()
                >= max_holding_minutes * 60
            ):
                close_position(open_price, timestamp, "max_holding_time")
                exited_at_open = True
            elif decision.action == "SELL":
                close_position(open_price, timestamp, decision.reason)
                exited_at_open = True
        elif position is None and decision.action == "BUY":
            requested_notional = min(
                trade_config.TRADE_AMOUNT_USD,
                trade_config.MAX_POSITION_USD,
            )
            entry_price = open_price * (1 + slippage)
            gross_quantity = requested_notional / entry_price if entry_price else 0.0
            buy_fee_quantity = gross_quantity * fee_rate
            quantity = gross_quantity - buy_fee_quantity

            # BUY fee is withheld in BTC; cash pays exactly the requested USD notional.
            if quantity > 0 and requested_notional <= cash:
                cash -= requested_notional
                position = OpenPosition(
                    entry_time=timestamp,
                    entry_price=entry_price,
                    entry_market_price=open_price,
                    requested_notional=requested_notional,
                    gross_quantity=gross_quantity,
                    quantity=quantity,
                    buy_fee_quantity=buy_fee_quantity,
                    entry_slippage_cost=(entry_price - open_price) * gross_quantity,
                    entry_reason=decision.reason,
                )
                entry_events.append(
                    {
                        "entry_time": timestamp,
                        "requested_notional": requested_notional,
                        "entry_market_price": open_price,
                        "entry_fill_price": entry_price,
                        "gross_quantity_before_fee": gross_quantity,
                        "quantity_after_buy_fee": quantity,
                        "entry_fee": buy_fee_quantity * open_price,
                        "entry_slippage_cost": (entry_price - open_price) * gross_quantity,
                    }
                )
                if cash < -1e-9:
                    raise AssertionError("Backtest cash became negative after entry")

        # Measure exposure at candle open; intrabar exit timing is unknown from OHLC.
        if position is not None:
            invested_bar_count += 1
            open_value = position.quantity * open_price
            open_equity = cash + open_value
            exposure_percent = open_value / open_equity * 100 if open_equity else 0.0
            capital_exposures.append(exposure_percent)
            exposure_samples.append(
                {
                    "time_invested": True,
                    "capital_exposure_percent": exposure_percent,
                    "capital_exposure_value": open_value,
                }
            )
        else:
            capital_exposures.append(0.0)
            exposure_samples.append(
                {
                    "time_invested": False,
                    "capital_exposure_percent": 0.0,
                    "capital_exposure_value": 0.0,
                }
            )

        if position is not None and not exited_at_open:
            stop_price = (
                position.entry_price * (1 - stop_loss_percent)
                if stop_loss_percent is not None
                else None
            )
            take_profit_price = (
                position.entry_price * (1 + take_profit_percent)
                if take_profit_percent is not None
                else None
            )
            low = float(candle["low"])
            high = float(candle["high"])
            stop_hit = stop_price is not None and low <= stop_price
            take_profit_hit = take_profit_price is not None and high >= take_profit_price

            # If OHLC cannot reveal which threshold came first, assume stop-loss first.
            if stop_hit and stop_price is not None:
                close_position(min(open_price, stop_price), timestamp, "stop_loss")
            elif take_profit_hit and take_profit_price is not None:
                close_position(
                    max(open_price, take_profit_price), timestamp, "take_profit"
                )

        if candle_index == len(bars) - 1 and position is not None and liquidate_at_end:
            close_position(
                float(candle["close"]), timestamp + bar_duration, "end_of_backtest"
            )
            equity_values.append(cash)
        else:
            marked_equity = cash
            if position is not None:
                marked_equity += position.quantity * float(candle["close"])
            equity_values.append(marked_equity)
        equity_times.append(timestamp + bar_duration)

    equity_curve = pd.Series(equity_values, index=pd.DatetimeIndex(equity_times))
    daily_equity = equity_curve.resample("1D").last().dropna()
    daily_returns = daily_equity.pct_change().dropna()
    daily_return_observations = len(daily_returns)
    if len(daily_returns) >= 2 and daily_returns.std(ddof=1) > 0:
        daily_sharpe = (
            daily_returns.mean() / daily_returns.std(ddof=1) * math.sqrt(365)
        )
    else:
        daily_sharpe = None

    peaks = equity_curve.cummax()
    drawdowns = (equity_curve - peaks) / peaks
    max_drawdown = abs(float(drawdowns.min())) * 100

    winning_trades = [trade for trade in completed_trades if trade["net_pnl"] > 0]
    losing_trades = [trade for trade in completed_trades if trade["net_pnl"] < 0]
    net_pnls = [trade["net_pnl"] for trade in completed_trades]
    net_profit_wins = sum(trade["net_pnl"] for trade in winning_trades)
    net_profit_losses = abs(sum(trade["net_pnl"] for trade in losing_trades))
    gross_profit_wins = sum(
        trade["gross_pnl"] for trade in completed_trades if trade["gross_pnl"] > 0
    )
    gross_profit_losses = abs(
        sum(
            trade["gross_pnl"]
            for trade in completed_trades
            if trade["gross_pnl"] < 0
        )
    )
    net_profit_factor = (
        net_profit_wins / net_profit_losses if net_profit_losses else None
    )
    gross_profit_factor = (
        gross_profit_wins / gross_profit_losses if gross_profit_losses else None
    )
    total_gross_pnl = sum(trade["gross_pnl"] for trade in completed_trades)
    total_fees = sum(trade["fees"] for trade in completed_trades)
    total_slippage = sum(trade["slippage_cost"] for trade in completed_trades)
    first_open = float(test_bars.iloc[0]["open"])
    last_close = float(test_bars.iloc[-1]["close"])
    if position is not None and not liquidate_at_end:
        total_gross_pnl += (last_close - position.entry_market_price) * position.quantity
        total_fees += position.buy_fee_quantity * position.entry_market_price
        total_slippage += position.entry_slippage_cost
    total_costs = total_fees + total_slippage

    full_buy_fill = first_open * (1 + slippage)
    full_buy_gross_quantity = starting_capital / full_buy_fill
    full_buy_quantity = full_buy_gross_quantity * (1 - fee_rate)
    full_sell_fill = last_close * (1 - slippage)
    full_buy_hold_ending = full_buy_quantity * full_sell_fill * (1 - fee_rate)

    same_notional = min(
        trade_config.TRADE_AMOUNT_USD,
        trade_config.MAX_POSITION_USD,
        starting_capital,
    )
    same_exposure_buy_fill = first_open * (1 + slippage)
    same_notional_gross_quantity = same_notional / same_exposure_buy_fill
    same_exposure_quantity = same_notional_gross_quantity * (1 - fee_rate)
    same_exposure_cash = starting_capital - same_notional
    same_exposure_sell_fill = last_close * (1 - slippage)
    same_notional_ending = (
        same_exposure_cash
        + same_exposure_quantity * same_exposure_sell_fill * (1 - fee_rate)
    )

    ending_cash = cash
    ending_equity = ending_cash + (
        position.quantity * last_close if position is not None else 0.0
    )
    ending_capital = ending_equity
    strategy_return = (ending_capital / starting_capital - 1) * 100
    same_path_zero_cost_pnl_total = sum(
        trade["same_path_zero_cost_pnl"] for trade in completed_trades
    )
    if position is not None and not liquidate_at_end and position.entry_market_price:
        same_path_zero_cost_pnl_total += (
            position.requested_notional
            / position.entry_market_price
            * (last_close - position.entry_market_price)
        )
    same_path_zero_cost_return = same_path_zero_cost_pnl_total / starting_capital * 100
    full_buy_hold_return = (full_buy_hold_ending / starting_capital - 1) * 100
    same_notional_return = (same_notional_ending / starting_capital - 1) * 100
    raw_market_return = (last_close / first_open - 1) * 100 if first_open else None
    trades_frame = pd.DataFrame(
        completed_trades,
        columns=[
            "entry_time",
            "entry_price",
            "entry_fill_price",
            "entry_market_price",
            "exit_time",
            "exit_price",
            "exit_fill_price",
            "exit_market_price",
            "requested_notional",
            "gross_quantity_before_fee",
            "quantity_after_buy_fee",
            "quantity",
            "entry_reason",
            "exit_reason",
            "gross_pnl",
            "entry_fee",
            "exit_fee",
            "fees",
            "entry_slippage_cost",
            "exit_slippage_cost",
            "slippage_cost",
            "net_pnl",
            "same_path_zero_cost_quantity",
            "same_path_zero_cost_pnl",
            "gross_return_percent",
            "return_percent",
            "holding_duration_minutes",
        ],
    )
    exposure_curve = pd.DataFrame(
        exposure_samples,
        index=pd.DatetimeIndex(test_bars.index),
        columns=[
            "time_invested",
            "capital_exposure_percent",
            "capital_exposure_value",
        ],
    )
    durations = [trade["holding_duration_minutes"] for trade in completed_trades]
    invested_bar_count = min(invested_bar_count, len(test_bars))
    measured_bars = max(len(test_bars), 1)

    available_pretest_bars = test_start_index
    required_strategy_warmup_bars = required_warmup_bars(strategy=selected_strategy)
    entry_events_frame = pd.DataFrame(
        entry_events,
        columns=[
            "entry_time",
            "requested_notional",
            "entry_market_price",
            "entry_fill_price",
            "gross_quantity_before_fee",
            "quantity_after_buy_fee",
            "entry_fee",
            "entry_slippage_cost",
        ],
    )
    return {
        "strategy_name": selected_strategy.name,
        "strategy_parameters": selected_strategy.parameters,
        "stop_loss_percent": stop_loss_percent,
        "take_profit_percent": take_profit_percent,
        "start_time": test_bars.index[0],
        "end_time": test_bars.index[-1] + bar_duration,
        # Keep warmup_bars as a compatibility alias for available pre-test bars.
        "warmup_bars": available_pretest_bars,
        "required_warmup_bars": required_strategy_warmup_bars,
        "available_pretest_bars": available_pretest_bars,
        "starting_capital": starting_capital,
        "ending_capital": ending_capital,
        "ending_cash": ending_cash,
        "ending_equity": ending_equity,
        "net_profit": ending_capital - starting_capital,
        "strategy_return": strategy_return,
        "same_path_zero_cost_pnl_total": same_path_zero_cost_pnl_total,
        "same_path_zero_cost_return_percent": same_path_zero_cost_return,
        "pure_cost_drag_percent": same_path_zero_cost_return - strategy_return,
        "trades": trades_frame,
        "entry_events": entry_events_frame,
        "open_position": (
            {
                "entry_time": position.entry_time,
                "entry_market_price": position.entry_market_price,
                "entry_fill_price": position.entry_price,
                "requested_notional": position.requested_notional,
                "gross_quantity_before_fee": position.gross_quantity,
                "quantity_after_buy_fee": position.quantity,
                "entry_reason": position.entry_reason,
            }
            if position is not None
            else None
        ),
        "total_trades": len(completed_trades),
        "winning_trades": len(winning_trades),
        "losing_trades": len(losing_trades),
        "win_rate": (
            len(winning_trades) / len(completed_trades) * 100
            if completed_trades
            else None
        ),
        "average_winning_trade": (
            sum(trade["net_pnl"] for trade in winning_trades) / len(winning_trades)
            if winning_trades
            else None
        ),
        "average_losing_trade": (
            sum(trade["net_pnl"] for trade in losing_trades) / len(losing_trades)
            if losing_trades
            else None
        ),
        "average_pnl_per_trade": (
            sum(net_pnls) / len(net_pnls) if net_pnls else None
        ),
        "net_profit_factor": net_profit_factor,
        "gross_profit_factor": gross_profit_factor,
        # Compatibility alias: profit_factor has always used net trade PnL.
        "profit_factor": net_profit_factor,
        "max_drawdown": max_drawdown,
        "candle_mark_max_drawdown_percent": max_drawdown,
        "daily_sharpe": float(daily_sharpe) if daily_sharpe is not None else None,
        "equity_curve": equity_curve,
        "exposure_curve": exposure_curve,
        "full_buy_hold_ending": full_buy_hold_ending,
        "full_buy_hold_return": full_buy_hold_return,
        "same_notional_ending": same_notional_ending,
        "same_notional_return": same_notional_return,
        "raw_market_return_percent": raw_market_return,
        "same_notional": same_notional,
        "gross_pnl": total_gross_pnl,
        "total_fees": total_fees,
        "total_slippage": total_slippage,
        "total_costs": total_costs,
        "cost_percent_of_gross_profit": (
            total_costs / total_gross_pnl * 100 if total_gross_pnl > 1e-12 else None
        ),
        "cost_percent_of_abs_gross_pnl": (
            total_costs / abs(total_gross_pnl) * 100
            if total_gross_pnl <= 0 and abs(total_gross_pnl) > 1e-12
            else None
        ),
        "daily_return_observations": daily_return_observations,
        "exit_reasons": tuple(
            dict.fromkeys(trade["exit_reason"] for trade in completed_trades)
        ),
        "average_holding_minutes": sum(durations) / len(durations) if durations else None,
        "median_holding_minutes": (
            float(pd.Series(durations).median()) if durations else None
        ),
        "shortest_holding_minutes": min(durations) if durations else None,
        "longest_holding_minutes": max(durations) if durations else None,
        "average_gross_return": (
            sum(trade["gross_return_percent"] for trade in completed_trades) / len(completed_trades)
            if completed_trades
            else None
        ),
        "average_net_return": (
            sum(trade["return_percent"] for trade in completed_trades) / len(completed_trades)
            if completed_trades
            else None
        ),
        "time_invested_percent": invested_bar_count / measured_bars * 100,
        "time_in_cash_percent": 100 - invested_bar_count / measured_bars * 100,
        "average_capital_exposure": (
            sum(capital_exposures) / len(capital_exposures)
            if capital_exposures
            else 0.0
        ),
        "maximum_capital_exposure": max(capital_exposures, default=0.0),
        "minutes_per_bar": minutes_per_bar,
    }


def _format_money(value: Optional[float]) -> str:
    return "N/A" if value is None else f"${value:,.2f}"


def _format_percent(value: Optional[float]) -> str:
    return "N/A" if value is None else f"{value:.2f}%"


def _format_duration(minutes: Optional[float]) -> str:
    if minutes is None:
        return "N/A"
    hours, remaining_minutes = divmod(minutes, 60)
    return f"{int(hours)}h {remaining_minutes:.1f}m"


def _exit_reason_rows(result: dict[str, Any]) -> None:
    trades = result["trades"]
    print("\nExit reason breakdown:")
    print("reason              count  wins  losses  win rate   gross PnL    net PnL")
    for reason in result["exit_reasons"]:
        rows = trades[trades["exit_reason"] == reason]
        pnl = rows["net_pnl"].tolist()
        count = len(rows)
        wins = sum(value > 0 for value in pnl)
        losses = sum(value < 0 for value in pnl)
        win_rate = f"{wins / count * 100:7.2f}%" if count else "     N/A"
        gross = rows["gross_pnl"].sum() if count else 0.0
        net = rows["net_pnl"].sum() if count else 0.0
        print(
            f"{reason:<19} {count:>5} {wins:>5} {losses:>7} {win_rate:>9} "
            f"{gross:>11.2f} {net:>10.2f}"
        )


def print_report(result: dict[str, Any], title=None) -> None:
    title = title or f"{config.SYMBOL} {result.get('strategy_name', 'BACKTEST')} BACKTEST"
    print(f"\n=== {title} ===")
    print(f"Period: {result['start_time']} to {result['end_time']}")
    print(f"Required warm-up bars: {result.get('required_warmup_bars', result['warmup_bars'])}")
    print(
        "Available pre-test bars excluded from metrics: "
        f"{result.get('available_pretest_bars', result['warmup_bars'])}"
    )
    print(f"Starting capital: {_format_money(result['starting_capital'])}")
    print(f"Ending capital: {_format_money(result['ending_capital'])}")
    print(f"Net profit: {_format_money(result['net_profit'])}")
    print(f"Total return: {_format_percent(result['strategy_return'])}")
    print(f"Total trades: {result['total_trades']}")
    print(f"Winning trades: {result['winning_trades']}")
    print(f"Losing trades: {result['losing_trades']}")
    print(f"Win rate: {_format_percent(result['win_rate'])}")
    print(f"Average winning trade: {_format_money(result['average_winning_trade'])}")
    print(f"Average losing trade: {_format_money(result['average_losing_trade'])}")
    print(f"Average PnL per trade: {_format_money(result['average_pnl_per_trade'])}")
    factor = result.get("net_profit_factor", result["profit_factor"])
    print("Net profit factor: N/A" if factor is None else f"Net profit factor: {factor:.3f}")
    gross_factor = result.get("gross_profit_factor")
    print(
        "Gross profit factor: N/A"
        if gross_factor is None
        else f"Gross profit factor: {gross_factor:.3f}"
    )
    print(
        "Candle-mark maximum drawdown (not tick-level worst case): "
        f"{_format_percent(result['candle_mark_max_drawdown_percent'])}"
    )
    sharpe = result["daily_sharpe"]
    print("Daily Sharpe ratio: N/A" if sharpe is None else f"Daily Sharpe ratio: {sharpe:.3f}")

    print("\nBenchmarks:")
    print(f"Raw BTC market return: {_format_percent(result['raw_market_return_percent'])}")
    print(
        "Full-capital buy & hold (all starting capital): "
        f"{_format_percent(result['full_buy_hold_return'])}"
    )
    print(
        f"Same-notional BTC buy & hold ({_format_money(result['same_notional'])} "
        f"invested, rest in cash): {_format_percent(result['same_notional_return'])}"
    )
    print(f"Strategy return: {_format_percent(result['strategy_return'])}")

    print("\nCosts:")
    print(f"Gross strategy PnL before costs: {_format_money(result['gross_pnl'])}")
    print(f"Total entry + exit fees: {_format_money(result['total_fees'])}")
    print(f"Estimated total slippage cost: {_format_money(result['total_slippage'])}")
    print(f"Net PnL: {_format_money(result['net_profit'])}")
    if result["gross_pnl"] > 0:
        print(
            "Costs as % of gross profit: "
            f"{_format_percent(result['cost_percent_of_gross_profit'])}"
        )
    else:
        print(
            "Costs as % of absolute gross PnL: "
            f"{_format_percent(result['cost_percent_of_abs_gross_pnl'])}"
        )

    print("\nTrade diagnostics:")
    print(f"Average holding duration: {_format_duration(result['average_holding_minutes'])}")
    print(f"Median holding duration: {_format_duration(result['median_holding_minutes'])}")
    print(f"Shortest holding duration: {_format_duration(result['shortest_holding_minutes'])}")
    print(f"Longest holding duration: {_format_duration(result['longest_holding_minutes'])}")
    print(f"Average gross return per trade: {_format_percent(result['average_gross_return'])}")
    print(f"Average net return per trade: {_format_percent(result['average_net_return'])}")
    print(f"Percentage of time invested: {_format_percent(result['time_invested_percent'])}")
    print(f"Percentage of time in cash: {_format_percent(result['time_in_cash_percent'])}")
    print(f"Average capital exposure: {_format_percent(result['average_capital_exposure'])}")
    print(f"Maximum capital exposure: {_format_percent(result['maximum_capital_exposure'])}")
    print("Exposure is sampled at each candle open; intrabar holding time is approximate.")

    _exit_reason_rows(result)
    print(
        "Daily Sharpe uses daily closing equity returns, a zero risk-free rate, "
        "and sqrt(365) annualization for 24/7 BTC trading."
    )
    print(f"Daily return observations: {result['daily_return_observations']}")
    if result["daily_return_observations"] < 60:
        print("WARNING: Daily Sharpe is based on a short sample and is not statistically reliable.")
    if result.get("stop_loss_percent") is not None or result.get("take_profit_percent") is not None:
        print(
            "Fixed risk exits use intrabar OHLC threshold detection. Live bot checks latest "
            "market price on each poll, so risk-exit execution may differ between backtest and live."
        )
    else:
        print("This strategy has no fixed stop-loss or take-profit; its signal defines exits.")
    print(
        "Accounting assumption: BUY fees are withheld in BTC and valued at entry market price; "
        "SELL fees are deducted from USD proceeds. Entry cash decreases by requested notional."
    )


def print_zero_cost_diagnostic(result: dict[str, Any]) -> None:
    print("\n=== ZERO-COST DIAGNOSTIC (not the main result) ===")
    print(f"Ending capital: {_format_money(result['ending_capital'])}")
    print(f"Net PnL: {_format_money(result['net_profit'])}")
    print(f"Return: {_format_percent(result['strategy_return'])}")
    print(f"Completed trades: {result['total_trades']}")
    print(f"Win rate: {_format_percent(result['win_rate'])}")
    print("Fees and slippage are both set to zero for this diagnostic only.")


def _run_main() -> int:
    parser = argparse.ArgumentParser(
        description=f"Backtest the {config.SYMBOL} MA/RSI strategy"
    )
    parser.add_argument(
        "--starting-capital",
        type=float,
        default=config.BACKTEST_STARTING_CAPITAL,
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=config.BACKTEST_LOOKBACK_DAYS,
    )
    parser.add_argument("--timeframe", default=trade_config.LIVE_TIMEFRAME)
    parser.add_argument(
        "--zero-cost-diagnostic",
        action="store_true",
        default=config.BACKTEST_ZERO_COST_DIAGNOSTIC,
        help="Also run the same strategy with zero fees and slippage",
    )
    parser.add_argument("--no-csv", action="store_true", help="Do not save trade history")
    args = parser.parse_args()
    if args.lookback_days <= 0:
        parser.error("--lookback-days must be positive")

    try:
        bars = fetch_history(args.lookback_days, args.timeframe)
        test_start = bars.attrs.get("test_start")
        result = run_backtest(
            bars, args.starting_capital, args.timeframe, test_start=test_start
        )
        zero_cost_result = (
            run_backtest(
                bars,
                args.starting_capital,
                args.timeframe,
                fee_rate=0,
                slippage=0,
                test_start=test_start,
            )
            if args.zero_cost_diagnostic
            else None
        )
    except Exception as error:
        print(f"BACKTEST ERROR: {error}")
        return 1

    print_report(result)
    if zero_cost_result is not None:
        print_zero_cost_diagnostic(zero_cost_result)
    if config.BACKTEST_SAVE_TRADES_CSV and not args.no_csv:
        output_path = Path("backtest_trades.csv")
        result["trades"].to_csv(output_path, index=False)
        print(f"Trade history saved to {output_path.resolve()}")
    return 0


def main() -> int:
    output = io.StringIO()
    exit_code = 0
    try:
        with redirect_stdout(output), redirect_stderr(output):
            exit_code = _run_main()
    except SystemExit as error:
        exit_code = error.code if isinstance(error.code, int) else 1
    finally:
        captured = output.getvalue()
        try:
            Path("last_backtest_output.txt").write_text(captured, encoding="utf-8")
        except OSError as error:
            print(f"Could not save backtest output: {error}", file=sys.stderr)
        sys.stdout.write(captured)
        sys.stdout.flush()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
