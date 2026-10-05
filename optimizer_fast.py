"""Array based CPU kernel for the MA/RSI research optimizer.

The ordinary backtester remains the reference implementation.  This module
contains an optimizer-specific equivalent loop over contiguous NumPy arrays;
Numba is used when installed, with a compatible Python fallback otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

try:  # Numba is an optional accelerator, not a project requirement.
    from numba import njit

    _NUMBA_NJIT = njit
    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - exercised when numba is missing or unusable
    _NUMBA_NJIT = None
    _KERNEL_JIT = None
    NUMBA_AVAILABLE = False

_KERNEL_JIT = None


# Trade matrix columns returned from the numerical loop.
ENTRY_NS = 0
EXIT_NS = 1
EXIT_REASON = 2
REQUESTED_NOTIONAL = 3
GROSS_QUANTITY = 4
QUANTITY = 5
ENTRY_MARKET = 6
ENTRY_FILL = 7
EXIT_MARKET = 8
EXIT_FILL = 9
GROSS_PNL = 10
ENTRY_FEE = 11
EXIT_FEE = 12
ENTRY_SLIPPAGE = 13
EXIT_SLIPPAGE = 14
NET_PNL = 15
NET_RETURN = 16
GROSS_RETURN = 17
HOLDING_MINUTES = 18
SAME_PATH_PNL = 19
TRADE_WIDTH = 20

REASON_NAMES = {
    1: "stop_loss",
    2: "take_profit",
    3: "bearish_ma_crossover",
    4: "end_of_backtest",
}


@dataclass(frozen=True)
class FastBars:
    timestamps_ns: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    index: pd.DatetimeIndex

    @classmethod
    def from_frame(cls, bars: pd.DataFrame) -> "FastBars":
        index = pd.DatetimeIndex(pd.to_datetime(bars.index, utc=True))
        return cls(
            np.ascontiguousarray(index.as_unit("ns").asi8, dtype=np.int64),
            np.ascontiguousarray(pd.to_numeric(bars["open"]).to_numpy(), dtype=np.float64),
            np.ascontiguousarray(pd.to_numeric(bars["high"]).to_numpy(), dtype=np.float64),
            np.ascontiguousarray(pd.to_numeric(bars["low"]).to_numpy(), dtype=np.float64),
            np.ascontiguousarray(pd.to_numeric(bars["close"]).to_numpy(), dtype=np.float64),
            index,
        )


@dataclass(frozen=True)
class FastPrepared:
    bars: FastBars
    ma_fast: np.ndarray
    ma_slow: np.ndarray
    rsi: np.ndarray

    @classmethod
    def from_frame(
        cls,
        bars: FastBars,
        prepared: pd.DataFrame,
    ) -> "FastPrepared":
        return cls(
            bars,
            np.ascontiguousarray(prepared["ma_fast"].to_numpy(), dtype=np.float64),
            np.ascontiguousarray(prepared["ma_slow"].to_numpy(), dtype=np.float64),
            np.ascontiguousarray(prepared["rsi"].to_numpy(), dtype=np.float64),
        )


def _close_position(
    cash: float,
    exit_market_price: float,
    exit_ns: int,
    entry_index: int,
    exit_index: float,
    reason: int,
    entry_ns: int,
    entry_market_price: float,
    entry_price: float,
    requested_notional: float,
    gross_quantity: float,
    quantity: float,
    buy_fee_quantity: float,
    entry_slippage_cost: float,
    fee_rate: float,
    slippage: float,
) -> tuple[float, np.ndarray]:
    exit_price = exit_market_price * (1.0 - slippage)
    exit_notional = exit_price * quantity
    exit_fee = exit_notional * fee_rate
    exit_slippage_cost = (exit_market_price - exit_price) * quantity
    cash += exit_notional - exit_fee
    gross_pnl = (exit_market_price - entry_market_price) * quantity
    entry_fee = buy_fee_quantity * entry_market_price
    fees = entry_fee + exit_fee
    slippage_cost = entry_slippage_cost + exit_slippage_cost
    net_pnl = gross_pnl - fees - slippage_cost
    gross_return = (
        gross_pnl / (entry_market_price * quantity) * 100.0 if quantity else 0.0
    )
    net_return = net_pnl / requested_notional * 100.0 if requested_notional else 0.0
    same_path_pnl = (
        requested_notional / entry_market_price * (exit_market_price - entry_market_price)
        if entry_market_price
        else 0.0
    )
    holding_minutes = (exit_ns - entry_ns) / 60_000_000_000.0
    row = np.empty(TRADE_WIDTH, dtype=np.float64)
    # Store row-relative candle indexes in float slots; epoch nanoseconds lose
    # precision in float64. Exact nanosecond timestamps are used for duration.
    row[ENTRY_NS] = entry_index
    row[EXIT_NS] = exit_index
    row[EXIT_REASON] = reason
    row[REQUESTED_NOTIONAL] = requested_notional
    row[GROSS_QUANTITY] = gross_quantity
    row[QUANTITY] = quantity
    row[ENTRY_MARKET] = entry_market_price
    row[ENTRY_FILL] = entry_price
    row[EXIT_MARKET] = exit_market_price
    row[EXIT_FILL] = exit_price
    row[GROSS_PNL] = gross_pnl
    row[ENTRY_FEE] = entry_fee
    row[EXIT_FEE] = exit_fee
    row[ENTRY_SLIPPAGE] = entry_slippage_cost
    row[EXIT_SLIPPAGE] = exit_slippage_cost
    row[NET_PNL] = net_pnl
    row[NET_RETURN] = net_return
    row[GROSS_RETURN] = gross_return
    row[HOLDING_MINUTES] = holding_minutes
    row[SAME_PATH_PNL] = same_path_pnl
    return cash, row


if NUMBA_AVAILABLE:
    _close_position = _NUMBA_NJIT(cache=True)(_close_position)


def _simulate_kernel(
    timestamps_ns: np.ndarray,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    ma_fast: np.ndarray,
    ma_slow: np.ndarray,
    rsi: np.ndarray,
    test_start_index: int,
    test_end_index: int,
    bar_minutes: float,
    rsi_threshold: float,
    stop_loss_percent: float,
    take_profit_percent: float,
    fee_rate: float,
    slippage: float,
    starting_capital: float,
    trade_amount: float,
    max_position: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate one candidate over [test_start_index, test_end_index)."""
    test_length = test_end_index - test_start_index
    trades = np.empty((test_length, TRADE_WIDTH), dtype=np.float64)
    equity = np.empty(test_length + 1, dtype=np.float64)
    equity[0] = starting_capital
    trade_count = 0
    invested_count = 0
    cash = starting_capital
    position_open = False
    entry_ns = 0
    entry_index = 0
    entry_market = 0.0
    entry_fill = 0.0
    requested_notional = 0.0
    gross_quantity = 0.0
    quantity = 0.0
    buy_fee_quantity = 0.0
    entry_slippage_cost = 0.0
    bar_ns = int(bar_minutes * 60_000_000_000.0)

    for candle_index in range(test_start_index, test_end_index):
        open_price = opens[candle_index]
        if candle_index == test_start_index:
            action = 0  # Each full run and segment begins flat without a carried signal.
        else:
            signal_index = candle_index - 1
            if signal_index < 1:
                action = 0
            else:
                previous_fast = ma_fast[signal_index - 1]
                previous_slow = ma_slow[signal_index - 1]
                current_fast = ma_fast[signal_index]
                current_slow = ma_slow[signal_index]
                current_rsi = rsi[signal_index]
                if not (
                    np.isfinite(previous_fast)
                    and np.isfinite(previous_slow)
                    and np.isfinite(current_fast)
                    and np.isfinite(current_slow)
                    and np.isfinite(current_rsi)
                ):
                    action = 0
                elif previous_fast <= previous_slow and current_fast > current_slow:
                    action = 1 if current_rsi < rsi_threshold else 0
                elif previous_fast >= previous_slow and current_fast < current_slow:
                    action = 2
                else:
                    action = 0

        exited_at_open = False
        if position_open:
            stop_at_open = entry_fill * (1.0 - stop_loss_percent)
            take_at_open = entry_fill * (1.0 + take_profit_percent)
            if open_price <= stop_at_open:
                cash, trades[trade_count] = _close_position(
                    cash, open_price, timestamps_ns[candle_index], entry_index, candle_index, 1, entry_ns,
                    entry_market, entry_fill, requested_notional, gross_quantity,
                    quantity, buy_fee_quantity, entry_slippage_cost, fee_rate, slippage,
                )
                trade_count += 1
                position_open = False
                exited_at_open = True
            elif open_price >= take_at_open:
                cash, trades[trade_count] = _close_position(
                    cash, open_price, timestamps_ns[candle_index], entry_index, candle_index, 2, entry_ns,
                    entry_market, entry_fill, requested_notional, gross_quantity,
                    quantity, buy_fee_quantity, entry_slippage_cost, fee_rate, slippage,
                )
                trade_count += 1
                position_open = False
                exited_at_open = True
            elif action == 2:
                cash, trades[trade_count] = _close_position(
                    cash, open_price, timestamps_ns[candle_index], entry_index, candle_index, 3, entry_ns,
                    entry_market, entry_fill, requested_notional, gross_quantity,
                    quantity, buy_fee_quantity, entry_slippage_cost, fee_rate, slippage,
                )
                trade_count += 1
                position_open = False
                exited_at_open = True
        elif action == 1:
            requested_notional = min(trade_amount, max_position)
            entry_fill = open_price * (1.0 + slippage)
            gross_quantity = requested_notional / entry_fill if entry_fill else 0.0
            buy_fee_quantity = gross_quantity * fee_rate
            quantity = gross_quantity - buy_fee_quantity
            if quantity > 0.0 and requested_notional <= cash:
                cash -= requested_notional
                position_open = True
                entry_ns = timestamps_ns[candle_index]
                entry_index = candle_index
                entry_market = open_price
                entry_slippage_cost = (entry_fill - open_price) * gross_quantity

        # Match the reference's exposure sampling before intrabar exits.
        if position_open:
            invested_count += 1

        if position_open and not exited_at_open:
            stop_price = entry_fill * (1.0 - stop_loss_percent)
            take_price = entry_fill * (1.0 + take_profit_percent)
            stop_hit = lows[candle_index] <= stop_price
            take_hit = highs[candle_index] >= take_price
            if stop_hit:
                exit_market = min(open_price, stop_price)
                cash, trades[trade_count] = _close_position(
                    cash, exit_market, timestamps_ns[candle_index], entry_index, candle_index, 1, entry_ns,
                    entry_market, entry_fill, requested_notional, gross_quantity,
                    quantity, buy_fee_quantity, entry_slippage_cost, fee_rate, slippage,
                )
                trade_count += 1
                position_open = False
            elif take_hit:
                exit_market = max(open_price, take_price)
                cash, trades[trade_count] = _close_position(
                    cash, exit_market, timestamps_ns[candle_index], entry_index, candle_index, 2, entry_ns,
                    entry_market, entry_fill, requested_notional, gross_quantity,
                    quantity, buy_fee_quantity, entry_slippage_cost, fee_rate, slippage,
                )
                trade_count += 1
                position_open = False

        is_last = candle_index == test_end_index - 1
        if is_last and position_open:
            exit_ns = timestamps_ns[candle_index] + bar_ns
            cash, trades[trade_count] = _close_position(
                cash, closes[candle_index], exit_ns, entry_index, candle_index + 1.0, 4, entry_ns,
                entry_market, entry_fill, requested_notional, gross_quantity,
                quantity, buy_fee_quantity, entry_slippage_cost, fee_rate, slippage,
            )
            trade_count += 1
            position_open = False

        marked_equity = cash + (quantity * closes[candle_index] if position_open else 0.0)
        equity[candle_index - test_start_index + 1] = marked_equity

    return trades[:trade_count], equity, np.asarray(
        [invested_count, test_length], dtype=np.int64
    )


def _python_simulate(*args: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _simulate_kernel(*args)


def _numba_simulate(*args: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not NUMBA_AVAILABLE:
        return _python_simulate(*args)
    global _KERNEL_JIT
    if _KERNEL_JIT is None:
        _KERNEL_JIT = _NUMBA_NJIT(cache=True)(_simulate_kernel)
    return _KERNEL_JIT(*args)


def simulate(
    prepared: FastPrepared,
    *,
    start_time: Any,
    end_time: Any,
    bar_minutes: float,
    rsi_threshold: float,
    stop_loss_percent: float,
    take_profit_percent: float,
    fee_rate: float,
    slippage: float,
    starting_capital: float,
    trade_amount: float,
    max_position: float,
    use_numba: bool = True,
) -> dict[str, Any]:
    bars = prepared.bars
    start_ns = pd.Timestamp(start_time)
    start_ns = start_ns.tz_localize("UTC") if start_ns.tzinfo is None else start_ns.tz_convert("UTC")
    end_ns = pd.Timestamp(end_time)
    end_ns = end_ns.tz_localize("UTC") if end_ns.tzinfo is None else end_ns.tz_convert("UTC")
    start_index = int(np.searchsorted(bars.timestamps_ns, start_ns.value, side="left"))
    end_index = int(np.searchsorted(bars.timestamps_ns, end_ns.value, side="left"))
    if end_index - start_index < 2:
        raise ValueError("At least two candles must be available in the fast backtest period")
    kernel = _numba_simulate if use_numba else _python_simulate
    trades, equity, exposure_counts = kernel(
        bars.timestamps_ns, bars.open, bars.high, bars.low, bars.close,
        prepared.ma_fast, prepared.ma_slow, prepared.rsi,
        start_index, end_index, bar_minutes, rsi_threshold, stop_loss_percent,
        take_profit_percent, fee_rate, slippage, starting_capital,
        trade_amount, max_position,
    )
    trade_frame = _trade_frame(trades, bars.index, bar_minutes)
    wins = trade_frame.loc[trade_frame["net_pnl"] > 0, "net_pnl"]
    losses = trade_frame.loc[trade_frame["net_pnl"] < 0, "net_pnl"]
    gross_wins = trade_frame.loc[trade_frame["gross_pnl"] > 0, "gross_pnl"]
    gross_losses = trade_frame.loc[trade_frame["gross_pnl"] < 0, "gross_pnl"]
    net_profit_factor = float(wins.sum() / abs(losses.sum())) if len(losses) else None
    gross_profit_factor = float(gross_wins.sum() / abs(gross_losses.sum())) if len(gross_losses) else None
    peaks = np.maximum.accumulate(equity)
    drawdowns = np.divide(equity - peaks, peaks, out=np.zeros_like(equity), where=peaks != 0)
    max_drawdown = abs(float(drawdowns.min())) * 100.0
    total_fees = float(trade_frame["fees"].sum())
    total_slippage = float(trade_frame["slippage_cost"].sum())
    same_path_pnl = float(trade_frame["same_path_zero_cost_pnl"].sum())
    ending_equity = float(equity[-1])
    strategy_return = (ending_equity / starting_capital - 1.0) * 100.0
    same_path_return = same_path_pnl / starting_capital * 100.0
    total_trades = len(trade_frame)
    return {
        "trades": trade_frame,
        "total_trades": total_trades,
        "winning_trades": int((trade_frame["net_pnl"] > 0).sum()),
        "losing_trades": int((trade_frame["net_pnl"] < 0).sum()),
        "win_rate": float((trade_frame["net_pnl"] > 0).sum() / total_trades * 100.0) if total_trades else None,
        "net_profit_factor": net_profit_factor,
        "gross_profit_factor": gross_profit_factor,
        "profit_factor": net_profit_factor,
        "ending_equity": ending_equity,
        "strategy_return": strategy_return,
        "same_path_zero_cost_return_percent": same_path_return,
        "pure_cost_drag_percent": same_path_return - strategy_return,
        "total_fees": total_fees,
        "total_slippage": total_slippage,
        "total_costs": total_fees + total_slippage,
        "max_drawdown": max_drawdown,
        "average_holding_minutes": float(trade_frame["holding_duration_minutes"].mean()) if total_trades else None,
        "average_net_return": float(trade_frame["return_percent"].mean()) if total_trades else None,
        "average_gross_return": float(trade_frame["gross_return_percent"].mean()) if total_trades else None,
        "time_invested_percent": int(exposure_counts[0]) / max(int(exposure_counts[1]), 1) * 100.0,
    }


def _trade_frame(
    trades: np.ndarray, index: pd.DatetimeIndex, bar_minutes: float
) -> pd.DataFrame:
    columns = [
        "entry_time", "exit_time", "exit_reason", "requested_notional",
        "gross_quantity_before_fee", "quantity_after_buy_fee", "quantity",
        "entry_market_price", "entry_fill_price", "exit_market_price", "exit_fill_price",
        "gross_pnl", "entry_fee", "exit_fee", "fees", "entry_slippage_cost",
        "exit_slippage_cost", "slippage_cost", "net_pnl", "return_percent",
        "gross_return_percent", "holding_duration_minutes", "same_path_zero_cost_pnl",
    ]
    if len(trades) == 0:
        return pd.DataFrame(columns=columns)
    entry_times = [index[int(i)] for i in trades[:, ENTRY_NS]]
    exit_times = [
        index[int(row[EXIT_NS]) - 1] + pd.Timedelta(minutes=bar_minutes)
        if int(row[EXIT_REASON]) == 4
        else index[int(row[EXIT_NS])]
        for row in trades
    ]
    frame = pd.DataFrame(
        {
            "entry_time": entry_times,
            "exit_time": exit_times,
            "exit_reason": [REASON_NAMES[int(reason)] for reason in trades[:, EXIT_REASON]],
            "requested_notional": trades[:, REQUESTED_NOTIONAL],
            "gross_quantity_before_fee": trades[:, GROSS_QUANTITY],
            "quantity_after_buy_fee": trades[:, QUANTITY],
            "quantity": trades[:, QUANTITY],
            "entry_market_price": trades[:, ENTRY_MARKET],
            "entry_fill_price": trades[:, ENTRY_FILL],
            "exit_market_price": trades[:, EXIT_MARKET],
            "exit_fill_price": trades[:, EXIT_FILL],
            "gross_pnl": trades[:, GROSS_PNL],
            "entry_fee": trades[:, ENTRY_FEE],
            "exit_fee": trades[:, EXIT_FEE],
            "fees": trades[:, ENTRY_FEE] + trades[:, EXIT_FEE],
            "entry_slippage_cost": trades[:, ENTRY_SLIPPAGE],
            "exit_slippage_cost": trades[:, EXIT_SLIPPAGE],
            "slippage_cost": trades[:, ENTRY_SLIPPAGE] + trades[:, EXIT_SLIPPAGE],
            "net_pnl": trades[:, NET_PNL],
            "return_percent": trades[:, NET_RETURN],
            "gross_return_percent": trades[:, GROSS_RETURN],
            "holding_duration_minutes": trades[:, HOLDING_MINUTES],
            "same_path_zero_cost_pnl": trades[:, SAME_PATH_PNL],
        }
    )
    return frame[columns]
