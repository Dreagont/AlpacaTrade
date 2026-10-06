"""Focused causal regime-filter research on continuous historical simulations."""

import argparse
import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

import pandas as pd

import backtest
import config
import regime_research as block_tools
from research import align_research_end_time
from strategy import create_regime_only_strategy, get_strategy
from timeframes import parse_timeframe
import trade_config


DEFAULT_BLOCK_DAYS = 90
DEFAULT_BLOCKS = 16
DEFAULT_TIMEFRAME = "4Hour"
STRATEGIES = (
    "ma_rsi_crossover",
    "donchian_breakout",
    "donchian_regime_filter",
    "regime_only_4h",
)

CSV_FIELDS = (
    "row_type",
    "block_index", "block_start", "block_end", "block_days", "timeframe",
    "strategy_name", "strategy_parameters_json", "error",
    "requested_research_time", "aligned_research_end", "fee_profile", "fee_rate", "slippage_rate",
    "starting_capital", "raw_btc_return_percent", "starting_equity", "ending_equity",
    "block_net_return_percent", "total_net_return_percent",
    "block_candle_mark_max_drawdown_percent",
    "net_pnl_usd", "pnl_percent_of_notional", "compounded_trade_return_percent",
    "compounded_trade_return_basis", "max_drawdown_usd",
    "max_drawdown_percent_of_notional", "total_trades", "trade_count_basis",
    "total_costs_usd", "btc_buy_hold_return_percent",
    "btc_buy_hold_return_after_costs_percent", "btc_buy_hold_max_drawdown_percent",
    "btc_buy_hold_costs_usd",
    "filtered_minus_regime_only_percent",
    "entries_in_block", "exits_in_block", "charged_costs_in_block",
    "time_invested_percent", "average_capital_exposure", "open_position_at_end",
    "expected_candle_count", "actual_candle_count", "missing_candle_count",
    "missing_candle_percent", "zero_volume_candle_count", "zero_volume_candle_percent",
    "bullish_bar_count", "non_bullish_bar_count", "bullish_bar_percent",
    "regime_on_transitions", "regime_off_transitions", "entries_in_bullish_regime",
    "exits_due_to_regime_filter", "exits_due_to_donchian_breakdown",
)


def _utc(value):
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def _quality_for_span(bars, timeframe, start, end):
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    span = bars.loc[(bars.index >= start_ts) & (bars.index < end_ts)]
    return backtest.data_quality_diagnostics(
        span, timeframe, start_time=start_ts, end_time=end_ts
    )


def _filter_diagnostics(result, prepared, block, timeframe):
    start, end = pd.Timestamp(block["block_start"]), pd.Timestamp(block["block_end"])
    index = pd.DatetimeIndex(pd.to_datetime(prepared.index, utc=True))
    prepared = prepared.copy()
    prepared.index = index
    block_mask = (index >= start) & (index < end)
    states = prepared["regime_bullish"]
    selected = states.loc[block_mask]
    bullish = selected.eq(True).fillna(False)
    bar_count = len(selected)
    previous = states.shift(1)
    ready_pair = states.notna() & previous.notna()
    on = states.eq(True) & previous.eq(False) & ready_pair
    off = states.eq(False) & previous.eq(True) & ready_pair
    on_transitions = int((on & block_mask).sum())
    off_transitions = int((off & block_mask).sum())

    _, minutes_per_bar = parse_timeframe(timeframe)
    duration = pd.Timedelta(minutes=minutes_per_bar)
    entries = result.get("entry_events", pd.DataFrame())
    entries_in_bullish = 0
    if not entries.empty:
        decision_times = pd.DatetimeIndex(
            pd.to_datetime(entries["entry_time"], utc=True) - duration
        )
        in_block = (decision_times >= start) & (decision_times < end)
        entry_states = states.reindex(decision_times)
        entries_in_bullish = int((in_block & entry_states.eq(True).fillna(False).to_numpy()).sum())

    trades = result.get("trades", pd.DataFrame())
    exits_regime = exits_donchian = 0
    if not trades.empty:
        exit_times = pd.to_datetime(trades["exit_time"], utc=True)
        in_block = (exit_times >= start) & (exit_times < end)
        exit_reasons = trades.loc[in_block, "exit_reason"]
        exits_regime = int((exit_reasons == "regime_filter_off").sum())
        exits_donchian = int((exit_reasons == "donchian_exit_breakdown").sum())

    return {
        "bullish_bar_count": int(bullish.sum()),
        "non_bullish_bar_count": int(bar_count - bullish.sum()),
        "bullish_bar_percent": float(bullish.mean() * 100) if bar_count else 0.0,
        "regime_on_transitions": on_transitions,
        "regime_off_transitions": off_transitions,
        "entries_in_bullish_regime": entries_in_bullish,
        "exits_due_to_regime_filter": exits_regime,
        "exits_due_to_donchian_breakdown": exits_donchian,
    }


def _max_drawdown_usd(equity):
    values = pd.to_numeric(equity, errors="coerce").dropna().astype(float)
    if values.empty:
        return 0.0
    return max(0.0, float((values.cummax() - values).max()))


def _marked_open_position_net_pnl(result, bars, fee_rate, slippage_rate):
    """Return hypothetical after-cost PnL for the open position at final close."""
    position = result.get("open_position")
    if not position:
        return None
    if bars.empty:
        raise ValueError("Cannot mark an open position without final market data")
    final_close = float(bars.sort_index().iloc[-1]["close"])
    quantity = float(position["quantity_after_buy_fee"])
    requested_notional = float(position["requested_notional"])
    exit_fill = final_close * (1 - slippage_rate)
    exit_fee = quantity * exit_fill * fee_rate
    # This cash-flow form includes the entry fee and entry slippage already paid
    # through the fixed requested notional and fee-reduced credited quantity.
    return quantity * exit_fill - exit_fee - requested_notional


def _compounded_trade_return_percent(result, bars, fee_rate, slippage_rate):
    factor = 1.0
    trades = result.get("trades", pd.DataFrame())
    if not trades.empty:
        for _, trade in trades.iterrows():
            notional = float(trade.get("requested_notional") or trade_config.TRADE_AMOUNT_USD)
            factor *= 1 + float(trade["net_pnl"]) / notional
    marked_pnl = _marked_open_position_net_pnl(
        result, bars, fee_rate, slippage_rate
    )
    if marked_pnl is not None:
        position_notional = float(
            result["open_position"].get("requested_notional")
            or trade_config.TRADE_AMOUNT_USD
        )
        factor *= 1 + marked_pnl / position_notional
    return (factor - 1) * 100


def _btc_buy_hold_metrics(bars, fee_rate, slippage_rate):
    if bars.empty:
        return {
            "btc_buy_hold_return_percent": None,
            "btc_buy_hold_return_after_costs_percent": None,
            "btc_buy_hold_max_drawdown_percent": None,
            "btc_buy_hold_costs_usd": None,
        }
    ordered = bars.sort_index()
    first_open = float(ordered.iloc[0]["open"])
    closes = pd.to_numeric(ordered["close"], errors="coerce").dropna()
    last_close = float(closes.iloc[-1]) if not closes.empty else None
    if not first_open or last_close is None:
        raw_return = after_costs = max_drawdown = costs = None
    else:
        raw_return = (last_close / first_open - 1) * 100
        high_water = closes.cummax()
        max_drawdown = abs(float((closes / high_water - 1).min())) * 100
        notional = float(trade_config.TRADE_AMOUNT_USD)
        buy_fill = first_open * (1 + slippage_rate)
        bought_quantity = notional / buy_fill * (1 - fee_rate)
        sell_fill = last_close * (1 - slippage_rate)
        ending_value = bought_quantity * sell_fill * (1 - fee_rate)
        after_costs = (ending_value / notional - 1) * 100
        gross_quantity = notional / buy_fill
        entry_fee = gross_quantity * fee_rate * first_open
        entry_slippage = (buy_fill - first_open) * gross_quantity
        exit_fee = bought_quantity * sell_fill * fee_rate
        exit_slippage = (last_close - sell_fill) * bought_quantity
        costs = entry_fee + entry_slippage + exit_fee + exit_slippage
    return {
        "btc_buy_hold_return_percent": raw_return,
        "btc_buy_hold_return_after_costs_percent": after_costs,
        "btc_buy_hold_max_drawdown_percent": max_drawdown,
        "btc_buy_hold_costs_usd": costs,
    }


def _total_summary(result, bars, fee_rate, slippage_rate):
    exposure = result["exposure_curve"]
    net_pnl = float(result["ending_equity"] - result["starting_capital"])
    equity = result["equity_curve"]
    marked_open_pnl = _marked_open_position_net_pnl(
        result, bars, fee_rate, slippage_rate
    )
    return {
        "starting_equity": result["starting_capital"],
        "ending_equity": result["ending_equity"],
        "total_net_return_percent": result["strategy_return"],
        "same_path_zero_cost_return_percent": result["same_path_zero_cost_return_percent"],
        "net_pnl_usd": net_pnl,
        "pnl_percent_of_notional": net_pnl / trade_config.TRADE_AMOUNT_USD * 100,
        "compounded_trade_return_percent": _compounded_trade_return_percent(
            result, bars, fee_rate, slippage_rate
        ),
        "compounded_trade_return_basis": (
            "closed trades plus final open position marked after hypothetical exit costs"
            if marked_open_pnl is not None
            else "closed trades"
        ),
        "max_drawdown_usd": _max_drawdown_usd(equity),
        "max_drawdown_percent_of_notional": (
            _max_drawdown_usd(equity) / trade_config.TRADE_AMOUNT_USD * 100
        ),
        "total_trades": result["total_trades"],
        "total_costs_usd": result["total_costs"],
        "candle_mark_max_drawdown_percent": result["candle_mark_max_drawdown_percent"],
        "time_invested_percent": float(exposure["time_invested"].mean() * 100),
        "average_capital_exposure": float(exposure["capital_exposure_percent"].mean()),
        "open_position_at_end": result.get("open_position") is not None,
    }


def _block_notional_metrics(result, block_metrics, block):
    equity = result["equity_curve"].copy()
    equity.index = pd.DatetimeIndex(pd.to_datetime(equity.index, utc=True))
    start, end = pd.Timestamp(block["block_start"]), pd.Timestamp(block["block_end"])
    block_equity = equity.loc[(equity.index >= start) & (equity.index <= end)]
    pnl = float(block_metrics["block_net_pnl"])
    drawdown = _max_drawdown_usd(block_equity)
    return {
        "net_pnl_usd": pnl,
        "pnl_percent_of_notional": pnl / trade_config.TRADE_AMOUNT_USD * 100,
        "max_drawdown_usd": drawdown,
        "max_drawdown_percent_of_notional": (
            drawdown / trade_config.TRADE_AMOUNT_USD * 100
        ),
        "total_trades": block_metrics["exits_in_block"],
        "trade_count_basis": "completed exits in block",
        "total_costs_usd": block_metrics["charged_costs_in_block"],
        "time_invested_percent": block_metrics["time_invested_percent"],
    }


def _span_bars(bars, start, end):
    normalized = bars.copy()
    normalized.index = pd.DatetimeIndex(pd.to_datetime(normalized.index, utc=True))
    return normalized.loc[
        (normalized.index >= pd.Timestamp(start))
        & (normalized.index < pd.Timestamp(end))
    ]


def run_regime_filter_research(
    *, block_days=DEFAULT_BLOCK_DAYS, blocks=DEFAULT_BLOCKS,
    timeframe=DEFAULT_TIMEFRAME, output="regime_filter_results.csv",
    research_end_time=None, starting_capital=config.BACKTEST_STARTING_CAPITAL,
    fee_profile: str = config.BACKTEST_FEE_PROFILE,
):
    if block_days <= 0 or blocks <= 0:
        raise ValueError("block-days and blocks must be positive")
    if starting_capital <= 0:
        raise ValueError("starting capital must be positive")
    requested_time = _utc(research_end_time or datetime.now(timezone.utc))
    aligned_end = align_research_end_time(requested_time, [timeframe])
    regime_blocks = block_tools.build_blocks(
        aligned_end, block_days=block_days, blocks=blocks
    )
    boundary_error = block_tools._boundary_error(
        timeframe, aligned_end, block_days, blocks
    )
    strategies = {
        name: (
            create_regime_only_strategy()
            if name == "regime_only_4h"
            else get_strategy(name)
        )
        for name in STRATEGIES
    }
    max_warmup = max(
        backtest.required_warmup_bars(strategy=strategy)
        for strategy in strategies.values()
    )
    oldest_start = regime_blocks[0]["block_start"]
    total_days = block_days * blocks
    rates = config.get_fee_profile(fee_profile)
    fee_rate = rates["fee_rate"]
    slippage_rate = rates["slippage_rate"]

    rows = []
    result_by_strategy = {}
    error_by_strategy = {}
    data_quality = None
    bars = None
    if boundary_error:
        error_by_strategy = {name: ValueError(boundary_error) for name in STRATEGIES}
    else:
        try:
            bars = backtest.fetch_history(
                total_days, timeframe, warmup_bars=max_warmup, end_time=aligned_end
            )
            data_quality = _quality_for_span(bars, timeframe, oldest_start, aligned_end)
            print(
                f"DATA QUALITY {timeframe}: expected={data_quality['expected_candle_count']} "
                f"actual={data_quality['actual_candle_count']} "
                f"missing={data_quality['missing_candle_count']} "
                f"({data_quality['missing_candle_percent']:.2f}%) "
                f"zero_volume={data_quality['zero_volume_candle_count']}"
            )
        except Exception as error:
            error_by_strategy = {name: error for name in STRATEGIES}

    if bars is not None:
        for name, strategy in strategies.items():
            try:
                available_warmup = int((bars.index < oldest_start).sum())
                required = backtest.required_warmup_bars(strategy=strategy)
                if available_warmup < required:
                    raise ValueError(
                        f"Insufficient warm-up history: {available_warmup} pre-test bars; "
                        f"{required} required for {name}"
                    )
                result = backtest.run_backtest(
                    bars, starting_capital=starting_capital, timeframe=timeframe,
                    fee_rate=fee_rate, slippage=slippage_rate, test_start=oldest_start,
                    strategy=strategy, liquidate_at_end=False,
                )
                if _utc(pd.Timestamp(result["start_time"]).to_pydatetime()) != oldest_start:
                    raise ValueError("Backtest start does not match requested block boundary")
                if _utc(pd.Timestamp(result["end_time"]).to_pydatetime()) != aligned_end:
                    raise ValueError("Backtest end does not match common aligned boundary")
                result_by_strategy[name] = result
            except Exception as error:
                error_by_strategy[name] = error

    filtered_prepared = (
        strategies["donchian_regime_filter"].prepare_indicators(bars)
        if bars is not None and "donchian_regime_filter" in result_by_strategy
        else None
    )
    for block in regime_blocks:
        block_quality = None
        market = pd.DataFrame()
        btc_metrics = {
            "btc_buy_hold_return_percent": None,
            "btc_buy_hold_return_after_costs_percent": None,
            "btc_buy_hold_max_drawdown_percent": None,
            "btc_buy_hold_costs_usd": None,
        }
        if bars is not None:
            block_quality = _quality_for_span(
                bars, timeframe, block["block_start"], block["block_end"]
            )
            _, minutes_per_bar = parse_timeframe(timeframe)
            duration = pd.Timedelta(minutes=minutes_per_bar)
            market = _span_bars(bars, block["block_start"], block["block_end"])
            if (
                not market.empty
                and market.index[0] == pd.Timestamp(block["block_start"])
                and market.index[-1] + duration == pd.Timestamp(block["block_end"])
            ):
                btc_metrics = _btc_buy_hold_metrics(
                    market, fee_rate, slippage_rate
                )
        for name in STRATEGIES:
            strategy = strategies[name]
            error = error_by_strategy.get(name)
            row = {field: None for field in CSV_FIELDS}
            row.update({
                "row_type": "BLOCK",
                "block_index": block["block_index"],
                "block_start": block["block_start"].isoformat(),
                "block_end": block["block_end"].isoformat(),
                "block_days": block_days,
                "timeframe": timeframe,
                "strategy_name": name,
                "strategy_parameters_json": json.dumps(
                    strategy.parameters, sort_keys=True, separators=(",", ":")
                ),
                "error": str(error) if error else "",
                "requested_research_time": requested_time.isoformat(),
                "aligned_research_end": aligned_end.isoformat(),
                "fee_rate": fee_rate,
                "slippage_rate": slippage_rate,
                "starting_capital": starting_capital,
                **(block_quality or {}),
                **btc_metrics,
            })
            row["raw_btc_return_percent"] = btc_metrics["btc_buy_hold_return_percent"]
            result = result_by_strategy.get(name)
            if result is not None and not error:
                try:
                    metrics = block_tools._block_metrics(
                        result, bars, block, timeframe=timeframe
                    )
                    notional_metrics = _block_notional_metrics(
                        result, metrics, block
                    )
                    row.update({
                        "starting_equity": metrics["starting_equity"],
                        "ending_equity": metrics["ending_equity"],
                        "block_net_return_percent": metrics["block_net_return_percent"],
                        "block_candle_mark_max_drawdown_percent": (
                            metrics["block_candle_mark_max_drawdown_percent"]
                        ),
                        "entries_in_block": metrics["entries_in_block"],
                        "exits_in_block": metrics["exits_in_block"],
                        "charged_costs_in_block": metrics["charged_costs_in_block"],
                        "time_invested_percent": metrics["time_invested_percent"],
                        "average_capital_exposure": metrics["average_capital_exposure"],
                        **notional_metrics,
                    })
                    if name == "donchian_regime_filter":
                        row.update(
                            _filter_diagnostics(
                                result, filtered_prepared, block, timeframe
                            )
                        )
                except Exception as block_error:
                    row["error"] = str(block_error)
            rows.append(row)

    total_btc_metrics = {
        "btc_buy_hold_return_percent": None,
        "btc_buy_hold_return_after_costs_percent": None,
        "btc_buy_hold_max_drawdown_percent": None,
        "btc_buy_hold_costs_usd": None,
    }
    total_quality = None
    total_bars = pd.DataFrame()
    if bars is not None:
        total_bars = _span_bars(bars, oldest_start, aligned_end)
        total_btc_metrics = _btc_buy_hold_metrics(
            total_bars, fee_rate, slippage_rate
        )
        total_quality = _quality_for_span(
            bars, timeframe, oldest_start, aligned_end
        )

    total_rows = []
    for name in STRATEGIES:
        strategy = strategies[name]
        error = error_by_strategy.get(name)
        result = result_by_strategy.get(name)
        row = {field: None for field in CSV_FIELDS}
        row.update({
            "row_type": "TOTAL",
            "block_start": oldest_start.isoformat(),
            "block_end": aligned_end.isoformat(),
            "block_days": total_days,
            "timeframe": timeframe,
            "strategy_name": name,
            "strategy_parameters_json": json.dumps(
                strategy.parameters, sort_keys=True, separators=(",", ":")
            ),
            "error": str(error) if error else "",
            "requested_research_time": requested_time.isoformat(),
            "aligned_research_end": aligned_end.isoformat(),
            "fee_rate": fee_rate,
            "slippage_rate": slippage_rate,
            "starting_capital": starting_capital,
            **(total_quality or {}),
            **total_btc_metrics,
            "raw_btc_return_percent": total_btc_metrics[
                "btc_buy_hold_return_percent"
            ],
        })
        if result is not None and not error:
            try:
                summary = _total_summary(
                    result, total_bars, fee_rate, slippage_rate
                )
                row.update({
                    "starting_equity": summary["starting_equity"],
                    "ending_equity": summary["ending_equity"],
                    "total_net_return_percent": summary["total_net_return_percent"],
                    "net_pnl_usd": summary["net_pnl_usd"],
                    "pnl_percent_of_notional": summary["pnl_percent_of_notional"],
                    "compounded_trade_return_percent": (
                        summary["compounded_trade_return_percent"]
                    ),
                    "compounded_trade_return_basis": (
                        summary["compounded_trade_return_basis"]
                    ),
                    "max_drawdown_usd": summary["max_drawdown_usd"],
                    "max_drawdown_percent_of_notional": (
                        summary["max_drawdown_percent_of_notional"]
                    ),
                    "total_trades": summary["total_trades"],
                    "trade_count_basis": "completed trades",
                    "total_costs_usd": summary["total_costs_usd"],
                    "time_invested_percent": summary["time_invested_percent"],
                    "average_capital_exposure": summary["average_capital_exposure"],
                    "open_position_at_end": summary["open_position_at_end"],
                })
            except Exception as total_error:
                row["error"] = str(total_error)
        total_rows.append(row)

    block_rows_by_key = {
        (row["block_index"], row["strategy_name"]): row for row in rows
    }
    for block in regime_blocks:
        filtered = block_rows_by_key[
            (block["block_index"], "donchian_regime_filter")
        ]
        regime_only = block_rows_by_key[(block["block_index"], "regime_only_4h")]
        if (
            not filtered["error"] and not regime_only["error"]
            and filtered["block_net_return_percent"] is not None
            and regime_only["block_net_return_percent"] is not None
        ):
            filtered["filtered_minus_regime_only_percent"] = (
                filtered["block_net_return_percent"]
                - regime_only["block_net_return_percent"]
            )

    report_rows = rows + total_rows
    for row in report_rows:
        row.update(fee_profile=fee_profile, fee_rate=fee_rate, slippage_rate=slippage_rate)

    _print_report(
        report_rows, result_by_strategy, regime_blocks, requested_time, aligned_end,
        starting_capital, fee_rate, slippage_rate,
    )
    if output is not None:
        output_path = Path(output)
        with output_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(report_rows)
        print(f"CSV saved to {output_path.resolve()}")
    return report_rows


def _print_report(
    rows, results, blocks, requested_time, aligned_end, starting_capital,
    fee_rate, slippage_rate,
):
    print("\n=== CAUSAL REGIME FILTER RESEARCH ===")
    if rows:
        config.print_fee_profile(rows[0]["fee_profile"], rows[0]["fee_rate"], rows[0]["slippage_rate"])
    print(f"Requested research time: {requested_time.isoformat()}")
    print(f"Common aligned end: {aligned_end.isoformat()}")
    print(f"Blocks: {len(blocks)} x {blocks[0]['block_days']} days")
    print(
        f"Fee: {fee_rate:.2%} | Slippage: {slippage_rate:.2%} | "
        f"Starting capital: {'$'}{starting_capital:,.2f} | "
        f"per-trade notional: {'$'}{trade_config.TRADE_AMOUNT_USD:,.2f}"
    )
    print("One continuous, non-liquidating simulation per strategy; block returns are candle-mark equity.")

    block_rows = [row for row in rows if row["row_type"] == "BLOCK"]
    total_rows = [row for row in rows if row["row_type"] == "TOTAL"]
    rows_by_key = {
        (row["block_index"], row["strategy_name"]): row for row in block_rows
    }
    totals_by_name = {row["strategy_name"]: row for row in total_rows}
    print(
        "Block  Period                          BTC%      MA%  PlainDonchian%  "
        "FilteredDonchian%  RegimeOnly%  FilterDelta%  FilteredMinusRegimeOnly%"
    )
    fmt = lambda value: "N/A" if value is None else f"{value:.2f}"
    for block in blocks:
        selected = {
            name: rows_by_key.get((block["block_index"], name), {})
            for name in STRATEGIES
        }
        btc = selected["ma_rsi_crossover"].get("raw_btc_return_percent")
        ma = selected["ma_rsi_crossover"].get("block_net_return_percent")
        plain = selected["donchian_breakout"].get("block_net_return_percent")
        filtered = selected["donchian_regime_filter"].get("block_net_return_percent")
        regime_only = selected["regime_only_4h"].get("block_net_return_percent")
        filter_delta = (
            filtered - plain if filtered is not None and plain is not None else None
        )
        filtered_regime_delta = (
            filtered - regime_only
            if filtered is not None and regime_only is not None
            else None
        )
        period = f"{block['block_start'].date()} -> {block['block_end'].date()}"
        print(
            f"{block['block_index']:<6} {period:<30} {fmt(btc):>7} "
            f"{fmt(ma):>9} {fmt(plain):>15} {fmt(filtered):>18} "
            f"{fmt(regime_only):>12} {fmt(filter_delta):>13} "
            f"{fmt(filtered_regime_delta):>27}"
        )

    print("\nTOTAL COMPARISON")
    print(
        "Series | Return % (notional for strategies; BTC price for benchmark) | "
        "Post-cost / compounded % | Max DD % | Invested % | Trades | Costs USD"
    )
    for name in STRATEGIES:
        row = totals_by_name.get(name, {})
        result = results.get(name)
        if result is None or row.get("error"):
            print(f"{name} | INVALID: {row.get('error', 'no result')}")
            continue
        print(
            f"{name} | {row['pnl_percent_of_notional']:.3f} | "
            f"{row['compounded_trade_return_percent']:.3f} "
            f"({row['compounded_trade_return_basis']}) | "
            f"{row['max_drawdown_percent_of_notional']:.3f} | "
            f"{row['time_invested_percent']:.2f} | {row['total_trades']} | "
            f"{'$'}{row['total_costs_usd']:.2f}"
        )
    benchmark = totals_by_name.get("ma_rsi_crossover", {})
    if benchmark and benchmark.get("btc_buy_hold_return_percent") is not None:
        print(
            f"BTC buy & hold | {benchmark['btc_buy_hold_return_percent']:.3f} | "
            f"after_costs={benchmark['btc_buy_hold_return_after_costs_percent']:.3f} | "
            f"max_dd={benchmark['btc_buy_hold_max_drawdown_percent']:.3f} | "
            f"invested=100.00 | trades=1 | "
            f"costs={'$'}{benchmark['btc_buy_hold_costs_usd']:.2f}"
        )

    for name in STRATEGIES:
        result = results.get(name)
        if result is None:
            continue
        total = totals_by_name[name]
        print(
            f"TOTAL {name}: start={'$'}{total['starting_equity']:.2f} "
            f"end={'$'}{total['ending_equity']:.2f} "
            f"net_capital_return={total['total_net_return_percent']:.2f}% "
            f"same_path_zero_cost={result['same_path_zero_cost_return_percent']:.2f}% "
            f"trades={total['total_trades']} costs={'$'}{total['total_costs_usd']:.2f} "
            f"notional_pnl={total['pnl_percent_of_notional']:.2f}% "
            f"notional_max_dd={total['max_drawdown_percent_of_notional']:.2f}% "
            f"invested={total['time_invested_percent']:.2f}% "
            f"avg_exposure={total['average_capital_exposure']:.2f}%"
        )
        strategy_rows = [
            row for row in block_rows
            if row["strategy_name"] == name and not row["error"]
        ]
        returns = [
            row["block_net_return_percent"] for row in strategy_rows
            if row["block_net_return_percent"] is not None
        ]
        dds = [
            row["block_candle_mark_max_drawdown_percent"] for row in strategy_rows
            if row["block_candle_mark_max_drawdown_percent"] is not None
        ]
        if returns:
            print(
                f"BLOCK SUMMARY {name}: profitable={sum(value > 0 for value in returns)} "
                f"losing={sum(value < 0 for value in returns)} avg={mean(returns):.2f}% "
                f"median={median(returns):.2f}% "
                f"avg_candle_max_dd={mean(dds):.2f}%"
            )

    plain_rows = [
        rows_by_key.get((block["block_index"], "donchian_breakout"), {})
        for block in blocks
    ]
    filter_rows = [
        rows_by_key.get((block["block_index"], "donchian_regime_filter"), {})
        for block in blocks
    ]
    deltas = [
        filtered["block_net_return_percent"] - plain["block_net_return_percent"]
        for plain, filtered in zip(plain_rows, filter_rows)
        if not plain.get("error") and not filtered.get("error")
        and plain.get("block_net_return_percent") is not None
        and filtered.get("block_net_return_percent") is not None
    ]
    if deltas:
        print(
            f"FILTER DELTA (filtered - plain Donchian): "
            f"helped={sum(value > 0 for value in deltas)} "
            f"hurt={sum(value < 0 for value in deltas)} "
            f"equal={sum(value == 0 for value in deltas)} "
            f"average={mean(deltas):.2f}%"
        )

    regime_deltas = [
        row["filtered_minus_regime_only_percent"]
        for row in filter_rows
        if not row.get("error")
        and row.get("filtered_minus_regime_only_percent") is not None
    ]
    if regime_deltas:
        print(
            f"FILTERED MINUS REGIME-ONLY: helped={sum(value > 0 for value in regime_deltas)} "
            f"hurt={sum(value < 0 for value in regime_deltas)} "
            f"equal={sum(value == 0 for value in regime_deltas)} "
            f"average={mean(regime_deltas):.2f}%"
        )

    for sign, label in ((1, "BTC-positive"), (-1, "BTC-negative")):
        market_blocks = [
            row for row in block_rows
            if row["strategy_name"] == "ma_rsi_crossover"
            and row["raw_btc_return_percent"] is not None
            and row["raw_btc_return_percent"] * sign > 0
        ]
        indexes = {row["block_index"] for row in market_blocks}
        print(f"POST-HOC {label} blocks:")
        for name in STRATEGIES:
            values = [
                row["block_net_return_percent"] for row in block_rows
                if row["strategy_name"] == name
                and row["block_index"] in indexes
                and not row["error"]
                and row["block_net_return_percent"] is not None
            ]
            print(
                f"  {name}: average={mean(values):.2f}% n={len(values)}"
                if values else f"  {name}: N/A"
            )
    print("\nThe displayed blocks summarize the historical period in this run.")
    print("This is exploratory / in-sample evidence, not untouched holdout validation or proof of profitability.")
    print("Any positive result still requires holdout, walk-forward, and out-of-sample validation.")


def _build_parser():
    parser = argparse.ArgumentParser(description="Compare fixed causal BTC regime-filter strategies")
    parser.add_argument("--block-days", type=int, default=DEFAULT_BLOCK_DAYS)
    parser.add_argument("--blocks", type=int, default=DEFAULT_BLOCKS)
    parser.add_argument("--timeframe", default=DEFAULT_TIMEFRAME)
    parser.add_argument(
        "--end-time",
        help="Freeze the common research end boundary (ISO-8601 UTC, e.g. 2022-10-26T04:00:00+00:00)",
    )
    parser.add_argument("--output", default="regime_filter_results.csv")
    parser.add_argument("--no-csv", action="store_true")
    config.add_fee_profile_argument(parser)
    return parser


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.block_days <= 0 or args.blocks <= 0:
        parser.error("--block-days and --blocks must be positive")
    end_time = None
    if args.end_time:
        try:
            end_time = datetime.fromisoformat(args.end_time.replace("Z", "+00:00"))
            if end_time.tzinfo is None or end_time.utcoffset() != timedelta(0):
                raise ValueError("timestamp must include a UTC timezone")
            end_time = end_time.astimezone(timezone.utc)
        except ValueError as error:
            parser.error(f"--end-time must be an ISO-8601 UTC timestamp: {error}")
    run_regime_filter_research(
        fee_profile=args.fee_profile,
        block_days=args.block_days, blocks=args.blocks, timeframe=args.timeframe,
        output=None if args.no_csv else args.output, research_end_time=end_time,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
