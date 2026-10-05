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
from strategy import get_strategy
from timeframes import parse_timeframe


DEFAULT_BLOCK_DAYS = 90
DEFAULT_BLOCKS = 16
DEFAULT_TIMEFRAME = "4Hour"
STRATEGIES = (
    "ma_rsi_crossover",
    "donchian_breakout",
    "donchian_regime_filter",
)

CSV_FIELDS = (
    "block_index", "block_start", "block_end", "block_days", "timeframe",
    "strategy_name", "strategy_parameters_json", "error",
    "requested_research_time", "aligned_research_end", "fee_rate", "slippage_rate",
    "starting_capital", "raw_btc_return_percent", "starting_equity", "ending_equity",
    "block_net_return_percent", "block_candle_mark_max_drawdown_percent",
    "entries_in_block", "exits_in_block", "charged_costs_in_block",
    "time_invested_percent", "average_capital_exposure",
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


def _total_summary(result):
    exposure = result["exposure_curve"]
    return {
        "starting_equity": result["starting_capital"],
        "ending_equity": result["ending_equity"],
        "total_net_return_percent": result["strategy_return"],
        "same_path_zero_cost_return_percent": result["same_path_zero_cost_return_percent"],
        "total_trades": result["total_trades"],
        "total_costs": result["total_costs"],
        "candle_mark_max_drawdown_percent": result["candle_mark_max_drawdown_percent"],
        "time_invested_percent": float(exposure["time_invested"].mean() * 100),
        "average_capital_exposure": float(exposure["capital_exposure_percent"].mean()),
    }


def run_regime_filter_research(
    *, block_days=DEFAULT_BLOCK_DAYS, blocks=DEFAULT_BLOCKS,
    timeframe=DEFAULT_TIMEFRAME, output="regime_filter_results.csv",
    research_end_time=None, starting_capital=config.BACKTEST_STARTING_CAPITAL,
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
    strategies = {name: get_strategy(name) for name in STRATEGIES}
    max_warmup = max(
        backtest.required_warmup_bars(strategy=strategy)
        for strategy in strategies.values()
    )
    oldest_start = regime_blocks[0]["block_start"]
    total_days = block_days * blocks
    fee_rate, slippage_rate = config.BACKTEST_FEE_PERCENT, config.BACKTEST_SLIPPAGE_PERCENT

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
    for block_index, block in enumerate(regime_blocks):
        raw_btc_return = None
        block_quality = None
        if bars is not None:
            block_quality = _quality_for_span(
                bars, timeframe, block["block_start"], block["block_end"]
            )
            _, minutes_per_bar = parse_timeframe(timeframe)
            duration = pd.Timedelta(minutes=minutes_per_bar)
            market = bars.copy()
            market.index = pd.DatetimeIndex(pd.to_datetime(market.index, utc=True))
            market = market.loc[
                (market.index >= pd.Timestamp(block["block_start"]))
                & (market.index < pd.Timestamp(block["block_end"]))
            ]
            if (
                not market.empty
                and market.index[0] == pd.Timestamp(block["block_start"])
                and market.index[-1] + duration == pd.Timestamp(block["block_end"])
            ):
                first_open, last_close = float(market.iloc[0]["open"]), float(market.iloc[-1]["close"])
                raw_btc_return = (last_close / first_open - 1) * 100 if first_open else None
        for name in STRATEGIES:
            strategy = strategies[name]
            error = error_by_strategy.get(name)
            row = {field: None for field in CSV_FIELDS}
            row.update({
                "block_index": block["block_index"],
                "block_start": block["block_start"].isoformat(),
                "block_end": block["block_end"].isoformat(),
                "block_days": block_days, "timeframe": timeframe,
                "strategy_name": name,
                "strategy_parameters_json": json.dumps(strategy.parameters, sort_keys=True, separators=(",", ":")),
                "error": str(error) if error else "",
                "requested_research_time": requested_time.isoformat(),
                "aligned_research_end": aligned_end.isoformat(),
                "fee_rate": fee_rate, "slippage_rate": slippage_rate,
                "starting_capital": starting_capital,
                **(block_quality or {}),
            })
            row["raw_btc_return_percent"] = raw_btc_return
            result = result_by_strategy.get(name)
            if result is not None and not error:
                try:
                    metrics = block_tools._block_metrics(result, bars, block, timeframe=timeframe)
                    row.update({
                        "starting_equity": metrics["starting_equity"],
                        "ending_equity": metrics["ending_equity"],
                        "block_net_return_percent": metrics["block_net_return_percent"],
                        "block_candle_mark_max_drawdown_percent": metrics["block_candle_mark_max_drawdown_percent"],
                        "entries_in_block": metrics["entries_in_block"],
                        "exits_in_block": metrics["exits_in_block"],
                        "charged_costs_in_block": metrics["charged_costs_in_block"],
                        "time_invested_percent": metrics["time_invested_percent"],
                        "average_capital_exposure": metrics["average_capital_exposure"],
                    })
                    if name == "donchian_regime_filter":
                        row.update(_filter_diagnostics(result, filtered_prepared, block, timeframe))
                except Exception as block_error:
                    row["error"] = str(block_error)
            rows.append(row)

    _print_report(rows, result_by_strategy, regime_blocks, requested_time, aligned_end, starting_capital)
    if output is not None:
        output_path = Path(output)
        with output_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"CSV saved to {output_path.resolve()}")
    return rows


def _print_report(rows, results, blocks, requested_time, aligned_end, starting_capital):
    print("\n=== CAUSAL REGIME FILTER RESEARCH ===")
    print(f"Requested research time: {requested_time.isoformat()}")
    print(f"Common aligned end: {aligned_end.isoformat()}")
    print(f"Blocks: {len(blocks)} x {blocks[0]['block_days']} days")
    print(
        f"Fee: {config.BACKTEST_FEE_PERCENT:.2%} | "
        f"Slippage: {config.BACKTEST_SLIPPAGE_PERCENT:.2%} | "
        f"Starting capital: ${starting_capital:,.2f}"
    )
    print("One continuous, non-liquidating simulation per strategy; block returns are candle-mark equity.")
    print("Block  Period                          BTC%      MA%  PlainDonchian%  FilteredDonchian%  FilterDelta%")
    rows_by_key = {(row["block_index"], row["strategy_name"]): row for row in rows}
    for block in blocks:
        selected = {name: rows_by_key.get((block["block_index"], name), {}) for name in STRATEGIES}
        btc = selected[STRATEGIES[0]].get("raw_btc_return_percent")
        ma = selected["ma_rsi_crossover"].get("block_net_return_percent")
        plain = selected["donchian_breakout"].get("block_net_return_percent")
        filtered = selected["donchian_regime_filter"].get("block_net_return_percent")
        delta = filtered - plain if filtered is not None and plain is not None else None
        period = f"{block['block_start'].date()} -> {block['block_end'].date()}"
        fmt = lambda value: "N/A" if value is None else f"{value:.2f}"
        print(f"{block['block_index']:<6} {period:<30} {fmt(btc):>7} {fmt(ma):>9} {fmt(plain):>15} {fmt(filtered):>18} {fmt(delta):>13}")

    for name in STRATEGIES:
        result = results.get(name)
        if result is None:
            continue
        total = _total_summary(result)
        print(
            f"TOTAL {name}: start=${total['starting_equity']:.2f} end=${total['ending_equity']:.2f} "
            f"net={total['total_net_return_percent']:.2f}% same_path_zero_cost="
            f"{total['same_path_zero_cost_return_percent']:.2f}% trades={total['total_trades']} "
            f"costs=${total['total_costs']:.2f} candle_max_dd={total['candle_mark_max_drawdown_percent']:.2f}% "
            f"invested={total['time_invested_percent']:.2f}% avg_exposure={total['average_capital_exposure']:.2f}%"
        )

        strategy_rows = [row for row in rows if row["strategy_name"] == name and not row["error"]]
        returns = [row["block_net_return_percent"] for row in strategy_rows if row["block_net_return_percent"] is not None]
        dds = [row["block_candle_mark_max_drawdown_percent"] for row in strategy_rows if row["block_candle_mark_max_drawdown_percent"] is not None]
        if returns:
            print(
                f"BLOCK SUMMARY {name}: profitable={sum(value > 0 for value in returns)} "
                f"losing={sum(value < 0 for value in returns)} avg={mean(returns):.2f}% "
                f"median={median(returns):.2f}% best={max(returns):.2f}% worst={min(returns):.2f}% "
                f"avg_candle_max_dd={mean(dds):.2f}%"
            )
    plain_rows = [row for row in rows if row["strategy_name"] == "donchian_breakout"]
    filter_rows = [row for row in rows if row["strategy_name"] == "donchian_regime_filter"]
    deltas = [f["block_net_return_percent"] - p["block_net_return_percent"] for p, f in zip(plain_rows, filter_rows) if not p["error"] and not f["error"] and p["block_net_return_percent"] is not None and f["block_net_return_percent"] is not None]
    if deltas:
        print(f"FILTER DELTA: helped={sum(value > 0 for value in deltas)} hurt={sum(value < 0 for value in deltas)} equal={sum(value == 0 for value in deltas)} average={mean(deltas):.2f}%")
    for sign, label in ((1, "BTC-positive"), (-1, "BTC-negative")):
        market_blocks = [row for row in rows if row["strategy_name"] == "ma_rsi_crossover" and row["raw_btc_return_percent"] is not None and row["raw_btc_return_percent"] * sign > 0]
        indexes = {row["block_index"] for row in market_blocks}
        print(f"POST-HOC {label} blocks:")
        for name in STRATEGIES:
            values = [row["block_net_return_percent"] for row in rows if row["strategy_name"] == name and row["block_index"] in indexes and not row["error"] and row["block_net_return_percent"] is not None]
            print(f"  {name}: average={mean(values):.2f}% n={len(values)}" if values else f"  {name}: N/A")
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
        block_days=args.block_days, blocks=args.blocks, timeframe=args.timeframe,
        output=None if args.no_csv else args.output, research_end_time=end_time,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
