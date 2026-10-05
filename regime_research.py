"""Sequential non-overlapping regime research over continuous backtests."""

import argparse
import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

import backtest
import config
from research import align_research_end_time
from strategy import MA_RSI_CROSSOVER, StrategySpec, get_strategy
from timeframes import parse_timeframe


DEFAULT_BLOCK_DAYS = 90
DEFAULT_BLOCKS = 8
DEFAULT_TIMEFRAMES = ("1Hour", "2Hour", "4Hour")
DEFAULT_STRATEGIES = (MA_RSI_CROSSOVER.name, "donchian_breakout")

REGIME_METRICS = (
    "raw_btc_return_percent",
    "starting_equity",
    "ending_equity",
    "block_net_return_percent",
    "block_net_pnl",
    "block_max_drawdown_percent",
    "block_candle_mark_max_drawdown_percent",
    "entries_in_block",
    "exits_in_block",
    "winning_exits_in_block",
    "losing_exits_in_block",
    "win_rate_of_exits_in_block",
    "closed_trade_net_pnl",
    "closed_trade_costs",
    "charged_costs_in_block",
    "time_invested_percent",
    "average_capital_exposure",
    "maximum_capital_exposure",
)

CONTEXT_FIELDS = (
    "requested_research_time",
    "aligned_research_end",
    "fee_rate",
    "slippage_rate",
    "starting_capital",
    "expected_candle_count",
    "actual_candle_count",
    "missing_candle_count",
    "missing_candle_percent",
    "zero_volume_candle_count",
    "zero_volume_candle_percent",
)

CSV_FIELDS = (
    "block_index",
    "block_start",
    "block_end",
    "block_days",
    "timeframe",
    "strategy_name",
    "strategy_parameters_json",
    "error",
    *CONTEXT_FIELDS,
    *REGIME_METRICS,
)


def _as_utc(value: datetime | pd.Timestamp) -> datetime:
    value = value.to_pydatetime() if isinstance(value, pd.Timestamp) else value
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _timestamp(value: Any) -> str:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _strategy_parameters_json(strategy: StrategySpec | None) -> str:
    if strategy is None:
        return "{}"
    return json.dumps(strategy.parameters, sort_keys=True, separators=(",", ":"))


def build_blocks(
    aligned_end: datetime,
    *,
    block_days: int = DEFAULT_BLOCK_DAYS,
    blocks: int = DEFAULT_BLOCKS,
) -> list[dict[str, Any]]:
    """Build chronological adjacent blocks ending at the common aligned end."""
    if block_days <= 0:
        raise ValueError("block_days must be greater than zero")
    if blocks <= 0:
        raise ValueError("blocks must be greater than zero")
    aligned_end = _as_utc(aligned_end)
    block_delta = timedelta(days=block_days)
    total_delta = block_delta * blocks
    oldest_start = aligned_end - total_delta
    return [
        {
            "block_index": index + 1,
            "block_start": oldest_start + block_delta * index,
            "block_end": oldest_start + block_delta * (index + 1),
            "block_days": block_days,
        }
        for index in range(blocks)
    ]


def _duration_seconds(timeframe: str) -> int:
    frame, minutes_per_bar = parse_timeframe(timeframe)
    if frame.unit.name in {"Week", "Month"}:
        raise ValueError(
            f"Regime boundary alignment for {frame.unit.name} timeframes is unsupported"
        )
    return int(round(minutes_per_bar * 60))


def _boundary_error(
    timeframe: str, aligned_end: datetime, block_days: int, blocks: int
) -> str | None:
    try:
        duration_seconds = _duration_seconds(timeframe)
    except ValueError as error:
        return str(error)
    epoch_seconds = int(aligned_end.timestamp())
    block_seconds = block_days * 86400
    if epoch_seconds % duration_seconds or block_seconds % duration_seconds:
        return (
            f"Block boundaries do not align to {timeframe} candle closes; "
            "choose block sizes and timeframes with common candle boundaries"
        )
    total_seconds = blocks * block_seconds
    if total_seconds % duration_seconds:
        return f"Total regime span does not align to {timeframe} candles"
    return None


def _utc_index(values: Any) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(values, utc=True))


def _exact_equity_at(equity: pd.Series, boundary: datetime) -> float:
    matches = equity.loc[equity.index == pd.Timestamp(boundary)]
    if matches.empty:
        raise ValueError(
            f"Backtest equity curve has no mark exactly at block boundary {boundary.isoformat()}"
        )
    return float(matches.iloc[-1])


def _in_block_mask(
    timestamps: pd.Series,
    start: datetime,
    end: datetime,
    *,
    include_end: bool,
) -> pd.Series:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    mask = (timestamps >= start_ts) & (timestamps < end_ts)
    if include_end:
        mask |= timestamps == end_ts
    return mask


def _block_metrics(
    result: dict[str, Any],
    bars: pd.DataFrame,
    block: dict[str, Any],
    *,
    timeframe: str,
) -> dict[str, Any]:
    start = block["block_start"]
    end = block["block_end"]
    equity = result["equity_curve"].copy()
    equity.index = _utc_index(equity.index)
    start_equity = _exact_equity_at(equity, start)
    end_equity = _exact_equity_at(equity, end)
    block_equity = equity.loc[(equity.index >= pd.Timestamp(start)) & (equity.index <= pd.Timestamp(end))]
    if block_equity.empty or start_equity <= 0:
        raise ValueError(f"No valid mark-to-market equity for block {block['block_index']}")
    drawdown = (block_equity / block_equity.cummax() - 1) * 100

    _, minutes_per_bar = parse_timeframe(timeframe)
    duration = pd.Timedelta(minutes=minutes_per_bar)
    market_bars = bars.copy()
    market_bars.index = _utc_index(market_bars.index)
    market_block = market_bars.loc[
        (market_bars.index >= pd.Timestamp(start))
        & (market_bars.index < pd.Timestamp(end))
    ]
    if (
        market_block.empty
        or market_block.index[0] != pd.Timestamp(start)
        or market_block.index[-1] + duration != pd.Timestamp(end)
    ):
        raise ValueError(
            f"Historical candles do not cover exact block boundaries for block {block['block_index']}"
        )
    first_market_price = float(market_block.iloc[0]["open"])
    final_market_price = float(market_block.iloc[-1]["close"])
    raw_btc_return = (
        (final_market_price / first_market_price - 1) * 100
        if first_market_price
        else None
    )

    trades = result["trades"].copy()
    entry_events = result.get("entry_events", pd.DataFrame()).copy()
    if entry_events.empty:
        entry_times = pd.Series(dtype="datetime64[ns, UTC]")
        entry_mask = pd.Series(dtype=bool)
    else:
        entry_times = pd.to_datetime(entry_events["entry_time"], utc=True)
        entry_mask = _in_block_mask(entry_times, start, end, include_end=False)
    if trades.empty:
        exit_mask = pd.Series(dtype=bool)
        exited = trades
    else:
        exit_times = pd.to_datetime(trades["exit_time"], utc=True)
        exit_mask = _in_block_mask(
            exit_times, start, end, include_end=False
        )
        exited = trades.loc[exit_mask]

    wins = int((exited["net_pnl"] > 0).sum()) if not exited.empty else 0
    losses = int((exited["net_pnl"] < 0).sum()) if not exited.empty else 0
    exits = int(exit_mask.sum()) if not trades.empty else 0
    closed_costs = (
        float((exited["fees"] + exited["slippage_cost"]).sum())
        if not exited.empty
        else 0.0
    )

    # Charge entry and exit transaction costs to the block where each event occurs.
    charged_costs = 0.0
    if not entry_events.empty:
        entered = entry_events.loc[entry_mask]
        charged_costs += float(
            (entered["entry_fee"] + entered["entry_slippage_cost"]).sum()
        )
    if not trades.empty:
        charged_exits = trades.loc[exit_mask]
        charged_costs += float(
            (charged_exits["exit_fee"] + charged_exits["exit_slippage_cost"]).sum()
        )

    exposure = result["exposure_curve"].copy()
    exposure.index = _utc_index(exposure.index)
    exposure = exposure.loc[
        (exposure.index >= pd.Timestamp(start))
        & (exposure.index < pd.Timestamp(end))
    ]
    if exposure.empty:
        raise ValueError(f"No exposure samples for block {block['block_index']}")

    block_net_pnl = end_equity - start_equity
    return {
        "raw_btc_return_percent": raw_btc_return,
        "starting_equity": start_equity,
        "ending_equity": end_equity,
        "block_net_return_percent": (end_equity / start_equity - 1) * 100,
        "block_net_pnl": block_net_pnl,
        "block_max_drawdown_percent": abs(float(drawdown.min())),
        "block_candle_mark_max_drawdown_percent": abs(float(drawdown.min())),
        "entries_in_block": int(entry_mask.sum()) if not entry_events.empty else 0,
        "exits_in_block": exits,
        "winning_exits_in_block": wins,
        "losing_exits_in_block": losses,
        "win_rate_of_exits_in_block": wins / exits * 100 if exits else None,
        "closed_trade_net_pnl": float(exited["net_pnl"].sum()) if not exited.empty else 0.0,
        "closed_trade_costs": closed_costs,
        "charged_costs_in_block": charged_costs,
        "time_invested_percent": float(exposure["time_invested"].mean()) * 100,
        "average_capital_exposure": float(exposure["capital_exposure_percent"].mean()),
        "maximum_capital_exposure": float(exposure["capital_exposure_percent"].max()),
    }


def _base_row(
    block: dict[str, Any],
    timeframe: str,
    strategy_name: str,
    strategy: StrategySpec | None,
    *,
    requested_research_time: datetime,
    aligned_end: datetime,
    fee_rate: float,
    slippage_rate: float,
    starting_capital: float,
    data_quality: dict[str, Any] | None = None,
    error: str = "",
) -> dict[str, Any]:
    row = {
        field: None
        for field in (*REGIME_METRICS, "expected_candle_count", "actual_candle_count", "missing_candle_count", "missing_candle_percent", "zero_volume_candle_count", "zero_volume_candle_percent")
    }
    row.update(
        {
            "block_index": block["block_index"],
            "block_start": _timestamp(block["block_start"]),
            "block_end": _timestamp(block["block_end"]),
            "block_days": block["block_days"],
            "timeframe": timeframe,
            "strategy_name": strategy_name,
            "strategy_parameters_json": _strategy_parameters_json(strategy),
            "error": error,
            "requested_research_time": _timestamp(requested_research_time),
            "aligned_research_end": _timestamp(aligned_end),
            "fee_rate": fee_rate,
            "slippage_rate": slippage_rate,
            "starting_capital": starting_capital,
            **(data_quality or {}),
        }
    )
    return row


def run_regime_research(
    *,
    block_days: int = DEFAULT_BLOCK_DAYS,
    blocks: int = DEFAULT_BLOCKS,
    timeframes: Iterable[str] = DEFAULT_TIMEFRAMES,
    strategies: Iterable[str] = DEFAULT_STRATEGIES,
    starting_capital: float = config.BACKTEST_STARTING_CAPITAL,
    output: str | Path | None = "regime_results.csv",
    research_end_time: datetime | None = None,
) -> list[dict[str, Any]]:
    """Run one continuous backtest per timeframe/strategy, then segment its marks."""
    if block_days <= 0 or blocks <= 0:
        raise ValueError("block_days and blocks must be greater than zero")
    if starting_capital <= 0:
        raise ValueError("starting_capital must be greater than zero")
    timeframes = list(timeframes)
    strategy_names = list(strategies)
    if not timeframes:
        raise ValueError("At least one timeframe is required")
    if not strategy_names:
        raise ValueError("At least one strategy is required")

    requested_research_time = _as_utc(research_end_time or datetime.now(timezone.utc))
    timeframe_errors: dict[str, Exception] = {}
    alignable_timeframes: list[str] = []
    for timeframe in dict.fromkeys(timeframes):
        try:
            _duration_seconds(timeframe)
            alignable_timeframes.append(timeframe)
        except Exception as error:
            timeframe_errors[timeframe] = error
    aligned_end = align_research_end_time(requested_research_time, alignable_timeframes)
    regime_blocks = build_blocks(aligned_end, block_days=block_days, blocks=blocks)
    oldest_start = regime_blocks[0]["block_start"]
    total_days = block_days * blocks

    strategies_by_name: dict[str, StrategySpec] = {}
    strategy_errors: dict[str, Exception] = {}
    for strategy_name in strategy_names:
        try:
            strategies_by_name[strategy_name] = get_strategy(strategy_name)
        except Exception as error:
            strategy_errors[strategy_name] = error
    max_warmup = max(
        (
            backtest.required_warmup_bars(strategy=strategy)
            for strategy in dict.fromkeys(strategies_by_name.values())
        ),
        default=0,
    )

    fee_rate = config.BACKTEST_FEE_PERCENT
    slippage_rate = config.BACKTEST_SLIPPAGE_PERCENT
    bars_by_timeframe: dict[str, pd.DataFrame] = {}
    data_quality_by_timeframe: dict[str, dict[str, Any]] = {}
    fetch_errors: dict[str, Exception] = {}
    for timeframe in dict.fromkeys(timeframes):
        if timeframe in timeframe_errors:
            continue
        alignment_error = _boundary_error(timeframe, aligned_end, block_days, blocks)
        if alignment_error:
            timeframe_errors[timeframe] = ValueError(alignment_error)
            continue
        if not strategies_by_name:
            continue
        try:
            bars_by_timeframe[timeframe] = backtest.fetch_history(
                total_days,
                timeframe,
                warmup_bars=max_warmup,
                end_time=aligned_end,
            )
            data_quality_by_timeframe[timeframe] = backtest.data_quality_diagnostics(
                bars_by_timeframe[timeframe], timeframe
            )
            quality = data_quality_by_timeframe[timeframe]
            print(
                f"DATA QUALITY {timeframe}: expected={quality['expected_candle_count']} "
                f"actual={quality['actual_candle_count']} missing="
                f"{quality['missing_candle_count']} ({quality['missing_candle_percent']:.2f}%) "
                f"zero_volume={quality['zero_volume_candle_count']}"
            )
        except Exception as error:
            fetch_errors[timeframe] = error

    block_metrics_by_combo: dict[tuple[str, str], list[dict[str, Any]] | Exception] = {}
    for timeframe in dict.fromkeys(timeframes):
        if timeframe in timeframe_errors or timeframe in fetch_errors:
            continue
        bars = bars_by_timeframe.get(timeframe)
        if bars is None:
            continue
        for strategy_name in dict.fromkeys(strategy_names):
            if strategy_name in strategy_errors:
                continue
            strategy = strategies_by_name[strategy_name]
            try:
                available_warmup = int((bars.index < oldest_start).sum())
                required_warmup = backtest.required_warmup_bars(strategy=strategy)
                if available_warmup < required_warmup:
                    raise ValueError(
                        "Insufficient warm-up history: received "
                        f"{available_warmup} pre-test bars; {required_warmup} required "
                        f"for {strategy_name}"
                    )
                result = backtest.run_backtest(
                    bars,
                    starting_capital=starting_capital,
                    timeframe=timeframe,
                    fee_rate=fee_rate,
                    slippage=slippage_rate,
                    test_start=oldest_start,
                    strategy=strategy,
                    liquidate_at_end=False,
                )
                if _as_utc(result["start_time"]) != oldest_start:
                    raise ValueError(
                        "Historical start coverage mismatch: requested "
                        f"{oldest_start.isoformat()}, actual {_as_utc(result['start_time']).isoformat()}"
                    )
                if _as_utc(result["end_time"]) != aligned_end:
                    raise ValueError(
                        "Historical end coverage mismatch: requested "
                        f"{aligned_end.isoformat()}, actual {_as_utc(result['end_time']).isoformat()}"
                    )
                records = [
                    _block_metrics(
                        result,
                        bars,
                        block,
                        timeframe=timeframe,
                    )
                    for block in regime_blocks
                ]
                block_metrics_by_combo[(timeframe, strategy_name)] = records
            except Exception as error:
                block_metrics_by_combo[(timeframe, strategy_name)] = error

    rows = []
    for block_index, block in enumerate(regime_blocks):
        for timeframe in timeframes:
            for strategy_name in strategy_names:
                strategy = strategies_by_name.get(strategy_name)
                error: Exception | None = None
                metrics: dict[str, Any] = {}
                if strategy_name in strategy_errors:
                    error = strategy_errors[strategy_name]
                elif timeframe in timeframe_errors:
                    error = timeframe_errors[timeframe]
                elif timeframe in fetch_errors:
                    error = fetch_errors[timeframe]
                else:
                    combo_result = block_metrics_by_combo.get((timeframe, strategy_name))
                    if isinstance(combo_result, Exception):
                        error = combo_result
                    elif combo_result is None:
                        error = RuntimeError("No continuous backtest result was produced")
                    else:
                        metrics = combo_result[block_index]
                row = _base_row(
                    block,
                    timeframe,
                    strategy_name,
                    strategy,
                    requested_research_time=requested_research_time,
                    aligned_end=aligned_end,
                    fee_rate=fee_rate,
                    slippage_rate=slippage_rate,
                    starting_capital=starting_capital,
                    data_quality=data_quality_by_timeframe.get(timeframe),
                    error=str(error) if error else "",
                )
                row.update(metrics)
                rows.append(row)
                if error:
                    print(
                        f"Block {block['block_index']} {timeframe} {strategy_name}  ERROR: {error}"
                    )

    _print_report(rows, regime_blocks, requested_research_time, aligned_end, starting_capital)
    if output is not None:
        output_path = Path(output)
        with output_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"CSV saved to {output_path.resolve()}")
    return rows


def _fmt(value: Any) -> str:
    return "N/A" if value is None else f"{value:.2f}"


def _print_report(
    rows: list[dict[str, Any]],
    blocks: list[dict[str, Any]],
    requested_research_time: datetime,
    aligned_end: datetime,
    starting_capital: float,
) -> None:
    print("\n=== CONTINUOUS NON-OVERLAPPING REGIME RESEARCH ===")
    print(f"Requested research time: {_timestamp(requested_research_time)}")
    print(f"Common aligned end: {_timestamp(aligned_end)}")
    print(
        f"Blocks: {len(blocks)} x {blocks[0]['block_days']} days | "
        f"Fee: {config.BACKTEST_FEE_PERCENT:.2%} | "
        f"Slippage: {config.BACKTEST_SLIPPAGE_PERCENT:.2%} | "
        f"Starting capital: ${starting_capital:,.2f}"
    )
    print(
        "One uninterrupted backtest per timeframe/strategy; block returns use "
        "mark-to-market equity. Charged costs follow transaction timestamps; "
        "closed_trade_costs reports full costs for trades exiting in that block."
    )
    print(
        "research.py covers nested recency windows; regime_research.py segments "
        "sequential blocks from one continuous simulation."
    )
    print("Drawdown is candle-mark based, not tick-level intrabar worst case.")
    print("Block  Period                                  BTC%    TF     Strategy                 Net% CandleMaxDD% Entries Exits")
    for row in rows:
        if row["error"]:
            continue
        period = f"{row['block_start']} -> {row['block_end']}"
        print(
            f"{row['block_index']:<6} {period:<40} {_fmt(row['raw_btc_return_percent']):>6} "
            f"{row['timeframe']:<6} {row['strategy_name']:<24} "
            f"{_fmt(row['block_net_return_percent']):>6} "
            f"{_fmt(row['block_max_drawdown_percent']):>7} "
            f"{row['entries_in_block']:>7} {row['exits_in_block']:>5}"
        )
    print("\nBlock  BTC%    TF     Strategy                 Invested% AvgExp% ExitWin% ClosedCosts ChargedCosts")
    for row in rows:
        if row["error"]:
            continue
        print(
            f"{row['block_index']:<6} {_fmt(row['raw_btc_return_percent']):>6} "
            f"{row['timeframe']:<6} {row['strategy_name']:<24} "
            f"{_fmt(row['time_invested_percent']):>9} "
            f"{_fmt(row['average_capital_exposure']):>7} "
            f"{_fmt(row['win_rate_of_exits_in_block']):>8} "
            f"${_fmt(row['closed_trade_costs']):>11} "
            f"${_fmt(row['charged_costs_in_block']):>11}"
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure sequential market blocks from continuous strategy backtests"
    )
    parser.add_argument("--block-days", type=int, default=DEFAULT_BLOCK_DAYS)
    parser.add_argument("--blocks", type=int, default=DEFAULT_BLOCKS)
    parser.add_argument(
        "--timeframes", nargs="+", default=list(DEFAULT_TIMEFRAMES), metavar="TIMEFRAME"
    )
    parser.add_argument(
        "--strategies", nargs="+", default=list(DEFAULT_STRATEGIES), metavar="STRATEGY"
    )
    parser.add_argument(
        "--starting-capital", type=float, default=config.BACKTEST_STARTING_CAPITAL
    )
    parser.add_argument("--output", default="regime_results.csv")
    parser.add_argument("--no-csv", action="store_true", help="Do not save a CSV")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.block_days <= 0 or args.blocks <= 0:
        parser.error("--block-days and --blocks must be greater than zero")
    if args.starting_capital <= 0:
        parser.error("--starting-capital must be greater than zero")
    run_regime_research(
        block_days=args.block_days,
        blocks=args.blocks,
        timeframes=args.timeframes,
        strategies=args.strategies,
        starting_capital=args.starting_capital,
        output=None if args.no_csv else args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
