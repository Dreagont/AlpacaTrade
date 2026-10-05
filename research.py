"""Compare the configured strategy across candle timeframes."""

import argparse
import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import config
import backtest
from strategy import MA_RSI_CROSSOVER, StrategySpec, get_strategy
from timeframes import parse_timeframe


DEFAULT_TIMEFRAMES = ("5Min", "15Min", "30Min", "1Hour")
DEFAULT_LOOKBACKS = (90, 180, 365, 730)
DEFAULT_STRATEGIES = (MA_RSI_CROSSOVER.name,)

METRICS = (
    "start_time",
    "end_time",
    "warmup_bars",
    "required_warmup_bars",
    "available_pretest_bars",
    "total_trades",
    "trades_per_day",
    "winning_trades",
    "losing_trades",
    "win_rate",
    "gross_pnl",
    "gross_return_percent",
    "total_fees",
    "total_slippage",
    "total_costs",
    "net_pnl",
    "net_return_percent",
    "same_path_zero_cost_pnl_total",
    "same_path_zero_cost_return_percent",
    "pure_cost_drag_percent",
    "free_run_zero_cost_return_percent",
    "realistic_net_return_percent",
    "average_pnl_per_trade",
    "average_gross_return_per_trade",
    "average_net_return_per_trade",
    "profit_factor",
    "net_profit_factor",
    "gross_profit_factor",
    "max_drawdown",
    "candle_mark_max_drawdown_percent",
    "raw_market_return_percent",
    "daily_sharpe",
    "daily_return_observations",
    "average_holding_minutes",
    "median_holding_minutes",
    "time_invested_percent",
    "average_capital_exposure",
    "maximum_capital_exposure",
    "full_capital_buy_hold_return",
    "same_notional_buy_hold_return",
)

CONTEXT_FIELDS = (
    "requested_research_time",
    "research_end_time",
    "requested_start_time",
    "actual_start_time",
    "requested_end_time",
    "actual_end_time",
    "coverage_days",
    "symbol",
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
    "strategy_name",
    "lookback_days",
    "timeframe",
    "error",
    "strategy_parameters_json",
    *CONTEXT_FIELDS,
    *METRICS,
)


def _timestamp(value: Any) -> str:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def align_research_end_time(
    end_time: datetime,
    timeframes: Iterable[str],
) -> datetime:
    """Floor a UTC timestamp to the largest supported timeframe boundary."""
    end_time = _as_utc(end_time)
    parsed_timeframes = [parse_timeframe(timeframe) for timeframe in timeframes]
    if not parsed_timeframes:
        return end_time

    durations = []
    for timeframe, minutes_per_bar in parsed_timeframes:
        unit_name = timeframe.unit.name
        if unit_name in {"Week", "Month"}:
            raise ValueError(
                f"Common end-time alignment for {unit_name} timeframes is unsupported"
            )
        if unit_name not in {"Minute", "Hour", "Day"}:
            raise ValueError(
                f"Common end-time alignment for {unit_name} timeframes is unsupported"
            )
        durations.append(timedelta(minutes=minutes_per_bar))

    largest_duration = max(durations)
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    completed_intervals = (end_time - epoch) // largest_duration
    aligned_end = epoch + largest_duration * completed_intervals
    return end_time if aligned_end == end_time else aligned_end


def _strategy_parameters_json(strategy: StrategySpec) -> str:
    return json.dumps(strategy.parameters, sort_keys=True, separators=(",", ":"))


def _coverage_fields(
    result: dict[str, Any],
    *,
    requested_start: datetime,
    requested_end: datetime,
) -> dict[str, Any]:
    actual_start = _as_utc(result["start_time"])
    actual_end = _as_utc(result["end_time"])
    requested_start = _as_utc(requested_start)
    requested_end = _as_utc(requested_end)
    coverage_days = (actual_end - actual_start).total_seconds() / 86400
    return {
        "requested_start_time": _timestamp(requested_start),
        "actual_start_time": _timestamp(actual_start),
        "requested_end_time": _timestamp(requested_end),
        "actual_end_time": _timestamp(actual_end),
        "coverage_days": coverage_days,
    }


def _validate_coverage(coverage: dict[str, Any], timeframe: str) -> None:
    requested_start = datetime.fromisoformat(coverage["requested_start_time"])
    actual_start = datetime.fromisoformat(coverage["actual_start_time"])
    requested_end = datetime.fromisoformat(coverage["requested_end_time"])
    actual_end = datetime.fromisoformat(coverage["actual_end_time"])
    _frame, minutes_per_bar = parse_timeframe(timeframe)
    bar_duration = timedelta(minutes=minutes_per_bar)
    start_offset = actual_start - requested_start
    one_second = timedelta(seconds=1)

    if start_offset < -one_second or start_offset > bar_duration + one_second:
        raise ValueError(
            "Historical start coverage mismatch: "
            f"requested {requested_start.isoformat()}, actual {actual_start.isoformat()}"
        )
    if abs(actual_end - requested_end) > one_second:
        raise ValueError(
            "Historical end coverage mismatch: "
            f"requested {requested_end.isoformat()}, actual {actual_end.isoformat()}"
        )
    if actual_end < actual_start:
        raise ValueError("Historical coverage ends before it starts")


def _research_row(
    strategy: StrategySpec,
    realistic_result: dict[str, Any],
    zero_cost_result: dict[str, Any],
    *,
    timeframe: str,
    requested_research_time: datetime,
    research_end_time: datetime,
    requested_start: datetime,
    lookback_days: int,
    starting_capital: float,
    fee_rate: float,
    slippage_rate: float,
    coverage: dict[str, Any],
    data_quality: dict[str, Any],
) -> dict[str, Any]:
    duration_days = coverage["coverage_days"]
    gross_pnl = realistic_result["gross_pnl"]
    realistic_return = realistic_result["strategy_return"]
    same_path_zero_cost_return = realistic_result["same_path_zero_cost_return_percent"]
    pure_cost_drag = realistic_result["pure_cost_drag_percent"]
    net_profit_factor = realistic_result.get(
        "net_profit_factor", realistic_result.get("profit_factor")
    )
    return {
        "strategy_name": strategy.name,
        "lookback_days": lookback_days,
        "timeframe": timeframe,
        "error": "",
        "strategy_parameters_json": _strategy_parameters_json(strategy),
        "requested_research_time": _timestamp(requested_research_time),
        "research_end_time": _timestamp(research_end_time),
        **coverage,
        "symbol": config.SYMBOL,
        "fee_rate": fee_rate,
        "slippage_rate": slippage_rate,
        "starting_capital": starting_capital,
        **data_quality,
        "start_time": _timestamp(realistic_result["start_time"]),
        "end_time": _timestamp(realistic_result["end_time"]),
        "warmup_bars": realistic_result["warmup_bars"],
        "required_warmup_bars": realistic_result["required_warmup_bars"],
        "available_pretest_bars": realistic_result["available_pretest_bars"],
        "total_trades": realistic_result["total_trades"],
        "trades_per_day": (
            realistic_result["total_trades"] / duration_days
            if duration_days > 0
            else None
        ),
        "winning_trades": realistic_result["winning_trades"],
        "losing_trades": realistic_result["losing_trades"],
        "win_rate": realistic_result["win_rate"],
        "gross_pnl": gross_pnl,
        "gross_return_percent": gross_pnl / starting_capital * 100,
        "total_fees": realistic_result["total_fees"],
        "total_slippage": realistic_result["total_slippage"],
        "total_costs": realistic_result["total_costs"],
        "net_pnl": realistic_result["net_profit"],
        "net_return_percent": realistic_return,
        "same_path_zero_cost_pnl_total": realistic_result[
            "same_path_zero_cost_pnl_total"
        ],
        "same_path_zero_cost_return_percent": same_path_zero_cost_return,
        "pure_cost_drag_percent": pure_cost_drag,
        "free_run_zero_cost_return_percent": zero_cost_result["strategy_return"],
        "realistic_net_return_percent": realistic_return,
        "average_pnl_per_trade": realistic_result["average_pnl_per_trade"],
        "average_gross_return_per_trade": realistic_result["average_gross_return"],
        "average_net_return_per_trade": realistic_result["average_net_return"],
        "profit_factor": net_profit_factor,
        "net_profit_factor": net_profit_factor,
        "gross_profit_factor": realistic_result.get("gross_profit_factor"),
        "max_drawdown": realistic_result["max_drawdown"],
        "candle_mark_max_drawdown_percent": realistic_result.get(
            "candle_mark_max_drawdown_percent", realistic_result["max_drawdown"]
        ),
        "raw_market_return_percent": realistic_result["raw_market_return_percent"],
        "daily_sharpe": realistic_result["daily_sharpe"],
        "daily_return_observations": realistic_result["daily_return_observations"],
        "average_holding_minutes": realistic_result["average_holding_minutes"],
        "median_holding_minutes": realistic_result["median_holding_minutes"],
        "time_invested_percent": realistic_result["time_invested_percent"],
        "average_capital_exposure": realistic_result["average_capital_exposure"],
        "maximum_capital_exposure": realistic_result["maximum_capital_exposure"],
        "full_capital_buy_hold_return": realistic_result["full_buy_hold_return"],
        "same_notional_buy_hold_return": realistic_result["same_notional_return"],
    }


def _failed_row(
    strategy_name: str,
    strategy_parameters_json: str,
    error: Exception,
    *,
    timeframe: str,
    requested_research_time: datetime,
    research_end_time: datetime,
    requested_start: datetime,
    lookback_days: int,
    starting_capital: float,
    fee_rate: float,
    slippage_rate: float,
    coverage: dict[str, Any] | None = None,
    data_quality: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "strategy_name": strategy_name,
        "lookback_days": lookback_days,
        "timeframe": timeframe,
        "error": str(error),
        "strategy_parameters_json": strategy_parameters_json,
        "requested_research_time": _timestamp(requested_research_time),
        "research_end_time": _timestamp(research_end_time),
        "requested_start_time": _timestamp(requested_start),
        "actual_start_time": None,
        "requested_end_time": _timestamp(research_end_time),
        "actual_end_time": None,
        "coverage_days": None,
        **(coverage or {}),
        "symbol": config.SYMBOL,
        "fee_rate": fee_rate,
        "slippage_rate": slippage_rate,
        "starting_capital": starting_capital,
        **(
            data_quality
            or {
                field: None
                for field in (
                    "expected_candle_count",
                    "actual_candle_count",
                    "missing_candle_count",
                    "missing_candle_percent",
                    "zero_volume_candle_count",
                    "zero_volume_candle_percent",
                )
            }
        ),
        **{field: None for field in METRICS},
    }


def run_research(
    *,
    lookbacks: Iterable[int] | None = None,
    lookback_days: int | None = None,
    timeframes: Iterable[str] = DEFAULT_TIMEFRAMES,
    strategies: Iterable[str] = DEFAULT_STRATEGIES,
    starting_capital: float = config.BACKTEST_STARTING_CAPITAL,
    output: str | Path | None = "research_results.csv",
    research_end_time: datetime | None = None,
) -> list[dict[str, Any]]:
    """Run every lookback/timeframe/strategy combination in input order."""
    if lookbacks is not None and lookback_days is not None:
        raise ValueError("Pass either lookbacks or lookback_days, not both")
    if lookback_days is not None:
        lookbacks = [lookback_days]
    elif lookbacks is None:
        lookbacks = DEFAULT_LOOKBACKS
    else:
        lookbacks = list(lookbacks)
    if not lookbacks:
        raise ValueError("At least one lookback period is required")
    if any(days <= 0 for days in lookbacks):
        raise ValueError("All lookback periods must be positive")
    if starting_capital <= 0:
        raise ValueError("starting_capital must be greater than zero")

    lookbacks = list(lookbacks)
    timeframes = list(timeframes)
    strategy_names = list(strategies)
    if not strategy_names:
        raise ValueError("At least one strategy is required")

    strategy_by_name: dict[str, StrategySpec] = {}
    strategy_error_by_name: dict[str, Exception] = {}
    for strategy_name in strategy_names:
        try:
            strategy_by_name[strategy_name] = get_strategy(strategy_name)
        except Exception as error:
            strategy_error_by_name[strategy_name] = error
    selected_strategies = list(dict.fromkeys(strategy_by_name.values()))
    max_strategy_warmup = max(
        (
            backtest.required_warmup_bars(strategy=strategy)
            for strategy in selected_strategies
        ),
        default=0,
    )

    requested_research_time = _as_utc(
        research_end_time or datetime.now(timezone.utc)
    )
    alignment_timeframes = []
    for timeframe in timeframes:
        try:
            parse_timeframe(timeframe)
        except ValueError:
            # Keep invalid frames in the requested list so the normal per-frame
            # error row is still produced below.
            continue
        alignment_timeframes.append(timeframe)
    aligned_research_end = align_research_end_time(
        requested_research_time, alignment_timeframes
    )
    fee_rate = config.BACKTEST_FEE_PERCENT
    slippage_rate = config.BACKTEST_SLIPPAGE_PERCENT
    rows = []
    largest_lookback = max(lookbacks)
    history_by_timeframe = {}
    fetch_error_by_timeframe = {}
    data_quality_by_window = {}

    # Fetch each timeframe once for the largest window and the largest selected
    # strategy warm-up. All windows, strategies, and cost modes reuse this frame.
    for timeframe in dict.fromkeys(timeframes):
        if not selected_strategies:
            break
        try:
            parse_timeframe(timeframe)
            history_by_timeframe[timeframe] = backtest.fetch_history(
                largest_lookback,
                timeframe,
                warmup_bars=max_strategy_warmup,
                end_time=aligned_research_end,
            )
        except Exception as error:
            fetch_error_by_timeframe[timeframe] = error

    for lookback_days in lookbacks:
        requested_start = aligned_research_end - timedelta(days=lookback_days)
        for timeframe in timeframes:
            for strategy_name in strategy_names:
                strategy = strategy_by_name.get(strategy_name)
                parameters_json = (
                    _strategy_parameters_json(strategy) if strategy is not None else "{}"
                )
                coverage = None
                data_quality = data_quality_by_window.get((lookback_days, timeframe))
                try:
                    if timeframe in fetch_error_by_timeframe:
                        raise fetch_error_by_timeframe[timeframe]

                    bars = history_by_timeframe[timeframe]
                    if data_quality is None:
                        index = bars.index
                        window_bars = bars.loc[(index >= requested_start) & (index < aligned_research_end)]
                        data_quality = backtest.data_quality_diagnostics(
                            window_bars,
                            timeframe,
                            start_time=requested_start,
                            end_time=aligned_research_end,
                        )
                        data_quality_by_window[(lookback_days, timeframe)] = data_quality
                        print(
                            f"DATA QUALITY {lookback_days}d {timeframe}: "
                            f"expected={data_quality['expected_candle_count']} "
                            f"actual={data_quality['actual_candle_count']} missing="
                            f"{data_quality['missing_candle_count']} "
                            f"({data_quality['missing_candle_percent']:.2f}%) "
                            f"zero_volume={data_quality['zero_volume_candle_count']}"
                        )
                    if strategy_name in strategy_error_by_name:
                        raise strategy_error_by_name[strategy_name]
                    required_warmup = backtest.required_warmup_bars(strategy=strategy)
                    available_pretest_bars = int((bars.index < requested_start).sum())
                    if available_pretest_bars < required_warmup:
                        raise ValueError(
                            "Insufficient warm-up history: received "
                            f"{available_pretest_bars} pre-test bars; "
                            f"{required_warmup} required for {strategy_name}"
                        )

                    realistic_result = backtest.run_backtest(
                        bars,
                        starting_capital=starting_capital,
                        timeframe=timeframe,
                        fee_rate=fee_rate,
                        slippage=slippage_rate,
                        test_start=requested_start,
                        strategy=strategy,
                    )
                    coverage = _coverage_fields(
                        realistic_result,
                        requested_start=requested_start,
                        requested_end=aligned_research_end,
                    )
                    _validate_coverage(coverage, timeframe)
                    zero_cost_result = backtest.run_backtest(
                        bars,
                        starting_capital=starting_capital,
                        timeframe=timeframe,
                        fee_rate=0,
                        slippage=0,
                        test_start=requested_start,
                        strategy=strategy,
                    )
                    rows.append(
                        _research_row(
                            strategy,
                            realistic_result,
                            zero_cost_result,
                            timeframe=timeframe,
                            requested_research_time=requested_research_time,
                            research_end_time=aligned_research_end,
                            requested_start=requested_start,
                            lookback_days=lookback_days,
                            starting_capital=starting_capital,
                            fee_rate=fee_rate,
                            slippage_rate=slippage_rate,
                            coverage=coverage,
                            data_quality=data_quality,
                        )
                    )
                    print(f"{lookback_days}d {timeframe} {strategy_name}  OK")
                except Exception as error:
                    rows.append(
                        _failed_row(
                            strategy_name,
                            parameters_json,
                            error,
                            timeframe=timeframe,
                            requested_research_time=requested_research_time,
                            research_end_time=aligned_research_end,
                            requested_start=requested_start,
                            lookback_days=lookback_days,
                            starting_capital=starting_capital,
                            fee_rate=fee_rate,
                            slippage_rate=slippage_rate,
                            coverage=coverage,
                            data_quality=data_quality,
                        )
                    )
                    print(
                        f"{lookback_days}d {timeframe} {strategy_name}  ERROR: {error}"
                    )

    _print_report(
        rows,
        requested_research_time,
        aligned_research_end,
        lookbacks,
        strategy_names,
        starting_capital,
    )
    if output is not None:
        output_path = Path(output)
        with output_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"CSV saved to {output_path.resolve()}")
    failed = [
        f"{row['lookback_days']}d/{row['timeframe']}/{row['strategy_name']}"
        for row in rows
        if row["error"]
    ]
    if failed:
        print("Failed research runs: " + ", ".join(failed))
    return rows


def _fmt(value: Any, spec: str = ".2f") -> str:
    return "N/A" if value is None else format(value, spec)


def _print_report(
    rows: list[dict[str, Any]],
    requested_research_time: datetime,
    research_end_time: datetime,
    lookbacks: Iterable[int],
    strategy_names: Iterable[str],
    starting_capital: float,
) -> None:
    print("\n=== LOOKBACK / TIMEFRAME RESEARCH ===")
    print(f"Requested research time: {_timestamp(requested_research_time)}")
    print(f"Common aligned end: {_timestamp(research_end_time)}")
    print("Lookback windows: " + ", ".join(f"{days}d" for days in lookbacks))
    print("Strategies: " + ", ".join(strategy_names))
    print(f"Symbol: {config.SYMBOL}")
    print(
        f"Fee rate: {config.BACKTEST_FEE_PERCENT:.2%} | "
        f"Slippage rate: {config.BACKTEST_SLIPPAGE_PERCENT:.2%} | "
        f"Starting capital: ${starting_capital:,.2f}"
    )
    print(
        "Windows ending at the same time are nested and overlapping; they support "
        "recency analysis, not independent-regime conclusions."
    )
    print(
        "SamePath0% removes costs from realistic trades without rerunning signals; "
        "FreeRun0% reruns the strategy with zero costs and can follow a different path."
    )
    print("Drawdown is based on candle-mark equity, not tick-level intrabar worst case.")
    print(
        "Window  TF      Strategy              Trades  Trades/day  SamePath0% "
        " Net%    Drag%  FreeRun0% NetPF   Win%   Costs CandleMaxDD% BTC%"
    )
    for row in rows:
        window = f"{row['lookback_days']}d"
        if row["error"]:
            print(
                f"{window:<7} {row['timeframe']:<7} {row['strategy_name']:<21} "
                f"ERROR: {row['error']}"
            )
            continue
        print(
            f"{window:<7} {row['timeframe']:<7} {row['strategy_name']:<21} "
            f"{row['total_trades']:>6} {_fmt(row['trades_per_day']):>11} "
            f"{_fmt(row['same_path_zero_cost_return_percent']):>11} "
            f"{_fmt(row['realistic_net_return_percent']):>7} "
            f"{_fmt(row['pure_cost_drag_percent']):>7} "
            f"{_fmt(row['free_run_zero_cost_return_percent']):>9} "
            f"{_fmt(row['net_profit_factor']):>6} "
            f"{_fmt(row['win_rate']):>6} ${_fmt(row['total_costs']):>7} "
            f"{_fmt(row['max_drawdown']):>7} "
            f"{_fmt(row['raw_market_return_percent']):>8}"
        )
    print(
        "\nWindow  TF      Strategy              Avg hold  Median hold  Invested% "
        "Avg exp%  Max exp%  Same B&H%  Full B&H%  Daily Sharpe  Daily obs"
    )
    for row in rows:
        if row["error"]:
            continue
        window = f"{row['lookback_days']}d"
        print(
            f"{window:<7} {row['timeframe']:<7} {row['strategy_name']:<21} "
            f"{_fmt(row['average_holding_minutes']):>8} "
            f"{_fmt(row['median_holding_minutes']):>11} "
            f"{_fmt(row['time_invested_percent']):>9} "
            f"{_fmt(row['average_capital_exposure']):>8} "
            f"{_fmt(row['maximum_capital_exposure']):>8} "
            f"{_fmt(row['same_notional_buy_hold_return']):>10} "
            f"{_fmt(row['full_capital_buy_hold_return']):>10} "
            f"{_fmt(row['daily_sharpe']):>12} "
            f"{_fmt(row['daily_return_observations'], 'd'):>10}"
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare registered strategies across lookback windows and timeframes"
    )
    parser.add_argument(
        "--lookbacks",
        nargs="+",
        type=int,
        default=None,
        metavar="DAYS",
        help="Research windows in days (default: 90 180 365 730)",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=None,
        help="Backwards-compatible alias for a single lookback window",
    )
    parser.add_argument(
        "--timeframes", nargs="+", default=list(DEFAULT_TIMEFRAMES), metavar="TIMEFRAME"
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=list(DEFAULT_STRATEGIES),
        metavar="STRATEGY",
        help="Strategy names (default: ma_rsi_crossover)",
    )
    parser.add_argument(
        "--starting-capital", type=float, default=config.BACKTEST_STARTING_CAPITAL
    )
    parser.add_argument("--output", default="research_results.csv")
    parser.add_argument("--no-csv", action="store_true", help="Do not save a CSV")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.lookbacks is not None and args.lookback_days is not None:
        parser.error("--lookbacks and --lookback-days cannot be used together")
    lookbacks = args.lookbacks
    if lookbacks is None:
        lookbacks = (
            [args.lookback_days]
            if args.lookback_days is not None
            else list(DEFAULT_LOOKBACKS)
        )
    if any(lookback_days <= 0 for lookback_days in lookbacks):
        parser.error("All lookback periods must be positive")
    if args.starting_capital <= 0:
        parser.error("--starting-capital must be greater than zero")
    run_research(
        lookbacks=lookbacks,
        timeframes=args.timeframes,
        strategies=args.strategies,
        starting_capital=args.starting_capital,
        output=None if args.no_csv else args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
