"""Compare the configured strategy across candle timeframes."""

import argparse
import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import config
import backtest
import trade_config
from timeframes import parse_timeframe


DEFAULT_TIMEFRAMES = ("5Min", "15Min", "30Min", "1Hour")
DEFAULT_LOOKBACKS = (90, 180, 365, 730)
STRATEGY_NAME = "ma_rsi_crossover"

STRATEGY_PARAMETERS = (
    "fast_ma",
    "slow_ma",
    "rsi_period",
    "rsi_buy_threshold",
    "stop_loss_percent",
    "take_profit_percent",
    "trade_amount_usd",
    "max_position_usd",
)

METRICS = (
    "start_time",
    "end_time",
    "warmup_bars",
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
    "average_pnl_per_trade",
    "average_gross_return_per_trade",
    "average_net_return_per_trade",
    "profit_factor",
    "max_drawdown",
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
    "symbol",
    "fee_percent",
    "slippage_percent",
    "starting_capital",
)

CSV_FIELDS = (
    "strategy_name",
    "lookback_days",
    "timeframe",
    "error",
    *CONTEXT_FIELDS,
    *STRATEGY_PARAMETERS,
    *METRICS,
)


def _strategy_parameters() -> dict[str, Any]:
    return {
        "fast_ma": trade_config.FAST_MA,
        "slow_ma": trade_config.SLOW_MA,
        "rsi_period": trade_config.RSI_PERIOD,
        "rsi_buy_threshold": trade_config.RSI_BUY_THRESHOLD,
        "stop_loss_percent": trade_config.STOP_LOSS_PERCENT,
        "take_profit_percent": trade_config.TAKE_PROFIT_PERCENT,
        "trade_amount_usd": trade_config.TRADE_AMOUNT_USD,
        "max_position_usd": trade_config.MAX_POSITION_USD,
    }


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


def _research_row(
    timeframe: str,
    result: dict[str, Any],
    *,
    requested_research_time: datetime,
    research_end_time: datetime,
    lookback_days: int,
    starting_capital: float,
    fee_percent: float,
    slippage_percent: float,
) -> dict[str, Any]:
    start = result["start_time"]
    end = result["end_time"]
    duration_days = (end - start).total_seconds() / 86400
    total_trades = result["total_trades"]
    gross_pnl = result["gross_pnl"]
    return {
        "strategy_name": STRATEGY_NAME,
        "timeframe": timeframe,
        "error": "",
        "requested_research_time": _timestamp(requested_research_time),
        "research_end_time": _timestamp(research_end_time),
        "symbol": config.SYMBOL,
        "fee_percent": fee_percent,
        "slippage_percent": slippage_percent,
        "starting_capital": starting_capital,
        "lookback_days": lookback_days,
        **_strategy_parameters(),
        "start_time": _timestamp(start),
        "end_time": _timestamp(end),
        "warmup_bars": result["warmup_bars"],
        "total_trades": total_trades,
        "trades_per_day": total_trades / duration_days if duration_days > 0 else None,
        "winning_trades": result["winning_trades"],
        "losing_trades": result["losing_trades"],
        "win_rate": result["win_rate"],
        "gross_pnl": gross_pnl,
        "gross_return_percent": gross_pnl / starting_capital * 100,
        "total_fees": result["total_fees"],
        "total_slippage": result["total_slippage"],
        "total_costs": result["total_costs"],
        "net_pnl": result["net_profit"],
        "net_return_percent": result["strategy_return"],
        "average_pnl_per_trade": result["average_pnl_per_trade"],
        "average_gross_return_per_trade": result["average_gross_return"],
        "average_net_return_per_trade": result["average_net_return"],
        "profit_factor": result["profit_factor"],
        "max_drawdown": result["max_drawdown"],
        "daily_sharpe": result["daily_sharpe"],
        "daily_return_observations": result["daily_return_observations"],
        "average_holding_minutes": result["average_holding_minutes"],
        "median_holding_minutes": result["median_holding_minutes"],
        "time_invested_percent": result["time_invested_percent"],
        "average_capital_exposure": result["average_capital_exposure"],
        "maximum_capital_exposure": result["maximum_capital_exposure"],
        "full_capital_buy_hold_return": result["full_buy_hold_return"],
        "same_notional_buy_hold_return": result["same_notional_return"],
    }


def _failed_row(
    timeframe: str,
    error: Exception,
    *,
    requested_research_time: datetime,
    research_end_time: datetime,
    lookback_days: int,
    starting_capital: float,
    fee_percent: float,
    slippage_percent: float,
) -> dict[str, Any]:
    return {
        "strategy_name": STRATEGY_NAME,
        "timeframe": timeframe,
        "error": str(error),
        "requested_research_time": _timestamp(requested_research_time),
        "research_end_time": _timestamp(research_end_time),
        "symbol": config.SYMBOL,
        "fee_percent": fee_percent,
        "slippage_percent": slippage_percent,
        "starting_capital": starting_capital,
        "lookback_days": lookback_days,
        **_strategy_parameters(),
        **{field: None for field in METRICS},
    }


def run_research(
    *,
    lookbacks: Iterable[int] | None = None,
    lookback_days: int | None = None,
    timeframes: Iterable[str] = DEFAULT_TIMEFRAMES,
    starting_capital: float = config.BACKTEST_STARTING_CAPITAL,
    output: str | Path | None = "research_results.csv",
    research_end_time: datetime | None = None,
) -> list[dict[str, Any]]:
    """Run every requested lookback/timeframe pair in input order."""
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
    fee_percent = config.BACKTEST_FEE_PERCENT
    slippage_percent = config.BACKTEST_SLIPPAGE_PERCENT
    rows = []
    largest_lookback = max(lookbacks)
    history_by_timeframe = {}
    fetch_error_by_timeframe = {}

    # Fetch each distinct requested timeframe once for the largest window. Smaller
    # windows reuse this history and choose their own explicit test_start below.
    for timeframe in dict.fromkeys(timeframes):
        try:
            parse_timeframe(timeframe)
            history_by_timeframe[timeframe] = backtest.fetch_history(
                largest_lookback, timeframe, end_time=aligned_research_end
            )
        except Exception as error:
            fetch_error_by_timeframe[timeframe] = error

    for lookback_days in lookbacks:
        test_start = aligned_research_end - timedelta(days=lookback_days)
        for timeframe in timeframes:
            try:
                if timeframe in fetch_error_by_timeframe:
                    raise fetch_error_by_timeframe[timeframe]
                bars = history_by_timeframe[timeframe]
                warmup_count = int((bars.index < test_start).sum())
                required_warmup = backtest.required_warmup_bars()
                if warmup_count < required_warmup:
                    raise ValueError(
                        f"Insufficient warm-up history: received {warmup_count} "
                        f"pre-test bars; {required_warmup} required"
                    )
                result = backtest.run_backtest(
                    bars,
                    starting_capital=starting_capital,
                    timeframe=timeframe,
                    fee_rate=fee_percent,
                    slippage=slippage_percent,
                    test_start=test_start,
                )
                rows.append(
                    _research_row(
                        timeframe,
                        result,
                        requested_research_time=requested_research_time,
                        research_end_time=aligned_research_end,
                        lookback_days=lookback_days,
                        starting_capital=starting_capital,
                        fee_percent=fee_percent,
                        slippage_percent=slippage_percent,
                    )
                )
                print(f"{lookback_days}d {timeframe}  OK")
            except Exception as error:
                rows.append(
                    _failed_row(
                        timeframe,
                        error,
                        requested_research_time=requested_research_time,
                        research_end_time=aligned_research_end,
                        lookback_days=lookback_days,
                        starting_capital=starting_capital,
                        fee_percent=fee_percent,
                        slippage_percent=slippage_percent,
                    )
                )
                print(f"{lookback_days}d {timeframe}  ERROR: {error}")

    _print_report(
        rows,
        requested_research_time,
        aligned_research_end,
        lookbacks,
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
        f"{row['lookback_days']}d/{row['timeframe']}" for row in rows if row["error"]
    ]
    if failed:
        print("Failed research windows: " + ", ".join(failed))
    return rows


def _fmt(value: Any, spec: str = ".2f") -> str:
    return "N/A" if value is None else format(value, spec)


def _print_report(
    rows: list[dict[str, Any]],
    requested_research_time: datetime,
    research_end_time: datetime,
    lookbacks: Iterable[int],
    starting_capital: float,
) -> None:
    params = _strategy_parameters()
    print("\n=== LOOKBACK / TIMEFRAME RESEARCH ===")
    print(f"Requested research time: {_timestamp(requested_research_time)}")
    print(f"Common aligned end: {_timestamp(research_end_time)}")
    print(
        "Lookback windows: "
        + ", ".join(f"{lookback_days}d" for lookback_days in lookbacks)
    )
    print(f"Common period end: {_timestamp(research_end_time)}")
    print(
        f"Symbol: {config.SYMBOL} | Strategy: MA{params['fast_ma']}/{params['slow_ma']} "
        f"RSI{params['rsi_period']} threshold={params['rsi_buy_threshold']} "
        f"SL={params['stop_loss_percent']:.0%} TP={params['take_profit_percent']:.0%}"
    )
    print(
        f"Fee: {config.BACKTEST_FEE_PERCENT:.2%} | "
        f"Slippage: {config.BACKTEST_SLIPPAGE_PERCENT:.2%} | "
        f"Starting capital: ${starting_capital:,.2f}"
    )
    print(
        "Window  TF      Trades  Trades/day  Gross%    Net%      PF    Win% "
        " Costs    MaxDD%      B&H%"
    )
    for row in rows:
        window = f"{row['lookback_days']}d"
        if row["error"]:
            print(f"{window:<7} {row['timeframe']:<7} ERROR: {row['error']}")
            continue
        print(
            f"{window:<7} {row['timeframe']:<7} {row['total_trades']:>6} "
            f"{_fmt(row['trades_per_day']):>11} "
            f"{_fmt(row['gross_return_percent']):>8} "
            f"{_fmt(row['net_return_percent']):>8} "
            f"{_fmt(row['profit_factor']):>6} {_fmt(row['win_rate']):>7} "
            f"${_fmt(row['total_costs']):>7} {_fmt(row['max_drawdown']):>7} "
            f"{_fmt(row['full_capital_buy_hold_return']):>9}"
        )
    print(
        "\nWindow  TF      Avg hold (min)  Median hold (min)  Time invested "
        " Avg exposure  Max exposure  Daily Sharpe  Daily obs"
    )
    for row in rows:
        if row["error"]:
            continue
        window = f"{row['lookback_days']}d"
        print(
            f"{window:<7} {row['timeframe']:<7} "
            f"{_fmt(row['average_holding_minutes']):>14} "
            f"{_fmt(row['median_holding_minutes']):>18} "
            f"{_fmt(row['time_invested_percent']):>12}% "
            f"{_fmt(row['average_capital_exposure']):>11}% "
            f"{_fmt(row['maximum_capital_exposure']):>12}% "
            f"{_fmt(row['daily_sharpe']):>12} "
            f"{_fmt(row['daily_return_observations'], 'd'):>10}"
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare the configured MA/RSI strategy across lookback windows and timeframes"
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
        starting_capital=args.starting_capital,
        output=None if args.no_csv else args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
