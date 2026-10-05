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
    "research_end_time",
    "symbol",
    "fee_percent",
    "slippage_percent",
    "starting_capital",
    "lookback_days",
)

CSV_FIELDS = ("timeframe", "error", *CONTEXT_FIELDS, *STRATEGY_PARAMETERS, *METRICS)


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


def _research_row(
    timeframe: str,
    result: dict[str, Any],
    *,
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
        "timeframe": timeframe,
        "error": "",
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
    research_end_time: datetime,
    lookback_days: int,
    starting_capital: float,
    fee_percent: float,
    slippage_percent: float,
) -> dict[str, Any]:
    return {
        "timeframe": timeframe,
        "error": str(error),
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
    lookback_days: int = 365,
    timeframes: Iterable[str] = DEFAULT_TIMEFRAMES,
    starting_capital: float = config.BACKTEST_STARTING_CAPITAL,
    output: str | Path | None = "research_results.csv",
    research_end_time: datetime | None = None,
) -> list[dict[str, Any]]:
    """Run one backtest per timeframe, preserving the supplied timeframe order."""
    if lookback_days <= 0:
        raise ValueError("lookback_days must be positive")
    if starting_capital <= 0:
        raise ValueError("starting_capital must be greater than zero")

    research_end_time = research_end_time or datetime.now(timezone.utc)
    if research_end_time.tzinfo is None:
        research_end_time = research_end_time.replace(tzinfo=timezone.utc)
    else:
        research_end_time = research_end_time.astimezone(timezone.utc)
    fee_percent = config.BACKTEST_FEE_PERCENT
    slippage_percent = config.BACKTEST_SLIPPAGE_PERCENT
    rows = []

    for timeframe in timeframes:
        try:
            # Validate through the shared parser; it also ensures the frame is usable by Alpaca.
            parse_timeframe(timeframe)
            bars = backtest.fetch_history(
                lookback_days, timeframe, end_time=research_end_time
            )
            test_start = bars.attrs.get(
                "test_start", research_end_time - timedelta(days=lookback_days)
            )
            warmup_count = int((bars.index < test_start).sum())
            required_warmup = backtest.required_warmup_bars()
            if warmup_count < required_warmup:
                raise ValueError(
                    f"Insufficient warm-up history: received {warmup_count} pre-test bars; "
                    f"{required_warmup} required"
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
                    research_end_time=research_end_time,
                    lookback_days=lookback_days,
                    starting_capital=starting_capital,
                    fee_percent=fee_percent,
                    slippage_percent=slippage_percent,
                )
            )
            print(f"{timeframe}  OK")
        except Exception as error:
            rows.append(
                _failed_row(
                    timeframe,
                    error,
                    research_end_time=research_end_time,
                    lookback_days=lookback_days,
                    starting_capital=starting_capital,
                    fee_percent=fee_percent,
                    slippage_percent=slippage_percent,
                )
            )
            print(f"{timeframe}  ERROR: {error}")

    _print_report(rows, research_end_time, lookback_days, starting_capital)
    if output is not None:
        output_path = Path(output)
        with output_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"CSV saved to {output_path.resolve()}")
    failed = [row["timeframe"] for row in rows if row["error"]]
    if failed:
        print("Failed timeframes: " + ", ".join(failed))
    return rows


def _fmt(value: Any, spec: str = ".2f") -> str:
    return "N/A" if value is None else format(value, spec)


def _print_report(
    rows: list[dict[str, Any]],
    research_end_time: datetime,
    lookback_days: int,
    starting_capital: float,
) -> None:
    params = _strategy_parameters()
    print("\n=== TIMEFRAME RESEARCH ===")
    print(f"Period: {lookback_days} calendar days ending {_timestamp(research_end_time)} UTC")
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
    print("TF      Trades  Trades/day  Gross%    Net%      PF    Win%    Costs    MaxDD%  Sharpe")
    for row in rows:
        if row["error"]:
            print(f"{row['timeframe']:<7} ERROR: {row['error']}")
            continue
        print(
            f"{row['timeframe']:<7} {row['total_trades']:>6} "
            f"{_fmt(row['trades_per_day']):>11} { _fmt(row['gross_return_percent']):>8} "
            f"{_fmt(row['net_return_percent']):>8} { _fmt(row['profit_factor']):>6} "
            f"{_fmt(row['win_rate']):>7} ${row['total_costs']:>7.2f} "
            f"{_fmt(row['max_drawdown']):>7} { _fmt(row['daily_sharpe']):>7}"
        )
    print("\nTF      Avg hold (min)  Median hold (min)  Time invested  Avg exposure  Same-notional B&H")
    for row in rows:
        if row["error"]:
            continue
        print(
            f"{row['timeframe']:<7} {_fmt(row['average_holding_minutes']):>14} "
            f"{_fmt(row['median_holding_minutes']):>18} "
            f"{_fmt(row['time_invested_percent']):>12}% "
            f"{_fmt(row['average_capital_exposure']):>11}% "
            f"{_fmt(row['same_notional_buy_hold_return']):>16}%"
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare the configured MA/RSI strategy across timeframes"
    )
    parser.add_argument("--lookback-days", type=int, default=365)
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
    if args.lookback_days <= 0:
        parser.error("--lookback-days must be positive")
    if args.starting_capital <= 0:
        parser.error("--starting-capital must be greater than zero")
    run_research(
        lookback_days=args.lookback_days,
        timeframes=args.timeframes,
        starting_capital=args.starting_capital,
        output=None if args.no_csv else args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
