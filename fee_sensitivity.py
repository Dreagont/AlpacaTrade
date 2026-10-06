"""Research-only full-path fee comparison; one history fetch, no trading DB access."""

import argparse
import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path

import backtest
import config
import regime_research as block_tools
from regime_filter_research import (
    DEFAULT_BLOCK_DAYS, DEFAULT_BLOCKS, _btc_buy_hold_metrics,
    _compounded_trade_return_percent, _span_bars,
)
from research import align_research_end_time
from report_output import csv_output_path
from timeframes import parse_timeframe
from strategy import get_strategy


def utc_end_time(value):
    try:
        end = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if end.tzinfo is None or end.utcoffset() != timedelta(0):
            raise ValueError("timestamp must include a UTC timezone")
        return end.astimezone(timezone.utc)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"Expected ISO-8601 UTC timestamp: {error}") from error


def run_fee_sensitivity(*, strategy_name="regime_only_4h", timeframe="4Hour",
                        blocks=DEFAULT_BLOCKS, block_days=DEFAULT_BLOCK_DAYS,
                        end_time=None, starting_capital=config.BACKTEST_STARTING_CAPITAL,
                        output=None):
    if blocks <= 0 or block_days <= 0 or starting_capital <= 0:
        raise ValueError("blocks, block_days and starting_capital must be positive")
    if output is not None:
        csv_output_path(output)
    strategy = get_strategy(strategy_name)
    aligned_end = align_research_end_time(end_time or datetime.now(timezone.utc), [timeframe])
    boundaries = block_tools.build_blocks(aligned_end, blocks=blocks, block_days=block_days)
    error = block_tools._boundary_error(timeframe, aligned_end, block_days, blocks)
    if error:
        raise ValueError(error)
    start = boundaries[0]["block_start"]
    warmup = backtest.required_warmup_bars(strategy=strategy)
    bars = backtest.fetch_history(blocks * block_days, timeframe,
                                 warmup_bars=warmup, end_time=aligned_end)
    if int((bars.index < start).sum()) < warmup:
        raise ValueError("Insufficient pre-test warm-up history")
    market = _span_bars(bars, start, aligned_end)
    duration = timedelta(minutes=parse_timeframe(timeframe)[1])
    if market.empty or market.index[0] != start or market.index[-1] + duration != aligned_end:
        raise ValueError("History does not cover the requested period boundaries")
    print(f"\n=== FEE SENSITIVITY: {strategy_name} {timeframe} ===")
    print(f"Period: {start.isoformat()} to {aligned_end.isoformat()}")
    print("Each profile reruns the entire strategy. Open positions are marked at final close "
          "with hypothetical exit fees/slippage; total costs include that exit.")
    rows = []
    for name in config.FEE_PROFILES:
        rates = config.get_fee_profile(name)
        fee, slip = rates["fee_rate"], rates["slippage_rate"]
        result = backtest.run_backtest(
            bars, starting_capital=starting_capital, timeframe=timeframe,
            fee_rate=fee, slippage=slip, fee_profile=name,
            test_start=start, strategy=strategy, liquidate_at_end=False,
        )
        costs = result["total_costs"]
        position = result.get("open_position")
        if position:
            close = float(market.iloc[-1]["close"])
            quantity = position["quantity_after_buy_fee"]
            costs += quantity * close * slip + quantity * close * (1 - slip) * fee
        row = {
            "fee_profile": name, **rates, "strategy_name": strategy_name,
            "timeframe": timeframe, "start_time": start.isoformat(),
            "end_time": aligned_end.isoformat(),
            "compounded_trade_return_percent": _compounded_trade_return_percent(result, market, fee, slip),
            "trades": result["total_trades"], "open_position_marked": bool(position),
            "total_costs_usd": costs, "time_invested_percent": result["time_invested_percent"],
            **_btc_buy_hold_metrics(market, fee, slip),
        }
        rows.append(row)
        config.print_fee_profile(name)
        print(f"  Compounded trade return: {row['compounded_trade_return_percent']:.4f}% | "
              f"Completed trades: {row['trades']} | Open marked: {bool(position)} | "
              f"Costs: ${costs:.4f} | Invested: {row['time_invested_percent']:.2f}% | "
              f"BTC buy & hold: {row['btc_buy_hold_return_percent']:.4f}% "
              f"(after costs: {row['btc_buy_hold_return_after_costs_percent']:.4f}%)")
    if output is not None:
        with csv_output_path(output).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"CSV saved to {Path(output).resolve()}")
    return rows


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", default="regime_only_4h")
    parser.add_argument("--timeframe", default="4Hour")
    parser.add_argument("--blocks", type=int, default=DEFAULT_BLOCKS)
    parser.add_argument("--block-days", type=int, default=DEFAULT_BLOCK_DAYS)
    parser.add_argument("--end-time", type=utc_end_time)
    parser.add_argument("--starting-capital", type=float, default=config.BACKTEST_STARTING_CAPITAL)
    parser.add_argument("--csv", nargs="?", const="fee_sensitivity.csv", help="Optional CSV path")
    return parser


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        run_fee_sensitivity(strategy_name=args.strategy, timeframe=args.timeframe,
                            blocks=args.blocks, block_days=args.block_days,
                            end_time=args.end_time, starting_capital=args.starting_capital,
                            output=args.csv)
    except (ValueError, RuntimeError) as error:
        parser.exit(1, f"FEE SENSITIVITY ERROR: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
