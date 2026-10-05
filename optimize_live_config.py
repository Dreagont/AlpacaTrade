"""Research-only 30-day MA/RSI configuration search; never changes live settings."""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import sys
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

import backtest
import config
import trade_config
from research import align_research_end_time
from strategy import StrategySpec, create_ma_rsi_strategy


DEFAULT_LOOKBACK_DAYS = 30
DEFAULT_TIMEFRAMES = ("5Min", "15Min", "30Min", "1Hour")
FAST_MA_VALUES = (5, 8, 10, 12, 15)
SLOW_MA_VALUES = (20, 30, 40, 50, 60)
RSI_PERIOD_VALUES = (7, 14, 21)
RSI_THRESHOLD_VALUES = (55, 60, 65, 70)
STOP_LOSS_VALUES = (0.01, 0.015, 0.02, 0.03)
TAKE_PROFIT_VALUES = (0.015, 0.02, 0.03, 0.04, 0.06)
SEGMENT_DAYS = 10
FETCH_ROBUSTNESS_BUFFER_BARS = 20
OUTPUT_PATH = "live_config_optimization_30d.csv"


@dataclass(frozen=True)
class Candidate:
    timeframe: str
    fast_ma: int
    slow_ma: int
    rsi_period: int
    rsi_threshold: int
    stop_loss: float
    take_profit: float


@dataclass(frozen=True)
class Segment:
    number: int
    start: pd.Timestamp
    end: pd.Timestamp


METRIC_FIELDS = (
    "completed_trades",
    "trades_per_day",
    "win_rate",
    "total_net_return_percent",
    "same_path_zero_cost_return_percent",
    "pure_cost_drag_percent",
    "net_profit_factor",
    "gross_profit_factor",
    "average_net_return_per_trade",
    "average_gross_return_per_trade",
    "total_transaction_costs",
    "max_drawdown_percent",
    "average_holding_minutes",
    "time_invested_percent",
)


def candidate_count(
    timeframes: Iterable[str] = DEFAULT_TIMEFRAMES,
) -> int:
    valid_pairs = sum(
        slow > fast for fast in FAST_MA_VALUES for slow in SLOW_MA_VALUES
    )
    return (
        len(tuple(timeframes))
        * valid_pairs
        * len(RSI_PERIOD_VALUES)
        * len(RSI_THRESHOLD_VALUES)
        * len(STOP_LOSS_VALUES)
        * len(TAKE_PROFIT_VALUES)
    )


def validate_candidate(candidate: Candidate) -> Candidate:
    if candidate.slow_ma <= candidate.fast_ma:
        raise ValueError("slow_ma must be greater than fast_ma")
    return candidate


def generate_candidates(
    timeframes: Iterable[str] = DEFAULT_TIMEFRAMES,
) -> list[Candidate]:
    candidates = []
    for timeframe, fast, slow, rsi_period, threshold, stop, target in itertools.product(
        timeframes,
        FAST_MA_VALUES,
        SLOW_MA_VALUES,
        RSI_PERIOD_VALUES,
        RSI_THRESHOLD_VALUES,
        STOP_LOSS_VALUES,
        TAKE_PROFIT_VALUES,
    ):
        if slow <= fast:
            continue
        candidates.append(
            Candidate(timeframe, fast, slow, rsi_period, threshold, stop, target)
        )
    return candidates


def build_segments(
    end_time: datetime | pd.Timestamp,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> tuple[Segment, Segment, Segment]:
    if lookback_days != 3 * SEGMENT_DAYS:
        raise ValueError("The stability check requires exactly 30 days")
    end = pd.Timestamp(end_time)
    if end.tzinfo is None:
        end = end.tz_localize("UTC")
    else:
        end = end.tz_convert("UTC")
    start = end - pd.Timedelta(days=lookback_days)
    return tuple(
        Segment(i + 1, start + pd.Timedelta(days=SEGMENT_DAYS * i),
                start + pd.Timedelta(days=SEGMENT_DAYS * (i + 1)))
        for i in range(3)
    )


def slice_segment(bars: pd.DataFrame, segment: Segment) -> pd.DataFrame:
    """Keep available warm-up plus this segment only; the backtest starts flat."""
    index = pd.DatetimeIndex(pd.to_datetime(bars.index, utc=True))
    start = segment.start
    end = segment.end
    selected = bars.loc[(index < end)].copy()
    selected.attrs.update(bars.attrs)
    selected.attrs["test_start"] = start
    selected.attrs["segment_end"] = end
    return selected


def minimum_sample_status(
    total_trades: int, segment_trade_counts: Iterable[int]
) -> tuple[bool, str]:
    counts = tuple(segment_trade_counts)
    reasons = []
    if total_trades < 10:
        reasons.append("fewer than 10 completed trades over 30 days")
    if sum(count >= 2 for count in counts) < 2:
        reasons.append("fewer than 2 segments with at least 2 completed trades")
    return not reasons, "; ".join(reasons)


def classify_candidate(
    *,
    eligible: bool,
    net_return: float | None,
    same_path_return: float | None,
    net_profit_factor: float | None,
    profitable_segment_count: int,
) -> str:
    if not eligible:
        return "INELIGIBLE"
    positive = (
        net_return is not None
        and net_return > 0
        and same_path_return is not None
        and same_path_return > 0
        and net_profit_factor is not None
        and net_profit_factor > 1
    )
    if positive and profitable_segment_count == 3:
        return "STRONG"
    if positive and profitable_segment_count >= 2:
        return "PROMISING"
    return "WEAK"


CLASS_RANK = {"STRONG": 3, "PROMISING": 2, "WEAK": 1, "INELIGIBLE": 0}


def _sort_number(value: Any, *, missing: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return missing
    return number if math.isfinite(number) else missing


def ranking_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        -CLASS_RANK[row["classification"]],
        -int(row["profitable_segment_count"]),
        -_sort_number(row["median_segment_return"], missing=-math.inf),
        -_sort_number(row["worst_segment_return"], missing=-math.inf),
        -_sort_number(row["total_net_return_percent"], missing=-math.inf),
        -_sort_number(row["net_profit_factor"], missing=-math.inf),
        _sort_number(row["max_drawdown_percent"], missing=math.inf),
    )


def _baseline_candidate(timeframe: str) -> Candidate:
    return Candidate(
        timeframe=timeframe,
        fast_ma=trade_config.FAST_MA,
        slow_ma=trade_config.SLOW_MA,
        rsi_period=trade_config.RSI_PERIOD,
        rsi_threshold=trade_config.RSI_BUY_THRESHOLD,
        stop_loss=trade_config.STOP_LOSS_PERCENT,
        take_profit=trade_config.TAKE_PROFIT_PERCENT,
    )


def _is_current_baseline(candidate: Candidate) -> bool:
    return (
        candidate.timeframe == trade_config.LIVE_TIMEFRAME
        and candidate == _baseline_candidate(trade_config.LIVE_TIMEFRAME)
    )


def _result_metrics(result: dict[str, Any], days: int) -> dict[str, Any]:
    return {
        "completed_trades": int(result["total_trades"]),
        "trades_per_day": result["total_trades"] / days,
        "win_rate": result["win_rate"],
        "total_net_return_percent": result["strategy_return"],
        "same_path_zero_cost_return_percent": result[
            "same_path_zero_cost_return_percent"
        ],
        "pure_cost_drag_percent": result["pure_cost_drag_percent"],
        "net_profit_factor": result["net_profit_factor"],
        "gross_profit_factor": result["gross_profit_factor"],
        "average_net_return_per_trade": result["average_net_return"],
        "average_gross_return_per_trade": result["average_gross_return"],
        "total_transaction_costs": result["total_costs"],
        "max_drawdown_percent": result["max_drawdown"],
        "average_holding_minutes": result["average_holding_minutes"],
        "time_invested_percent": result["time_invested_percent"],
    }


def _candidate_strategy(
    candidate: Candidate,
    bars: pd.DataFrame,
    indicator_cache: dict[tuple[str, int, int, int], pd.DataFrame],
) -> StrategySpec:
    strategy = create_ma_rsi_strategy(
        candidate.fast_ma,
        candidate.slow_ma,
        candidate.rsi_period,
        candidate.rsi_threshold,
        candidate.stop_loss,
        candidate.take_profit,
    )
    key = (
        candidate.timeframe,
        candidate.fast_ma,
        candidate.slow_ma,
        candidate.rsi_period,
    )
    if key not in indicator_cache:
        indicator_cache[key] = strategy.prepare_indicators(bars)
    prepared = indicator_cache[key]
    return replace(
        strategy,
        _prepare_indicators=lambda candidate_bars, cached=prepared: cached.loc[
            candidate_bars.index
        ],
    )


def _evaluate_candidate(
    candidate: Candidate,
    bars: pd.DataFrame,
    segments: tuple[Segment, Segment, Segment],
    indicator_cache: dict[tuple[str, int, int, int], pd.DataFrame],
) -> dict[str, Any]:
    strategy = _candidate_strategy(candidate, bars, indicator_cache)
    full = backtest.run_backtest(
        bars,
        starting_capital=config.BACKTEST_STARTING_CAPITAL,
        timeframe=candidate.timeframe,
        fee_rate=config.BACKTEST_FEE_PERCENT,
        slippage=config.BACKTEST_SLIPPAGE_PERCENT,
        test_start=segments[0].start,
        strategy=strategy,
    )
    row: dict[str, Any] = asdict(candidate)
    row["rsi_threshold"] = candidate.rsi_threshold
    row["is_current_live_baseline"] = _is_current_baseline(candidate)
    row.update(_result_metrics(full, DEFAULT_LOOKBACK_DAYS))
    segment_returns: list[float] = []
    segment_trade_counts: list[int] = []
    for segment in segments:
        segment_bars = slice_segment(bars, segment)
        outcome = backtest.run_backtest(
            segment_bars,
            starting_capital=config.BACKTEST_STARTING_CAPITAL,
            timeframe=candidate.timeframe,
            fee_rate=config.BACKTEST_FEE_PERCENT,
            slippage=config.BACKTEST_SLIPPAGE_PERCENT,
            test_start=segment.start,
            strategy=strategy,
        )
        segment_returns.append(float(outcome["strategy_return"]))
        segment_trade_counts.append(int(outcome["total_trades"]))
        row[f"segment_{segment.number}_net_return_percent"] = outcome["strategy_return"]
        row[f"segment_{segment.number}_completed_trades"] = outcome["total_trades"]

    profitable_count = sum(value > 0 for value in segment_returns)
    eligible, reason = minimum_sample_status(
        row["completed_trades"], segment_trade_counts
    )
    row.update(
        {
            "profitable_segment_count": profitable_count,
            "worst_segment_return": min(segment_returns),
            "median_segment_return": float(pd.Series(segment_returns).median()),
            "eligible_for_ranking": eligible,
            "ineligible_reason": reason,
            "classification": classify_candidate(
                eligible=eligible,
                net_return=row["total_net_return_percent"],
                same_path_return=row["same_path_zero_cost_return_percent"],
                net_profit_factor=row["net_profit_factor"],
                profitable_segment_count=profitable_count,
            ),
        }
    )
    return row


def _quality_line(timeframe: str, diagnostics: dict[str, Any]) -> None:
    print(
        f"DATA QUALITY {timeframe}: expected={diagnostics['expected_candle_count']} "
        f"actual={diagnostics['actual_candle_count']} "
        f"missing={diagnostics['missing_candle_count']} "
        f"({diagnostics['missing_candle_percent']:.2f}%) "
        f"zero_volume={diagnostics['zero_volume_candle_count']}"
    )


def _fmt(value: Any, fmt: str = ".2f") -> str:
    return "N/A" if value is None else format(value, fmt)


def _print_candidates(title: str, rows: list[dict[str, Any]]) -> None:
    print(title)
    headers = (
        "Rank", "Class", "TF", "Fast/Slow", "RSI P/T", "SL / TP", "Trades",
        "Net%", "SamePath0%", "NetPF", "MaxDD%", "S1%", "S2%", "S3%", "Prof Seg",
    )
    print(" | ".join(headers))
    for row in rows:
        print(
            " | ".join(
                (
                    str(row.get("rank", "BASELINE")), row["classification"], row["timeframe"],
                    f"{row['fast_ma']}/{row['slow_ma']}",
                    f"{row['rsi_period']}/{row['rsi_threshold']}",
                    f"{row['stop_loss']:.3f}/{row['take_profit']:.3f}",
                    str(row["completed_trades"]), _fmt(row["total_net_return_percent"]),
                    _fmt(row["same_path_zero_cost_return_percent"]),
                    _fmt(row["net_profit_factor"]), _fmt(row["max_drawdown_percent"]),
                    _fmt(row["segment_1_net_return_percent"]),
                    _fmt(row["segment_2_net_return_percent"]),
                    _fmt(row["segment_3_net_return_percent"]),
                    str(row["profitable_segment_count"]),
                )
            )
        )


def run_optimizer(
    *,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    timeframes: Iterable[str] = DEFAULT_TIMEFRAMES,
    end_time: datetime | None = None,
    output: str | Path | None = OUTPUT_PATH,
) -> list[dict[str, Any]]:
    if lookback_days != DEFAULT_LOOKBACK_DAYS:
        raise ValueError("This optimizer searches the fixed 30-day development window")
    selected_timeframes = tuple(dict.fromkeys(timeframes))
    if not selected_timeframes:
        raise ValueError("At least one timeframe is required")
    candidates = generate_candidates(selected_timeframes)
    print(f"Total candidates: {len(candidates)}")
    end = end_time or datetime.now(timezone.utc)
    aligned_end = align_research_end_time(end, selected_timeframes)
    segments = build_segments(aligned_end, lookback_days)

    max_lookback = max(max(FAST_MA_VALUES), max(SLOW_MA_VALUES), max(RSI_PERIOD_VALUES))
    max_required_warmup = max_lookback + 10
    bars_by_timeframe: dict[str, pd.DataFrame] = {}
    for timeframe in selected_timeframes:
        bars = backtest.fetch_history(
            lookback_days,
            timeframe,
            warmup_bars=max_required_warmup + FETCH_ROBUSTNESS_BUFFER_BARS,
            end_time=aligned_end,
        )
        bars_index = pd.DatetimeIndex(pd.to_datetime(bars.index, utc=True))
        available_warmup = int((bars_index < segments[0].start).sum())
        if available_warmup < max_required_warmup:
            raise RuntimeError(
                f"Insufficient warm-up history for {timeframe}: received "
                f"{available_warmup} pre-test bars; {max_required_warmup} required"
            )
        bars_by_timeframe[timeframe] = bars
        diagnostics = backtest.data_quality_diagnostics(
            bars,
            timeframe,
            start_time=segments[0].start,
            end_time=segments[-1].end,
        )
        _quality_line(timeframe, diagnostics)

    print(
        "This optimizer uses the last 30 days as DEVELOPMENT data.\n"
        "The winning configuration is selected from this data and its reported\n"
        "performance is therefore optimistically biased."
    )
    print(
        f"Fees={config.BACKTEST_FEE_PERCENT:g}, slippage={config.BACKTEST_SLIPPAGE_PERCENT:g}, "
        f"starting_capital={config.BACKTEST_STARTING_CAPITAL:g}, "
        f"TRADE_AMOUNT_USD={trade_config.TRADE_AMOUNT_USD:g}, "
        f"MAX_POSITION_USD={trade_config.MAX_POSITION_USD:g}"
    )

    indicator_cache: dict[tuple[str, int, int, int], pd.DataFrame] = {}
    rows = []
    total = len(candidates)
    for position, candidate in enumerate(candidates, 1):
        rows.append(
            _evaluate_candidate(
                candidate,
                bars_by_timeframe[candidate.timeframe],
                segments,
                indicator_cache,
            )
        )
        if position % 500 == 0 or position == total:
            print(f"evaluated {position} / {total} candidates")

    eligible = [row for row in rows if row["eligible_for_ranking"]]
    eligible.sort(key=ranking_key)
    for rank, row in enumerate(eligible, 1):
        row["rank"] = rank

    if output is not None:
        columns = [
            "rank", "classification", "eligible_for_ranking", "ineligible_reason",
            "is_current_live_baseline", "timeframe", "fast_ma", "slow_ma", "rsi_period",
            "rsi_threshold", "stop_loss", "take_profit", *METRIC_FIELDS,
            "segment_1_net_return_percent", "segment_2_net_return_percent",
            "segment_3_net_return_percent", "segment_1_completed_trades",
            "segment_2_completed_trades", "segment_3_completed_trades",
            "profitable_segment_count", "worst_segment_return", "median_segment_return",
        ]
        with Path(output).open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    _print_candidates("TOP 20 eligible candidates", eligible[:20])
    baseline = next((row for row in rows if row["is_current_live_baseline"]), None)
    print("CURRENT LIVE BASELINE")
    if baseline:
        _print_candidates("CURRENT LIVE BASELINE", [baseline])
    else:
        print("Current live settings are outside this timeframe's configured search space.")
    robust = next(
        (row for row in eligible if row["classification"] in {"STRONG", "PROMISING"}),
        None,
    )
    print("BEST ROBUST CANDIDATE")
    if robust:
        print(
            f"timeframe={robust['timeframe']} fast_ma={robust['fast_ma']} "
            f"slow_ma={robust['slow_ma']} rsi_period={robust['rsi_period']} "
            f"rsi_buy_threshold={robust['rsi_threshold']} "
            f"stop_loss_percent={robust['stop_loss']} "
            f"take_profit_percent={robust['take_profit']}"
        )
    else:
        print("NO ROBUSTLY PROFITABLE CONFIG FOUND IN THIS 30-DAY DEVELOPMENT WINDOW")
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS)
    parser.add_argument("--end-time", help="Freeze the common research end (ISO-8601 UTC)")
    parser.add_argument("--output", default=OUTPUT_PATH)
    parser.add_argument("--no-csv", action="store_true")
    args = parser.parse_args(argv)
    end_time = None
    if args.end_time:
        try:
            end_time = datetime.fromisoformat(args.end_time.replace("Z", "+00:00"))
        except ValueError as error:
            parser.error(f"invalid --end-time: {error}")
        if end_time.tzinfo is None:
            parser.error("--end-time must include a UTC timezone")
        if end_time.utcoffset() != timedelta(0):
            parser.error("--end-time must be UTC")
    try:
        run_optimizer(
            lookback_days=args.lookback_days,
            end_time=end_time,
            output=None if args.no_csv else args.output,
        )
    except (ValueError, RuntimeError) as error:
        print(f"Optimizer failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
