"""Research-only 30-minute Donchian experiments with a causal 4-hour regime."""

import argparse
import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd

import backtest
import config
import regime_research as block_tools
from research import align_research_end_time
from strategy import (
    Decision,
    StrategySpec,
    WARMUP_SAFETY_BARS,
    calculate_regime_filter,
    create_donchian_breakout_strategy,
)
from timeframes import parse_timeframe


DEFAULT_BLOCK_DAYS = 90
DEFAULT_BLOCKS = 16
EXECUTION_TIMEFRAME = "30Min"
REGIME_TIMEFRAME = "4Hour"
DONCHIAN_ENTRY_LOOKBACK = 20
DONCHIAN_EXIT_LOOKBACK = 10
STOP_LOSS_PERCENT = 0.0075
TAKE_PROFIT_PERCENT = 0.015
MAX_HOLDING_MINUTES = 1440
REGIME_SMA_PERIOD = 200
REGIME_SLOPE_LOOKBACK = 20
REGIME_STALE_AFTER_MINUTES = 240
BASELINE_NAME = "short_term_donchian_30m"
GATED_NAME = "short_term_donchian_30m_regime"
DEFAULT_OUTPUT = "short_term_research_results.csv"

QUALITY_METRICS = (
    "expected_candle_count",
    "actual_candle_count",
    "missing_candle_count",
    "missing_candle_percent",
    "zero_volume_candle_count",
    "zero_volume_candle_percent",
)
QUALITY_FIELDS = tuple(
    f"{prefix}_{metric}"
    for prefix in ("execution", "regime")
    for metric in QUALITY_METRICS
)
EXIT_REASONS = (
    "stop_loss",
    "take_profit",
    "donchian_exit_breakdown",
    "max_holding_time",
    "regime_filter_off",
    "regime_unavailable",
)
TOTAL_METRICS = (
    "actual_start_time",
    "actual_end_time",
    "total_net_return_percent",
    "same_path_zero_cost_return_percent",
    "total_trades",
    "trades_per_30_days",
    "win_rate_percent",
    "net_profit_factor",
    "gross_profit_factor",
    "average_gross_return_percent_per_trade",
    "average_net_return_percent_per_trade",
    "expectancy_usd_per_completed_trade",
    "expectancy_percent_per_completed_trade",
    "average_winner_percent",
    "average_loser_percent",
    "total_fees_usd",
    "total_slippage_usd",
    "total_transaction_costs_usd",
    "average_transaction_cost_usd_per_completed_trade",
    "average_transaction_cost_percent_of_requested_notional_per_trade",
    "pure_cost_drag_percent",
    "average_same_path_zero_cost_return_percent_per_trade",
    "average_realized_cost_percent_of_requested_notional_per_trade",
    "average_cost_headroom_percent_per_trade",
    "break_even_total_cost_budget_percent_per_trade",
    "break_even_cost_budget_note",
    "average_holding_minutes",
    "median_holding_minutes",
    "shortest_holding_minutes",
    "longest_holding_minutes",
    "candle_mark_max_drawdown_percent",
    "time_invested_percent",
    "average_capital_exposure_percent",
    "ending_equity",
    "open_position_at_end",
    *(f"total_{reason}_exit_count" for reason in EXIT_REASONS),
)
BLOCK_METRICS = (
    "raw_btc_return_percent",
    "starting_equity",
    "ending_equity",
    "block_net_return_percent",
    "block_candle_mark_max_drawdown_percent",
    "entries_in_block",
    "exits_in_block",
    "completed_trades_in_block",
    "win_rate_percent_in_block",
    "charged_costs_in_block",
    "time_invested_percent",
    "average_capital_exposure_percent",
    "regime_delta_percent",
    *(f"block_{reason}_exit_count" for reason in EXIT_REASONS),
)
CSV_FIELDS = tuple(dict.fromkeys((
    "row_type",
    "strategy_name",
    "strategy_parameters_json",
    "error",
    "requested_research_time",
    "aligned_research_end",
    "execution_timeframe",
    "regime_timeframe",
    "fee_profile", "fee_rate",
    "slippage_rate",
    "configured_nominal_round_trip_friction_percent",
    "starting_capital",
    "block_index",
    "block_start",
    "block_end",
    "block_days",
    *QUALITY_FIELDS,
    *TOTAL_METRICS,
    *BLOCK_METRICS,
)))


def _as_utc(value: datetime | pd.Timestamp) -> datetime:
    value = value.to_pydatetime() if isinstance(value, pd.Timestamp) else value
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _utc_bars(bars: pd.DataFrame, label: str) -> pd.DataFrame:
    if bars.empty:
        raise ValueError(f"No {label} candles were returned")
    normalized = bars.copy()
    normalized.index = pd.DatetimeIndex(pd.to_datetime(normalized.index, utc=True))
    if normalized.index.has_duplicates:
        raise ValueError(f"{label} candles contain duplicate timestamps")
    return normalized.sort_index()


def create_short_term_strategies() -> tuple[StrategySpec, StrategySpec]:
    """Create the frozen A/B research pair without touching the live registry."""
    donchian = create_donchian_breakout_strategy(
        DONCHIAN_ENTRY_LOOKBACK, DONCHIAN_EXIT_LOOKBACK
    )

    def shared_parameters() -> dict[str, Any]:
        return {
            "entry_lookback": DONCHIAN_ENTRY_LOOKBACK,
            "exit_lookback": DONCHIAN_EXIT_LOOKBACK,
            "stop_loss_percent": STOP_LOSS_PERCENT,
            "take_profit_percent": TAKE_PROFIT_PERCENT,
            "max_holding_minutes": MAX_HOLDING_MINUTES,
        }

    baseline = StrategySpec(
        name=BASELINE_NAME,
        _prepare_indicators=donchian.prepare_indicators,
        _decide_at=donchian.decide_at,
        _warmup_lookback=lambda: max(
            DONCHIAN_ENTRY_LOOKBACK, DONCHIAN_EXIT_LOOKBACK
        ),
        _parameters=shared_parameters,
        _stop_loss=lambda: STOP_LOSS_PERCENT,
        _take_profit=lambda: TAKE_PROFIT_PERCENT,
        max_holding_minutes=MAX_HOLDING_MINUTES,
    )

    def prepare_gated(bars: pd.DataFrame) -> pd.DataFrame:
        if "regime_bullish_4h" not in bars.columns:
            raise ValueError(
                "Causal 4Hour regime data must be merged first; missing "
                "'regime_bullish_4h' column"
            )
        return donchian.prepare_indicators(bars)

    def decide_gated(prepared: pd.DataFrame, index: int) -> Decision:
        if "regime_bullish_4h" not in prepared.columns:
            raise ValueError(
                "Causal 4Hour regime data must be merged first; missing "
                "'regime_bullish_4h' column"
            )
        if index < 0 or index >= len(prepared):
            return Decision("SELL", "regime_unavailable")
        regime = prepared.iloc[index]["regime_bullish_4h"]
        if pd.isna(regime):
            return Decision("SELL", "regime_unavailable")
        if not bool(regime):
            return Decision("SELL", "regime_filter_off")
        return donchian.decide_at(prepared, index)

    def gated_parameters() -> dict[str, Any]:
        return {
            **shared_parameters(),
            "regime_sma_period": REGIME_SMA_PERIOD,
            "regime_slope_lookback": REGIME_SLOPE_LOOKBACK,
            "regime_stale_after_minutes": REGIME_STALE_AFTER_MINUTES,
        }

    gated = StrategySpec(
        name=GATED_NAME,
        _prepare_indicators=prepare_gated,
        _decide_at=decide_gated,
        _warmup_lookback=lambda: max(
            DONCHIAN_ENTRY_LOOKBACK, DONCHIAN_EXIT_LOOKBACK
        ),
        _parameters=gated_parameters,
        _stop_loss=lambda: STOP_LOSS_PERCENT,
        _take_profit=lambda: TAKE_PROFIT_PERCENT,
        max_holding_minutes=MAX_HOLDING_MINUTES,
    )
    assert_ab_invariant(baseline, gated)
    return baseline, gated


def assert_ab_invariant(
    baseline: StrategySpec | None = None,
    gated: StrategySpec | None = None,
) -> None:
    """Fail loudly if the regime gate changes any frozen execution rule."""
    if baseline is None or gated is None:
        baseline, gated = create_short_term_strategies()
    comparable = (
        "entry_lookback",
        "exit_lookback",
        "stop_loss_percent",
        "take_profit_percent",
        "max_holding_minutes",
    )
    baseline_parameters = baseline.parameters
    gated_parameters = gated.parameters
    for key in comparable:
        if baseline_parameters.get(key) != gated_parameters.get(key):
            raise AssertionError(f"A/B strategies differ in {key}")
    if baseline.stop_loss_percent != gated.stop_loss_percent:
        raise AssertionError("A/B stop loss rules must be identical")
    if baseline.take_profit_percent != gated.take_profit_percent:
        raise AssertionError("A/B take profit rules must be identical")
    if baseline.max_holding_minutes != gated.max_holding_minutes:
        raise AssertionError("A/B max_holding_minutes must be identical")
    if baseline_parameters.get("entry_lookback") != DONCHIAN_ENTRY_LOOKBACK:
        raise AssertionError("A/B entry lookback differs from frozen v1")
    if baseline_parameters.get("exit_lookback") != DONCHIAN_EXIT_LOOKBACK:
        raise AssertionError("A/B exit lookback differs from frozen v1")
    if baseline.stop_loss_percent != STOP_LOSS_PERCENT:
        raise AssertionError("A/B stop loss differs from frozen v1")
    if baseline.take_profit_percent != TAKE_PROFIT_PERCENT:
        raise AssertionError("A/B take profit differs from frozen v1")
    if baseline.max_holding_minutes != MAX_HOLDING_MINUTES:
        raise AssertionError("A/B max holding differs from frozen v1")


def merge_completed_4h_regime(
    execution_bars: pd.DataFrame,
    regime_bars: pd.DataFrame,
) -> pd.DataFrame:
    """Causally map completed 4Hour regime states to 30Min close decisions."""
    execution = _utc_bars(execution_bars, "30Min execution")
    regime = _utc_bars(regime_bars, "4Hour regime")
    execution_minutes = int(parse_timeframe(EXECUTION_TIMEFRAME)[1])
    regime_minutes = int(parse_timeframe(REGIME_TIMEFRAME)[1])

    prepared_regime = calculate_regime_filter(
        regime,
        sma_period=REGIME_SMA_PERIOD,
        slope_lookback=REGIME_SLOPE_LOOKBACK,
    )
    starts_4h = pd.DatetimeIndex(prepared_regime.index)
    right = pd.DataFrame(
        {
            "regime_source_close_time": starts_4h
            + pd.Timedelta(minutes=regime_minutes),
            "regime_bullish_4h": pd.array(
                prepared_regime["regime_bullish"], dtype="boolean"
            ),
        }
    ).sort_values("regime_source_close_time", kind="mergesort")

    left = execution.copy()
    left.index = pd.DatetimeIndex(execution.index)
    reserved = {
        "_execution_bar_start",
        "execution_decision_time",
        "regime_source_close_time",
        "regime_bullish_4h",
        "regime_age_minutes",
    }
    left = left.drop(columns=[name for name in reserved if name in left.columns])
    left["_execution_bar_start"] = execution.index
    left["execution_decision_time"] = execution.index + pd.Timedelta(
        minutes=execution_minutes
    )
    left = left.sort_values("execution_decision_time", kind="mergesort").reset_index(drop=True)

    merged = pd.merge_asof(
        left,
        right,
        left_on="execution_decision_time",
        right_on="regime_source_close_time",
        direction="backward",
        allow_exact_matches=True,
    )
    age = (
        merged["execution_decision_time"]
        - merged["regime_source_close_time"]
    ).dt.total_seconds() / 60
    merged["regime_age_minutes"] = age
    states = pd.array(merged["regime_bullish_4h"], dtype="boolean")
    states[age.to_numpy() >= REGIME_STALE_AFTER_MINUTES] = pd.NA
    merged["regime_bullish_4h"] = states
    merged.index = pd.DatetimeIndex(merged.pop("_execution_bar_start"))
    merged.index.name = execution.index.name
    merged = merged.sort_index()
    merged.attrs.update(execution_bars.attrs)
    return merged


def _bar_duration(timeframe: str) -> pd.Timedelta:
    return pd.Timedelta(minutes=parse_timeframe(timeframe)[1])


def _expected_bar_starts(start: Any, end: Any, timeframe: str) -> pd.DatetimeIndex:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    start_ts = start_ts.tz_localize("UTC") if start_ts.tzinfo is None else start_ts.tz_convert("UTC")
    end_ts = end_ts.tz_localize("UTC") if end_ts.tzinfo is None else end_ts.tz_convert("UTC")
    duration = _bar_duration(timeframe)
    duration_ns = duration.value
    epoch_ns = start_ts.value
    first_ns = ((epoch_ns + duration_ns - 1) // duration_ns) * duration_ns
    first = pd.Timestamp(first_ns, tz="UTC")
    last = end_ts - duration
    if last < first:
        return pd.DatetimeIndex([], tz="UTC")
    return pd.date_range(start=first, end=last, freq=duration)


def _missing_timestamps(
    bars: pd.DataFrame, timeframe: str, start: Any, end: Any
) -> pd.DatetimeIndex:
    expected = _expected_bar_starts(start, end, timeframe)
    index = pd.DatetimeIndex(pd.to_datetime(bars.index, utc=True))
    return expected.difference(index)


def _quality_for_span(
    bars: pd.DataFrame, timeframe: str, start: Any, end: Any
) -> dict[str, int | float]:
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    span = bars.loc[(bars.index >= start_ts) & (bars.index < end_ts)]
    return backtest.data_quality_diagnostics(
        span, timeframe, start_time=start_ts, end_time=end_ts
    )


def _print_data_quality(
    label: str,
    metrics: dict[str, Any],
    bars: pd.DataFrame,
    timeframe: str,
    start: Any,
    end: Any,
) -> None:
    print(
        f"DATA QUALITY {label}: expected={metrics['expected_candle_count']} "
        f"actual={metrics['actual_candle_count']} "
        f"missing={metrics['missing_candle_count']} "
        f"({metrics['missing_candle_percent']:.2f}%) "
        f"zero_volume={metrics['zero_volume_candle_count']}"
    )
    missing = _missing_timestamps(bars, timeframe, start, end)
    if len(missing) <= 20:
        values = [timestamp.isoformat() for timestamp in missing]
        print(f"  missing timestamps: {values if values else 'none'}")
    else:
        first = [timestamp.isoformat() for timestamp in missing[:3]]
        last = [timestamp.isoformat() for timestamp in missing[-3:]]
        print(
            f"  missing timestamp count={len(missing)} first={first} last={last}"
        )


def _validate_fetch_window(
    bars: pd.DataFrame,
    timeframe: str,
    test_start: datetime,
    warmup_bars: int,
    label: str,
) -> None:
    expected_start = pd.Timestamp(test_start) - _bar_duration(timeframe) * warmup_bars
    requested_start = bars.attrs.get("requested_start_time")
    if requested_start is None:
        raise ValueError(
            f"Insufficient {label} warm-up: fetch metadata does not confirm "
            "the requested warm-up window"
        )
    requested_start = pd.Timestamp(requested_start)
    requested_start = (
        requested_start.tz_localize("UTC")
        if requested_start.tzinfo is None
        else requested_start.tz_convert("UTC")
    )
    if requested_start > expected_start:
        raise ValueError(
            f"Insufficient {label} warm-up requested: history starts at "
            f"{requested_start.isoformat()}, expected at or before {expected_start.isoformat()}"
        )


def _coverage_error(
    bars: pd.DataFrame,
    timeframe: str,
    block: dict[str, Any],
    label: str,
) -> str | None:
    start = pd.Timestamp(block["block_start"])
    end = pd.Timestamp(block["block_end"])
    duration = _bar_duration(timeframe)
    span = bars.loc[(bars.index >= start) & (bars.index < end)]
    if span.empty:
        return f"{label} has no candles inside block {block['block_index']}"
    if span.index[0] != start or span.index[-1] + duration != end:
        return (
            f"{label} does not cover exact block boundaries for block "
            f"{block['block_index']} [{start.isoformat()}, {end.isoformat()})"
        )
    return None


def _total_metrics(result: dict[str, Any], span_days: float, fee_rate: float, slippage: float) -> dict[str, Any]:
    trades = result["trades"].copy()
    trade_count = len(trades)
    if trade_count:
        winners = trades.loc[trades["net_pnl"] > 0]
        losers = trades.loc[trades["net_pnl"] < 0]
        requested = pd.to_numeric(trades["requested_notional"], errors="coerce")
        valid_requested = requested.where(requested > 0)
        same_path_per_trade = (
            trades["same_path_zero_cost_pnl"] / valid_requested * 100
        )
        costs_per_trade = (
            (trades["fees"] + trades["slippage_cost"]) / valid_requested * 100
        )
        expectancy_percent = trades["net_pnl"] / valid_requested * 100
        avg_same_path = float(same_path_per_trade.mean())
        avg_realized_cost_percent = float(costs_per_trade.mean())
        avg_headroom = avg_same_path - avg_realized_cost_percent
        break_even_note = (
            "no non-negative transaction cost can rescue the average trade path"
            if avg_same_path <= 0
            else ""
        )
        exit_counts = {
            f"total_{reason}_exit_count": int((trades["exit_reason"] == reason).sum())
            for reason in EXIT_REASONS
        }
        average_transaction_cost = float(
            (trades["fees"] + trades["slippage_cost"]).mean()
        )
        average_transaction_cost_percent = avg_realized_cost_percent
        average_gross_return = float(trades["gross_return_percent"].mean())
        average_net_return = float(trades["return_percent"].mean())
        expectancy_usd = float(trades["net_pnl"].mean())
        average_winner = (
            float(winners["return_percent"].mean()) if not winners.empty else None
        )
        average_loser = (
            float(losers["return_percent"].mean()) if not losers.empty else None
        )
    else:
        winners = losers = trades
        avg_same_path = avg_realized_cost_percent = avg_headroom = None
        break_even_note = ""
        exit_counts = {f"total_{reason}_exit_count": 0 for reason in EXIT_REASONS}
        average_transaction_cost = average_transaction_cost_percent = None
        average_gross_return = average_net_return = expectancy_usd = None
        expectancy_percent = average_winner = average_loser = None

    break_even_budget = avg_same_path
    return {
        "actual_start_time": result["start_time"].isoformat(),
        "actual_end_time": result["end_time"].isoformat(),
        "total_net_return_percent": result["strategy_return"],
        "same_path_zero_cost_return_percent": result[
            "same_path_zero_cost_return_percent"
        ],
        "total_trades": result["total_trades"],
        "trades_per_30_days": (
            result["total_trades"] / span_days * 30 if span_days > 0 else None
        ),
        "win_rate_percent": result["win_rate"],
        "net_profit_factor": result["net_profit_factor"],
        "gross_profit_factor": result["gross_profit_factor"],
        "average_gross_return_percent_per_trade": average_gross_return,
        "average_net_return_percent_per_trade": average_net_return,
        "expectancy_usd_per_completed_trade": expectancy_usd,
        "expectancy_percent_per_completed_trade": (
            float(expectancy_percent.mean()) if trade_count else None
        ),
        "average_winner_percent": average_winner,
        "average_loser_percent": average_loser,
        "total_fees_usd": result["total_fees"],
        "total_slippage_usd": result["total_slippage"],
        "total_transaction_costs_usd": result["total_costs"],
        "average_transaction_cost_usd_per_completed_trade": average_transaction_cost,
        "average_transaction_cost_percent_of_requested_notional_per_trade": average_transaction_cost_percent,
        "pure_cost_drag_percent": result["pure_cost_drag_percent"],
        "configured_nominal_round_trip_friction_percent": 2
        * (fee_rate + slippage)
        * 100,
        "average_same_path_zero_cost_return_percent_per_trade": avg_same_path,
        "average_realized_cost_percent_of_requested_notional_per_trade": avg_realized_cost_percent,
        "average_cost_headroom_percent_per_trade": avg_headroom,
        "break_even_total_cost_budget_percent_per_trade": break_even_budget,
        "break_even_cost_budget_note": break_even_note,
        "average_holding_minutes": result["average_holding_minutes"],
        "median_holding_minutes": result["median_holding_minutes"],
        "shortest_holding_minutes": result["shortest_holding_minutes"],
        "longest_holding_minutes": result["longest_holding_minutes"],
        "candle_mark_max_drawdown_percent": result[
            "candle_mark_max_drawdown_percent"
        ],
        "time_invested_percent": result["time_invested_percent"],
        "average_capital_exposure_percent": result["average_capital_exposure"],
        "ending_equity": result["ending_equity"],
        "open_position_at_end": result["open_position"] is not None,
        **exit_counts,
    }


def _block_metrics(
    result: dict[str, Any],
    execution_bars: pd.DataFrame,
    block: dict[str, Any],
) -> dict[str, Any]:
    metrics = block_tools._block_metrics(
        result, execution_bars, block, timeframe=EXECUTION_TIMEFRAME
    )
    trades = result["trades"].copy()
    if trades.empty:
        exited = trades
    else:
        exit_times = pd.to_datetime(trades["exit_time"], utc=True)
        start, end = pd.Timestamp(block["block_start"]), pd.Timestamp(block["block_end"])
        exited = trades.loc[(exit_times >= start) & (exit_times < end)]
    exits = len(exited)
    return {
        "raw_btc_return_percent": metrics["raw_btc_return_percent"],
        "starting_equity": metrics["starting_equity"],
        "ending_equity": metrics["ending_equity"],
        "block_net_return_percent": metrics["block_net_return_percent"],
        "block_candle_mark_max_drawdown_percent": metrics[
            "block_candle_mark_max_drawdown_percent"
        ],
        "entries_in_block": metrics["entries_in_block"],
        "exits_in_block": metrics["exits_in_block"],
        "completed_trades_in_block": exits,
        "win_rate_percent_in_block": (
            float((exited["net_pnl"] > 0).sum()) / exits * 100 if exits else None
        ),
        "charged_costs_in_block": metrics["charged_costs_in_block"],
        "time_invested_percent": metrics["time_invested_percent"],
        "average_capital_exposure_percent": metrics["average_capital_exposure"],
        **{
            f"block_{reason}_exit_count": int((exited["exit_reason"] == reason).sum())
            for reason in EXIT_REASONS
        },
    }


def _quality_columns(
    execution_quality: dict[str, Any], regime_quality: dict[str, Any]
) -> dict[str, Any]:
    return {
        **{f"execution_{key}": value for key, value in execution_quality.items()},
        **{f"regime_{key}": value for key, value in regime_quality.items()},
    }


def _base_row(
    *, row_type: str, strategy: StrategySpec, requested_time: datetime,
    aligned_end: datetime, fee_rate: float, slippage: float,
    starting_capital: float, block: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = {field: None for field in CSV_FIELDS}
    row.update(
        {
            "row_type": row_type,
            "strategy_name": strategy.name,
            "strategy_parameters_json": json.dumps(
                strategy.parameters, sort_keys=True, separators=(",", ":")
            ),
            "error": "",
            "requested_research_time": requested_time.isoformat(),
            "aligned_research_end": aligned_end.isoformat(),
            "execution_timeframe": EXECUTION_TIMEFRAME,
            "regime_timeframe": REGIME_TIMEFRAME,
            "fee_rate": fee_rate,
            "slippage_rate": slippage,
            "configured_nominal_round_trip_friction_percent": 2
            * (fee_rate + slippage)
            * 100,
            "starting_capital": starting_capital,
        }
    )
    if block is not None:
        row.update(
            {
                "block_index": block["block_index"],
                "block_start": block["block_start"].isoformat(),
                "block_end": block["block_end"].isoformat(),
                "block_days": block["block_days"],
            }
        )
    return row


def _check_first_regime_decision(
    merged_bars: pd.DataFrame,
    regime_bars: pd.DataFrame,
    test_start: datetime,
    requested_regime_start: pd.Timestamp,
) -> None:
    start = pd.Timestamp(test_start)
    if start not in merged_bars.index:
        return  # The corresponding block will be reported invalid by boundary checks.
    first_decision = start + _bar_duration(EXECUTION_TIMEFRAME)
    state = merged_bars.loc[start, "regime_bullish_4h"]
    if not pd.isna(state):
        return

    missing = _missing_timestamps(
        regime_bars, REGIME_TIMEFRAME, requested_regime_start, first_decision
    )
    if len(missing) == 0:
        raise ValueError(
            "Insufficient 4Hour warm-up: the first 30Min execution decision has "
            "no fully initialized causal regime despite complete requested history"
        )
    if len(missing) <= 20:
        gap_report = [timestamp.isoformat() for timestamp in missing]
    else:
        gap_report = {
            "count": len(missing),
            "first": [timestamp.isoformat() for timestamp in missing[:3]],
            "last": [timestamp.isoformat() for timestamp in missing[-3:]],
        }
    print(
        "4Hour warm-up readiness: first execution decision is unavailable because "
        f"of missing historical regime candle(s) {gap_report}; state will remain NA "
        "until a fresh, completed regime bar is available."
    )


def run_short_term_research(
    *,
    block_days: int = DEFAULT_BLOCK_DAYS,
    blocks: int = DEFAULT_BLOCKS,
    output: str | Path | None = DEFAULT_OUTPUT,
    research_end_time: datetime | None = None,
    fee_profile: str = config.BACKTEST_FEE_PROFILE,
) -> list[dict[str, Any]]:
    """Run both frozen strategies once continuously, then report 90-day blocks."""
    if block_days <= 0 or blocks <= 0:
        raise ValueError("block-days and blocks must be greater than zero")
    requested_time = _as_utc(research_end_time or datetime.now(timezone.utc))
    aligned_end = align_research_end_time(
        requested_time, [EXECUTION_TIMEFRAME, REGIME_TIMEFRAME]
    )
    regime_blocks = block_tools.build_blocks(
        aligned_end, block_days=block_days, blocks=blocks
    )
    for timeframe in (EXECUTION_TIMEFRAME, REGIME_TIMEFRAME):
        boundary_error = block_tools._boundary_error(
            timeframe, aligned_end, block_days, blocks
        )
        if boundary_error:
            raise ValueError(boundary_error)

    baseline, gated = create_short_term_strategies()
    strategies = (baseline, gated)
    assert_ab_invariant(baseline, gated)
    test_start = regime_blocks[0]["block_start"]
    total_days = block_days * blocks
    starting_capital = config.BACKTEST_STARTING_CAPITAL
    rates = config.get_fee_profile(fee_profile)
    fee_rate = rates["fee_rate"]
    slippage = rates["slippage_rate"]
    execution_warmup = backtest.required_warmup_bars(strategy=baseline)
    regime_warmup = (
        REGIME_SMA_PERIOD
        + REGIME_SLOPE_LOOKBACK
        + WARMUP_SAFETY_BARS
    )
    requested_regime_start = (
        pd.Timestamp(test_start)
        - _bar_duration(REGIME_TIMEFRAME) * regime_warmup
    )

    execution_bars = _utc_bars(
        backtest.fetch_history(
            total_days,
            EXECUTION_TIMEFRAME,
            warmup_bars=execution_warmup,
            end_time=aligned_end,
        ),
        "30Min execution",
    )
    regime_bars = _utc_bars(
        backtest.fetch_history(
            total_days,
            REGIME_TIMEFRAME,
            warmup_bars=regime_warmup,
            end_time=aligned_end,
        ),
        "4Hour regime",
    )
    _validate_fetch_window(
        execution_bars, EXECUTION_TIMEFRAME, test_start, execution_warmup,
        "30Min execution",
    )
    _validate_fetch_window(
        regime_bars, REGIME_TIMEFRAME, test_start, regime_warmup,
        "4Hour regime",
    )
    execution_available = int((execution_bars.index < pd.Timestamp(test_start)).sum())
    if execution_available < DONCHIAN_ENTRY_LOOKBACK:
        raise ValueError(
            f"Insufficient 30Min Donchian warm-up: {execution_available} pre-test "
            f"candles; {DONCHIAN_ENTRY_LOOKBACK} are required"
        )
    merged_bars = merge_completed_4h_regime(execution_bars, regime_bars)
    _check_first_regime_decision(
        merged_bars, regime_bars, test_start, requested_regime_start
    )

    execution_span_quality = _quality_for_span(
        execution_bars, EXECUTION_TIMEFRAME, test_start, aligned_end
    )
    regime_span_quality = _quality_for_span(
        regime_bars, REGIME_TIMEFRAME, test_start, aligned_end
    )
    _print_data_quality(
        "30Min whole research span", execution_span_quality,
        execution_bars, EXECUTION_TIMEFRAME, test_start, aligned_end,
    )
    _print_data_quality(
        "4Hour whole research span", regime_span_quality,
        regime_bars, REGIME_TIMEFRAME, test_start, aligned_end,
    )

    results: dict[str, dict[str, Any]] = {}
    for strategy in strategies:
        results[strategy.name] = backtest.run_backtest(
            merged_bars,
            starting_capital=starting_capital,
            timeframe=EXECUTION_TIMEFRAME,
            fee_rate=fee_rate,
            slippage=slippage,
            test_start=test_start,
            strategy=strategy,
            liquidate_at_end=False,
        )

    span_days = total_days
    rows: list[dict[str, Any]] = []
    total_row_by_strategy: dict[str, dict[str, Any]] = {}
    quality_total = _quality_columns(execution_span_quality, regime_span_quality)
    for strategy in strategies:
        result = results[strategy.name]
        row = _base_row(
            row_type="TOTAL", strategy=strategy,
            requested_time=requested_time, aligned_end=aligned_end,
            fee_rate=fee_rate, slippage=slippage,
            starting_capital=starting_capital,
        )
        row.update(quality_total)
        actual_span_days = (
            result["end_time"] - result["start_time"]
        ).total_seconds() / 86400
        row.update(_total_metrics(result, actual_span_days or span_days, fee_rate, slippage))
        total_row_by_strategy[strategy.name] = row
        rows.append(row)

    baseline_total = total_row_by_strategy[BASELINE_NAME]["total_net_return_percent"]
    gated_total = total_row_by_strategy[GATED_NAME]["total_net_return_percent"]
    total_row_by_strategy[GATED_NAME]["regime_delta_percent"] = gated_total - baseline_total

    block_rows: list[dict[str, Any]] = []
    for block in regime_blocks:
        execution_quality = _quality_for_span(
            execution_bars, EXECUTION_TIMEFRAME,
            block["block_start"], block["block_end"],
        )
        regime_quality = _quality_for_span(
            regime_bars, REGIME_TIMEFRAME,
            block["block_start"], block["block_end"],
        )
        _print_data_quality(
            f"30Min block {block['block_index']}", execution_quality,
            execution_bars, EXECUTION_TIMEFRAME,
            block["block_start"], block["block_end"],
        )
        _print_data_quality(
            f"4Hour block {block['block_index']}", regime_quality,
            regime_bars, REGIME_TIMEFRAME,
            block["block_start"], block["block_end"],
        )
        quality = _quality_columns(execution_quality, regime_quality)
        errors = [
            _coverage_error(execution_bars, EXECUTION_TIMEFRAME, block, "30Min execution data"),
            _coverage_error(regime_bars, REGIME_TIMEFRAME, block, "4Hour regime data"),
        ]
        block_error = next((error for error in errors if error), None)
        for strategy in strategies:
            row = _base_row(
                row_type="BLOCK", strategy=strategy,
                requested_time=requested_time, aligned_end=aligned_end,
                fee_rate=fee_rate, slippage=slippage,
                starting_capital=starting_capital, block=block,
            )
            row.update(quality)
            if block_error:
                row["error"] = block_error
            else:
                row.update(_block_metrics(results[strategy.name], execution_bars, block))
            block_rows.append(row)

    rows.extend(block_rows)
    for block in regime_blocks:
        baseline_row = next(
            row for row in block_rows
            if row["block_index"] == block["block_index"]
            and row["strategy_name"] == BASELINE_NAME
        )
        gated_row = next(
            row for row in block_rows
            if row["block_index"] == block["block_index"]
            and row["strategy_name"] == GATED_NAME
        )
        if not baseline_row["error"] and not gated_row["error"]:
            gated_row["regime_delta_percent"] = (
                gated_row["block_net_return_percent"]
                - baseline_row["block_net_return_percent"]
            )

    for row in rows:
        row.update(fee_profile=fee_profile, fee_rate=fee_rate, slippage_rate=slippage)

    _print_report(
        rows, results, regime_blocks, requested_time, aligned_end,
        starting_capital, fee_rate, slippage,
        default_window=research_end_time is None,
    )
    if output is not None:
        output_path = Path(output)
        with output_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"CSV saved to {output_path.resolve()}")
    return rows


def _format(value: Any, suffix: str = "") -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:.2f}{suffix}"


def _print_report(
    rows: list[dict[str, Any]],
    results: dict[str, dict[str, Any]],
    blocks: list[dict[str, Any]],
    requested_time: datetime,
    aligned_end: datetime,
    starting_capital: float,
    fee_rate: float,
    slippage: float,
    *,
    default_window: bool,
) -> None:
    print("\n=== SHORT-TERM BTC RESEARCH: 30Min DONCHIAN A/B ===")
    if rows:
        config.print_fee_profile(rows[0]["fee_profile"], rows[0]["fee_rate"], rows[0]["slippage_rate"])
    print(f"Requested research time: {requested_time.isoformat()}")
    print(f"Common aligned end: {aligned_end.isoformat()}")
    print(f"Test start: {blocks[0]['block_start'].isoformat()}")
    print(
        f"Execution timeframe: {EXECUTION_TIMEFRAME} | "
        f"Regime timeframe: {REGIME_TIMEFRAME}"
    )
    print(f"Blocks: {len(blocks)} x {blocks[0]['block_days']} days")
    print(
        f"Fee: {fee_rate:.2%} | Slippage: {slippage:.2%} | "
        f"Starting capital: ${starting_capital:,.2f}"
    )
    if default_window:
        print(
            "WARNING: the default development span has been heavily inspected; "
            "treat it as exploratory, not untouched holdout evidence."
        )
    else:
        print(
            "This run uses an explicitly supplied end-time; no other historical "
            "holdout interval is evaluated automatically."
        )
    print(
        "Holdout note: the 4Hour regime concept was previously inspected before "
        "this 30Min execution experiment; pre-2022 is therefore not a pristine "
        "regime-concept holdout. No earlier interval is evaluated automatically."
    )
    print(
        "Frozen A/B rules: Donchian 20/10, SL 0.75%, TP 1.5%, max hold 24h; "
        "the only difference is the frozen 4Hour regime gate."
    )
    print(
        "configured_nominal_round_trip_friction_percent (approximate): "
        f"{2 * (fee_rate + slippage) * 100:.3f}%"
    )

    for strategy_name in (BASELINE_NAME, GATED_NAME):
        row = next(
            item for item in rows
            if item["row_type"] == "TOTAL" and item["strategy_name"] == strategy_name
        )
        result = results[strategy_name]
        print(
            f"TOTAL {strategy_name}: net={_format(row['total_net_return_percent'], '%')} "
            f"zero_cost_same_path={_format(row['same_path_zero_cost_return_percent'], '%')} "
            f"completed_trades={row['total_trades']} "
            f"trades/30d={_format(row['trades_per_30_days'])} "
            f"win_rate={_format(row['win_rate_percent'], '%')} "
            f"net_PF={_format(row['net_profit_factor'])} "
            f"gross_PF={_format(row['gross_profit_factor'])} "
            f"max_DD={_format(row['candle_mark_max_drawdown_percent'], '%')} "
            f"costs=${row['total_transaction_costs_usd']:.2f} "
            f"invested={_format(row['time_invested_percent'], '%')} "
            f"avg_exposure={_format(row['average_capital_exposure_percent'], '%')}"
        )
        print(
            f"  expectancy=${_format(row['expectancy_usd_per_completed_trade'])} / "
            f"{_format(row['expectancy_percent_per_completed_trade'], '%')} per trade; "
            f"avg gross/net return="
            f"{_format(row['average_gross_return_percent_per_trade'], '%')}/"
            f"{_format(row['average_net_return_percent_per_trade'], '%')}; "
            f"avg winner/loser="
            f"{_format(row['average_winner_percent'], '%')}/"
            f"{_format(row['average_loser_percent'], '%')}; "
            f"fees/slippage=${row['total_fees_usd']:.2f}/"
            f"${row['total_slippage_usd']:.2f}; "
            f"avg cost/trade="
            f"${_format(row['average_transaction_cost_usd_per_completed_trade'])}/"
            f"{_format(row['average_transaction_cost_percent_of_requested_notional_per_trade'], '%')}; "
            f"pure_cost_drag={_format(row['pure_cost_drag_percent'], '%')}; "
            f"holding avg/median/min/max="
            f"{_format(row['average_holding_minutes'])}/"
            f"{_format(row['median_holding_minutes'])}/"
            f"{_format(row['shortest_holding_minutes'])}/"
            f"{_format(row['longest_holding_minutes'])} minutes; "
            f"avg same-path zero-cost edge="
            f"{_format(row['average_same_path_zero_cost_return_percent_per_trade'], '%')}; "
            f"avg realized cost="
            f"{_format(row['average_realized_cost_percent_of_requested_notional_per_trade'], '%')}; "
            f"avg headroom={_format(row['average_cost_headroom_percent_per_trade'], '%')}"
        )
        if row["break_even_cost_budget_note"]:
            print(f"  NOTE: {row['break_even_cost_budget_note']}")
        exit_summary = ", ".join(
            f"{reason}={row[f'total_{reason}_exit_count']}" for reason in EXIT_REASONS
        )
        print(f"  exits: {exit_summary}; open_at_end={result['open_position'] is not None}")

    gated_total_row = next(
        row for row in rows
        if row["row_type"] == "TOTAL" and row["strategy_name"] == GATED_NAME
    )
    print(
        "TOTAL RegimeDelta (gated - ungated): "
        f"{_format(gated_total_row['regime_delta_percent'], '%')}"
    )

    print("\nBlock comparison: RegimeDelta = gated block return - ungated block return")
    block_rows = [row for row in rows if row["row_type"] == "BLOCK"]
    deltas = []
    for block in blocks:
        gated = next(
            row for row in block_rows
            if row["block_index"] == block["block_index"]
            and row["strategy_name"] == GATED_NAME
        )
        baseline = next(
            row for row in block_rows
            if row["block_index"] == block["block_index"]
            and row["strategy_name"] == BASELINE_NAME
        )
        if gated["error"] or baseline["error"]:
            print(f"  block {block['block_index']}: INVALID {gated['error'] or baseline['error']}")
            continue
        delta = gated["regime_delta_percent"]
        deltas.append(delta)
        print(
            f"  block {block['block_index']:>2}: "
            f"ungated={_format(baseline['block_net_return_percent'], '%')} "
            f"gated={_format(gated['block_net_return_percent'], '%')} "
            f"RegimeDelta={_format(delta, '%')} "
            f"gated_entries={gated['entries_in_block']} "
            f"gated_exits={gated['exits_in_block']} "
            f"gated_costs=${gated['charged_costs_in_block']:.2f}"
        )
    if deltas:
        print(
            f"RegimeDelta summary: helped={sum(value > 0 for value in deltas)} "
            f"hurt={sum(value < 0 for value in deltas)} "
            f"equal={sum(value == 0 for value in deltas)} "
            f"average={mean(deltas):.3f}%"
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare frozen research-only 30Min Donchian strategies"
    )
    parser.add_argument("--block-days", type=int, default=DEFAULT_BLOCK_DAYS)
    parser.add_argument("--blocks", type=int, default=DEFAULT_BLOCKS)
    parser.add_argument(
        "--end-time",
        help="Freeze the common end boundary (ISO-8601 UTC timestamp)",
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--no-csv", action="store_true")
    config.add_fee_profile_argument(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
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
    run_short_term_research(
        fee_profile=args.fee_profile,
        block_days=args.block_days,
        blocks=args.blocks,
        output=None if args.no_csv else args.output,
        research_end_time=end_time,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
