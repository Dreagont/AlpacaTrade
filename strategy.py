from dataclasses import dataclass
from math import isfinite
from typing import Any, Callable

import pandas as pd

import trade_config

WARMUP_SAFETY_BARS = 10


@dataclass(frozen=True)
class Decision:
    action: str
    reason: str


@dataclass(frozen=True)
class StrategySpec:
    """Small adapter between a signal strategy and the shared backtest engine."""

    name: str
    _prepare_indicators: Callable
    _decide_at: Callable
    _warmup_lookback: Callable[[], int]
    _parameters: Callable[[], dict[str, Any]]
    _stop_loss: Callable[[], float | None]
    _take_profit: Callable[[], float | None]
    max_holding_minutes: float | None = None

    def required_warmup_bars(
        self, safety_margin: int = WARMUP_SAFETY_BARS
    ) -> int:
        return max(int(self._warmup_lookback()), 0) + max(int(safety_margin), 0)

    def prepare_indicators(self, df):
        return self._prepare_indicators(df)

    def decide_at(self, prepared_df, index: int) -> Decision:
        return self._decide_at(prepared_df, index)

    @property
    def stop_loss_percent(self) -> float | None:
        return self._stop_loss()

    @property
    def take_profit_percent(self) -> float | None:
        return self._take_profit()

    @property
    def parameters(self) -> dict[str, Any]:
        return dict(self._parameters())


def _calculate_ma_rsi_indicators(df, fast_ma: int, slow_ma: int, rsi_period: int):
    df = df.copy()

    df["ma_fast"] = df["close"].rolling(fast_ma).mean()
    df["ma_slow"] = df["close"].rolling(slow_ma).mean()

    delta = df["close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # This is simple moving-average RSI smoothing, not Wilder's standard smoothing.
    avg_gain = gain.rolling(rsi_period).mean()
    avg_loss = loss.rolling(rsi_period).mean()

    rs = avg_gain / avg_loss

    df["rsi"] = 100 - (100 / (1 + rs))

    return df


def calculate_indicators(df):
    """Prepare the current live MA/RSI indicators (simple-average RSI)."""
    return _calculate_ma_rsi_indicators(
        df, trade_config.FAST_MA, trade_config.SLOW_MA, trade_config.RSI_PERIOD
    )


def _ma_rsi_decide_at(df, index, rsi_buy_threshold: float):
    if index < 1 or index >= len(df):
        return Decision("HOLD", "not_enough_data")

    row = df.iloc[index]
    prev = df.iloc[index - 1]

    indicator_values = (
        prev["ma_fast"],
        prev["ma_slow"],
        row["ma_fast"],
        row["ma_slow"],
        row["rsi"],
    )
    if not all(isfinite(float(value)) for value in indicator_values):
        return Decision("HOLD", "indicators_not_ready")

    bullish_cross = (
        prev["ma_fast"] <= prev["ma_slow"]
        and row["ma_fast"] > row["ma_slow"]
    )

    bearish_cross = (
        prev["ma_fast"] >= prev["ma_slow"]
        and row["ma_fast"] < row["ma_slow"]
    )

    if bullish_cross:
        if row["rsi"] < rsi_buy_threshold:
            return Decision(
                "BUY",
                "bullish_ma_crossover_rsi_filter_passed",
            )
        return Decision("HOLD", "bullish_ma_crossover_rsi_filter_failed")

    if bearish_cross:
        return Decision("SELL", "bearish_ma_crossover")

    return Decision("HOLD", "no_crossover")


def decide_at(df, index):
    """Decide using the current live MA/RSI threshold."""
    return _ma_rsi_decide_at(df, index, trade_config.RSI_BUY_THRESHOLD)


def decide(df):
    return decide_at(df, len(df) - 1)


def _ma_warmup_lookback() -> int:
    return max(
        trade_config.FAST_MA,
        trade_config.SLOW_MA,
        trade_config.RSI_PERIOD,
    )


def _ma_parameters() -> dict[str, Any]:
    return {
        "fast_ma": trade_config.FAST_MA,
        "slow_ma": trade_config.SLOW_MA,
        "rsi_period": trade_config.RSI_PERIOD,
        "rsi_buy_threshold": trade_config.RSI_BUY_THRESHOLD,
        "stop_loss_percent": trade_config.STOP_LOSS_PERCENT,
        "take_profit_percent": trade_config.TAKE_PROFIT_PERCENT,
    }


def create_ma_rsi_strategy(
    fast_ma: int,
    slow_ma: int,
    rsi_period: int,
    rsi_buy_threshold: float,
    stop_loss_percent: float,
    take_profit_percent: float,
) -> StrategySpec:
    """Build an MA/RSI StrategySpec with the existing live decision semantics."""
    if min(fast_ma, slow_ma, rsi_period) <= 0:
        raise ValueError("MA periods and RSI period must be positive")
    if slow_ma <= fast_ma:
        raise ValueError("slow_ma must be greater than fast_ma")
    if not 0 <= rsi_buy_threshold <= 100:
        raise ValueError("RSI buy threshold must be between 0 and 100")
    if not 0 < stop_loss_percent < 1 or not 0 < take_profit_percent < 1:
        raise ValueError("Stop-loss and take-profit percentages must be in (0, 1)")

    def parameters() -> dict[str, Any]:
        return {
            "fast_ma": fast_ma,
            "slow_ma": slow_ma,
            "rsi_period": rsi_period,
            "rsi_buy_threshold": rsi_buy_threshold,
            "stop_loss_percent": stop_loss_percent,
            "take_profit_percent": take_profit_percent,
        }

    return StrategySpec(
        name="ma_rsi_crossover",
        _prepare_indicators=lambda df: _calculate_ma_rsi_indicators(
            df, fast_ma, slow_ma, rsi_period
        ),
        _decide_at=lambda df, index: _ma_rsi_decide_at(
            df, index, rsi_buy_threshold
        ),
        _warmup_lookback=lambda: max(fast_ma, slow_ma, rsi_period),
        _parameters=parameters,
        _stop_loss=lambda: stop_loss_percent,
        _take_profit=lambda: take_profit_percent,
    )


MA_RSI_CROSSOVER = StrategySpec(
    name="ma_rsi_crossover",
    _prepare_indicators=calculate_indicators,
    _decide_at=decide_at,
    _warmup_lookback=_ma_warmup_lookback,
    _parameters=_ma_parameters,
    _stop_loss=lambda: trade_config.STOP_LOSS_PERCENT,
    _take_profit=lambda: trade_config.TAKE_PROFIT_PERCENT,
)


def _prepare_donchian(entry_lookback: int, exit_lookback: int):
    def prepare(df):
        prepared = df.copy()
        prepared["donchian_entry_high"] = (
            prepared["high"].shift(1).rolling(entry_lookback).max()
        )
        prepared["donchian_exit_low"] = (
            prepared["low"].shift(1).rolling(exit_lookback).min()
        )
        return prepared

    return prepare


def _donchian_decide_at(prepared_df, index: int) -> Decision:
    if index < 0 or index >= len(prepared_df):
        return Decision("HOLD", "not_enough_data")

    row = prepared_df.iloc[index]
    values = (row["close"], row["donchian_entry_high"], row["donchian_exit_low"])
    try:
        if not all(isfinite(float(value)) for value in values):
            return Decision("HOLD", "indicators_not_ready")
    except (TypeError, ValueError):
        return Decision("HOLD", "indicators_not_ready")

    if row["close"] > row["donchian_entry_high"]:
        return Decision("BUY", "donchian_entry_breakout")
    if row["close"] < row["donchian_exit_low"]:
        return Decision("SELL", "donchian_exit_breakdown")
    return Decision("HOLD", "no_donchian_breakout")


def create_donchian_breakout_strategy(
    entry_lookback: int = 20,
    exit_lookback: int = 10,
) -> StrategySpec:
    if entry_lookback <= 0 or exit_lookback <= 0:
        raise ValueError("Donchian lookbacks must be positive")

    def parameters() -> dict[str, Any]:
        return {
            "entry_lookback": entry_lookback,
            "exit_lookback": exit_lookback,
            "stop_loss_percent": None,
            "take_profit_percent": None,
        }

    return StrategySpec(
        name="donchian_breakout",
        _prepare_indicators=_prepare_donchian(entry_lookback, exit_lookback),
        _decide_at=_donchian_decide_at,
        _warmup_lookback=lambda: max(entry_lookback, exit_lookback),
        _parameters=parameters,
        _stop_loss=lambda: None,
        _take_profit=lambda: None,
    )


def calculate_regime_filter(df, sma_period: int = 200, slope_lookback: int = 20):
    """Add a causal bullish regime using only the current and prior closes."""
    if sma_period <= 0 or slope_lookback <= 0:
        raise ValueError("Regime SMA period and slope lookback must be positive")
    prepared = df.copy()
    close = prepared["close"]
    prepared["regime_sma"] = close.rolling(sma_period, min_periods=sma_period).mean()
    prepared["regime_sma_previous"] = prepared["regime_sma"].shift(slope_lookback)
    ready = (
        prepared["close"].notna()
        & prepared["regime_sma"].notna()
        & prepared["regime_sma_previous"].notna()
    )
    bullish = pd.Series(pd.NA, index=prepared.index, dtype="boolean")
    bullish.loc[ready] = (
        (prepared.loc[ready, "close"] > prepared.loc[ready, "regime_sma"])
        & (prepared.loc[ready, "regime_sma"] > prepared.loc[ready, "regime_sma_previous"])
    ).to_numpy()
    prepared["regime_bullish"] = bullish
    return prepared


def create_regime_filtered_donchian_strategy(
    entry_lookback: int = 20,
    exit_lookback: int = 10,
    regime_sma_period: int = 200,
    regime_slope_lookback: int = 20,
) -> StrategySpec:
    if min(entry_lookback, exit_lookback, regime_sma_period, regime_slope_lookback) <= 0:
        raise ValueError("Donchian and regime lookbacks must be positive")
    prepare_donchian = _prepare_donchian(entry_lookback, exit_lookback)

    def prepare(df):
        return calculate_regime_filter(
            prepare_donchian(df), regime_sma_period, regime_slope_lookback
        )

    def decide(prepared_df, index):
        if index < 0 or index >= len(prepared_df):
            return Decision("HOLD", "regime_filter_not_ready")
        row = prepared_df.iloc[index]
        try:
            if not all(isfinite(float(row.get(name))) for name in (
                "regime_sma", "regime_sma_previous",
            )):
                return Decision("HOLD", "regime_filter_not_ready")
            regime = row.get("regime_bullish")
            if regime is None or regime is pd.NA or pd.isna(regime):
                return Decision("HOLD", "regime_filter_not_ready")
        except (TypeError, ValueError):
            return Decision("HOLD", "regime_filter_not_ready")

        if not bool(regime):
            return Decision("SELL", "regime_filter_off")
        return _donchian_decide_at(prepared_df, index)

    def parameters():
        return {
            "entry_lookback": entry_lookback,
            "exit_lookback": exit_lookback,
            "regime_sma_period": regime_sma_period,
            "regime_slope_lookback": regime_slope_lookback,
            "stop_loss_percent": None,
            "take_profit_percent": None,
        }

    return StrategySpec(
        name="donchian_regime_filter",
        _prepare_indicators=prepare,
        _decide_at=decide,
        _warmup_lookback=lambda: max(
            entry_lookback, exit_lookback,
            regime_sma_period + regime_slope_lookback,
        ),
        _parameters=parameters,
        _stop_loss=lambda: None,
        _take_profit=lambda: None,
    )


def create_regime_only_strategy() -> StrategySpec:
    """Create the frozen 4Hour regime-only strategy for research experiments."""
    sma_period = 200
    slope_lookback = 20

    def prepare(df):
        return calculate_regime_filter(
            df, sma_period=sma_period, slope_lookback=slope_lookback
        )

    def decide(prepared_df, index):
        if index < 0 or index >= len(prepared_df):
            return Decision("HOLD", "regime_filter_not_ready")
        regime = prepared_df.iloc[index].get("regime_bullish")
        if regime is None or regime is pd.NA or pd.isna(regime):
            return Decision("HOLD", "regime_filter_not_ready")
        if bool(regime):
            return Decision("BUY", "regime_on")
        return Decision("SELL", "regime_filter_off")

    return StrategySpec(
        name="regime_only_4h",
        _prepare_indicators=prepare,
        _decide_at=decide,
        _warmup_lookback=lambda: sma_period + slope_lookback,
        _parameters=lambda: {
            "regime_sma_period": sma_period,
            "regime_slope_lookback": slope_lookback,
            "stop_loss_percent": None,
            "take_profit_percent": None,
            "max_holding_minutes": None,
        },
        _stop_loss=lambda: None,
        _take_profit=lambda: None,
    )


DONCHIAN_BREAKOUT = create_donchian_breakout_strategy()
DONCHIAN_REGIME_FILTER = create_regime_filtered_donchian_strategy()
STRATEGY_REGISTRY = {
    MA_RSI_CROSSOVER.name: MA_RSI_CROSSOVER,
    DONCHIAN_BREAKOUT.name: DONCHIAN_BREAKOUT,
    DONCHIAN_REGIME_FILTER.name: DONCHIAN_REGIME_FILTER,
}


def get_strategy(name: str) -> StrategySpec:
    try:
        return STRATEGY_REGISTRY[name]
    except KeyError as error:
        choices = ", ".join(STRATEGY_REGISTRY)
        raise ValueError(f"Unknown strategy {name!r}; choose from: {choices}") from error
