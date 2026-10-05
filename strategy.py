from dataclasses import dataclass
from math import isfinite
from typing import Any, Callable

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


def calculate_indicators(df):
    df = df.copy()

    df["ma_fast"] = df["close"].rolling(trade_config.FAST_MA).mean()
    df["ma_slow"] = df["close"].rolling(trade_config.SLOW_MA).mean()

    delta = df["close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # This is simple moving-average RSI smoothing, not Wilder's standard smoothing.
    avg_gain = gain.rolling(trade_config.RSI_PERIOD).mean()
    avg_loss = loss.rolling(trade_config.RSI_PERIOD).mean()

    rs = avg_gain / avg_loss

    df["rsi"] = 100 - (100 / (1 + rs))

    return df


def decide_at(df, index):
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
        if row["rsi"] < trade_config.RSI_BUY_THRESHOLD:
            return Decision(
                "BUY",
                "bullish_ma_crossover_rsi_filter_passed",
            )
        return Decision("HOLD", "bullish_ma_crossover_rsi_filter_failed")

    if bearish_cross:
        return Decision("SELL", "bearish_ma_crossover")

    return Decision("HOLD", "no_crossover")


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


DONCHIAN_BREAKOUT = create_donchian_breakout_strategy()
STRATEGY_REGISTRY = {
    MA_RSI_CROSSOVER.name: MA_RSI_CROSSOVER,
    DONCHIAN_BREAKOUT.name: DONCHIAN_BREAKOUT,
}


def get_strategy(name: str) -> StrategySpec:
    try:
        return STRATEGY_REGISTRY[name]
    except KeyError as error:
        choices = ", ".join(STRATEGY_REGISTRY)
        raise ValueError(f"Unknown strategy {name!r}; choose from: {choices}") from error
