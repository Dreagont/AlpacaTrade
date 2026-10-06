"""Frozen research search space; no changes to the existing strategy registry."""

from dataclasses import asdict, dataclass
from itertools import product

import numpy as np
import pandas as pd

from short_term_research import REGIME_SMA_PERIOD, REGIME_SLOPE_LOOKBACK, REGIME_STALE_AFTER_MINUTES
from strategy import Decision, StrategySpec, calculate_regime_filter

TIMEFRAMES = ("1Hour", "2Hour", "4Hour")
FAMILIES = ("trend_sma", "ma_cross", "donchian", "rsi_pullback")
REASONS = {1: "stop_loss", 2: "take_profit", 3: "max_holding_time",
           4: "signal_sell", 5: "regime_filter_off", 6: "regime_unavailable", 7: "end_of_backtest"}


@dataclass(frozen=True)
class SearchConfig:
    config_id: str
    timeframe: str
    family: str
    first: int
    second: int
    regime_gate: bool
    stop_loss: float | None
    max_hold_hours: int | None

    def to_dict(self):
        return asdict(self)


def enumerate_configs():
    bases = [*(('trend_sma', n, k) for n, k in product((50, 100, 200), (0, 10, 20))),
             *(('ma_cross', f, s) for f, s in ((10, 50), (10, 100), (20, 100), (20, 200), (50, 200))),
             *(('donchian', n, m) for n, m in ((20, 10), (55, 20), (100, 50))),
             *(('rsi_pullback', p, th) for p, th in product((2, 3, 5), (10, 20, 30)))]
    configs = []
    for timeframe in TIMEFRAMES:
        for family, first, second in bases:
            for gate, stop, hold in product((False, True), (None, 0.03, 0.06), (None, 120)):
                identity = f"c{len(configs):04d}"
                configs.append(SearchConfig(identity, timeframe, family, first, second, gate, stop, hold))
    assert len(configs) == 936
    return tuple(configs)


def merge_regime(execution, regime_4h, timeframe):
    """Same completed-close merge, 200/20 v1 and 240-minute stale rule."""
    prepared = calculate_regime_filter(regime_4h, sma_period=REGIME_SMA_PERIOD, slope_lookback=REGIME_SLOPE_LOOKBACK)
    hours = {"1Hour": 1, "2Hour": 2, "4Hour": 4}[timeframe]
    right = pd.DataFrame({"source_close": prepared.index + pd.Timedelta(hours=4),
                          "regime": pd.array(prepared["regime_bullish"], dtype="boolean")}).sort_values("source_close")
    left = pd.DataFrame({"decision_close": execution.index + pd.Timedelta(hours=hours)})
    merged = pd.merge_asof(left, right, left_on="decision_close", right_on="source_close", direction="backward")
    age = (merged["decision_close"] - merged["source_close"]).dt.total_seconds() / 60
    states = pd.array(merged["regime"], dtype="boolean")
    states[age.to_numpy() >= REGIME_STALE_AFTER_MINUTES] = pd.NA
    return pd.Series(states, index=execution.index)


def wilder_rsi(close, period):
    delta = close.diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    # Seed with the first period's simple average, then Wilder recursive smoothing.
    for values in (gain, loss):
        seed = values.iloc[1:period + 1].mean()
        values.iloc[:period] = np.nan
        if len(values) > period:
            values.iloc[period] = seed
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=1).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=1).mean()
    rsi = 100 - 100 / (1 + avg_gain / avg_loss)
    return rsi.mask((avg_gain == 0) & (avg_loss == 0), 50)


def family_signals(bars, config):
    close = bars["close"]
    first, second = config.first, config.second
    action = np.zeros(len(bars), dtype=np.int8)  # HOLD=0, BUY=1, SELL=-1
    if config.family == "trend_sma":
        average = close.rolling(first).mean()
        valid = average.notna()
        buy = close > average
        if second:
            lag = average.shift(second)
            valid &= lag.notna()
            buy &= average > lag
        action[valid & buy], action[valid & ~buy] = 1, -1
    elif config.family == "ma_cross":
        fast, slow = close.rolling(first).mean(), close.rolling(second).mean()
        valid = fast.notna() & slow.notna()
        action[valid & (fast > slow)], action[valid & (fast <= slow)] = 1, -1
    elif config.family == "donchian":
        high = bars["high"].rolling(first).max().shift(1)
        low = bars["low"].rolling(second).min().shift(1)
        valid = high.notna() & low.notna()
        action[valid & (close > high)], action[valid & (close < low)] = 1, -1
    elif config.family == "rsi_pullback":
        rsi, average = wilder_rsi(close, first), close.rolling(200).mean()
        valid = rsi.notna() & average.notna()
        action[valid & (rsi < second) & (close > average)] = 1
        action[valid & (rsi > 70)] = -1
    else:
        raise ValueError(f"Unknown family: {config.family}")
    return action


def apply_regime(signals, states, enabled):
    actions = np.array(signals, copy=True)
    reasons = np.full(len(actions), 4, dtype=np.int8)
    if enabled:
        missing = states.isna().to_numpy()
        off = ~states.fillna(False).to_numpy(dtype=bool)
        actions[off] = -1
        reasons[off] = 5
        reasons[missing] = 6
    return actions, reasons


def reference_strategy(config, actions, reasons):
    def prepare(bars):
        if len(bars) != len(actions):
            raise ValueError("Reference signals must align with bars")
        return bars
    def decide(_prepared, index):
        action = actions[index]
        return Decision("BUY" if action == 1 else "SELL" if action == -1 else "HOLD", REASONS[int(reasons[index])])
    warmup = max(config.first + (config.second if config.family == 'trend_sma' else 0), config.second,
                 200 if config.family == 'rsi_pullback' else 0)
    return StrategySpec(config.config_id, prepare, decide, lambda: warmup, config.to_dict,
                        lambda: config.stop_loss, lambda: None,
                        max_holding_minutes=config.max_hold_hours * 60 if config.max_hold_hours else None)
