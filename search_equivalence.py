"""Fail-closed family/overlay/timeframe gate against the unchanged reference engine."""

import numpy as np
import pandas as pd

import backtest
from binance_data import resample_bars
from fast_search_engine import simulate
from search_space import FAMILIES, apply_regime, enumerate_configs, family_signals, merge_regime, reference_strategy


def synthetic_history(seed=20261006, periods=4096):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0001, 0.022, periods)))
    opens = np.r_[100.0, close[:-1]] * np.exp(rng.normal(0, 0.005, periods))
    highs = np.maximum(opens, close) * (1 + rng.uniform(0.001, 0.045, periods))
    lows = np.minimum(opens, close) * (1 - rng.uniform(0.001, 0.045, periods))
    return pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": close, "volume": 1.0},
                        index=pd.date_range("2018-01-01", periods=periods, freq="h", tz="UTC"))


def gate_configs(configs=None):
    # Long-lookback representative from each family, ALL 12 overlays and ALL frames.
    params = {"trend_sma": (200, 20), "ma_cross": (20, 200), "donchian": (100, 50), "rsi_pullback": (5, 30)}
    return tuple(c for c in (configs or enumerate_configs()) if (c.first, c.second) == params[c.family])


def compare_one(bars, actions, reasons, config, *, test_start, fee=0.00075, slip=0.0002, liquidate=False):
    spec = reference_strategy(config, actions, reasons)
    # Large cash ensures fixed reference notional never blocks an otherwise valid entry.
    reference = backtest.run_backtest(bars, starting_capital=1e9, timeframe=config.timeframe,
                                     fee_rate=fee, slippage=slip, test_start=test_start,
                                     strategy=spec, liquidate_at_end=liquidate)
    fast = simulate(bars, actions, reasons, timeframe=config.timeframe, fee_rate=fee, slippage_rate=slip,
                    test_start=test_start, stop_loss=config.stop_loss, max_hold_hours=config.max_hold_hours,
                    liquidate_at_end=liquidate)
    expected, actual = reference["trades"], fast.trades
    if len(expected) != len(actual):
        raise AssertionError(f"{config.config_id}: trade counts differ: {len(expected)} vs {len(actual)}")
    for field in ("entry_time", "exit_time", "exit_reason"):
        if list(expected[field]) != list(actual[field]):
            raise AssertionError(f"{config.config_id}: {field} differs")
    if not np.allclose(expected["return_percent"].to_numpy(dtype=float), actual["return_percent"].to_numpy(dtype=float), rtol=0, atol=1e-9):
        raise AssertionError(f"{config.config_id}: per-trade net return differs beyond 1e-9")
    return len(actual)


def equivalence_gate(real_hourly, *, configs=None, synthetic=None, progress=print):
    if real_hourly is None or len(real_hourly) < 1000:
        raise ValueError("Equivalence gate requires a cached real-data sample (>=1000 hourly bars); run --download first")
    selected = gate_configs(configs)
    statuses = {family: {"synthetic": 0, "cached_real": 0, "trades_checked": 0} for family in FAMILIES}
    for dataset_name, hourly in (("synthetic", synthetic if synthetic is not None else synthetic_history()),
                                 ("cached_real", real_hourly.iloc[:4096])):
        frames = {tf: resample_bars(hourly, tf) for tf in {c.timeframe for c in selected}}
        regime = resample_bars(hourly, "4Hour")
        states = {tf: merge_regime(bars, regime, tf) for tf, bars in frames.items()}
        for family in FAMILIES:
            family_configs = [c for c in selected if c.family == family]
            signal_cache = {}
            for c in family_configs:
                bars = frames[c.timeframe]
                key = (c.timeframe, c.first, c.second)
                if key not in signal_cache:
                    signal_cache[key] = family_signals(bars, c)
                actions, reasons = apply_regime(signal_cache[key], states[c.timeframe], c.regime_gate)
                warmup = max(reference_strategy(c, actions, reasons).required_warmup_bars(), 230)
                if warmup >= len(bars) - 1:
                    raise ValueError(f"Cached sample too short to verify {family} on {c.timeframe}")
                try:
                    trades = compare_one(bars, actions, reasons, c, test_start=bars.index[warmup])
                except AssertionError as error:
                    progress(f"EQUIVALENCE FAIL {dataset_name} {family}: {error}")
                    raise RuntimeError("Equivalence gate failed; search stopped") from error
                statuses[family][dataset_name] += 1
                statuses[family]["trades_checked"] += trades
            if not family_configs:
                raise ValueError(f"Gate is missing family {family}")
            progress(f"EQUIVALENCE PASS {dataset_name} {family}: {len(family_configs)} overlay/timeframe cases")
    for family, status in statuses.items():
        if not status["trades_checked"]:
            raise RuntimeError(f"Equivalence gate inconclusive: no completed trades for {family}")
    return statuses
