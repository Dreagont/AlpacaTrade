import unittest

import numpy as np
import pandas as pd

from backtest import required_warmup_bars, run_backtest
from strategy import (
    Decision,
    DONCHIAN_BREAKOUT,
    MA_RSI_CROSSOVER,
    calculate_regime_filter,
    create_regime_filtered_donchian_strategy,
    get_strategy,
)


class CausalRegimeFilterTests(unittest.TestCase):
    def test_sma_and_regime_use_only_current_and_past_closes(self):
        close = np.linspace(100, 300, 300)
        frame = pd.DataFrame({"close": close})
        prepared = calculate_regime_filter(frame, sma_period=200, slope_lookback=20)
        self.assertAlmostEqual(prepared.loc[250, "regime_sma"], frame.close.iloc[51:251].mean())
        changed = frame.copy()
        changed.loc[280:, "close"] *= 10
        recomputed = calculate_regime_filter(changed, sma_period=200, slope_lookback=20)
        pd.testing.assert_series_equal(
            prepared.loc[:270, "regime_sma"], recomputed.loc[:270, "regime_sma"]
        )
        pd.testing.assert_series_equal(
            prepared.loc[:270, "regime_bullish"], recomputed.loc[:270, "regime_bullish"]
        )

    def test_regime_not_ready_until_sma_plus_slope_history_exists(self):
        frame = pd.DataFrame({"close": np.arange(1.0, 241.0)})
        prepared = calculate_regime_filter(frame, 200, 20)
        self.assertTrue(prepared.regime_bullish.iloc[:219].isna().all())
        self.assertFalse(pd.isna(prepared.regime_bullish.iloc[219]))

    def test_close_above_sma_without_rising_sma_is_non_bullish(self):
        close = [200.0] * 200 + [100.0] * 20 + [195.0]
        prepared = calculate_regime_filter(pd.DataFrame({"close": close}), 200, 20)
        row = prepared.iloc[-1]
        self.assertGreater(row.close, row.regime_sma)
        self.assertLessEqual(row.regime_sma, row.regime_sma_previous)
        self.assertFalse(bool(row.regime_bullish))

    def test_rising_sma_without_close_above_sma_is_non_bullish(self):
        close = [100.0] * 200 + [200.0] * 19 + [100.0]
        prepared = calculate_regime_filter(pd.DataFrame({"close": close}), 200, 20)
        row = prepared.iloc[-1]
        self.assertGreater(row.regime_sma, row.regime_sma_previous)
        self.assertLessEqual(row.close, row.regime_sma)
        self.assertFalse(bool(row.regime_bullish))

    def test_both_close_and_sma_slope_conditions_make_bullish_regime(self):
        prepared = calculate_regime_filter(
            pd.DataFrame({"close": np.arange(1.0, 241.0)}), 200, 20
        )
        self.assertTrue(bool(prepared.regime_bullish.iloc[-1]))

    def test_filtered_donchian_channels_exclude_current_high_and_low(self):
        strategy = create_regime_filtered_donchian_strategy(3, 2, 3, 1)
        bars = pd.DataFrame({
            "high": [3, 4, 5, 100], "low": [2, 2, 2, 0],
            "close": [2, 3, 4, 50],
        })
        prepared = strategy.prepare_indicators(bars)
        self.assertEqual(prepared.donchian_entry_high.iloc[-1], 5)
        self.assertEqual(prepared.donchian_exit_low.iloc[-1], 2)

    def test_not_ready_and_nonbullish_states_never_buy(self):
        strategy = create_regime_filtered_donchian_strategy(2, 2, 2, 1)
        frame = pd.DataFrame({
            "close": [20.0], "donchian_entry_high": [10.0],
            "donchian_exit_low": [5.0], "regime_sma": [np.nan],
            "regime_sma_previous": [np.nan], "regime_bullish": [pd.NA],
        })
        self.assertEqual(strategy.decide_at(frame, 0).reason, "regime_filter_not_ready")
        frame.loc[0, ["regime_sma", "regime_sma_previous", "regime_bullish"]] = [15, 12, False]
        decision = strategy.decide_at(frame, 0)
        self.assertEqual((decision.action, decision.reason), ("SELL", "regime_filter_off"))

    def test_bullish_regime_matches_plain_donchian_without_transition_delay(self):
        strategy = create_regime_filtered_donchian_strategy(2, 2, 2, 1)
        frame = pd.DataFrame({
            "close": [10.0, 12.0, 4.0, 8.0],
            "donchian_entry_high": [11.0, 11.0, 11.0, 11.0],
            "donchian_exit_low": [5.0, 5.0, 5.0, 5.0],
            "regime_sma": [9.0] * 4, "regime_sma_previous": [8.0] * 4,
            "regime_bullish": [False, True, True, True],
        })
        plain = DONCHIAN_BREAKOUT
        for index in (1, 2, 3):
            self.assertEqual(strategy.decide_at(frame, index), plain.decide_at(frame, index))
        self.assertEqual(strategy.decide_at(frame, 1).action, "BUY")
        self.assertEqual(strategy.decide_at(frame, 2).action, "SELL")
        self.assertEqual(strategy.decide_at(frame, 3).action, "HOLD")

    def test_nonbullish_regime_overrides_to_cash_without_entry(self):
        strategy = create_regime_filtered_donchian_strategy(2, 2, 2, 1)
        frame = pd.DataFrame({
            "close": [12.0, 12.0], "donchian_entry_high": [11.0, 11.0],
            "donchian_exit_low": [5.0, 5.0], "regime_sma": [9.0, 9.0],
            "regime_sma_previous": [8.0, 8.0], "regime_bullish": [True, False],
        })
        self.assertEqual(strategy.decide_at(frame, 1), Decision("SELL", "regime_filter_off"))
        frame.loc[1, ["donchian_entry_high", "donchian_exit_low"]] = [np.nan, np.nan]
        self.assertEqual(strategy.decide_at(frame, 1), Decision("SELL", "regime_filter_off"))
        frame.loc[1, ["donchian_entry_high", "donchian_exit_low"]] = [np.nan, np.nan]
        self.assertEqual(strategy.decide_at(frame, 1), Decision("SELL", "regime_filter_off"))

    def test_nonbullish_holding_sells_and_bullish_breakdown_sells(self):
        strategy = create_regime_filtered_donchian_strategy(2, 2, 2, 1)
        frame = pd.DataFrame({
            "close": [11.0, 12.0], "donchian_entry_high": [10.0, 10.0],
            "donchian_exit_low": [5.0, 5.0], "regime_sma": [9.0, 10.0],
            "regime_sma_previous": [8.0, 9.0], "regime_bullish": [True, False],
        })
        decision = strategy.decide_at(frame, 1)
        self.assertEqual((decision.action, decision.reason), ("SELL", "regime_filter_off"))
        frame.loc[1, ["close", "regime_sma", "regime_sma_previous", "regime_bullish"]] = [4.0, 3.0, 2.0, True]
        decision = strategy.decide_at(frame, 1)
        self.assertEqual((decision.action, decision.reason), ("SELL", "donchian_exit_breakdown"))

    def test_regime_sell_is_executed_at_next_candle_open(self):
        strategy = create_regime_filtered_donchian_strategy(3, 2, 10, 2)
        close = [100.0] * 20 + [110.0, 111.0, 112.0, 113.0, 114.0] + [80.0] * 3
        index = pd.date_range("2026-01-01", periods=len(close), freq="4h", tz="UTC")
        bars = pd.DataFrame({
            "open": close, "high": [value + 1 for value in close],
            "low": [value - 1 for value in close], "close": close,
            "volume": [1.0] * len(close),
        }, index=index)
        signals = strategy.prepare_indicators(bars)
        self.assertEqual(strategy.decide_at(signals, 20).action, "BUY")
        self.assertEqual(strategy.decide_at(signals, 25).reason, "regime_filter_off")
        result = run_backtest(
            bars, timeframe="4Hour", starting_capital=1000,
            fee_rate=0, slippage=0, test_start=index[0], strategy=strategy,
        )
        self.assertEqual(result["trades"].iloc[0].entry_time, index[21])
        self.assertEqual(result["trades"].iloc[0].exit_time, index[26])
        self.assertEqual(result["trades"].iloc[0].exit_reason, "regime_filter_off")

    def test_changing_future_market_data_cannot_change_past_trades(self):
        strategy = create_regime_filtered_donchian_strategy(3, 2, 10, 2)
        close = [100.0] * 20 + [110.0, 111.0, 112.0, 113.0, 114.0] + [80.0] * 45
        index = pd.date_range("2026-01-01", periods=len(close), freq="4h", tz="UTC")
        bars = pd.DataFrame({
            "open": close, "high": [value + 1 for value in close],
            "low": [value - 1 for value in close], "close": close,
            "volume": [1.0] * len(close),
        }, index=index)
        changed_future = bars.copy()
        changed_future.loc[index[50]:, ["open", "high", "low", "close"]] *= 4
        common = dict(
            timeframe="4Hour", starting_capital=1000, fee_rate=0,
            slippage=0, test_start=index[0], strategy=strategy,
        )
        original = run_backtest(bars, **common)
        changed = run_backtest(changed_future, **common)
        cutoff = index[50]
        original_trades = original["trades"]
        changed_trades = changed["trades"]
        original_past = original_trades.loc[original_trades.exit_time < cutoff]
        changed_past = changed_trades.loc[changed_trades.exit_time < cutoff]
        pd.testing.assert_frame_equal(
            original_past.reset_index(drop=True), changed_past.reset_index(drop=True)
        )
        self.assertGreater(len(original_past), 0)
        self.assertTrue((original_past.exit_reason == "regime_filter_off").all())

    def test_strategy_metadata_warmup_and_existing_strategies_unchanged(self):
        filtered = get_strategy("donchian_regime_filter")
        self.assertEqual(filtered.required_warmup_bars() - 10, 220)
        self.assertIsNone(filtered.stop_loss_percent)
        self.assertIsNone(filtered.take_profit_percent)
        self.assertEqual(filtered.parameters["entry_lookback"], 20)
        self.assertEqual(filtered.parameters["exit_lookback"], 10)
        self.assertIs(get_strategy("donchian_breakout"), DONCHIAN_BREAKOUT)
        self.assertIs(get_strategy("ma_rsi_crossover"), MA_RSI_CROSSOVER)


if __name__ == "__main__":
    unittest.main()
