import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pandas as pd

import backtest
import trade_config
from strategy import Decision


def warmup_sample_bars():
    index = pd.date_range("2026-01-01", periods=6, freq="5min", tz="UTC")
    return pd.DataFrame(
        {
            "open": [10.0, 11.0, 12.0, 100.0, 110.0, 115.0],
            "high": [10.5, 11.5, 12.5, 105.0, 114.0, 120.0],
            "low": [9.5, 10.5, 11.5, 95.0, 105.0, 110.0],
            "close": [10.0, 11.0, 12.0, 100.0, 110.0, 115.0],
            "volume": [1.0] * 6,
        },
        index=index,
    )


class WarmupTests(unittest.TestCase):
    def test_warmup_count_tracks_configured_indicators(self):
        with (
            patch.object(trade_config, "FAST_MA", 40),
            patch.object(trade_config, "SLOW_MA", 80),
            patch.object(trade_config, "RSI_PERIOD", 25),
        ):
            self.assertEqual(backtest.required_warmup_bars(), 90)

    def test_first_test_signal_uses_warmed_indicators_but_executes_next_open(self):
        bars = warmup_sample_bars()
        test_start = bars.index[3]
        indicator_inputs = []
        decision_indexes = []

        def fake_indicators(frame):
            indicator_inputs.append((frame.index[0], len(frame)))
            result = frame.copy()
            result["ma_fast"] = result["close"]
            result["ma_slow"] = result["close"]
            result["rsi"] = 50.0
            return result

        def fake_decide(indicators, index):
            decision_indexes.append(index)
            return Decision("BUY", "first_test_signal") if index == 3 else Decision("HOLD", "hold")

        with (
            patch("backtest.calculate_indicators", side_effect=fake_indicators),
            patch("backtest.decide_at", side_effect=fake_decide),
            patch.object(trade_config, "STOP_LOSS_PERCENT", 0.5),
            patch.object(trade_config, "TAKE_PROFIT_PERCENT", 1.0),
        ):
            result = backtest.run_backtest(
                bars,
                starting_capital=1000,
                timeframe="5Min",
                fee_rate=0,
                slippage=0,
                test_start=test_start,
            )

        self.assertEqual(indicator_inputs, [(bars.index[0], len(bars))])
        self.assertEqual(decision_indexes[0], 3)
        self.assertEqual(result["trades"].iloc[0]["entry_time"], bars.index[4])
        self.assertEqual(result["start_time"], test_start)
        self.assertEqual(result["end_time"], bars.index[-1] + pd.Timedelta(minutes=5))
        self.assertAlmostEqual(result["full_buy_hold_return"], 15.0)
        self.assertEqual(result["warmup_bars"], 3)

    def test_pretest_signal_cannot_open_a_test_period_position(self):
        bars = warmup_sample_bars()
        test_start = bars.index[3]

        def pretest_only_buy(_indicators, index):
            return Decision("BUY", "pretest_buy") if index == 2 else Decision("HOLD", "hold")

        with (
            patch("backtest.decide_at", side_effect=pretest_only_buy),
            patch.object(trade_config, "STOP_LOSS_PERCENT", 0.5),
            patch.object(trade_config, "TAKE_PROFIT_PERCENT", 1.0),
        ):
            result = backtest.run_backtest(
                bars,
                starting_capital=1000,
                timeframe="5Min",
                fee_rate=0,
                slippage=0,
                test_start=test_start,
            )

        self.assertEqual(result["total_trades"], 0)
        self.assertEqual(result["ending_capital"], 1000)
        self.assertEqual(result["time_invested_percent"], 0)

    def test_fetch_requests_dynamic_warmup_before_test_start(self):
        end = datetime(2026, 2, 2, tzinfo=timezone.utc)
        test_start = end - timedelta(days=1)
        warmup = backtest.required_warmup_bars()
        request_start = test_start - timedelta(minutes=5 * warmup)
        index = pd.date_range(request_start, end, freq="5min", inclusive="left")
        candles = pd.DataFrame(
            {
                "open": [100.0] * len(index),
                "high": [101.0] * len(index),
                "low": [99.0] * len(index),
                "close": [100.0] * len(index),
                "volume": [1.0] * len(index),
            },
            index=index,
        )
        with patch("backtest.CryptoHistoricalDataClient") as client_class:
            client_class.return_value.get_crypto_bars.return_value.df = candles
            history = backtest.fetch_history(
                1, "5Min", end_time=end
            )
            request = client_class.return_value.get_crypto_bars.call_args.args[0]

        actual_start = pd.Timestamp(request.start)
        if actual_start.tzinfo is None:
            actual_start = actual_start.tz_localize("UTC")
        self.assertEqual(actual_start, pd.Timestamp(request_start))
        self.assertEqual(history.attrs["test_start"], pd.Timestamp(test_start))
        self.assertLess(history.index[0], pd.Timestamp(test_start))
        self.assertEqual(history.attrs["warmup_bars_requested"], warmup)


if __name__ == "__main__":
    unittest.main()
