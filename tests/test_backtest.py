import unittest

import pandas as pd

from backtest import run_backtest
from strategy import Decision, StrategySpec
from unittest.mock import patch


def flat_bars(count=4):
    index = pd.date_range("2026-01-01", periods=count, freq="5min", tz="UTC")
    return pd.DataFrame(
        {
            "open": [100.0] * count,
            "high": [101.0] * count,
            "low": [99.0] * count,
            "close": [100.0] * count,
            "volume": [1.0] * count,
        },
        index=index,
    )


def test_strategy(
    decide_at, prepare_indicators=None, *, stop_loss=None, take_profit=None,
    max_holding_minutes=None,
):
    return StrategySpec(
        name="test_strategy",
        _prepare_indicators=prepare_indicators or (lambda bars: bars.copy()),
        _decide_at=decide_at,
        _warmup_lookback=lambda: 0,
        _parameters=lambda: {},
        _stop_loss=lambda: stop_loss,
        _take_profit=lambda: take_profit,
        max_holding_minutes=max_holding_minutes,
    )


class BacktestAccountingTests(unittest.TestCase):
    def test_max_holding_exits_at_first_open_at_or_after_limit(self):
        bars = flat_bars(7)
        bars.index = pd.date_range("2026-01-01", periods=7, freq="30min", tz="UTC")
        strategy = test_strategy(
            lambda _indicators, index: (
                Decision("BUY", "entry") if index == 0 else Decision("HOLD", "hold")
            ),
            max_holding_minutes=75,
        )
        result = run_backtest(
            bars, starting_capital=1000, timeframe="30Min", fee_rate=0,
            slippage=0, strategy=strategy,
        )
        trade = result["trades"].iloc[0]
        self.assertEqual(trade["exit_reason"], "max_holding_time")
        self.assertEqual(trade["entry_time"], bars.index[1])
        self.assertEqual(trade["exit_time"], bars.index[4])  # 90 minutes after entry.
        self.assertEqual(strategy.max_holding_minutes, 75)

    def test_open_exit_priority_is_gap_stop_then_gap_target_then_max_then_signal(self):
        expected = (
            ("stop", 80.0, "stop_loss"),
            ("target", 120.0, "take_profit"),
            ("max", 100.0, "max_holding_time"),
        )
        for label, open_price, expected_reason in expected:
            with self.subTest(label=label):
                bars = flat_bars(5)
                bars.index = pd.date_range(
                    "2026-01-01", periods=5, freq="30min", tz="UTC"
                )
                bars.iloc[3, bars.columns.get_loc("open")] = open_price
                bars.iloc[3, bars.columns.get_loc("high")] = open_price + 1
                bars.iloc[3, bars.columns.get_loc("low")] = open_price - 1
                strategy = test_strategy(
                    lambda _indicators, index: (
                        Decision("BUY", "entry") if index == 0
                        else Decision("SELL", "ordinary_sell") if index == 2
                        else Decision("HOLD", "hold")
                    ),
                    stop_loss=0.10,
                    take_profit=0.10,
                    max_holding_minutes=60,
                )
                result = run_backtest(
                    bars, starting_capital=1000, timeframe="30Min", fee_rate=0,
                    slippage=0, strategy=strategy,
                )
                self.assertEqual(result["trades"].iloc[0]["exit_reason"], expected_reason)

    def test_existing_strategy_specs_default_to_no_max_holding(self):
        strategy = test_strategy(
            lambda _indicators, index: (
                Decision("BUY", "entry") if index == 0
                else Decision("HOLD", "hold")
            )
        )
        result = run_backtest(
            flat_bars(), starting_capital=100, timeframe="5Min", fee_rate=0,
            slippage=0,
            strategy=strategy,
        )
        self.assertIsNone(strategy.max_holding_minutes)
        self.assertEqual(result["trades"].iloc[0]["exit_reason"], "end_of_backtest")

    def test_liquidate_at_end_defaults_true_and_can_be_disabled(self):
        bars = flat_bars(4)
        bars.iloc[-1, bars.columns.get_loc("close")] = 120.0
        strategy = test_strategy(
            lambda _indicators, index: (
                Decision("BUY", "test_buy") if index == 0 else Decision("HOLD", "hold")
            )
        )
        with (
            patch("trade_config.TRADE_AMOUNT_USD", 100.0),
            patch("trade_config.MAX_POSITION_USD", 100.0),
        ):
            liquidated = run_backtest(
                bars,
                starting_capital=1000,
                timeframe="5Min",
                fee_rate=0,
                slippage=0,
                strategy=strategy,
            )
            marked = run_backtest(
                bars,
                starting_capital=1000,
                timeframe="5Min",
                fee_rate=0,
                slippage=0,
                strategy=strategy,
                liquidate_at_end=False,
            )

        self.assertEqual(liquidated["trades"].iloc[0]["exit_reason"], "end_of_backtest")
        self.assertEqual(marked["total_trades"], 0)
        self.assertIsNotNone(marked["open_position"])
        self.assertEqual(marked["ending_cash"], 900.0)
        self.assertEqual(marked["ending_equity"], 1020.0)
        self.assertEqual(marked["ending_capital"], marked["ending_equity"])
        self.assertAlmostEqual(marked["strategy_return"], 2.0)

    def test_historical_data_quality_counts_gaps_and_zero_volume(self):
        bars = flat_bars(5).drop(index=flat_bars(5).index[2])
        bars.loc[bars.index[1], "volume"] = 0
        bars.attrs["requested_start_time"] = flat_bars(5).index[0]
        bars.attrs["requested_end_time"] = flat_bars(5).index[-1] + pd.Timedelta(minutes=5)

        from backtest import data_quality_diagnostics

        metrics = data_quality_diagnostics(bars, "5Min")
        self.assertEqual(metrics["expected_candle_count"], 5)
        self.assertEqual(metrics["actual_candle_count"], 4)
        self.assertEqual(metrics["missing_candle_count"], 1)
        self.assertEqual(metrics["missing_candle_percent"], 20.0)
        self.assertEqual(metrics["zero_volume_candle_count"], 1)
        self.assertEqual(metrics["zero_volume_candle_percent"], 25.0)

    def test_fees_and_slippage_are_counted_once_and_cash_stays_nonnegative(self):
        def decisions(_indicators, index):
            return Decision("BUY", "test_buy") if index == 0 else Decision("SELL", "test_sell")

        bars = flat_bars(3)
        bars.iloc[2, bars.columns.get_loc("open")] = 100.5
        bars.iloc[2, bars.columns.get_loc("high")] = 101.0
        bars.iloc[2, bars.columns.get_loc("low")] = 100.0
        result = run_backtest(
            bars,
            starting_capital=100,
            timeframe="5Min",
            fee_rate=0.01,
            slippage=0.01,
            strategy=test_strategy(decisions),
        )

        trade = result["trades"].iloc[0]
        self.assertGreaterEqual(result["ending_capital"], 0)
        self.assertAlmostEqual(
            trade["gross_pnl"] - trade["fees"] - trade["slippage_cost"],
            trade["net_pnl"],
        )
        self.assertAlmostEqual(result["net_profit"], trade["net_pnl"])

    def test_backtest_allows_only_one_open_position(self):
        result = run_backtest(
            flat_bars(),
            starting_capital=100,
            timeframe="5Min",
            fee_rate=0,
            slippage=0,
            strategy=test_strategy(lambda _indicators, _index: Decision("BUY", "test_buy")),
        )
        self.assertEqual(result["total_trades"], 1)
        self.assertGreaterEqual(result["ending_capital"], 0)

    def test_equity_curve_timestamps_are_candle_closes(self):
        result = run_backtest(
            flat_bars(2),
            starting_capital=100,
            timeframe="5Min",
            fee_rate=0,
            slippage=0,
            strategy=test_strategy(lambda _indicators, _index: Decision("HOLD", "test_hold")),
        )
        self.assertEqual(
            result["equity_curve"].index[0],
            flat_bars(2).index[0],
        )


if __name__ == "__main__":
    unittest.main()
