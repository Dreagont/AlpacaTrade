import unittest

import pandas as pd

from backtest import run_backtest
from strategy import (
    DONCHIAN_BREAKOUT,
    MA_RSI_CROSSOVER,
    Decision,
    StrategySpec,
    create_donchian_breakout_strategy,
)


def donchian_trend_bars(*, breakdown=False):
    index = pd.date_range("2026-01-01", periods=36, freq="1h", tz="UTC")
    rows = []
    for _ in range(20):
        rows.append((100.0, 101.0, 99.0, 100.0))
    rows.append((101.0, 103.0, 100.0, 102.0))  # breakout signal on candle 20
    rows.append((110.0, 111.0, 109.0, 110.0))  # entry executes here, at candle 21 open
    previous_close = 110.0
    for _ in range(10):
        open_price = previous_close + 1.0
        close = open_price + 1.0
        rows.append((open_price, close + 0.5, open_price - 0.5, close))
        previous_close = close
    if breakdown:
        rows.append((90.0, 91.0, 79.0, 80.0))  # signal candle; execute next candle
        rows.append((78.0, 81.0, 77.0, 80.0))  # breakdown exit executes at this open
        while len(rows) < len(index):
            rows.append((80.0, 81.0, 79.0, 80.0))
    else:
        while len(rows) < len(index):
            open_price = previous_close + 1.0
            close = open_price + 1.0
            rows.append((open_price, close + 0.5, open_price - 0.5, close))
            previous_close = close
    return pd.DataFrame(
        {
            "open": [row[0] for row in rows],
            "high": [row[1] for row in rows],
            "low": [row[2] for row in rows],
            "close": [row[3] for row in rows],
            "volume": [1.0] * len(rows),
        },
        index=index,
    )


def profit_factor_test_strategy():
    decisions = {
        0: Decision("BUY", "first_buy"),
        1: Decision("SELL", "first_sell"),
        2: Decision("BUY", "second_buy"),
        3: Decision("SELL", "second_sell"),
    }
    return StrategySpec(
        name="profit_factor_fixture",
        _prepare_indicators=lambda bars: bars.copy(),
        _decide_at=lambda _prepared, index: decisions.get(
            index, Decision("HOLD", "hold")
        ),
        _warmup_lookback=lambda: 0,
        _parameters=lambda: {},
        _stop_loss=lambda: None,
        _take_profit=lambda: None,
    )


class StrategyEngineTests(unittest.TestCase):
    def test_donchian_entry_channel_excludes_current_candle_high(self):
        index = pd.date_range("2026-01-01", periods=21, freq="1h", tz="UTC")
        bars = pd.DataFrame(
            {
                "open": [9.0] * 20 + [10.0],
                "high": [10.0] * 20 + [15.0],
                "low": [8.0] * 21,
                "close": [9.0] * 20 + [11.0],
                "volume": [1.0] * 21,
            },
            index=index,
        )
        prepared = DONCHIAN_BREAKOUT.prepare_indicators(bars)

        self.assertEqual(prepared.iloc[20]["donchian_entry_high"], 10.0)
        self.assertEqual(
            DONCHIAN_BREAKOUT.decide_at(prepared, 20),
            Decision("BUY", "donchian_entry_breakout"),
        )

    def test_donchian_exit_channel_excludes_current_candle_low(self):
        index = pd.date_range("2026-01-01", periods=21, freq="1h", tz="UTC")
        bars = pd.DataFrame(
            {
                "open": [11.0] * 20 + [9.0],
                "high": [12.0] * 20 + [15.0],
                "low": [10.0] * 20 + [5.0],
                "close": [11.0] * 20 + [9.0],
                "volume": [1.0] * 21,
            },
            index=index,
        )
        prepared = DONCHIAN_BREAKOUT.prepare_indicators(bars)

        self.assertEqual(prepared.iloc[20]["donchian_exit_low"], 10.0)
        self.assertEqual(
            DONCHIAN_BREAKOUT.decide_at(prepared, 20),
            Decision("SELL", "donchian_exit_breakdown"),
        )

    def test_donchian_has_no_fixed_risk_exits_and_entry_executes_next_open(self):
        bars = donchian_trend_bars()
        result = run_backtest(
            bars,
            starting_capital=1000,
            timeframe="1Hour",
            fee_rate=0,
            slippage=0,
            strategy=DONCHIAN_BREAKOUT,
        )
        trade = result["trades"].iloc[0]

        self.assertIsNone(DONCHIAN_BREAKOUT.stop_loss_percent)
        self.assertIsNone(DONCHIAN_BREAKOUT.take_profit_percent)
        self.assertEqual(trade["entry_reason"], "donchian_entry_breakout")
        self.assertEqual(trade["entry_time"], bars.index[21])
        self.assertEqual(trade["entry_price"], bars.iloc[21]["open"])
        self.assertEqual(trade["exit_reason"], "end_of_backtest")
        self.assertGreater(trade["exit_price"] / trade["entry_price"], 1.04)

    def test_zero_cost_return_is_at_least_realistic_return_with_positive_costs(self):
        bars = donchian_trend_bars()
        realistic = run_backtest(
            bars,
            starting_capital=1000,
            timeframe="1Hour",
            fee_rate=0.005,
            slippage=0.005,
            strategy=DONCHIAN_BREAKOUT,
        )
        zero_cost = run_backtest(
            bars,
            starting_capital=1000,
            timeframe="1Hour",
            fee_rate=0,
            slippage=0,
            strategy=DONCHIAN_BREAKOUT,
        )

        self.assertGreater(zero_cost["strategy_return"], realistic["strategy_return"])

    def test_donchian_exit_executes_at_next_candle_open_after_10_bar_breakdown(self):
        bars = donchian_trend_bars(breakdown=True)
        result = run_backtest(
            bars,
            starting_capital=1000,
            timeframe="1Hour",
            fee_rate=0,
            slippage=0,
            strategy=DONCHIAN_BREAKOUT,
        )
        trade = result["trades"].iloc[0]

        self.assertEqual(trade["exit_reason"], "donchian_exit_breakdown")
        self.assertEqual(trade["exit_time"], bars.index[33])
        self.assertEqual(trade["exit_price"], bars.iloc[33]["open"])

    def test_same_engine_accepts_ma_and_donchian_specs(self):
        bars = donchian_trend_bars()
        ma_result = run_backtest(
            bars, timeframe="1Hour", fee_rate=0, slippage=0, strategy=MA_RSI_CROSSOVER
        )
        donchian_result = run_backtest(
            bars, timeframe="1Hour", fee_rate=0, slippage=0, strategy=DONCHIAN_BREAKOUT
        )

        self.assertEqual(ma_result["strategy_name"], "ma_rsi_crossover")
        self.assertEqual(donchian_result["strategy_name"], "donchian_breakout")
        self.assertEqual(DONCHIAN_BREAKOUT.required_warmup_bars(), 30)
        self.assertEqual(
            create_donchian_breakout_strategy(20, 10).parameters,
            {"entry_lookback": 20, "exit_lookback": 10, "stop_loss_percent": None, "take_profit_percent": None},
        )

    def test_net_and_gross_profit_factors_use_their_named_pnl(self):
        index = pd.date_range("2026-01-01", periods=5, freq="1h", tz="UTC")
        bars = pd.DataFrame(
            {
                "open": [100.0, 100.0, 102.0, 100.0, 99.5],
                "high": [100.0, 100.0, 102.0, 100.0, 99.5],
                "low": [100.0, 100.0, 102.0, 100.0, 99.5],
                "close": [100.0, 100.0, 102.0, 100.0, 99.5],
                "volume": [1.0] * 5,
            },
            index=index,
        )
        result = run_backtest(
            bars,
            starting_capital=1000,
            timeframe="1Hour",
            fee_rate=0.01,
            slippage=0,
            strategy=profit_factor_test_strategy(),
        )

        self.assertEqual(result["net_profit_factor"], 0.0)
        self.assertEqual(result["profit_factor"], 0.0)
        self.assertGreater(result["gross_profit_factor"], 1.0)

    def test_raw_market_return_and_same_notional_benchmark_are_available(self):
        bars = donchian_trend_bars()
        result = run_backtest(
            bars,
            timeframe="1Hour",
            fee_rate=0,
            slippage=0,
            strategy=DONCHIAN_BREAKOUT,
        )
        expected_market_return = (bars.iloc[-1]["close"] / bars.iloc[0]["open"] - 1) * 100

        self.assertAlmostEqual(result["raw_market_return_percent"], expected_market_return)
        self.assertIn("same_notional_return", result)
        self.assertIn("full_buy_hold_return", result)

    def test_strategy_metadata_distinguishes_required_and_available_warmup(self):
        bars = donchian_trend_bars()
        test_start = bars.index[5]
        result = run_backtest(
            bars,
            timeframe="1Hour",
            fee_rate=0,
            slippage=0,
            test_start=test_start,
            strategy=DONCHIAN_BREAKOUT,
        )

        self.assertEqual(result["required_warmup_bars"], 30)
        self.assertEqual(result["available_pretest_bars"], 5)
        self.assertEqual(result["warmup_bars"], result["available_pretest_bars"])


if __name__ == "__main__":
    unittest.main()
