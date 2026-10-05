import unittest

import pandas as pd

from backtest import run_backtest
from strategy import Decision, StrategySpec


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


def test_strategy(decide_at, prepare_indicators=None):
    return StrategySpec(
        name="test_strategy",
        _prepare_indicators=prepare_indicators or (lambda bars: bars.copy()),
        _decide_at=decide_at,
        _warmup_lookback=lambda: 0,
        _parameters=lambda: {},
        _stop_loss=lambda: None,
        _take_profit=lambda: None,
    )


class BacktestAccountingTests(unittest.TestCase):
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
