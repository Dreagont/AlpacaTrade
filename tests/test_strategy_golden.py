"""Golden-output guard for the frozen strategies at HEAD 3eec069."""

import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

import config
import backtest
from short_term_research import create_short_term_strategies
from strategy import get_strategy


FIXTURE_PATH = (
    Path(__file__).parent / "fixtures" / "strategy_golden_head_3eec069.json"
)


def _seeded_random_walk_bars(freq: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    periods = 5000
    index = pd.date_range("2018-01-01T00:00:00Z", periods=periods, freq=freq)
    opens = np.empty(periods)
    highs = np.empty(periods)
    lows = np.empty(periods)
    closes = np.empty(periods)
    opens[0] = closes[0] = 10000.0
    for index_at in range(periods):
        if index_at:
            opens[index_at] = closes[index_at - 1] * np.exp(rng.normal(0, 0.002))
            closes[index_at] = opens[index_at] * np.exp(rng.normal(0.00003, 0.008))
        wick_up = abs(rng.normal(0.0025, 0.001))
        wick_down = abs(rng.normal(0.0025, 0.001))
        highs[index_at] = max(opens[index_at], closes[index_at]) * (1 + wick_up)
        lows[index_at] = min(opens[index_at], closes[index_at]) * max(
            0.01, 1 - wick_down
        )
    return pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": np.full(periods, 1.0),
        },
        index=index,
    )


def _capture_run(strategy, bars, timeframe, liquidate_at_end):
    result = backtest.run_backtest(
        bars,
        starting_capital=config.BACKTEST_STARTING_CAPITAL,
        timeframe=timeframe,
        fee_rate=config.BACKTEST_FEE_PERCENT,
        slippage=config.BACKTEST_SLIPPAGE_PERCENT,
        strategy=strategy,
        liquidate_at_end=liquidate_at_end,
    )
    trades = [
        {
            "entry_time": pd.Timestamp(trade["entry_time"]).isoformat(),
            "exit_time": pd.Timestamp(trade["exit_time"]).isoformat(),
            "exit_reason": trade["exit_reason"],
            "net_pnl": round(float(trade["net_pnl"]), 10),
        }
        for _, trade in result["trades"].iterrows()
    ]
    return {
        "trades": trades,
        "ending_equity": round(float(result["ending_equity"]), 10),
    }


class StrategyGoldenOutputTests(unittest.TestCase):
    def test_frozen_strategy_outputs_match_head_3eec069_fixture(self):
        with FIXTURE_PATH.open(encoding="utf-8") as fixture_file:
            fixture = json.load(fixture_file)

        bars_4h = _seeded_random_walk_bars("4h", 20261005)
        bars_30m = _seeded_random_walk_bars("30min", 20261006)
        short_term_baseline = create_short_term_strategies()[0]
        strategies = (
            ("ma_rsi_crossover", get_strategy("ma_rsi_crossover"), bars_4h, "4Hour"),
            ("donchian_breakout", get_strategy("donchian_breakout"), bars_4h, "4Hour"),
            (
                "donchian_regime_filter",
                get_strategy("donchian_regime_filter"),
                bars_4h,
                "4Hour",
            ),
            ("short_term_donchian_30m", short_term_baseline, bars_30m, "30Min"),
        )
        actual = {}
        for name, strategy, bars, timeframe in strategies:
            for liquidate_at_end in (True, False):
                key = f"{name}|liquidate={str(liquidate_at_end).lower()}"
                actual[key] = _capture_run(
                    strategy, bars, timeframe, liquidate_at_end
                )

        expected = fixture["runs"]
        self.assertEqual(set(actual), set(expected))
        for run_key in expected:
            with self.subTest(run=run_key):
                actual_run = actual[run_key]
                expected_run = expected[run_key]
                self.assertEqual(len(actual_run["trades"]), len(expected_run["trades"]))
                for trade_index, (actual_trade, expected_trade) in enumerate(
                    zip(actual_run["trades"], expected_run["trades"])
                ):
                    with self.subTest(trade=trade_index):
                        for field in ("entry_time", "exit_time", "exit_reason"):
                            self.assertEqual(actual_trade[field], expected_trade[field])
                        self.assertAlmostEqual(
                            actual_trade["net_pnl"], expected_trade["net_pnl"], places=9
                        )
                self.assertAlmostEqual(
                    actual_run["ending_equity"], expected_run["ending_equity"], places=9
                )


if __name__ == "__main__":
    unittest.main()
