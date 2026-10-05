import unittest

import pandas as pd

from strategy import decide_at


def indicators(previous_fast, previous_slow, current_fast, current_slow, rsi):
    return pd.DataFrame(
        {
            "ma_fast": [previous_fast, current_fast],
            "ma_slow": [previous_slow, current_slow],
            "rsi": [50.0, rsi],
        }
    )


class StrategyDecisionTests(unittest.TestCase):
    def test_bullish_crossover_with_rsi_below_threshold_buys(self):
        decision = decide_at(indicators(10, 11, 12, 11, 50), 1)
        self.assertEqual(
            (decision.action, decision.reason),
            ("BUY", "bullish_ma_crossover_rsi_filter_passed"),
        )

    def test_bullish_crossover_with_rsi_above_threshold_holds(self):
        decision = decide_at(indicators(10, 11, 12, 11, 80), 1)
        self.assertEqual(
            (decision.action, decision.reason),
            ("HOLD", "bullish_ma_crossover_rsi_filter_failed"),
        )

    def test_bearish_crossover_sells(self):
        decision = decide_at(indicators(12, 11, 10, 11, 50), 1)
        self.assertEqual((decision.action, decision.reason), ("SELL", "bearish_ma_crossover"))

    def test_no_crossover_holds(self):
        decision = decide_at(indicators(10, 11, 10, 11, 50), 1)
        self.assertEqual((decision.action, decision.reason), ("HOLD", "no_crossover"))


if __name__ == "__main__":
    unittest.main()
