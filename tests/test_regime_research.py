import csv
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import regime_research
from backtest import required_warmup_bars
from strategy import Decision, StrategySpec, get_strategy


def constant_bars(start, end, timeframe, warmup_bars):
    _, minutes_per_bar = regime_research.parse_timeframe(timeframe)
    duration = pd.Timedelta(minutes=minutes_per_bar)
    test_start = pd.Timestamp(end) - pd.Timedelta(days=2)
    first = test_start - duration * warmup_bars
    index = pd.date_range(start=first, end=pd.Timestamp(end) - duration, freq=duration)
    return pd.DataFrame(
        {
            "open": [100.0] * len(index),
            "high": [101.0] * len(index),
            "low": [99.0] * len(index),
            "close": [100.0] * len(index),
            "volume": [1.0] * len(index),
        },
        index=index,
    )


def persistent_position_bars(start, end, warmup_bars):
    test_start = pd.Timestamp(start)
    first = test_start - pd.Timedelta(hours=warmup_bars)
    index = pd.date_range(start=first, end=pd.Timestamp(end) - pd.Timedelta(hours=1), freq="1h")
    prices = [100.0] * len(index)
    start_index = warmup_bars
    # The position is entered early in Block 1, then marked up at both boundaries.
    prices[start_index + 23] = 120.0
    prices[start_index + 24] = 120.0
    prices[-1] = 130.0
    return pd.DataFrame(
        {
            "open": prices,
            "high": [price + 1.0 for price in prices],
            "low": [price - 1.0 for price in prices],
            "close": prices,
            "volume": [1.0] * len(index),
        },
        index=index,
    )


def buy_once_strategy(name="buy_once"):
    return StrategySpec(
        name=name,
        _prepare_indicators=lambda bars: bars.copy(),
        _decide_at=lambda _bars, index: (
            Decision("BUY", "buy_once") if index == 10 else Decision("HOLD", "hold")
        ),
        _warmup_lookback=lambda: 0,
        _parameters=lambda: {},
        _stop_loss=lambda: None,
        _take_profit=lambda: None,
    )


class RegimeResearchTests(unittest.TestCase):
    def test_default_blocks_are_720_days_chronological_adjacent_and_share_end(self):
        aligned_end = datetime(2026, 10, 5, tzinfo=timezone.utc)
        blocks = regime_research.build_blocks(aligned_end)

        self.assertEqual(len(blocks), 8)
        self.assertEqual(blocks[0]["block_start"], aligned_end - timedelta(days=720))
        self.assertEqual(blocks[-1]["block_end"], aligned_end)
        self.assertEqual(
            sum((block["block_end"] - block["block_start"]).days for block in blocks),
            720,
        )
        for current, following in zip(blocks, blocks[1:]):
            self.assertEqual(current["block_end"], following["block_start"])
            self.assertLess(current["block_start"], current["block_end"])
        self.assertTrue(all(block["block_end"] == aligned_end - timedelta(days=90 * (7-i)) for i, block in enumerate(blocks)))

    def test_one_continuous_backtest_preserves_position_and_marks_each_block(self):
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        end = start + timedelta(days=2)
        strategy = buy_once_strategy()
        bars = persistent_position_bars(start, end, required_warmup_bars(strategy=strategy))

        with (
            patch("regime_research.get_strategy", return_value=strategy),
            patch("regime_research.backtest.fetch_history", return_value=bars) as fetch_mock,
            patch("regime_research.config.BACKTEST_FEE_PERCENT", 0.0),
            patch("regime_research.config.BACKTEST_SLIPPAGE_PERCENT", 0.0),
            patch("trade_config.TRADE_AMOUNT_USD", 100.0),
            patch("trade_config.MAX_POSITION_USD", 100.0),
            redirect_stdout(io.StringIO()),
        ):
            rows = regime_research.run_regime_research(
                block_days=1,
                blocks=2,
                timeframes=["1Hour"],
                strategies=["buy_once"],
                starting_capital=1000,
                output=None,
                research_end_time=end,
            )

        self.assertEqual(fetch_mock.call_count, 1)
        first, second = rows
        self.assertEqual((first["entries_in_block"], first["exits_in_block"]), (1, 0))
        self.assertEqual((second["entries_in_block"], second["exits_in_block"]), (0, 1))
        self.assertEqual(second["winning_exits_in_block"], 1)
        self.assertEqual(second["losing_exits_in_block"], 0)
        self.assertEqual(first["starting_equity"], 1000.0)
        self.assertAlmostEqual(first["ending_equity"], 1020.0)
        self.assertAlmostEqual(first["block_net_pnl"], 20.0)
        self.assertAlmostEqual(first["block_net_return_percent"], 2.0)
        self.assertEqual(first["closed_trade_net_pnl"], 0.0)
        self.assertEqual(second["starting_equity"], first["ending_equity"])
        self.assertAlmostEqual(second["ending_equity"], 1030.0)
        self.assertAlmostEqual(first["raw_btc_return_percent"], 20.0)
        self.assertAlmostEqual(second["raw_btc_return_percent"], (130 / 120 - 1) * 100)
        self.assertEqual(second["closed_trade_net_pnl"], 30.0)
        self.assertEqual([row["block_start"] for row in rows], [
            start.isoformat(), (start + timedelta(days=1)).isoformat()
        ])

    def test_charged_costs_follow_transaction_blocks_and_closed_costs_are_labeled(self):
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        end = start + timedelta(days=2)
        strategy = buy_once_strategy()
        bars = persistent_position_bars(start, end, required_warmup_bars(strategy=strategy))

        with (
            patch("regime_research.get_strategy", return_value=strategy),
            patch("regime_research.backtest.fetch_history", return_value=bars),
            patch("regime_research.config.BACKTEST_FEE_PERCENT", 0.01),
            patch("regime_research.config.BACKTEST_SLIPPAGE_PERCENT", 0.02),
            patch("trade_config.TRADE_AMOUNT_USD", 100.0),
            patch("trade_config.MAX_POSITION_USD", 100.0),
            redirect_stdout(io.StringIO()),
        ):
            rows = regime_research.run_regime_research(
                block_days=1,
                blocks=2,
                timeframes=["1Hour"],
                strategies=["buy_once"],
                starting_capital=1000,
                output=None,
                research_end_time=end,
            )

        entry_block, exit_block = rows
        self.assertGreater(entry_block["charged_costs_in_block"], 0)
        self.assertGreater(exit_block["charged_costs_in_block"], 0)
        self.assertGreater(
            exit_block["closed_trade_costs"], exit_block["charged_costs_in_block"]
        )
        self.assertAlmostEqual(
            entry_block["charged_costs_in_block"]
            + exit_block["charged_costs_in_block"],
            exit_block["closed_trade_costs"],
        )

    def test_fetch_warmup_order_errors_and_csv_are_isolated(self):
        aligned = datetime(2026, 1, 3, tzinfo=timezone.utc)
        timeframes = ["2Hour", "1Hour", "badFrame"]
        strategies = ["ma_rsi_crossover", "donchian_breakout", "missing_strategy"]
        expected_warmup = max(
            required_warmup_bars(strategy=get_strategy(name))
            for name in strategies[:2]
        )
        bars_by_tf = {
            timeframe: constant_bars(
                aligned - timedelta(days=2), aligned, timeframe, expected_warmup
            )
            for timeframe in ("1Hour", "2Hour")
        }

        def fetch(days, timeframe, *, warmup_bars, end_time):
            self.assertEqual(days, 2)
            self.assertEqual(warmup_bars, expected_warmup)
            self.assertEqual(end_time, aligned)
            if timeframe == "2Hour":
                raise RuntimeError("simulated 2Hour history outage")
            return bars_by_tf[timeframe]

        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "regimes.csv"
            with (
                patch("regime_research.backtest.fetch_history", side_effect=fetch) as fetch_mock,
                redirect_stdout(io.StringIO()) as console,
            ):
                rows = regime_research.run_regime_research(
                    block_days=1,
                    blocks=2,
                    timeframes=timeframes,
                    strategies=strategies,
                    output=csv_path,
                    research_end_time=aligned,
                )
            with csv_path.open(newline="", encoding="utf-8") as csv_file:
                saved = list(csv.DictReader(csv_file))

        expected_keys = [
            (block_index, timeframe, strategy)
            for block_index in (1, 2)
            for timeframe in timeframes
            for strategy in strategies
        ]
        self.assertEqual(
            [(row["block_index"], row["timeframe"], row["strategy_name"]) for row in rows],
            expected_keys,
        )
        self.assertEqual(fetch_mock.call_count, 2)
        self.assertEqual(
            [call.args[1] for call in fetch_mock.call_args_list], ["2Hour", "1Hour"]
        )
        self.assertEqual(len(rows), 2 * len(timeframes) * len(strategies))
        successful = [row for row in rows if not row["error"]]
        self.assertEqual(len(successful), 4)
        self.assertTrue(all(row["timeframe"] == "1Hour" for row in successful))
        self.assertTrue(all(row["aligned_research_end"] == aligned.isoformat() for row in rows))
        self.assertTrue(all(row["error"] for row in rows if row["timeframe"] == "2Hour"))
        self.assertTrue(all(row["error"] for row in rows if row["strategy_name"] == "missing_strategy"))
        self.assertEqual(len(saved), len(rows))
        self.assertIn("strategy_parameters_json", saved[0])
        self.assertIn("fee_rate", saved[0])
        self.assertIn("slippage_rate", saved[0])
        self.assertEqual(
            {(row["block_start"], row["block_end"]) for row in rows},
            {
                ((aligned - timedelta(days=2)).isoformat(), (aligned - timedelta(days=1)).isoformat()),
                ((aligned - timedelta(days=1)).isoformat(), aligned.isoformat()),
            },
        )
        self.assertNotRegex(console.getvalue().lower(), r"\b(best|winner|optimal|recommended)\b")


if __name__ == "__main__":
    unittest.main()
