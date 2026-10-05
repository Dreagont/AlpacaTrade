import csv
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import backtest
import regime_filter_research as experiment


class RegimeFilterResearchTests(unittest.TestCase):
    def test_default_cli_values_and_help_options(self):
        args = experiment._build_parser().parse_args([])
        self.assertEqual(args.block_days, 90)
        self.assertEqual(args.blocks, 16)
        self.assertEqual(args.timeframe, "4Hour")
        with self.assertRaises(SystemExit) as help_exit:
            experiment.main(["--help"])
        self.assertEqual(help_exit.exception.code, 0)

    def test_cli_accepts_explicit_utc_end_time_without_running_research(self):
        with patch.object(experiment, "run_regime_filter_research") as run:
            self.assertEqual(experiment.main([
                "--end-time", "2022-10-26T04:00:00+00:00", "--no-csv",
            ]), 0)
        self.assertEqual(
            run.call_args.kwargs["research_end_time"],
            datetime(2022, 10, 26, 4, tzinfo=timezone.utc),
        )

    def test_cli_rejects_non_utc_end_time(self):
        with self.assertRaises(SystemExit) as error:
            experiment.main(["--end-time", "2022-10-26T04:00:00+02:00"])
        self.assertEqual(error.exception.code, 2)

    def test_fixed_strategies_fetch_once_run_continuously_and_write_block_csv(self):
        aligned = datetime(2026, 10, 5, 0, tzinfo=timezone.utc)
        block_days, blocks = 1, 2
        oldest = aligned - timedelta(days=block_days * blocks)
        required = max(
            backtest.required_warmup_bars(strategy=experiment.get_strategy(name))
            for name in experiment.STRATEGIES
        )
        index = pd.date_range(
            start=oldest - timedelta(hours=4 * required),
            end=aligned - timedelta(hours=4),
            freq="4h",
        )
        bars = pd.DataFrame(
            {
                "open": [100.0] * len(index), "high": [101.0] * len(index),
                "low": [99.0] * len(index), "close": [100.0] * len(index),
                "volume": [1.0] * len(index),
            },
            index=index,
        )
        original_backtest = backtest.run_backtest
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "filter.csv"
            with (
                patch("regime_filter_research.backtest.fetch_history", return_value=bars) as fetch,
                patch("regime_filter_research.backtest.run_backtest", wraps=original_backtest) as run,
                patch("config.BACKTEST_FEE_PERCENT", 0.0),
                patch("config.BACKTEST_SLIPPAGE_PERCENT", 0.0),
                redirect_stdout(io.StringIO()) as console,
            ):
                rows = experiment.run_regime_filter_research(
                    block_days=block_days, blocks=blocks, timeframe="4Hour",
                    output=output, research_end_time=aligned, starting_capital=1000,
                )
            with output.open(newline="", encoding="utf-8") as handle:
                saved = list(csv.DictReader(handle))

        self.assertEqual(fetch.call_count, 1)
        self.assertEqual(fetch.call_args.args[:2], (2, "4Hour"))
        self.assertEqual(fetch.call_args.kwargs["warmup_bars"], required)
        self.assertEqual(fetch.call_args.kwargs["end_time"], aligned)
        self.assertEqual(run.call_count, 3)
        self.assertTrue(all(call.kwargs["liquidate_at_end"] is False for call in run.call_args_list))
        self.assertEqual(len(rows), 2 * len(experiment.STRATEGIES))
        self.assertEqual(len(saved), len(rows))
        self.assertEqual(
            [(row["block_index"], row["strategy_name"]) for row in rows],
            [(block, strategy) for block in (1, 2) for strategy in experiment.STRATEGIES],
        )
        filtered = [row for row in rows if row["strategy_name"] == "donchian_regime_filter"]
        self.assertTrue(all(row["bullish_bar_count"] == 0 for row in filtered))
        self.assertTrue(all(row["non_bullish_bar_count"] == 6 for row in filtered))
        self.assertTrue(all(row["bullish_bar_percent"] == 0 for row in filtered))
        self.assertTrue(all(row["error"] == "" for row in rows))
        report = console.getvalue()
        self.assertIn("FILTER DELTA:", report)
        self.assertIn("POST-HOC BTC-positive", report)
        self.assertIn("in-sample evidence", report)
        self.assertNotRegex(report.lower(), r"\b(best strategy|winner|optimal)\b")
        self.assertIn("bullish_bar_percent", saved[0])
        self.assertIn("exits_due_to_regime_filter", saved[0])

    def test_strategy_position_and_equity_continue_across_block_boundary(self):
        aligned = datetime(2026, 10, 5, 0, tzinfo=timezone.utc)
        block_days, blocks = 1, 2
        oldest = aligned - timedelta(days=2)
        required = max(
            backtest.required_warmup_bars(strategy=experiment.get_strategy(name))
            for name in experiment.STRATEGIES
        )
        values = [100.0] * required + [101.0 + n for n in range(12)]
        index = pd.date_range(
            start=oldest - timedelta(hours=required * 4),
            end=aligned - timedelta(hours=4), freq="4h",
        )
        bars = pd.DataFrame({
            "open": values, "high": values,
            "low": [value - 1 for value in values], "close": values,
            "volume": [1.0] * len(values),
        }, index=index)
        with (
            patch("regime_filter_research.backtest.fetch_history", return_value=bars),
            patch("config.BACKTEST_FEE_PERCENT", 0.0),
            patch("config.BACKTEST_SLIPPAGE_PERCENT", 0.0),
            redirect_stdout(io.StringIO()),
        ):
            rows = experiment.run_regime_filter_research(
                block_days=block_days, blocks=blocks, timeframe="4Hour",
                output=None, research_end_time=aligned, starting_capital=1000,
            )
        filtered = [row for row in rows if row["strategy_name"] == "donchian_regime_filter"]
        self.assertEqual(len(filtered), 2)
        self.assertGreater(filtered[0]["ending_equity"], filtered[0]["starting_equity"])
        self.assertAlmostEqual(filtered[1]["starting_equity"], filtered[0]["ending_equity"])
        self.assertGreater(filtered[0]["time_invested_percent"], 0)
        self.assertGreater(filtered[1]["time_invested_percent"], 0)


if __name__ == "__main__":
    unittest.main()
