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


def _research_strategy(name):
    if name == "regime_only_4h":
        return experiment.create_regime_only_strategy()
    return experiment.get_strategy(name)


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
            backtest.required_warmup_bars(strategy=_research_strategy(name))
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
        # One interior gap belongs only to block 1. Each CSV row must report
        # its own [block_start, block_end) data-quality counts.
        bars = bars.drop(index=index[required + 2])
        original_backtest = backtest.run_backtest
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "filter.csv"
            with (
                patch("regime_filter_research.backtest.fetch_history", return_value=bars) as fetch,
                patch("regime_filter_research.backtest.run_backtest", wraps=original_backtest) as run,
                patch.dict("config.FEE_PROFILES", {"binance_spot_bnb": {"fee_rate": 0.0, "slippage_rate": 0.0}}),
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
        self.assertEqual(run.call_count, 4)
        self.assertTrue(all(call.args[0] is bars for call in run.call_args_list))
        self.assertTrue(all(call.kwargs["liquidate_at_end"] is False for call in run.call_args_list))
        block_rows = [row for row in rows if row["row_type"] == "BLOCK"]
        total_rows = [row for row in rows if row["row_type"] == "TOTAL"]
        self.assertEqual(len(block_rows), 2 * len(experiment.STRATEGIES))
        self.assertEqual(len(total_rows), len(experiment.STRATEGIES))
        self.assertEqual(len(saved), len(rows))
        self.assertEqual(
            [(row["block_index"], row["strategy_name"]) for row in block_rows],
            [(block, strategy) for block in (1, 2) for strategy in experiment.STRATEGIES],
        )
        self.assertTrue(all(row["row_type"] == "BLOCK" for row in saved[:len(block_rows)]))
        filtered = [
            row for row in block_rows
            if row["strategy_name"] == "donchian_regime_filter"
        ]
        self.assertTrue(all(row["bullish_bar_count"] == 0 for row in filtered))
        self.assertEqual([row["non_bullish_bar_count"] for row in filtered], [5, 6])
        self.assertTrue(all(row["bullish_bar_percent"] == 0 for row in filtered))
        self.assertTrue(all(row["error"] == "" for row in rows))
        self.assertEqual([row["expected_candle_count"] for row in block_rows[:4]], [6] * 4)
        self.assertEqual([row["actual_candle_count"] for row in block_rows[:4]], [5] * 4)
        self.assertEqual([row["missing_candle_count"] for row in block_rows[:4]], [1] * 4)
        self.assertEqual([row["expected_candle_count"] for row in block_rows[4:]], [6] * 4)
        self.assertEqual([row["actual_candle_count"] for row in block_rows[4:]], [6] * 4)
        self.assertEqual([row["missing_candle_count"] for row in block_rows[4:]], [0] * 4)
        self.assertTrue(all(row["pnl_percent_of_notional"] is not None for row in block_rows))
        self.assertTrue(all(row["max_drawdown_percent_of_notional"] is not None for row in block_rows))
        self.assertTrue(all(row["compounded_trade_return_percent"] is None for row in block_rows))
        self.assertTrue(all(row["compounded_trade_return_percent"] is not None for row in total_rows))
        self.assertTrue(all(row["btc_buy_hold_return_after_costs_percent"] is not None for row in total_rows))
        report = console.getvalue()
        self.assertIn("FILTER DELTA (filtered - plain Donchian):", report)
        self.assertIn("FILTERED MINUS REGIME-ONLY:", report)
        self.assertIn("TOTAL COMPARISON", report)
        self.assertIn("POST-HOC BTC-positive", report)
        self.assertIn("in-sample evidence", report)
        self.assertIn("displayed blocks summarize the historical period in this run", report)
        self.assertNotIn("These 16 historical blocks", report)
        self.assertNotRegex(report.lower(), r"\b(best strategy|winner|optimal)\b")
        self.assertIn("bullish_bar_percent", saved[0])
        self.assertIn("exits_due_to_regime_filter", saved[0])

    def test_strategy_position_and_equity_continue_across_block_boundary(self):
        aligned = datetime(2026, 10, 5, 0, tzinfo=timezone.utc)
        block_days, blocks = 1, 2
        oldest = aligned - timedelta(days=2)
        required = max(
            backtest.required_warmup_bars(strategy=_research_strategy(name))
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
            patch.dict("config.FEE_PROFILES", {"binance_spot_bnb": {"fee_rate": 0.0, "slippage_rate": 0.0}}),
            redirect_stdout(io.StringIO()),
        ):
            rows = experiment.run_regime_filter_research(
                block_days=block_days, blocks=blocks, timeframe="4Hour",
                output=None, research_end_time=aligned, starting_capital=1000,
            )
        filtered = [
            row for row in rows
            if row["row_type"] == "BLOCK"
            and row["strategy_name"] == "donchian_regime_filter"
        ]
        self.assertEqual(len(filtered), 2)
        self.assertGreater(filtered[0]["ending_equity"], filtered[0]["starting_equity"])
        self.assertAlmostEqual(filtered[1]["starting_equity"], filtered[0]["ending_equity"])
        self.assertGreater(filtered[0]["time_invested_percent"], 0)
        self.assertGreater(filtered[1]["time_invested_percent"], 0)


if __name__ == "__main__":
    unittest.main()
