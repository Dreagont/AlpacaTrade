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
import config
import fee_sensitivity
import regime_filter_research
import regime_research
import research
import short_term_research
from strategy import Decision, StrategySpec
from timeframes import parse_timeframe


def history(days, timeframe, *, warmup_bars=0, end_time=None):
    end = pd.Timestamp(end_time or datetime(2026, 1, 3, tzinfo=timezone.utc))
    start = end - pd.Timedelta(days=days)
    duration = pd.Timedelta(minutes=parse_timeframe(timeframe)[1])
    index = pd.date_range(start - duration * warmup_bars, end - duration, freq=duration)
    bars = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                         "close": 100.0, "volume": 1.0}, index=index)
    bars.attrs.update(test_start=start, requested_start_time=index[0], requested_end_time=end)
    return bars


def buy_once(entry_index=0):
    return StrategySpec(
        name="buy_once", _prepare_indicators=lambda bars: bars,
        _decide_at=lambda bars, i: Decision("BUY", "entry") if i == entry_index else Decision("HOLD", "hold"),
        _warmup_lookback=lambda: 0, _parameters=lambda: {},
        _stop_loss=lambda: None, _take_profit=lambda: None,
    )


class FeeProfileTests(unittest.TestCase):
    def test_exact_table_default_and_compatibility_aliases(self):
        self.assertEqual(config.FEE_PROFILES, {
            "alpaca": {"fee_rate": 0.0025, "slippage_rate": 0.0005},
            "binance_spot_bnb": {"fee_rate": 0.00075, "slippage_rate": 0.0002},
            "mexc_spot_taker": {"fee_rate": 0.0005, "slippage_rate": 0.0002},
            "zero_cost": {"fee_rate": 0.0, "slippage_rate": 0.0},
        })
        self.assertEqual(config.BACKTEST_FEE_PROFILE, "binance_spot_bnb")
        self.assertEqual(config.BACKTEST_FEE_PERCENT, 0.00075)
        self.assertEqual(config.BACKTEST_SLIPPAGE_PERCENT, 0.0002)

    def test_get_profile_returns_copy_and_invalid_lists_names(self):
        rates = config.get_fee_profile("alpaca")
        rates["fee_rate"] = 0
        self.assertEqual(config.get_fee_profile("alpaca")["fee_rate"], 0.0025)
        with self.assertRaises(ValueError) as caught:
            config.get_fee_profile("typo")
        for name in config.FEE_PROFILES:
            self.assertIn(name, str(caught.exception))

    def test_all_five_cli_parsers_default_explicit_and_invalid(self):
        for module in (backtest, research, regime_research, regime_filter_research, short_term_research):
            with self.subTest(cli=module.__name__):
                parser = module._build_parser()
                self.assertEqual(parser.parse_args([]).fee_profile, "binance_spot_bnb")
                self.assertEqual(parser.parse_args(["--fee-profile", "alpaca"]).fee_profile, "alpaca")
                with redirect_stdout(io.StringIO()), patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
                    parser.parse_args(["--fee-profile", "bad"])

    def test_backtest_profile_metadata_and_default_printed(self):
        bars = history(1, "4Hour")
        result = backtest.run_backtest(bars, timeframe="4Hour", strategy=buy_once())
        console = io.StringIO()
        with redirect_stdout(console):
            backtest.print_report(result)
        self.assertIn("fee_profile=binance_spot_bnb", console.getvalue())
        self.assertIn("fee_rate=0.00075", console.getvalue())
        self.assertIn("slippage_rate=0.0002", console.getvalue())
        self.assertIn("approximates fees paid separately in BNB", console.getvalue())
        self.assertTrue((result["trades"]["fee_profile"] == "binance_spot_bnb").all())

    def test_backtest_cli_selected_profile_reaches_engine(self):
        real = backtest.run_backtest
        with patch.object(backtest, "fetch_history", side_effect=history), \
             patch.object(backtest, "run_backtest", wraps=real) as run, redirect_stdout(io.StringIO()):
            self.assertEqual(backtest._run_main(["--fee-profile", "alpaca", "--no-csv", "--lookback-days", "1"]), 0)
        self.assertEqual(run.call_args.kwargs["fee_profile"], "alpaca")

    def test_research_cli_profiles_propagate_to_engine_headers_and_rows(self):
        cases = [
            (research, ["--lookbacks", "2", "--timeframes", "4Hour", "--strategies", "regime_only_4h"]),
            (regime_research, ["--blocks", "1", "--block-days", "2", "--timeframes", "4Hour", "--strategies", "regime_only_4h"]),
            (regime_filter_research, ["--blocks", "1", "--block-days", "2"]),
            (short_term_research, ["--blocks", "1", "--block-days", "2"]),
        ]
        real = backtest.run_backtest
        for module, argv in cases:
            with self.subTest(cli=module.__name__), tempfile.TemporaryDirectory() as directory:
                csv_path = Path(directory) / "report.csv"
                console = io.StringIO()
                with patch.object(backtest, "fetch_history", side_effect=history), \
                     patch.object(backtest, "run_backtest", wraps=real) as run, redirect_stdout(console):
                    self.assertEqual(module.main(argv + ["--fee-profile", "alpaca", "--output", str(csv_path)]), 0)
                self.assertTrue(run.called)
                paid_calls = [call for call in run.call_args_list if call.kwargs["fee_rate"] != 0]
                self.assertTrue(paid_calls)
                for call in paid_calls:
                    self.assertEqual(call.kwargs["fee_rate"], 0.0025)
                    self.assertEqual(call.kwargs["slippage"], 0.0005)
                self.assertIn("fee_profile=alpaca", console.getvalue())
                with csv_path.open() as handle:
                    rows = list(csv.DictReader(handle))
                self.assertTrue(rows)
                for row in rows:
                    self.assertEqual(row["fee_profile"], "alpaca")
                    self.assertEqual(float(row["fee_rate"]), 0.0025)
                    self.assertEqual(float(row["slippage_rate"]), 0.0005)

    def test_explicit_rate_conflict_and_custom_rates_metadata(self):
        bars = history(1, "4Hour")
        with self.assertRaisesRegex(ValueError, "conflict"):
            backtest.run_backtest(bars, fee_profile="alpaca", fee_rate=0)
        result = backtest.run_backtest(bars, timeframe="4Hour", fee_rate=0.01, slippage=0.03)
        self.assertEqual(result["fee_profile"], "custom")
        self.assertEqual(result["fee_rate"], 0.01)


class FeeSensitivityTests(unittest.TestCase):
    def test_fetch_once_full_run_per_profile_and_cost_monotonicity(self):
        end = datetime(2026, 1, 3, tzinfo=timezone.utc)
        bars = history(2, "4Hour", warmup_bars=10, end_time=end)
        # A fixed path without SL/TP: BUY followed by HOLD, ending open.
        strategy = buy_once(entry_index=10)
        real = backtest.run_backtest
        console = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "fees.csv"
            with patch.object(fee_sensitivity, "get_strategy", return_value=strategy), \
                 patch.object(backtest, "fetch_history", return_value=bars) as fetch, \
                 patch.object(backtest, "run_backtest", wraps=real) as run, redirect_stdout(console):
                rows = fee_sensitivity.run_fee_sensitivity(blocks=2, block_days=1, end_time=end, output=output)
            self.assertEqual(fetch.call_count, 1)
            self.assertEqual(run.call_count, len(config.FEE_PROFILES))
            self.assertEqual(fetch.call_args.args, (2, "4Hour"))
            self.assertEqual(fetch.call_args.kwargs["end_time"], end)
            free = next(row for row in rows if row["fee_profile"] == "zero_cost")
            self.assertTrue(all(free["compounded_trade_return_percent"] >= row["compounded_trade_return_percent"] for row in rows))
            self.assertEqual(free["total_costs_usd"], 0)
            for row, call in zip(rows, run.call_args_list):
                self.assertTrue(row["open_position_marked"])
                self.assertFalse(call.kwargs["liquidate_at_end"])
                fee, slip = row["fee_rate"], row["slippage_rate"]
                expected_return = ((1 - fee) ** 2 * (1 - slip) / (1 + slip) - 1) * 100
                self.assertAlmostEqual(row["compounded_trade_return_percent"], expected_return)
                self.assertEqual(row["btc_buy_hold_return_percent"], 0)
                if fee:
                    # Entry plus hypothetical exit costs, with $20 fixed notional.
                    gross_q = 20 / (100 * (1 + slip))
                    q = gross_q * (1 - fee)
                    expected_cost = gross_q * 100 * fee + gross_q * 100 * slip + q * 100 * slip + q * 100 * (1 - slip) * fee
                    self.assertAlmostEqual(row["total_costs_usd"], expected_cost)
            with output.open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), len(config.FEE_PROFILES))
        self.assertIn("fee_profile=binance_spot_bnb", console.getvalue())

    def test_cli_defaults_end_time_and_optional_csv(self):
        parser = fee_sensitivity._build_parser()
        args = parser.parse_args([])
        self.assertEqual((args.strategy, args.timeframe, args.blocks, args.block_days), ("regime_only_4h", "4Hour", 16, 90))
        with patch.object(fee_sensitivity, "run_fee_sensitivity") as run:
            self.assertEqual(fee_sensitivity.main(["--blocks", "2", "--block-days", "3", "--end-time", "2026-01-03T00:00:00Z", "--csv"]), 0)
        self.assertEqual(run.call_args.kwargs["output"], "fee_sensitivity.csv")
        self.assertEqual(run.call_args.kwargs["end_time"].tzinfo, timezone.utc)
        with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["--end-time", "2026-01-03T00:00:00+02:00"])

    def test_invalid_span_or_output_fails_before_fetch(self):
        with patch.object(backtest, "fetch_history") as fetch:
            with self.assertRaises(ValueError):
                fee_sensitivity.run_fee_sensitivity(blocks=0)
            with self.assertRaisesRegex(ValueError, "database"):
                fee_sensitivity.run_fee_sensitivity(output=config.DATABASE_PATH)
        fetch.assert_not_called()

    def test_missing_boundary_history_rejected(self):
        bars = history(2, "4Hour", warmup_bars=10)
        bars = bars.iloc[:-1]
        with patch.object(backtest, "fetch_history", return_value=bars), \
             patch.object(fee_sensitivity, "get_strategy", return_value=buy_once()), self.assertRaisesRegex(ValueError, "boundaries"):
            fee_sensitivity.run_fee_sensitivity(blocks=2, block_days=1, end_time=datetime(2026, 1, 3, tzinfo=timezone.utc))


if __name__ == "__main__":
    unittest.main()
