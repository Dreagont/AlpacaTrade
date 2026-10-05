import csv
import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import config
import research
import trade_config


def history_frame(end_time, lookback_days=2):
    test_start = end_time - timedelta(days=lookback_days)
    index = pd.date_range(
        end=test_start - timedelta(minutes=5),
        periods=research.backtest.required_warmup_bars() + 1,
        freq="5min",
    ).append(pd.date_range(test_start, periods=577, freq="5min"))
    bars = pd.DataFrame(
        {name: [100.0] * len(index) for name in ("open", "high", "low", "close", "volume")},
        index=index,
    )
    bars.attrs["test_start"] = test_start
    return bars


def backtest_result(start, end):
    return {
        "start_time": start,
        "end_time": end,
        "warmup_bars": research.backtest.required_warmup_bars(),
        "total_trades": 4,
        "winning_trades": 3,
        "losing_trades": 1,
        "win_rate": 75.0,
        "gross_pnl": 50.0,
        "total_fees": 3.0,
        "total_slippage": 1.0,
        "total_costs": 4.0,
        "net_profit": 46.0,
        "strategy_return": 4.6,
        "average_pnl_per_trade": 11.5,
        "average_gross_return": 2.0,
        "average_net_return": 1.8,
        "profit_factor": 2.0,
        "max_drawdown": 5.0,
        "daily_sharpe": 1.2,
        "daily_return_observations": 2,
        "average_holding_minutes": 30.0,
        "median_holding_minutes": 20.0,
        "time_invested_percent": 10.0,
        "average_capital_exposure": 2.0,
        "maximum_capital_exposure": 5.0,
        "full_buy_hold_return": 8.0,
        "same_notional_return": 1.0,
    }


class ResearchTests(unittest.TestCase):
    def setUp(self):
        self.end = datetime(2026, 10, 5, 12, tzinfo=timezone.utc)
        self.test_start = self.end - timedelta(days=2)

    def run_with_mocks(self, *, timeframes=research.DEFAULT_TIMEFRAMES, output=None, fetch=None):
        bars = history_frame(self.end)
        with (
            patch("research.backtest.fetch_history", side_effect=fetch or (lambda *a, **k: bars)) as fetch_mock,
            patch(
                "research.backtest.run_backtest",
                return_value=backtest_result(self.test_start, self.end),
            ) as run_mock,
            redirect_stdout(io.StringIO()) as console,
        ):
            rows = research.run_research(
                lookback_days=2,
                timeframes=timeframes,
                output=output,
                research_end_time=self.end,
            )
        return rows, fetch_mock, run_mock, console.getvalue()

    def test_default_timeframe_order_is_preserved_and_no_winner_is_selected(self):
        rows, _, _, output = self.run_with_mocks()
        self.assertEqual([row["timeframe"] for row in rows], ["5Min", "15Min", "30Min", "1Hour"])
        self.assertNotRegex(output.lower(), r"\b(best timeframe|winner|recommended timeframe|optimal)\b")

    def test_every_fetch_uses_same_end_time_and_warmup_aware_api(self):
        rows, fetch_mock, run_mock, _ = self.run_with_mocks()
        self.assertEqual(len(rows), 4)
        self.assertEqual(fetch_mock.call_count, 4)
        for call, timeframe in zip(fetch_mock.call_args_list, research.DEFAULT_TIMEFRAMES):
            self.assertEqual(call.args, (2, timeframe))
            self.assertIs(call.kwargs["end_time"], self.end)
        self.assertEqual(run_mock.call_count, 4)
        self.assertTrue(all(call.kwargs["test_start"] == self.test_start for call in run_mock.call_args_list))

    def test_aligns_hour_timeframes_to_common_4_hour_boundary(self):
        current = datetime(2026, 10, 5, 1, 44, tzinfo=timezone.utc)
        aligned = research.align_research_end_time(
            current, ["1Hour", "2Hour", "4Hour"]
        )
        self.assertEqual(aligned, datetime(2026, 10, 5, 0, tzinfo=timezone.utc))

    def test_aligns_minute_timeframes_to_common_boundary(self):
        current = datetime(2026, 10, 5, 10, 37, tzinfo=timezone.utc)
        aligned = research.align_research_end_time(
            current, ["5Min", "15Min", "30Min"]
        )
        self.assertEqual(aligned, datetime(2026, 10, 5, 10, 30, tzinfo=timezone.utc))

    def test_already_aligned_time_remains_unchanged(self):
        current = datetime(2026, 10, 5, 0, tzinfo=timezone.utc)
        aligned = research.align_research_end_time(
            current, ["1Hour", "2Hour", "4Hour"]
        )
        self.assertIs(aligned, current)

    def test_existing_timeframe_parser_supports_1_2_and_4_hours(self):
        self.assertEqual(research.parse_timeframe("1Hour")[1], 60)
        self.assertEqual(research.parse_timeframe("2Hour")[1], 120)
        self.assertEqual(research.parse_timeframe("4Hour")[1], 240)

    def test_calendar_month_alignment_is_rejected_explicitly(self):
        current = datetime(2026, 10, 5, 1, 44, tzinfo=timezone.utc)
        with self.assertRaisesRegex(ValueError, "Month timeframes is unsupported"):
            research.align_research_end_time(current, ["1Month"])

    def test_research_uses_common_aligned_end_for_fetch_and_csv(self):
        requested = datetime(2026, 10, 5, 1, 44, tzinfo=timezone.utc)
        aligned = datetime(2026, 10, 5, 0, tzinfo=timezone.utc)
        test_start = aligned - timedelta(days=365)
        bars = history_frame(aligned, lookback_days=365)
        timeframes = ["1Hour", "2Hour", "4Hour"]

        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "results.csv"
            with (
                patch("research.backtest.fetch_history", return_value=bars) as fetch_mock,
                patch(
                    "research.backtest.run_backtest",
                    return_value=backtest_result(test_start, aligned),
                ) as run_mock,
                redirect_stdout(io.StringIO()) as console,
            ):
                rows = research.run_research(
                    lookback_days=365,
                    timeframes=timeframes,
                    output=csv_path,
                    research_end_time=requested,
                )

            with csv_path.open(newline="", encoding="utf-8") as csv_file:
                saved_rows = list(csv.DictReader(csv_file))

        self.assertEqual(fetch_mock.call_count, len(timeframes))
        for call, timeframe in zip(fetch_mock.call_args_list, timeframes):
            self.assertEqual(call.args, (365, timeframe))
            self.assertEqual(call.kwargs["end_time"], aligned)
        self.assertTrue(
            all(call.kwargs["test_start"] == test_start for call in run_mock.call_args_list)
        )
        self.assertEqual(
            [row["research_end_time"] for row in rows], [aligned.isoformat()] * 3
        )
        self.assertEqual(
            [row["requested_research_time"] for row in rows], [requested.isoformat()] * 3
        )
        self.assertEqual(
            [row["research_end_time"] for row in saved_rows], [aligned.isoformat()] * 3
        )
        self.assertEqual(
            [row["requested_research_time"] for row in saved_rows],
            [requested.isoformat()] * 3,
        )
        self.assertIn(f"Common aligned end: {aligned.isoformat()}", console.getvalue())
        self.assertIn(f"Requested research time: {requested.isoformat()}", console.getvalue())

    def test_lookback_matrix_order_fetch_reuse_alignment_csv_and_summary(self):
        requested = datetime(2026, 10, 5, 2, 30, tzinfo=timezone.utc)
        aligned = datetime(2026, 10, 5, 0, tzinfo=timezone.utc)
        lookbacks = [90, 180, 365, 730]
        timeframes = ["1Hour", "2Hour", "4Hour"]
        bars = history_frame(aligned, lookback_days=max(lookbacks))

        def run_one_window(_bars, **kwargs):
            return backtest_result(kwargs["test_start"], aligned)

        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "matrix.csv"
            with (
                patch("research.backtest.fetch_history", return_value=bars) as fetch_mock,
                patch("research.backtest.run_backtest", side_effect=run_one_window) as run_mock,
                redirect_stdout(io.StringIO()) as console,
            ):
                rows = research.run_research(
                    lookbacks=lookbacks,
                    timeframes=timeframes,
                    output=csv_path,
                    research_end_time=requested,
                )
            with csv_path.open(newline="", encoding="utf-8") as csv_file:
                saved_rows = list(csv.DictReader(csv_file))

        expected_pairs = [
            (lookback, timeframe)
            for lookback in lookbacks
            for timeframe in timeframes
        ]
        self.assertEqual(
            [(row["lookback_days"], row["timeframe"]) for row in rows],
            expected_pairs,
        )
        self.assertEqual(fetch_mock.call_count, len(timeframes))
        for call, timeframe in zip(fetch_mock.call_args_list, timeframes):
            self.assertEqual(call.args, (max(lookbacks), timeframe))
            self.assertEqual(call.kwargs["end_time"], aligned)
        self.assertEqual(run_mock.call_count, len(expected_pairs))
        self.assertEqual(
            [call.kwargs["test_start"] for call in run_mock.call_args_list],
            [aligned - timedelta(days=lookback) for lookback, _ in expected_pairs],
        )
        self.assertTrue(all(row["research_end_time"] == aligned.isoformat() for row in rows))
        self.assertTrue(all(row["requested_research_time"] == requested.isoformat() for row in rows))
        self.assertTrue(all(row["strategy_name"] == "ma_rsi_crossover" for row in rows))
        self.assertTrue(all(row["full_capital_buy_hold_return"] == 8.0 for row in rows))
        self.assertEqual(len(saved_rows), len(expected_pairs))
        self.assertEqual(
            [(int(row["lookback_days"]), row["timeframe"]) for row in saved_rows],
            expected_pairs,
        )
        self.assertTrue(all(row["strategy_name"] == "ma_rsi_crossover" for row in saved_rows))
        self.assertIn("B&H%", console.getvalue())
        self.assertNotRegex(
            console.getvalue().lower(),
            r"\b(best|winner|optimal)\b",
        )

    def test_warmup_is_validated_separately_for_each_lookback(self):
        aligned = datetime(2026, 10, 5, 0, tzinfo=timezone.utc)
        long_start = aligned - timedelta(days=730)
        short_start = aligned - timedelta(days=90)
        required = research.backtest.required_warmup_bars()
        early_warmup = pd.date_range(
            end=long_start - timedelta(hours=1), periods=required - 1, freq="1h"
        )
        bars_inside_long_window = pd.date_range(
            start=long_start,
            end=short_start - timedelta(days=1),
            freq="1D",
        )
        short_window_bars = pd.date_range(start=short_start, periods=5, freq="1h")
        index = early_warmup.append(bars_inside_long_window).append(short_window_bars)
        bars = pd.DataFrame(
            {name: [100.0] * len(index) for name in ("open", "high", "low", "close", "volume")},
            index=index,
        )
        bars.attrs["test_start"] = long_start

        with (
            patch("research.backtest.fetch_history", return_value=bars) as fetch_mock,
            patch(
                "research.backtest.run_backtest",
                return_value=backtest_result(short_start, aligned),
            ) as run_mock,
            redirect_stdout(io.StringIO()),
        ):
            rows = research.run_research(
                lookbacks=[90, 730],
                timeframes=["1Hour"],
                output=None,
                research_end_time=aligned,
            )

        self.assertEqual(fetch_mock.call_count, 1)
        self.assertEqual([row["lookback_days"] for row in rows], [90, 730])
        self.assertEqual(rows[0]["error"], "")
        self.assertIn("Insufficient warm-up history", rows[1]["error"])
        self.assertEqual(run_mock.call_count, 1)
        self.assertEqual(run_mock.call_args.kwargs["test_start"], short_start)

    def test_fetch_failure_creates_errors_for_only_that_timeframe(self):
        aligned = datetime(2026, 10, 5, 0, tzinfo=timezone.utc)
        lookbacks = [90, 180]
        timeframes = ["1Hour", "2Hour", "4Hour"]
        bars = history_frame(aligned, lookback_days=max(lookbacks))

        def fetch_once(_days, timeframe, **_kwargs):
            if timeframe == "2Hour":
                raise RuntimeError("simulated 2Hour fetch failure")
            return bars

        with (
            patch("research.backtest.fetch_history", side_effect=fetch_once) as fetch_mock,
            patch(
                "research.backtest.run_backtest",
                side_effect=lambda _bars, **kwargs: backtest_result(
                    kwargs["test_start"], aligned
                ),
            ) as run_mock,
            redirect_stdout(io.StringIO()),
        ):
            rows = research.run_research(
                lookbacks=lookbacks,
                timeframes=timeframes,
                output=None,
                research_end_time=aligned,
            )

        self.assertEqual(fetch_mock.call_count, len(timeframes))
        self.assertEqual(run_mock.call_count, len(lookbacks) * 2)
        self.assertEqual(
            [(row["lookback_days"], row["timeframe"]) for row in rows],
            [(lookback, timeframe) for lookback in lookbacks for timeframe in timeframes],
        )
        self.assertTrue(
            all(
                row["error"] == "simulated 2Hour fetch failure"
                for row in rows
                if row["timeframe"] == "2Hour"
            )
        )
        self.assertTrue(
            all(not row["error"] for row in rows if row["timeframe"] != "2Hour")
        )

    def test_one_backtest_failure_does_not_affect_other_matrix_rows(self):
        aligned = datetime(2026, 10, 5, 0, tzinfo=timezone.utc)
        lookbacks = [90, 180]
        timeframes = ["1Hour", "2Hour"]
        bars = history_frame(aligned, lookback_days=max(lookbacks))
        failing_start = aligned - timedelta(days=180)

        def run_one_window(_bars, **kwargs):
            if kwargs["timeframe"] == "2Hour" and kwargs["test_start"] == failing_start:
                raise RuntimeError("simulated 180d 2Hour backtest failure")
            return backtest_result(kwargs["test_start"], aligned)

        with (
            patch("research.backtest.fetch_history", return_value=bars),
            patch("research.backtest.run_backtest", side_effect=run_one_window),
            redirect_stdout(io.StringIO()),
        ):
            rows = research.run_research(
                lookbacks=lookbacks,
                timeframes=timeframes,
                output=None,
                research_end_time=aligned,
            )

        self.assertEqual(len(rows), 4)
        failed = [row for row in rows if row["error"]]
        self.assertEqual(len(failed), 1)
        self.assertEqual((failed[0]["lookback_days"], failed[0]["timeframe"]), (180, "2Hour"))
        self.assertEqual(failed[0]["error"], "simulated 180d 2Hour backtest failure")

    def test_cli_defaults_alias_and_conflicting_lookback_options(self):
        with patch("research.run_research") as run_mock:
            research.main(["--no-csv", "--timeframes", "1Hour"])
            self.assertEqual(run_mock.call_args.kwargs["lookbacks"], [90, 180, 365, 730])

            research.main(
                [
                    "--no-csv",
                    "--lookbacks",
                    "90",
                    "180",
                    "--timeframes",
                    "4Hour",
                ]
            )
            self.assertEqual(run_mock.call_args.kwargs["lookbacks"], [90, 180])
            self.assertEqual(run_mock.call_args.kwargs["timeframes"], ["4Hour"])

            research.main(["--no-csv", "--lookback-days", "365", "--timeframes", "1Hour"])
            self.assertEqual(run_mock.call_args.kwargs["lookbacks"], [365])

            with (
                redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit) as raised,
            ):
                research.main(
                    [
                        "--lookbacks",
                        "90",
                        "180",
                        "--lookback-days",
                        "365",
                    ]
                )
            self.assertEqual(raised.exception.code, 2)

    def test_metrics_gross_return_trade_rate_and_strategy_parameters(self):
        rows, _, _, _ = self.run_with_mocks(timeframes=["15Min"])
        row = rows[0]
        self.assertEqual(row["gross_return_percent"], 5.0)
        self.assertAlmostEqual(row["trades_per_day"], 2.0)
        self.assertEqual(row["fast_ma"], trade_config.FAST_MA)
        self.assertEqual(row["slow_ma"], trade_config.SLOW_MA)
        self.assertEqual(row["rsi_period"], trade_config.RSI_PERIOD)
        self.assertEqual(row["rsi_buy_threshold"], trade_config.RSI_BUY_THRESHOLD)
        self.assertEqual(row["stop_loss_percent"], trade_config.STOP_LOSS_PERCENT)
        self.assertEqual(row["take_profit_percent"], trade_config.TAKE_PROFIT_PERCENT)
        self.assertEqual(row["trade_amount_usd"], trade_config.TRADE_AMOUNT_USD)
        self.assertEqual(row["max_position_usd"], trade_config.MAX_POSITION_USD)
        self.assertEqual(row["fee_percent"], config.BACKTEST_FEE_PERCENT)
        self.assertEqual(row["slippage_percent"], config.BACKTEST_SLIPPAGE_PERCENT)
        self.assertEqual(row["research_end_time"], self.end.isoformat())

    def test_failed_timeframe_does_not_prevent_later_timeframe_and_has_error(self):
        bars = history_frame(self.end)

        def fetch(_days, timeframe, **_kwargs):
            if timeframe == "15Min":
                raise RuntimeError("simulated data failure")
            return bars

        rows, _, run_mock, output = self.run_with_mocks(
            timeframes=["5Min", "15Min", "1Hour"], fetch=fetch
        )
        self.assertEqual([row["timeframe"] for row in rows], ["5Min", "15Min", "1Hour"])
        self.assertEqual(rows[1]["error"], "simulated data failure")
        self.assertIsNone(rows[1]["total_trades"])
        self.assertEqual(run_mock.call_count, 2)
        self.assertIn("1Hour  OK", output)
        self.assertIn("Failed research windows: 2d/15Min", output)

    def test_csv_has_one_row_per_requested_timeframe(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "results.csv"
            rows, _, _, _ = self.run_with_mocks(
                timeframes=["15Min", "30Min"], output=csv_path
            )
            with csv_path.open(newline="", encoding="utf-8") as csv_file:
                saved_rows = list(csv.DictReader(csv_file))
        self.assertEqual(len(saved_rows), len(rows))
        self.assertEqual([row["timeframe"] for row in saved_rows], ["15Min", "30Min"])
        self.assertIn("gross_return_percent", saved_rows[0])
        self.assertIn("fast_ma", saved_rows[0])


if __name__ == "__main__":
    unittest.main()
