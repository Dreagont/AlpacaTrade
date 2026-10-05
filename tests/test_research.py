import csv
import io
import json
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
from backtest import required_warmup_bars
from strategy import DONCHIAN_BREAKOUT, MA_RSI_CROSSOVER, get_strategy


def history_frame(end_time, lookback_days=2, *, warmup_bars=None):
    test_start = end_time - timedelta(days=lookback_days)
    warmup_bars = (
        required_warmup_bars(strategy=MA_RSI_CROSSOVER) + 1
        if warmup_bars is None
        else warmup_bars
    )
    index = pd.date_range(
        end=test_start - timedelta(minutes=5),
        periods=warmup_bars,
        freq="5min",
    ).append(pd.date_range(test_start, periods=577, freq="5min"))
    bars = pd.DataFrame(
        {name: [100.0] * len(index) for name in ("open", "high", "low", "close", "volume")},
        index=index,
    )
    bars.attrs["test_start"] = test_start
    return bars


def backtest_result(start, end, *, strategy_return=4.6):
    return {
        "strategy_name": "test_strategy",
        "strategy_parameters": {},
        "start_time": start,
        "end_time": end,
        "warmup_bars": required_warmup_bars(),
        "required_warmup_bars": required_warmup_bars(),
        "available_pretest_bars": required_warmup_bars() + 3,
        "total_trades": 4,
        "winning_trades": 3,
        "losing_trades": 1,
        "win_rate": 75.0,
        "gross_pnl": 50.0,
        "total_fees": 3.0,
        "total_slippage": 1.0,
        "total_costs": 4.0,
        "net_profit": strategy_return,
        "strategy_return": strategy_return,
        "same_path_zero_cost_pnl_total": strategy_return + 2.5,
        "same_path_zero_cost_return_percent": strategy_return + 2.5,
        "pure_cost_drag_percent": 2.5,
        "average_pnl_per_trade": 11.5,
        "average_gross_return": 2.0,
        "average_net_return": 1.8,
        "profit_factor": 2.0,
        "net_profit_factor": 2.0,
        "gross_profit_factor": 3.0,
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
        "raw_market_return_percent": 12.0,
    }


class ResearchTests(unittest.TestCase):
    def setUp(self):
        self.end = datetime(2026, 10, 5, 12, tzinfo=timezone.utc)
        self.test_start = self.end - timedelta(days=2)

    def test_alignment_uses_largest_frame_and_keeps_common_boundaries(self):
        hourly_now = datetime(2026, 10, 5, 1, 44, tzinfo=timezone.utc)
        self.assertEqual(
            research.align_research_end_time(hourly_now, ["1Hour", "2Hour", "4Hour"]),
            datetime(2026, 10, 5, 0, tzinfo=timezone.utc),
        )
        minute_now = datetime(2026, 10, 5, 10, 37, tzinfo=timezone.utc)
        self.assertEqual(
            research.align_research_end_time(minute_now, ["5Min", "15Min", "30Min"]),
            datetime(2026, 10, 5, 10, 30, tzinfo=timezone.utc),
        )
        boundary = datetime(2026, 10, 5, 0, tzinfo=timezone.utc)
        self.assertIs(
            research.align_research_end_time(boundary, ["1Hour", "2Hour", "4Hour"]),
            boundary,
        )

    def test_calendar_month_alignment_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, "Month timeframes is unsupported"):
            research.align_research_end_time(self.end, ["1Month"])

    def test_1_2_and_4_hour_parser_support_remains(self):
        self.assertEqual(research.parse_timeframe("1Hour")[1], 60)
        self.assertEqual(research.parse_timeframe("2Hour")[1], 120)
        self.assertEqual(research.parse_timeframe("4Hour")[1], 240)

    def test_default_run_preserves_timeframe_order_and_uses_ma_strategy(self):
        bars = history_frame(self.end)
        def run_same_window(_bars, **kwargs):
            return backtest_result(kwargs["test_start"], self.end)

        with (
            patch("research.backtest.fetch_history", return_value=bars) as fetch_mock,
            patch("research.backtest.run_backtest", side_effect=run_same_window) as run_mock,
            redirect_stdout(io.StringIO()) as console,
        ):
            rows = research.run_research(
                lookback_days=2, output=None, research_end_time=self.end
            )

        self.assertEqual(
            [(row["lookback_days"], row["timeframe"], row["strategy_name"]) for row in rows],
            [(2, timeframe, "ma_rsi_crossover") for timeframe in research.DEFAULT_TIMEFRAMES],
        )
        self.assertEqual(fetch_mock.call_count, len(research.DEFAULT_TIMEFRAMES))
        self.assertEqual(run_mock.call_count, 2 * len(research.DEFAULT_TIMEFRAMES))
        self.assertNotRegex(console.getvalue().lower(), r"\b(best|winner|optimal)\b")

    def test_matrix_order_common_end_warmup_reuse_cost_runs_and_csv(self):
        requested = datetime(2026, 10, 5, 2, 30, tzinfo=timezone.utc)
        aligned = datetime(2026, 10, 5, 0, tzinfo=timezone.utc)
        lookbacks = [90, 180, 365, 730]
        timeframes = ["1Hour", "2Hour", "4Hour"]
        strategies = ["ma_rsi_crossover", "donchian_breakout"]
        expected_pairs = [
            (lookback, timeframe, strategy_name)
            for lookback in lookbacks
            for timeframe in timeframes
            for strategy_name in strategies
        ]
        max_warmup = max(
            required_warmup_bars(strategy=get_strategy(name)) for name in strategies
        )
        bars = history_frame(aligned, max(lookbacks), warmup_bars=max_warmup + 2)

        def run_cost_case(_bars, **kwargs):
            return backtest_result(
                kwargs["test_start"],
                aligned,
                strategy_return=7.0 if kwargs["fee_rate"] == 0 else 4.5,
            )

        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "matrix.csv"
            with (
                patch("research.backtest.fetch_history", return_value=bars) as fetch_mock,
                patch("research.backtest.run_backtest", side_effect=run_cost_case) as run_mock,
                redirect_stdout(io.StringIO()) as console,
            ):
                rows = research.run_research(
                    lookbacks=lookbacks,
                    timeframes=timeframes,
                    strategies=strategies,
                    output=csv_path,
                    research_end_time=requested,
                )
            with csv_path.open(newline="", encoding="utf-8") as csv_file:
                saved_rows = list(csv.DictReader(csv_file))

        actual_pairs = [
            (row["lookback_days"], row["timeframe"], row["strategy_name"])
            for row in rows
        ]
        self.assertEqual(actual_pairs, expected_pairs)
        self.assertEqual(fetch_mock.call_count, len(timeframes))
        for call, timeframe in zip(fetch_mock.call_args_list, timeframes):
            self.assertEqual(call.args, (max(lookbacks), timeframe))
            self.assertEqual(call.kwargs["warmup_bars"], max_warmup)
            self.assertEqual(call.kwargs["end_time"], aligned)

        self.assertEqual(run_mock.call_count, len(expected_pairs) * 2)
        for pair_index, (lookback, timeframe, strategy_name) in enumerate(expected_pairs):
            real_call, zero_call = run_mock.call_args_list[pair_index * 2 : pair_index * 2 + 2]
            expected_start = aligned - timedelta(days=lookback)
            strategy = get_strategy(strategy_name)
            for call in (real_call, zero_call):
                self.assertIs(call.args[0], bars)
                self.assertEqual(call.kwargs["test_start"], expected_start)
                self.assertIs(call.kwargs["strategy"], strategy)
                self.assertEqual(call.kwargs["timeframe"], timeframe)
            self.assertEqual(real_call.kwargs["fee_rate"], config.BACKTEST_FEE_PERCENT)
            self.assertEqual(real_call.kwargs["slippage"], config.BACKTEST_SLIPPAGE_PERCENT)
            self.assertEqual(zero_call.kwargs["fee_rate"], 0)
            self.assertEqual(zero_call.kwargs["slippage"], 0)

        self.assertTrue(all(row["research_end_time"] == aligned.isoformat() for row in rows))
        self.assertTrue(all(row["requested_research_time"] == requested.isoformat() for row in rows))
        self.assertTrue(all(row["requested_start_time"] for row in rows))
        self.assertTrue(all(row["actual_start_time"] for row in rows))
        self.assertTrue(all(row["actual_end_time"] == aligned.isoformat() for row in rows))
        self.assertTrue(all(row["coverage_days"] > 0 for row in rows))
        self.assertTrue(all(row["same_path_zero_cost_return_percent"] == 7.0 for row in rows))
        self.assertTrue(all(row["realistic_net_return_percent"] == 4.5 for row in rows))
        self.assertTrue(all(row["pure_cost_drag_percent"] == 2.5 for row in rows))
        self.assertTrue(all(row["free_run_zero_cost_return_percent"] == 7.0 for row in rows))
        self.assertTrue(all(row["raw_market_return_percent"] == 12.0 for row in rows))
        self.assertTrue(all(row["fee_rate"] == config.BACKTEST_FEE_PERCENT for row in rows))
        self.assertTrue(all(row["slippage_rate"] == config.BACKTEST_SLIPPAGE_PERCENT for row in rows))
        self.assertEqual(len(saved_rows), len(expected_pairs))
        self.assertEqual(
            [(int(row["lookback_days"]), row["timeframe"], row["strategy_name"]) for row in saved_rows],
            expected_pairs,
        )
        self.assertEqual(
            json.loads(saved_rows[1]["strategy_parameters_json"]),
            DONCHIAN_BREAKOUT.parameters,
        )
        self.assertEqual(json.loads(saved_rows[0]["strategy_parameters_json"])["fast_ma"], trade_config.FAST_MA)
        self.assertNotIn("fee_percent", saved_rows[0])
        output = console.getvalue()
        self.assertIn("NetPF", output)
        self.assertIn("BTC%", output)
        self.assertIn("Same B&H%", output)
        self.assertIn("Full B&H%", output)
        self.assertIn("nested and overlapping", output)
        self.assertNotRegex(output.lower(), r"\b(best|winner|optimal|recommended strategy)\b")

    def test_data_quality_is_measured_inside_each_requested_lookback(self):
        aligned = datetime(2026, 10, 5, 0, tzinfo=timezone.utc)
        oldest = aligned - timedelta(days=730)
        warmup = required_warmup_bars(strategy=MA_RSI_CROSSOVER)
        index = pd.date_range(
            start=oldest - timedelta(hours=warmup),
            end=aligned - timedelta(hours=1),
            freq="1h",
        )
        bars = pd.DataFrame(
            {name: [100.0] * len(index) for name in ("open", "high", "low", "close", "volume")},
            index=index,
        )

        def run_success(_bars, **kwargs):
            return backtest_result(kwargs["test_start"], aligned)

        with (
            patch("research.backtest.fetch_history", return_value=bars) as fetch,
            patch("research.backtest.run_backtest", side_effect=run_success),
            redirect_stdout(io.StringIO()),
        ):
            rows = research.run_research(
                lookbacks=[90, 730], timeframes=["1Hour"], output=None,
                research_end_time=aligned,
            )
        self.assertEqual(fetch.call_count, 1)
        by_lookback = {row["lookback_days"]: row for row in rows}
        self.assertEqual(by_lookback[90]["expected_candle_count"], 90 * 24)
        self.assertEqual(by_lookback[730]["expected_candle_count"], 730 * 24)
        self.assertEqual(by_lookback[90]["actual_candle_count"], 90 * 24)
        self.assertEqual(by_lookback[730]["actual_candle_count"], 730 * 24)

    def test_warmup_shortfall_isolated_to_window_and_uses_clear_metadata(self):
        aligned = datetime(2026, 10, 5, 0, tzinfo=timezone.utc)
        long_start = aligned - timedelta(days=730)
        short_start = aligned - timedelta(days=90)
        required = required_warmup_bars(strategy=MA_RSI_CROSSOVER)
        early = pd.date_range(end=long_start - timedelta(hours=1), periods=required - 1, freq="1h")
        middle = pd.date_range(start=long_start, end=short_start - timedelta(days=1), freq="1D")
        late = pd.date_range(start=short_start, periods=5, freq="1h")
        index = early.append(middle).append(late)
        bars = pd.DataFrame(
            {name: [100.0] * len(index) for name in ("open", "high", "low", "close", "volume")},
            index=index,
        )
        bars.attrs["test_start"] = long_start

        def run_success(_bars, **kwargs):
            return backtest_result(kwargs["test_start"], aligned)

        with (
            patch("research.backtest.fetch_history", return_value=bars) as fetch_mock,
            patch("research.backtest.run_backtest", side_effect=run_success) as run_mock,
            redirect_stdout(io.StringIO()),
        ):
            rows = research.run_research(
                lookbacks=[90, 730],
                timeframes=["1Hour"],
                output=None,
                research_end_time=aligned,
            )

        self.assertEqual(fetch_mock.call_count, 1)
        self.assertEqual(fetch_mock.call_args.args, (730, "1Hour"))
        self.assertEqual(rows[0]["error"], "")
        self.assertIn("Insufficient warm-up history", rows[1]["error"])
        self.assertEqual(run_mock.call_count, 2)  # both cost modes for 90d only

    def test_one_timeframe_fetch_failure_does_not_stop_other_rows(self):
        aligned = self.end
        lookbacks = [90, 180]
        timeframes = ["1Hour", "2Hour", "4Hour"]
        strategies = ["ma_rsi_crossover", "donchian_breakout"]
        bars = history_frame(aligned, max(lookbacks))

        def fetch_once(days, timeframe, **kwargs):
            if timeframe == "2Hour":
                raise RuntimeError("simulated 2Hour fetch failure")
            return bars

        def run_success(_bars, **kwargs):
            return backtest_result(kwargs["test_start"], aligned)

        with (
            patch("research.backtest.fetch_history", side_effect=fetch_once) as fetch_mock,
            patch("research.backtest.run_backtest", side_effect=run_success) as run_mock,
            redirect_stdout(io.StringIO()),
        ):
            rows = research.run_research(
                lookbacks=lookbacks,
                timeframes=timeframes,
                strategies=strategies,
                output=None,
                research_end_time=aligned,
            )

        self.assertEqual(fetch_mock.call_count, len(timeframes))
        self.assertEqual(
            sum(bool(row["error"]) for row in rows),
            len(lookbacks) * len(strategies),
        )
        self.assertTrue(
            all("simulated 2Hour fetch failure" in row["error"] for row in rows if row["timeframe"] == "2Hour")
        )
        self.assertTrue(
            all(not row["error"] for row in rows if row["timeframe"] != "2Hour")
        )
        self.assertEqual(run_mock.call_count, len(lookbacks) * 2 * len(strategies) * 2)

    def test_one_strategy_window_failure_does_not_stop_remaining_matrix(self):
        aligned = self.end
        lookbacks = [90, 180]
        timeframes = ["1Hour", "2Hour"]
        strategies = ["ma_rsi_crossover", "donchian_breakout"]
        bars = history_frame(aligned, max(lookbacks))
        failing_start = aligned - timedelta(days=180)

        def run_one(_bars, **kwargs):
            if (
                kwargs["strategy"].name == "donchian_breakout"
                and kwargs["timeframe"] == "2Hour"
                and kwargs["test_start"] == failing_start
                and kwargs["fee_rate"] != 0
            ):
                raise RuntimeError("simulated Donchian row failure")
            return backtest_result(kwargs["test_start"], aligned)

        with (
            patch("research.backtest.fetch_history", return_value=bars),
            patch("research.backtest.run_backtest", side_effect=run_one),
            redirect_stdout(io.StringIO()),
        ):
            rows = research.run_research(
                lookbacks=lookbacks,
                timeframes=timeframes,
                strategies=strategies,
                output=None,
                research_end_time=aligned,
            )

        failed = [row for row in rows if row["error"]]
        self.assertEqual(len(rows), len(lookbacks) * len(timeframes) * len(strategies))
        self.assertEqual(len(failed), 1)
        self.assertEqual(
            (failed[0]["lookback_days"], failed[0]["timeframe"], failed[0]["strategy_name"]),
            (180, "2Hour", "donchian_breakout"),
        )

    def test_coverage_mismatch_is_an_error_and_reports_expected_vs_actual(self):
        aligned = self.end
        bars = history_frame(aligned)
        requested_start = aligned - timedelta(days=2)

        def short_history(_bars, **kwargs):
            return backtest_result(requested_start + timedelta(days=3), aligned)

        with (
            patch("research.backtest.fetch_history", return_value=bars),
            patch("research.backtest.run_backtest", side_effect=short_history),
            redirect_stdout(io.StringIO()),
        ):
            rows = research.run_research(
                lookbacks=[2],
                timeframes=["1Hour"],
                output=None,
                research_end_time=aligned,
            )

        self.assertIn("Historical start coverage mismatch", rows[0]["error"])
        self.assertIn(requested_start.isoformat(), rows[0]["error"])
        self.assertIn((requested_start + timedelta(days=3)).isoformat(), rows[0]["error"])
        self.assertIsNotNone(rows[0]["actual_start_time"])
        self.assertEqual(rows[0]["actual_end_time"], aligned.isoformat())

    def test_realistic_and_zero_cost_rows_calculate_drag_and_keep_net_pf(self):
        aligned = self.end
        bars = history_frame(aligned)
        calls = []

        def run_pair(_bars, **kwargs):
            calls.append(kwargs)
            result = backtest_result(
                kwargs["test_start"],
                aligned,
                strategy_return=8.0 if kwargs["fee_rate"] == 0 else 5.0,
            )
            result["net_profit_factor"] = 1.7
            result["gross_profit_factor"] = 2.4
            result["profit_factor"] = 1.7
            return result

        with (
            patch("research.backtest.fetch_history", return_value=bars),
            patch("research.backtest.run_backtest", side_effect=run_pair),
            redirect_stdout(io.StringIO()),
        ):
            rows = research.run_research(
                lookbacks=[2],
                timeframes=["1Hour"],
                output=None,
                research_end_time=aligned,
            )

        row = rows[0]
        self.assertEqual(len(calls), 2)
        self.assertIs(calls[0]["strategy"], calls[1]["strategy"])
        self.assertEqual(calls[0]["test_start"], calls[1]["test_start"])
        self.assertEqual(row["same_path_zero_cost_return_percent"], 7.5)
        self.assertEqual(row["free_run_zero_cost_return_percent"], 8.0)
        self.assertEqual(row["realistic_net_return_percent"], 5.0)
        self.assertGreaterEqual(
            row["same_path_zero_cost_return_percent"], row["realistic_net_return_percent"]
        )
        self.assertEqual(row["pure_cost_drag_percent"], 2.5)
        self.assertEqual(row["net_profit_factor"], 1.7)
        self.assertEqual(row["profit_factor"], 1.7)

    def test_csv_context_uses_rate_names_and_strategy_json(self):
        bars = history_frame(self.end)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            with (
                patch(
                    "research.backtest.fetch_history",
                    return_value=bars,
                ),
                patch(
                    "research.backtest.run_backtest",
                    side_effect=lambda _bars, **kwargs: backtest_result(
                        kwargs["test_start"], self.end
                    ),
                ),
                redirect_stdout(io.StringIO()),
            ):
                rows = research.run_research(
                    lookbacks=[2],
                    timeframes=["1Hour"],
                    strategies=["ma_rsi_crossover", "donchian_breakout"],
                    output=path,
                    research_end_time=self.end,
                )
            with path.open(newline="", encoding="utf-8") as csv_file:
                saved_rows = list(csv.DictReader(csv_file))

        self.assertEqual(len(rows), 2)
        self.assertEqual(len(saved_rows), 2)
        self.assertIn("fee_rate", saved_rows[0])
        self.assertIn("slippage_rate", saved_rows[0])
        self.assertIn("strategy_parameters_json", saved_rows[0])
        self.assertEqual(json.loads(saved_rows[1]["strategy_parameters_json"]), DONCHIAN_BREAKOUT.parameters)
        self.assertTrue(all(row["actual_end_time"] == self.end.isoformat() for row in saved_rows))

    def test_cli_defaults_strategy_and_accepts_strategy_list(self):
        with patch("research.run_research") as run_mock:
            research.main(["--no-csv"])
            self.assertEqual(run_mock.call_args.kwargs["lookbacks"], [90, 180, 365, 730])
            self.assertEqual(run_mock.call_args.kwargs["strategies"], ["ma_rsi_crossover"])

            research.main(
                [
                    "--no-csv",
                    "--lookbacks",
                    "90",
                    "180",
                    "--timeframes",
                    "4Hour",
                    "--strategies",
                    "ma_rsi_crossover",
                    "donchian_breakout",
                ]
            )
            self.assertEqual(
                run_mock.call_args.kwargs["strategies"],
                ["ma_rsi_crossover", "donchian_breakout"],
            )
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                research.main(["--lookbacks", "90", "--lookback-days", "365"])
            self.assertEqual(raised.exception.code, 2)

    def test_ma_parameters_fee_assumptions_and_strategy_name_are_recorded(self):
        bars = history_frame(self.end)
        with (
            patch("research.backtest.fetch_history", return_value=bars),
            patch(
                "research.backtest.run_backtest",
                side_effect=lambda _bars, **kwargs: backtest_result(
                    kwargs["test_start"], self.end
                ),
            ),
            redirect_stdout(io.StringIO()),
        ):
            rows = research.run_research(
                lookbacks=[2],
                timeframes=["1Hour"],
                output=None,
                research_end_time=self.end,
            )

        row = rows[0]
        parameters = json.loads(row["strategy_parameters_json"])
        self.assertEqual(row["strategy_name"], "ma_rsi_crossover")
        self.assertEqual(parameters["fast_ma"], trade_config.FAST_MA)
        self.assertEqual(parameters["slow_ma"], trade_config.SLOW_MA)
        self.assertEqual(parameters["rsi_period"], trade_config.RSI_PERIOD)
        self.assertEqual(parameters["rsi_buy_threshold"], trade_config.RSI_BUY_THRESHOLD)
        self.assertEqual(row["fee_rate"], config.BACKTEST_FEE_PERCENT)
        self.assertEqual(row["slippage_rate"], config.BACKTEST_SLIPPAGE_PERCENT)
        self.assertEqual(row["same_notional_buy_hold_return"], 1.0)
        self.assertEqual(row["full_capital_buy_hold_return"], 8.0)


if __name__ == "__main__":
    unittest.main()
