import io
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import backtest
import short_term_research as experiment
import strategy as strategy_module
from backtest import run_backtest
from strategy import Decision
from timeframes import parse_timeframe


def rising_4h_bars(index):
    close = np.linspace(100.0, 400.0, len(index))
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.ones(len(index)),
        },
        index=index,
    )


def flat_30m_bars(index):
    count = len(index)
    return pd.DataFrame(
        {
            "open": np.full(count, 100.0),
            "high": np.full(count, 101.0),
            "low": np.full(count, 99.0),
            "close": np.full(count, 100.0),
            "volume": np.ones(count),
        },
        index=index,
    )


class ShortTermCausalAlignmentTests(unittest.TestCase):
    def test_future_4h_bar_is_not_visible_but_exact_close_boundary_is(self):
        last_start = pd.Timestamp("2026-01-10T08:00:00Z")
        regime_index = pd.date_range(end=last_start, periods=221, freq="4h")
        regime = rising_4h_bars(regime_index)
        execution_index = pd.date_range(
            "2026-01-10T11:00:00Z", periods=2, freq="30min"
        )
        execution = flat_30m_bars(execution_index)

        merged = experiment.merge_completed_4h_regime(execution, regime)

        self.assertEqual(
            merged["execution_decision_time"].tolist(),
            list(execution_index + pd.Timedelta(minutes=30)),
        )
        self.assertEqual(
            merged["regime_source_close_time"].tolist(),
            [pd.Timestamp("2026-01-10T08:00:00Z"), pd.Timestamp("2026-01-10T12:00:00Z")],
        )
        self.assertTrue(bool(merged["regime_bullish_4h"].iloc[0]))
        self.assertTrue(bool(merged["regime_bullish_4h"].iloc[1]))

    def test_missing_4h_candle_makes_state_unavailable_at_240_minutes(self):
        regime_index = pd.date_range(
            end=pd.Timestamp("2026-02-10T16:00:00Z"), periods=225, freq="4h"
        )
        missing_start = pd.Timestamp("2026-02-10T12:00:00Z")
        regime = rising_4h_bars(regime_index).drop(index=missing_start)
        execution_index = pd.DatetimeIndex(
            [
                "2026-02-10T15:00:00Z",  # decision at 15:30; age 210m
                "2026-02-10T15:30:00Z",  # decision at 16:00; age 240m
                "2026-02-10T16:00:00Z",  # still stale
                "2026-02-10T19:30:00Z",  # fresh close at 20:00
            ]
        )
        merged = experiment.merge_completed_4h_regime(
            flat_30m_bars(execution_index), regime
        )

        self.assertEqual(merged["regime_age_minutes"].iloc[0], 210)
        self.assertTrue(bool(merged["regime_bullish_4h"].iloc[0]))
        self.assertEqual(merged["regime_age_minutes"].iloc[1], 240)
        self.assertTrue(pd.isna(merged["regime_bullish_4h"].iloc[1]))
        self.assertTrue(pd.isna(merged["regime_bullish_4h"].iloc[2]))
        self.assertEqual(
            merged["regime_source_close_time"].iloc[3],
            pd.Timestamp("2026-02-10T20:00:00Z"),
        )
        self.assertTrue(bool(merged["regime_bullish_4h"].iloc[3]))

    def test_future_4h_candles_cannot_change_prior_assignments(self):
        regime_index = pd.date_range(
            end=pd.Timestamp("2026-03-10T16:00:00Z"), periods=230, freq="4h"
        )
        regime = rising_4h_bars(regime_index)
        changed = regime.copy()
        first_future_start = regime_index[-4]
        changed.loc[changed.index >= first_future_start, "close"] *= 0.1
        execution_index = pd.date_range(
            first_future_start - pd.Timedelta(hours=8),
            first_future_start + pd.Timedelta(hours=3, minutes=30),
            freq="30min",
        )
        execution = flat_30m_bars(execution_index)

        before = experiment.merge_completed_4h_regime(execution, regime)
        after = experiment.merge_completed_4h_regime(execution, changed)
        cutoff = first_future_start + pd.Timedelta(hours=4)
        prior = before["execution_decision_time"] < cutoff

        pd.testing.assert_series_equal(
            before.loc[prior, "regime_bullish_4h"].reset_index(drop=True),
            after.loc[prior, "regime_bullish_4h"].reset_index(drop=True),
        )
        pd.testing.assert_series_equal(
            before.loc[prior, "regime_source_close_time"].reset_index(drop=True),
            after.loc[prior, "regime_source_close_time"].reset_index(drop=True),
        )

    def test_future_30m_candles_cannot_change_an_earlier_trade(self):
        index = pd.date_range("2026-01-01", periods=40, freq="30min", tz="UTC")
        bars = flat_30m_bars(index)
        bars.iloc[20, bars.columns.get_loc("close")] = 102.0
        bars.iloc[20, bars.columns.get_loc("high")] = 103.0
        for i in range(21, 30):
            bars.iloc[i, bars.columns.get_loc("low")] = 99.5
        bars.iloc[30, bars.columns.get_loc("close")] = 98.0
        bars.iloc[30, bars.columns.get_loc("low")] = 97.0
        changed = bars.copy()
        changed.loc[index[31]:, "open"] = 300.0
        changed.loc[index[31]:, "high"] = 350.0
        changed.loc[index[31]:, "low"] = 250.0
        changed.loc[index[31]:, "close"] = 325.0
        baseline, _gated = experiment.create_short_term_strategies()

        first = run_backtest(
            bars, starting_capital=1000, timeframe="30Min", fee_rate=0,
            slippage=0, strategy=baseline,
        )
        second = run_backtest(
            changed, starting_capital=1000, timeframe="30Min", fee_rate=0,
            slippage=0, strategy=baseline,
        )

        self.assertGreaterEqual(len(first["trades"]), 1)
        pd.testing.assert_series_equal(
            first["trades"].iloc[0], second["trades"].iloc[0]
        )


class ShortTermStrategyTests(unittest.TestCase):
    def setUp(self):
        self.baseline, self.gated = experiment.create_short_term_strategies()

    def test_strategies_are_research_only_and_ab_rules_match(self):
        self.assertNotIn(experiment.BASELINE_NAME, strategy_module.STRATEGY_REGISTRY)
        self.assertNotIn(experiment.GATED_NAME, strategy_module.STRATEGY_REGISTRY)
        experiment.assert_ab_invariant(self.baseline, self.gated)
        self.assertEqual(self.baseline.max_holding_minutes, 1440)
        self.assertEqual(self.gated.max_holding_minutes, 1440)
        self.assertEqual(self.baseline.stop_loss_percent, 0.0075)
        self.assertEqual(self.gated.take_profit_percent, 0.015)

    def test_ab_invariant_rejects_a_changed_execution_parameter(self):
        changed = replace(self.gated, max_holding_minutes=1200)
        with self.assertRaisesRegex(AssertionError, "max_holding_minutes"):
            experiment.assert_ab_invariant(self.baseline, changed)

    def test_cost_headroom_uses_same_path_zero_cost_pnl_and_requested_notional(self):
        trades = pd.DataFrame(
            [
                {
                    "net_pnl": -1.0,
                    "requested_notional": 100.0,
                    "same_path_zero_cost_pnl": 0.0,
                    "fees": 0.4,
                    "slippage_cost": 0.2,
                    "gross_return_percent": 0.0,
                    "return_percent": -1.0,
                    "exit_reason": "stop_loss",
                }
            ]
        )
        result = {
            "trades": trades,
            "start_time": pd.Timestamp("2026-01-01T00:00:00Z"),
            "end_time": pd.Timestamp("2026-01-31T00:00:00Z"),
            "strategy_return": -0.1,
            "same_path_zero_cost_return_percent": 0.0,
            "total_trades": 1,
            "win_rate": 0.0,
            "net_profit_factor": 0.0,
            "gross_profit_factor": None,
            "total_fees": 0.4,
            "total_slippage": 0.2,
            "total_costs": 0.6,
            "pure_cost_drag_percent": 0.1,
            "average_holding_minutes": 30.0,
            "median_holding_minutes": 30.0,
            "shortest_holding_minutes": 30.0,
            "longest_holding_minutes": 30.0,
            "candle_mark_max_drawdown_percent": 0.1,
            "time_invested_percent": 1.0,
            "average_capital_exposure": 2.0,
            "ending_equity": 999.0,
            "open_position": None,
        }

        metrics = experiment._total_metrics(result, 30, 0.0025, 0.0005)

        self.assertEqual(metrics["average_same_path_zero_cost_return_percent_per_trade"], 0.0)
        self.assertAlmostEqual(
            metrics["average_realized_cost_percent_of_requested_notional_per_trade"], 0.6
        )
        self.assertAlmostEqual(metrics["average_cost_headroom_percent_per_trade"], -0.6)
        self.assertEqual(
            metrics["break_even_cost_budget_note"],
            "no non-negative transaction cost can rescue the average trade path",
        )
        self.assertEqual(metrics["total_stop_loss_exit_count"], 1)

    def test_bullish_gate_matches_plain_donchian_for_buy_sell_and_hold(self):
        prepared = pd.DataFrame(
            {
                "close": [12.0, 4.0, 8.0],
                "donchian_entry_high": [10.0, 10.0, 10.0],
                "donchian_exit_low": [5.0, 5.0, 5.0],
                "regime_bullish_4h": pd.array([True, True, True], dtype="boolean"),
            }
        )
        for index in range(len(prepared)):
            self.assertEqual(
                self.gated.decide_at(prepared, index),
                self.baseline.decide_at(prepared, index),
            )
        self.assertEqual(
            [self.gated.decide_at(prepared, i).action for i in range(3)],
            ["BUY", "SELL", "HOLD"],
        )

    def _gated_fixture(self, entry_regime, holding_regime=None):
        index = pd.date_range("2026-01-01", periods=25, freq="30min", tz="UTC")
        bars = flat_30m_bars(index)
        bars["regime_bullish_4h"] = pd.array([True] * len(bars), dtype="boolean")
        bars.iloc[20, bars.columns.get_loc("close")] = 102.0
        bars.iloc[20, bars.columns.get_loc("high")] = 103.0
        bars.iloc[20, bars.columns.get_loc("regime_bullish_4h")] = entry_regime
        bars.iloc[21, bars.columns.get_loc("open")] = 102.0
        bars.iloc[21, bars.columns.get_loc("high")] = 103.0
        bars.iloc[21, bars.columns.get_loc("low")] = 101.5
        bars.iloc[21, bars.columns.get_loc("close")] = 102.0
        if holding_regime is not None:
            bars.iloc[21, bars.columns.get_loc("regime_bullish_4h")] = holding_regime
        bars.iloc[22, bars.columns.get_loc("open")] = 102.0
        bars.iloc[22, bars.columns.get_loc("high")] = 103.0
        bars.iloc[22, bars.columns.get_loc("low")] = 101.5
        bars.iloc[22, bars.columns.get_loc("close")] = 102.0
        return bars

    def test_nonbullish_gate_blocks_flat_entry_and_exits_holding_next_open(self):
        flat = self._gated_fixture(False)
        no_entry = run_backtest(
            flat, starting_capital=1000, timeframe="30Min", fee_rate=0,
            slippage=0, strategy=self.gated,
        )
        self.assertEqual(no_entry["total_trades"], 0)
        self.assertIsNone(no_entry["open_position"])

        holding = self._gated_fixture(True, False)
        exits = run_backtest(
            holding, starting_capital=1000, timeframe="30Min", fee_rate=0,
            slippage=0, strategy=self.gated,
        )
        self.assertEqual(exits["trades"].iloc[0]["exit_reason"], "regime_filter_off")
        self.assertEqual(exits["trades"].iloc[0]["exit_time"], holding.index[22])

    def test_unavailable_regime_blocks_entry_and_exits_holding_next_open(self):
        unavailable = self._gated_fixture(pd.NA)
        no_entry = run_backtest(
            unavailable, starting_capital=1000, timeframe="30Min", fee_rate=0,
            slippage=0, strategy=self.gated,
        )
        self.assertEqual(no_entry["total_trades"], 0)

        holding = self._gated_fixture(True, pd.NA)
        exits = run_backtest(
            holding, starting_capital=1000, timeframe="30Min", fee_rate=0,
            slippage=0, strategy=self.gated,
        )
        self.assertEqual(exits["trades"].iloc[0]["exit_reason"], "regime_unavailable")
        self.assertEqual(exits["trades"].iloc[0]["exit_time"], holding.index[22])

    def test_missing_regime_column_fails_loudly(self):
        bars = flat_30m_bars(
            pd.date_range("2026-01-01", periods=3, freq="30min", tz="UTC")
        )
        with self.assertRaisesRegex(ValueError, "Causal 4Hour regime data must be merged first"):
            self.gated.prepare_indicators(bars)
        with self.assertRaisesRegex(ValueError, "regime_bullish_4h"):
            self.gated.decide_at(bars, 0)

    def test_short_term_research_fetches_each_timeframe_once_and_writes_totals_blocks(self):
        aligned_end = datetime(2026, 10, 5, 0, tzinfo=timezone.utc)

        def fake_fetch(lookback_days, timeframe, *, warmup_bars, end_time):
            duration = pd.Timedelta(minutes=parse_minutes(timeframe))
            requested_start = pd.Timestamp(end_time) - pd.Timedelta(days=lookback_days)
            requested_start -= duration * warmup_bars
            index = pd.date_range(
                requested_start, pd.Timestamp(end_time) - duration, freq=duration
            )
            if timeframe == experiment.EXECUTION_TIMEFRAME:
                bars = flat_30m_bars(index)
            else:
                bars = rising_4h_bars(index)
            bars.attrs["requested_start_time"] = requested_start
            bars.attrs["requested_end_time"] = pd.Timestamp(end_time)
            bars.attrs["test_start"] = pd.Timestamp(end_time) - pd.Timedelta(days=lookback_days)
            return bars

        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "short-term.csv"
            original_run_backtest = backtest.run_backtest
            with (
                patch.object(backtest, "fetch_history", side_effect=fake_fetch) as fetch,
                patch.object(
                    backtest, "run_backtest", wraps=original_run_backtest
                ) as run,
                redirect_stdout(io.StringIO()) as console,
            ):
                rows = experiment.run_short_term_research(
                    block_days=1, blocks=1, output=output,
                    research_end_time=aligned_end,
                )
            saved = pd.read_csv(output)

        self.assertEqual(fetch.call_count, 2)
        self.assertEqual(run.call_count, 2)
        self.assertIs(run.call_args_list[0].args[0], run.call_args_list[1].args[0])
        self.assertTrue(
            all(call.kwargs["liquidate_at_end"] is False for call in run.call_args_list)
        )
        self.assertEqual(
            [call.kwargs["timeframe"] for call in run.call_args_list], ["30Min", "30Min"]
        )
        self.assertEqual(
            [call.args[1] for call in fetch.call_args_list],
            ["30Min", "4Hour"],
        )
        self.assertEqual(
            [call.kwargs["warmup_bars"] for call in fetch.call_args_list],
            [30, 230],
        )
        self.assertEqual(len(rows), 4)  # two TOTAL and two BLOCK rows
        self.assertEqual(len(saved), 4)
        self.assertEqual(set(saved["row_type"]), {"TOTAL", "BLOCK"})
        totals = [row for row in rows if row["row_type"] == "TOTAL"]
        blocks = [row for row in rows if row["row_type"] == "BLOCK"]
        self.assertEqual(
            [row["execution_expected_candle_count"] for row in totals], [48, 48]
        )
        self.assertEqual(
            [row["regime_expected_candle_count"] for row in totals], [6, 6]
        )
        self.assertEqual(
            [row["execution_expected_candle_count"] for row in blocks], [48, 48]
        )
        self.assertTrue(all(row["error"] == "" for row in rows))
        self.assertIn("Common aligned end:", console.getvalue())
        self.assertIn("RegimeDelta summary:", console.getvalue())
        self.assertIn("gated_entries=", console.getvalue())
        self.assertEqual(
            totals[0]["configured_nominal_round_trip_friction_percent"], 0.6
        )
        self.assertIn("same-path zero-cost edge", console.getvalue())

    def test_under_requested_warmup_window_fails_instead_of_silently_starting_cold(self):
        test_start = pd.Timestamp("2026-10-05T00:00:00Z")
        bars = flat_30m_bars(
            pd.date_range(test_start - pd.Timedelta(hours=8), periods=16, freq="30min")
        )
        bars.attrs["requested_start_time"] = test_start - pd.Timedelta(hours=7)
        with self.assertRaisesRegex(ValueError, "Insufficient 30Min execution warm-up requested"):
            experiment._validate_fetch_window(
                bars, "30Min", test_start, 30, "30Min execution"
            )

    def test_real_4h_gap_allows_unavailable_initial_state_and_reports_it(self):
        test_start = pd.Timestamp("2026-10-05T00:00:00Z")
        requested_start = test_start - pd.Timedelta(hours=4 * 230)
        regime_index = pd.date_range(requested_start, test_start, freq="4h")
        missing_start = test_start - pd.Timedelta(hours=4)
        regime = rising_4h_bars(regime_index).drop(index=missing_start)
        execution = flat_30m_bars(pd.DatetimeIndex([test_start]))
        merged = experiment.merge_completed_4h_regime(execution, regime)
        self.assertTrue(pd.isna(merged["regime_bullish_4h"].iloc[0]))
        with redirect_stdout(io.StringIO()) as output:
            experiment._check_first_regime_decision(
                merged, regime, test_start.to_pydatetime(), requested_start
            )
        self.assertIn(missing_start.isoformat(), output.getvalue())

    def test_missing_block_boundary_candles_mark_the_block_invalid(self):
        aligned_end = datetime(2026, 10, 5, 0, tzinfo=timezone.utc)

        def fake_fetch(lookback_days, timeframe, *, warmup_bars, end_time):
            duration = pd.Timedelta(minutes=parse_minutes(timeframe))
            test_start = pd.Timestamp(end_time) - pd.Timedelta(days=lookback_days)
            requested_start = test_start - duration * warmup_bars
            index = pd.date_range(
                requested_start, pd.Timestamp(end_time) - duration, freq=duration
            )
            if timeframe == "30Min":
                bars = flat_30m_bars(index)
                bars = bars.iloc[:-1]  # Missing final bar: cannot mark end boundary.
            else:
                bars = rising_4h_bars(index)
            bars.attrs["requested_start_time"] = requested_start
            bars.attrs["requested_end_time"] = pd.Timestamp(end_time)
            return bars

        with (
            patch.object(backtest, "fetch_history", side_effect=fake_fetch),
            redirect_stdout(io.StringIO()),
        ):
            rows = experiment.run_short_term_research(
                block_days=1, blocks=1, output=None, research_end_time=aligned_end
            )

        block_rows = [row for row in rows if row["row_type"] == "BLOCK"]
        self.assertEqual(len(block_rows), 2)
        self.assertTrue(
            all("does not cover exact block boundaries" in row["error"] for row in block_rows)
        )
        self.assertTrue(all(row["block_net_return_percent"] is None for row in block_rows))

    def test_cli_exposes_fixed_research_options_and_accepts_utc_end_time(self):
        with patch.object(experiment, "run_short_term_research") as run:
            self.assertEqual(
                experiment.main(
                    ["--block-days", "1", "--blocks", "2", "--end-time",
                     "2022-10-26T04:00:00+00:00", "--no-csv"]
                ),
                0,
            )
        self.assertEqual(run.call_args.kwargs["block_days"], 1)
        self.assertEqual(run.call_args.kwargs["blocks"], 2)
        self.assertEqual(
            run.call_args.kwargs["research_end_time"],
            datetime(2022, 10, 26, 4, tzinfo=timezone.utc),
        )
        self.assertNotIn("--entry-lookback", experiment._build_parser().format_help())


def parse_minutes(timeframe):
    return int(parse_timeframe(timeframe)[1])


if __name__ == "__main__":
    unittest.main()
