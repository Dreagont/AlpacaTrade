import io
import json
import sqlite3
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import binance_data as data
import search_equivalence as gate
import strategy_search as search
from fast_search_engine import Simulation, simulate
from search_space import (FAMILIES, TIMEFRAMES, SearchConfig, apply_regime, enumerate_configs,
                          family_signals, merge_regime, reference_strategy, wilder_rsi)
from walk_forward import (Fold, build_folds, compound, eligible, holdout_bounds, pass_rule,
                          random_baseline, select_top, selection_score, window_metrics)
from strategy import Decision, StrategySpec
import backtest


def metric(profit=10, dd=5, trades=60, days=365, invested=10, average=0.2):
    return {"return_percent": profit, "max_drawdown_percent": dd, "trades": trades,
            "trades_per_week": trades / days * 7, "trades_per_day": trades / days,
            "time_invested_percent": invested, "trade_return_sum": trades * average,
            "average_trade_return_percent": average}


def flat_bars(count=9):
    return pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1.0},
                        index=pd.date_range("2020-01-01", periods=count, freq="h", tz="UTC"))


class SearchSpaceAndEngineTests(unittest.TestCase):
    def test_deterministic_exact_space_all_overlays(self):
        configs = enumerate_configs()
        self.assertEqual(len(configs), 936)
        self.assertEqual(len({c.config_id for c in configs}), 936)
        self.assertEqual(configs, enumerate_configs())
        self.assertEqual({c.family for c in configs}, set(FAMILIES))
        self.assertEqual({c.timeframe for c in configs}, set(TIMEFRAMES))
        self.assertEqual({c.stop_loss for c in configs}, {None, .03, .06})
        self.assertEqual({c.max_hold_hours for c in configs}, {None, 120})
        for timeframe in TIMEFRAMES:
            self.assertEqual(sum(c.timeframe == timeframe for c in configs), 312)

    def test_signal_formula_trend_cross_donchian_and_rsi(self):
        bars = flat_bars(260)
        bars["close"] = np.arange(260, dtype=float) + 100
        bars["high"], bars["low"] = bars["close"] + 1, bars["close"] - 1
        for family, first, second in (("trend_sma", 50, 10), ("ma_cross", 10, 50), ("donchian", 20, 10)):
            config = SearchConfig("test", "1Hour", family, first, second, False, None, None)
            actions = family_signals(bars, config)
            if family == "donchian":
                bars.loc[bars.index[-1], "close"] += 20
                actions = family_signals(bars, config)
            self.assertEqual(actions[-1], 1)
        rsi = wilder_rsi(pd.Series([100, 101, 102, 101, 101, 102], dtype=float), 2)
        self.assertAlmostEqual(rsi.iloc[2], 100)
        self.assertAlmostEqual(rsi.iloc[3], 50)
        self.assertTrue(wilder_rsi(pd.Series([100.] * 10), 2).iloc[2:].eq(50).all())
        rsi_config = SearchConfig("rsi", "1Hour", "rsi_pullback", 2, 30, False, None, None)
        actions = family_signals(bars, rsi_config)
        self.assertEqual(actions[-1], -1)

    def test_prior_channel_excludes_current_high_and_low(self):
        bars = flat_bars(30)
        bars.loc[bars.index[-1], ["close", "high"]] = [102, 1000]
        c = SearchConfig("channel", "1Hour", "donchian", 20, 10, False, None, None)
        self.assertEqual(family_signals(bars, c)[-1], 1)
        bars.loc[bars.index[-1], ["close", "low"]] = [98, 1]
        self.assertEqual(family_signals(bars, c)[-1], -1)

    def test_regime_merge_matches_frozen_short_term_mapping(self):
        from short_term_research import merge_completed_4h_regime
        hourly = gate.synthetic_history(periods=4000)
        regime = data.resample_bars(hourly, "4Hour")
        execution = hourly.resample("30min").ffill()  # Test scaffold only, not production OHLC.
        expected = merge_completed_4h_regime(execution, regime)["regime_bullish_4h"]
        states = merge_regime(hourly, regime, "1Hour")
        # 1H decision close = 30Min bar at hourly start+30Min decision close.
        corresponding = expected.reindex(hourly.index + pd.Timedelta(minutes=30))
        pd.testing.assert_series_equal(states.iloc[:-1].reset_index(drop=True), corresponding.iloc[:-1].reset_index(drop=True), check_names=False)
        missing = regime.drop(regime.index[-2:])
        mapped = merge_regime(hourly, missing, "1Hour")
        self.assertTrue(mapped.iloc[-4:].isna().all())

    def test_gate_runs_every_family_overlay_and_timeframe_offline(self):
        # Cache surrogate is independent seeded OHLC; production also gates real klines.
        sample = gate.synthetic_history(seed=114, periods=4096)
        messages = []
        statuses = gate.equivalence_gate(sample, progress=messages.append)
        for family in FAMILIES:
            self.assertEqual(statuses[family]["synthetic"], 36)
            self.assertEqual(statuses[family]["cached_real"], 36)
            self.assertGreater(statuses[family]["trades_checked"], 0)
        self.assertEqual(len(messages), 8)

    def test_gate_failure_stops_and_missing_cache_never_searches(self):
        sample = gate.synthetic_history(periods=4096)
        with patch.object(gate, "compare_one", side_effect=AssertionError("mismatch")), self.assertRaisesRegex(RuntimeError, "search stopped"):
            gate.equivalence_gate(sample, progress=lambda _: None)
        with self.assertRaisesRegex(ValueError, "cached real"):
            gate.equivalence_gate(None)

    def test_candle_mark_compounding_fully_reinvests(self):
        bars = flat_bars(7)
        bars["open"] = [100, 100, 110, 110, 100, 100, 120]
        bars["close"] = bars["open"]
        bars["high"], bars["low"] = bars["open"] + 1, bars["open"] - 1
        actions = np.array([1, -1, 0, 1, -1, 0, 0])
        result = simulate(bars, actions, np.full(7, 4), timeframe="1Hour", fee_rate=0, slippage_rate=0)
        self.assertEqual(len(result.trades), 2)
        self.assertAlmostEqual(result.trades.iloc[0]["return_percent"], 10)
        self.assertAlmostEqual(result.equity.iloc[-1], 1.1)
        self.assertAlmostEqual(result.equity.iloc[2], 1.0)
        self.assertAlmostEqual(result.equity.iloc[3], 1.1)

    def test_gap_exit_priority_and_intrabar_stop_before_target(self):
        for label, exit_open, low, high, expected in (
            ("gap_stop", 90, 85, 120, "stop_loss"),
            ("gap_take", 115, 90, 120, "take_profit"),
            ("max_hold", 100, 99, 101, "max_holding_time"),
            ("intrabar_both", 100, 80, 120, "stop_loss"),
        ):
            with self.subTest(label=label):
                bars = flat_bars(5)
                bars.loc[bars.index[2], ["open", "low", "high"]] = [exit_open, low, high]
                actions = np.array([1, 0, 0, 0, 0])
                hold = 1 if label != "intrabar_both" else None
                fast = simulate(bars, actions, np.full(5, 4), timeframe="1Hour", fee_rate=.00075, slippage_rate=.0002,
                                stop_loss=.03, take_profit=.1, max_hold_hours=hold)
                spec = StrategySpec("overlay", lambda bars: bars, lambda _bars, i: Decision("BUY" if actions[i] == 1 else "HOLD", "signal_sell"),
                                    lambda: 0, lambda: {}, lambda: .03, lambda: .1, hold * 60 if hold else None)
                reference = backtest.run_backtest(bars, timeframe="1Hour", starting_capital=1e9, fee_rate=.00075, slippage=.0002,
                                                  strategy=spec, liquidate_at_end=False)
                self.assertEqual(fast.trades.iloc[0]["exit_reason"], expected)
                self.assertEqual(fast.trades.iloc[0]["exit_reason"], reference["trades"].iloc[0]["exit_reason"])
                self.assertAlmostEqual(fast.trades.iloc[0]["return_percent"], reference["trades"].iloc[0]["return_percent"], places=9)

    def test_open_position_and_hypothetical_end_exit_parity(self):
        bars = flat_bars()
        actions, reasons = np.ones(9, dtype=int), np.full(9, 4)
        c = SearchConfig("flat", "1Hour", "trend_sma", 50, 0, False, None, None)
        for liquidate in (False, True):
            gate.compare_one(bars, actions, reasons, c, test_start=bars.index[0], liquidate=liquidate)
        fast = simulate(bars, actions, reasons, timeframe="1Hour", fee_rate=.001, slippage_rate=.002)
        self.assertTrue(fast.trades.empty)
        self.assertAlmostEqual(fast.equity.iloc[-1], (1 - .001) / (1 + .002))

    def test_gate_sell_overrides_family_buy_and_stale_is_distinct(self):
        actions, reasons = apply_regime(np.ones(3), pd.Series(pd.array([True, False, pd.NA], dtype="boolean")), True)
        self.assertEqual(actions.tolist(), [1, -1, -1])
        self.assertEqual(reasons.tolist(), [4, 5, 6])


class WalkForwardTests(unittest.TestCase):
    def test_calendar_folds_count_adjacency_train_before_test_holdout_excluded(self):
        start, end = holdout_bounds("2026-10-06T03:13:00Z")
        self.assertEqual(start, pd.Timestamp("2026-04-06T00:00:00Z"))
        folds = build_folds(start)
        self.assertEqual(len(folds), 25)
        for i, fold in enumerate(folds):
            self.assertEqual(fold.train_end, fold.test_start)
            self.assertEqual(fold.train_start + pd.DateOffset(months=12), fold.train_end)
            self.assertEqual(fold.test_start + pd.DateOffset(months=3), fold.test_end)
            self.assertLessEqual(fold.test_end, start)
            if i:
                self.assertEqual(folds[i - 1].test_end, fold.test_start)
        self.assertEqual(folds[0].test_start, pd.Timestamp("2020-01-01", tz="UTC"))
        self.assertEqual(folds[-1].test_end, pd.Timestamp("2026-04-01", tz="UTC"))

    def test_eligibility_all_thresholds_inclusive(self):
        self.assertTrue(eligible(metric()))
        for values in ({"trades": 19}, {"trades_per_week": .999}, {"trades_per_day": 5.001}, {"time_invested_percent": 4.999}):
            metrics = metric()
            metrics.update(values)
            self.assertFalse(eligible(metrics))
        metrics = metric()
        metrics.update(trades=20, trades_per_week=1, trades_per_day=5, time_invested_percent=5)
        self.assertTrue(eligible(metrics))

    def test_score_return_over_drawdown_ties_more_trades_then_id(self):
        self.assertEqual(selection_score(metric(10, 5)), 2)
        self.assertEqual(selection_score(metric(-10, 5)), -2)
        self.assertEqual(selection_score(metric(10, 0)), np.inf)
        self.assertEqual(selection_score(metric(0, 0)), 0)
        metrics = {"b": metric(trades=61), "a": metric(trades=60), "c": metric(trades=61), "excluded": metric(trades=19)}
        selected, names = select_top(metrics)
        self.assertEqual(selected, ["b", "c", "a"])
        self.assertNotIn("excluded", names)

    def test_hand_built_window_curve_and_completed_trade_accounting(self):
        start = pd.Timestamp("2020-01-01", tz="UTC")
        times = pd.date_range(start, periods=5, freq="6h")
        result = Simulation(pd.Series([1., 1.1, 1., 1.2, 1.3], index=times), pd.Series([False, True, True, False], index=times[:-1]),
                            pd.DataFrame([{"entry_time": start, "exit_time": times[2], "return_percent": 10},
                                          {"entry_time": start, "exit_time": times[-1], "return_percent": 20}]))
        metrics = window_metrics(result, start, times[-1])
        self.assertAlmostEqual(metrics["return_percent"], 30)
        self.assertAlmostEqual(metrics["max_drawdown_percent"], (1 - 1 / 1.1) * 100)
        self.assertEqual(metrics["trades"], 1)  # [start,end): next-window exit excluded.
        self.assertEqual(metrics["time_invested_percent"], 50)
        self.assertEqual(metrics["trade_return_sum"], 10)

    def test_random_reproducible_uniform_eligible_only_five_distinct(self):
        names = [f"c{i}" for i in range(7)]
        windows = [{name: i * 2 for i, name in enumerate(names)}] * 2
        one, choices = random_baseline([names, names], windows)
        two, choices2 = random_baseline([list(reversed(names)), names], windows)
        np.testing.assert_array_equal(one, two)
        self.assertEqual(choices, choices2)
        self.assertEqual(len(one), 1000)
        for fold in choices:
            for chosen in fold:
                self.assertEqual(len(set(chosen)), 5)
                self.assertTrue(set(chosen) <= set(names))
        with self.assertRaisesRegex(ValueError, "five eligible"):
            random_baseline([names[:4]], windows[:1])

    def test_pass_rule_all_conditions_thresholds_exact(self):
        values = dict(compounded_return=10, random_p95=10, average_trade_return=.1, positive_fold_percent=55, double_cost_return=.01)
        self.assertTrue(pass_rule(**values)[0])
        for key, value in (("compounded_return", 0), ("random_p95", 10.001), ("average_trade_return", 0),
                           ("positive_fold_percent", 54.99), ("double_cost_return", 0), ("compounded_return", np.nan)):
            self.assertFalse(pass_rule(**{**values, key: value})[0])
        self.assertAlmostEqual(compound([10, -10]), -1)

    def test_top_five_equal_weight_random_and_secondary_costs_freeze_selection(self):
        fold = build_folds("2020-04-01T00:00:00Z")[0]
        ids = [f"c{i:04d}" for i in range(7)]
        summaries = {}
        for cost_name, scale in (("binance_spot_bnb", 1), ("alpaca", .5), ("2x_binance", .1)):
            train = {name: metric(profit=100 - i * 10) for i, name in enumerate(ids)}
            if cost_name != "binance_spot_bnb":
                train[ids[-1]] = metric(profit=10000)  # Must not alter selections.
            test = {name: metric(profit=(10 - i) * scale) for i, name in enumerate(ids)}
            summaries[cost_name] = {"folds": [{"train": train, "test": test}]}
        report = search.summarize_walk_forward(summaries, [fold])
        self.assertEqual(report["selections"], [ids[:5]])
        self.assertAlmostEqual(report["portfolio"]["binance_spot_bnb"]["return_percent"], 8)
        self.assertAlmostEqual(report["portfolio"]["alpaca"]["return_percent"], 4)
        self.assertAlmostEqual(report["portfolio"]["2x_binance"]["return_percent"], .8)
        self.assertTrue(report["passed"])
        self.assertTrue(any(row["row_type"] == "IS_OOS_DEGRADATION" for row in report["rows"]))

    def test_insufficient_eligible_never_silently_shrinks_portfolio_or_passes(self):
        fold = build_folds("2020-04-01T00:00:00Z")[0]
        metrics = {"c0000": metric()}
        summaries = {name: {"folds": [{"train": metrics, "test": metrics}]} for name in search.costs()}
        report = search.summarize_walk_forward(summaries, [fold])
        self.assertFalse(report["passed"])
        self.assertTrue(any(row["row_type"] == "INSUFFICIENT_ELIGIBLE" for row in report["rows"]))

    def test_changing_future_after_fold_test_end_cannot_change_selection_or_oos(self):
        hourly = gate.synthetic_history(periods=24 * 980)
        cutoff = pd.Timestamp("2020-07-01", tz="UTC")
        folds = build_folds(cutoff)
        configs = tuple(c for c in enumerate_configs() if c.timeframe == "4Hour" and c.family == "trend_sma" and c.first == 50)[:6]
        one, _ = search.evaluate_configs(hourly, folds, cutoff=cutoff, configs=configs, progress=False)
        altered = hourly.copy()
        altered.loc[altered.index >= folds[0].test_end, ["open", "high", "low", "close"]] *= 1000
        two, _ = search.evaluate_configs(altered, folds, cutoff=cutoff, configs=configs, progress=False)
        for name in search.costs():
            self.assertEqual(one[name]["folds"][0], two[name]["folds"][0])
        self.assertEqual(select_top(one["binance_spot_bnb"]["folds"][0]["train"]), select_top(two["binance_spot_bnb"]["folds"][0]["train"]))

    def test_default_search_cache_access_bounded_and_gate_failure_stops_search(self):
        start, end = holdout_bounds("2026-10-06T00:00:00Z")
        # Sparse fixture covers endpoints; gate is mocked to stop before simulations.
        bars = pd.concat([flat_bars(2).set_axis(pd.date_range(data.HISTORY_START, periods=2, freq="h")),
                          flat_bars(2).set_axis(pd.date_range(start - pd.Timedelta(hours=2), periods=2, freq="h"))])
        def bounded_loader(**kwargs):
            self.assertEqual(kwargs["end"], start)
            self.assertLess(kwargs["end"], end)
            return bars
        with patch.object(data, "load_history", side_effect=bounded_loader) as loader, \
             patch.object(search, "equivalence_gate", side_effect=RuntimeError("gate failed")), \
             patch.object(search, "evaluate_configs") as evaluate, patch("sys.stdout", io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "gate failed"):
                search.run_search(as_of=end)
        loader.assert_called_once()
        evaluate.assert_not_called()

    def test_final_holdout_requires_frozen_candidate_no_auto_run(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.json"
            path.write_text(json.dumps({"passed": False}), encoding="utf-8")
            with patch.object(data, "load_history") as loader, self.assertRaisesRegex(ValueError, "frozen PASS"):
                search.run_final_holdout(candidate_path=path)
            loader.assert_not_called()
        with patch.object(search, "run_search") as run, patch.object(search, "run_final_holdout") as holdout:
            self.assertEqual(search.main([]), 0)
        run.assert_called_once()
        holdout.assert_not_called()

    def test_cli_modes_and_frozen_holdout_explicit_access(self):
        with patch.object(data, "download_history") as download:
            self.assertEqual(search.main(["--download"]), 0)
        download.assert_called_once()
        with patch.object(search, "run_final_holdout") as final:
            self.assertEqual(search.main(["--final-holdout"]), 0)
        final.assert_called_once()
        with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
            search._build_parser().parse_args(["--download", "--final-holdout"])

    def test_benchmarks_same_oos_segments_and_costs_btc_roundtrip(self):
        hourly = pd.DataFrame({"open": 100., "high": 101., "low": 99., "close": 100., "volume": 1.},
                              index=pd.date_range("2018-01-01", "2020-04-01", inclusive="left", freq="h", tz="UTC"))
        frames, _ = search.prepare_history(hourly)
        folds = build_folds("2020-04-01T00:00:00Z")
        rows = search.benchmark_rows(frames, folds)
        totals = [row for row in rows if row["row_type"] == "BENCHMARK_TOTAL"]
        self.assertEqual(len(totals), 6)
        for row in totals:
            if row["benchmark"] == "BTC_buy_hold":
                rates = search.costs()[row["fee_profile"]]
                expected = ((1 - rates["fee_rate"]) ** 2 * (1 - rates["slippage_rate"]) / (1 + rates["slippage_rate"]) - 1) * 100
                self.assertAlmostEqual(row["return_percent"], expected)

    def test_default_success_report_freezes_last_registered_train_and_never_loads_holdout(self):
        holdout_start, holdout_end = holdout_bounds("2026-10-06T00:00:00Z")
        folds = build_folds(holdout_start)
        ids = [f"c{i:04d}" for i in range(7)]
        windows = {"train": {name: metric(profit=100 - i * 10) for i, name in enumerate(ids)},
                   "test": {name: metric(profit=10 - i) for i, name in enumerate(ids)}}
        summaries = {name: {"folds": [windows for _ in folds]} for name in search.costs()}
        bars = pd.concat([flat_bars(2).set_axis(pd.date_range(data.HISTORY_START, periods=2, freq="h")),
                          flat_bars(4).set_axis(pd.date_range(holdout_start - pd.Timedelta(hours=4), periods=4, freq="h"))])
        frames = {tf: data.resample_bars(bars, tf) for tf in TIMEFRAMES}
        status = {family: {"synthetic": 36, "cached_real": 36, "trades_checked": 10} for family in FAMILIES}
        with tempfile.TemporaryDirectory() as directory:
            candidate, output = Path(directory) / "frozen.json", Path(directory) / "result.csv"
            with patch.object(data, "load_history", return_value=bars) as loader, \
                 patch.object(search, "equivalence_gate", return_value=status), \
                 patch.object(data, "parity_check", return_value={}), \
                 patch.object(search, "evaluate_configs", return_value=(summaries, frames)), \
                 patch.object(search, "benchmark_rows", return_value=[]), \
                 patch.object(search, "run_final_holdout") as final, patch("sys.stdout", io.StringIO()):
                report = search.run_search(as_of=holdout_end, candidate_path=candidate, output=output)
            self.assertTrue(report["passed"])
            final.assert_not_called()
            loader.assert_called_once_with(end=holdout_start, cache_path=data.CACHE_PATH)
            with candidate.open(encoding="utf-8") as handle:
                frozen = json.load(handle)
            self.assertEqual(frozen["config_ids"], ids[:5])
            self.assertEqual(data.utc(frozen["train_start"]), folds[-1].train_start)
            self.assertEqual(data.utc(frozen["train_end"]), folds[-1].train_end)
            with output.open(encoding="utf-8") as handle:
                content = handle.read()
            self.assertIn("PASS_RULE", content)
            self.assertIn("RANDOM_PERCENTILE", content)
            self.assertIn("SELECTION", content)
            candidate.unlink()
            output.unlink()

    def test_explicit_holdout_uses_only_frozen_ids_and_original_boundaries(self):
        start, end = holdout_bounds("2026-10-06T00:00:00Z")
        last_train = build_folds(start)[-1]
        candidate = {"passed": True, "protocol_hash": search.protocol_hash(),
                     "config_ids": [f"c{i:04d}" for i in range(5)],
                     "train_start": str(last_train.train_start), "train_end": str(last_train.train_end),
                     "holdout_start": str(start), "holdout_end": str(end)}
        index = pd.date_range(start, periods=230, freq="h")
        bars = flat_bars(230).set_axis(index)
        result = Simulation(pd.Series([1., 1.5], index=[start, end]), pd.Series(False, index=index), pd.DataFrame())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frozen.json"
            path.write_text(json.dumps(candidate), encoding="utf-8")
            original = path.read_bytes()
            with patch.object(data, "load_history", return_value=bars) as loader, \
                 patch.object(search, "equivalence_gate", return_value={}), \
                 patch.object(search, "prepare_history", return_value=({tf: bars for tf in TIMEFRAMES}, {tf: pd.Series(True, index=index) for tf in TIMEFRAMES})), \
                 patch.object(search, "simulate", return_value=result) as sim, \
                 patch.object(search, "evaluate_configs") as evaluate, patch("sys.stdout", io.StringIO()):
                # Add a last completed endpoint so coverage checks pass.
                completed = pd.concat([bars, bars.iloc[-1:].set_axis([end - pd.Timedelta(hours=1)])])
                loader.return_value = completed
                rows = search.run_final_holdout(candidate_path=path, output=None)
            self.assertEqual([call.kwargs["end"] for call in loader.call_args_list], [start, end])
            self.assertEqual(sim.call_count, 15)
            evaluate.assert_not_called()
            self.assertEqual(path.read_bytes(), original)
            self.assertTrue(all(row.get("config_id") in candidate["config_ids"] for row in rows if row["row_type"] == "HOLDOUT_CONFIG"))
            self.assertEqual([row["return_percent"] for row in rows if row["row_type"] == "HOLDOUT_PORTFOLIO"], [50., 50., 50.])

    def test_every_family_and_regime_signal_prefix_is_causal(self):
        hourly = gate.synthetic_history(periods=4096)
        boundary = hourly.index[3000]
        altered = hourly.copy()
        altered.loc[altered.index >= boundary, ["open", "high", "low", "close"]] *= 10
        original_frames, original_states = search.prepare_history(hourly)
        altered_frames, altered_states = search.prepare_history(altered)
        for c in gate.gate_configs():
            original = original_frames[c.timeframe]
            mask = original.index + pd.Timedelta(hours={"1Hour": 1, "2Hour": 2, "4Hour": 4}[c.timeframe]) <= boundary
            first, reason_one = apply_regime(family_signals(original, c), original_states[c.timeframe], c.regime_gate)
            second, reason_two = apply_regime(family_signals(altered_frames[c.timeframe], c), altered_states[c.timeframe], c.regime_gate)
            np.testing.assert_array_equal(first[mask], second[mask])
            np.testing.assert_array_equal(reason_one[mask], reason_two[mask])

    def test_artifact_outputs_cannot_overwrite_cache_or_each_other(self):
        with self.assertRaisesRegex(ValueError, "cache"):
            search.validate_paths("cache.sqlite", "cache.sqlite", "candidate.json")
        with self.assertRaisesRegex(ValueError, "differ"):
            search.validate_paths("cache.sqlite", "same.csv", "same.csv")


if __name__ == "__main__":
    unittest.main()
