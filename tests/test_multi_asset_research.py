import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from alpaca.data.enums import Adjustment, DataFeed

import multi_asset_data as data
import multi_asset_engine as engine
import multi_asset_research as research
from fast_search_engine import simulate


def bars(count=600, start="2016-01-04", close=None):
    dates = pd.bdate_range(start, periods=count)
    values = np.asarray(close if close is not None else 100 + 15 * np.sin(np.arange(count) / 30), dtype=float)
    return pd.DataFrame(dict(open=values, high=values * 1.01, low=values * .99, close=values, volume=1.), index=dates)


def targets(dates, value=True):
    return pd.Series(value, index=dates)


class DailyDataTests(unittest.TestCase):
    def test_completed_dates_before_after_close_and_dst(self):
        self.assertEqual(data.completed_end(now="2024-03-11 19:59Z"), pd.Timestamp("2024-03-10"))
        self.assertEqual(data.completed_end(now="2024-03-11 20:01Z"), pd.Timestamp("2024-03-11"))
        self.assertEqual(data.completed_end("2025-01-01", now="2024-11-04 20:59Z"), pd.Timestamp("2024-11-03"))
        self.assertEqual(data.completed_end(now="2024-11-04 21:01Z"), pd.Timestamp("2024-11-04"))

    def test_adjustment_all_sip_request_preserves_adjusted_ohlc(self):
        raw = bars(3)
        raw.index = raw.index.tz_localize("America/New_York").tz_convert("UTC")
        raw.index = pd.MultiIndex.from_product([["SPY"], raw.index])
        calls = []
        def fetch(request):
            calls.append(request)
            return SimpleNamespace(df=raw)
        result = data.fetch_etf(SimpleNamespace(get_stock_bars=fetch), "SPY", pd.Timestamp("2016-01-05"), lambda _: None)
        self.assertEqual(len(result), 2)
        self.assertEqual(calls[0].adjustment, Adjustment.ALL)
        self.assertEqual(calls[0].feed, DataFeed.SIP)
        self.assertEqual(result.iloc[0].open, 100)
        self.assertEqual(result.attrs["adjustment"], "all")

    def test_sip_denial_falls_back_and_reports_feed(self):
        calls, messages = [], []
        def fetch(request):
            calls.append(request)
            if request.feed == DataFeed.SIP:
                raise RuntimeError("subscription does not permit querying SIP data")
            return SimpleNamespace(df=bars(3))
        result = data.fetch_etf(SimpleNamespace(get_stock_bars=fetch), "SPY", pd.Timestamp("2016-01-06"), messages.append)
        self.assertEqual([r.feed for r in calls], [DataFeed.SIP, DataFeed.IEX])
        self.assertEqual(result.attrs["feed"], "iex")
        self.assertTrue(any("FALLING BACK TO IEX" in m for m in messages))
        self.assertTrue(any("feed=iex" in m for m in messages))

    def test_other_network_failure_does_not_fallback_or_echo_credentials(self):
        client = SimpleNamespace(get_stock_bars=lambda _: (_ for _ in ()).throw(RuntimeError("SECRET network failure")))
        with self.assertRaisesRegex(RuntimeError, "no feed substitution") as error:
            data.fetch_etf(client, "SPY", pd.Timestamp("2016-01-06"))
        self.assertNotIn("SECRET", str(error.exception))

    def test_missing_days_spy_calendar_excludes_holidays_and_counts_edges(self):
        reference = pd.DatetimeIndex(["2024-07-03", "2024-07-05", "2024-07-08"])
        frame = bars(2)
        frame.index = reference[1:]
        quality = data.quality_report("TLT", frame, reference)
        self.assertEqual(quality["missing_us_days"], 1)
        self.assertEqual(quality["missing_dates"], "2024-07-03")

    def test_cache_requires_adjustment_metadata_and_replays_feed(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = SimpleNamespace(get_stock_bars=lambda _: SimpleNamespace(df=bars(4)))
            data.load_etfs(pd.Timestamp("2016-01-07"), download=True, cache_dir=tmp, client=client, progress=lambda _: None)
            messages = []
            cached = data.load_etfs(pd.Timestamp("2016-01-07"), cache_dir=tmp, progress=messages.append)
            self.assertEqual(cached["GLD"].attrs["adjustment"], "all")
            self.assertEqual(len(messages), 4)
            meta = Path(tmp) / "SPY_1day_all.json"
            metadata = json.loads(meta.read_text())
            metadata["adjustment"] = "raw"
            meta.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "adjusted-price"):
                data.load_etfs(pd.Timestamp("2016-01-07"), cache_dir=tmp, progress=lambda _: None)

    def test_btc_daily_requires_complete_24_hours(self):
        hourly = bars(48)
        hourly.index = pd.date_range("2024-03-01", periods=48, freq="h", tz="UTC")
        daily = data.btc_daily(hourly.drop(hourly.index[4]))
        self.assertEqual(len(daily), 1)
        self.assertEqual(daily.index[0], pd.Timestamp("2024-03-02", tz="UTC"))
        self.assertEqual(daily.close_time.iloc[0], pd.Timestamp("2024-03-03", tz="UTC"))

    def test_btc_close_alignment_dst_weekends_and_no_future_source(self):
        for start, dates, hours in (("2024-03-05", ["2024-03-08", "2024-03-11", "2024-03-12"], [21, 20, 20]),
                                    ("2024-10-30", ["2024-11-01", "2024-11-04", "2024-11-05"], [20, 21, 21])):
            daily = bars(10, close=np.arange(10) + 100)
            daily.index = pd.date_range(start, periods=10, freq="D", tz="UTC")
            daily["close_time"] = daily.index + pd.Timedelta(days=1)
            us = pd.DatetimeIndex(dates)
            aligned = data.align_btc(daily, us)
            cutoffs = data.session_time(us)
            self.assertEqual(list(cutoffs.hour), hours)
            self.assertTrue((pd.DatetimeIndex(aligned.source_close_time) <= cutoffs).all())
            self.assertTrue((pd.DatetimeIndex(aligned.execution_time)[1:] > cutoffs[:-1]).all())
            self.assertEqual(aligned.execution_time.iloc[1].dayofweek, 5)  # Saturday fill
            for stamp, cutoff in zip(aligned.source_close_time, cutoffs):
                self.assertEqual(stamp, max(daily.loc[daily.close_time <= cutoff, "close_time"]))

    def test_btc_missing_close_uses_latest_older_completed_bar(self):
        daily = bars(7)
        daily.index = pd.date_range("2024-03-05", periods=7, freq="D", tz="UTC")
        daily["close_time"] = daily.index + pd.Timedelta(days=1)
        aligned = data.align_btc(daily.drop(pd.Timestamp("2024-03-07", tz="UTC")), pd.DatetimeIndex(["2024-03-08"]))
        self.assertEqual(aligned.source_close_time.iloc[0], pd.Timestamp("2024-03-07", tz="UTC"))

    def test_btc_missing_midday_hour_does_not_erase_observed_midnight_open(self):
        hourly = bars(96)
        hourly.index = pd.date_range("2024-03-05", periods=96, freq="h", tz="UTC")
        hourly = hourly.drop(pd.Timestamp("2024-03-07 10:00", tz="UTC"))
        daily = data.btc_daily(hourly)
        us = pd.DatetimeIndex(["2024-03-06", "2024-03-07", "2024-03-08"])
        aligned = data.align_btc(daily, us, execution_opens=hourly.open)
        self.assertIn(pd.Timestamp("2024-03-07"), aligned.index)
        self.assertEqual(aligned.loc["2024-03-07", "open"], hourly.loc["2024-03-07 00:00Z", "open"])
        self.assertEqual(aligned.loc["2024-03-08", "source_close_time"], pd.Timestamp("2024-03-07", tz="UTC"))

    def test_btc_unobserved_execution_open_remains_missing(self):
        hourly = bars(96)
        hourly.index = pd.date_range("2024-03-05", periods=96, freq="h", tz="UTC")
        hourly = hourly.drop(pd.Timestamp("2024-03-07", tz="UTC"))
        aligned = data.align_btc(data.btc_daily(hourly), pd.DatetimeIndex(["2024-03-06", "2024-03-07"]), execution_opens=hourly.open)
        self.assertNotIn(pd.Timestamp("2024-03-07"), aligned.index)


class DailyEngineTests(unittest.TestCase):
    def test_warmup_and_prefix_causality_both_rules(self):
        frame = bars(500, close=np.arange(500) + 100)
        for rule in engine.RULES:
            actions = engine.trend_actions(frame, rule)
            self.assertTrue((actions[:220] == 0).all())
            self.assertEqual(actions[220], 1)
            changed = frame.copy()
            changed.loc[changed.index[300]:, "close"] = .01
            np.testing.assert_array_equal(actions[:300], engine.trend_actions(changed, rule)[:300])
            np.testing.assert_array_equal(actions[:300], engine.trend_actions(frame.iloc[:300], rule))

    def test_signal_fill_next_session_open_across_gap(self):
        frame = bars(3)
        frame.index = pd.DatetimeIndex(["2024-03-28", "2024-04-01", "2024-04-02"])
        frame.loc[frame.index[1], ["open", "high", "low", "close"]] = [200, 220, 190, 210]
        result = simulate(frame, [1, -1, 0], [4, 4, 4], timeframe="1Day", fee_rate=0, slippage_rate=0, use_jit=False)
        self.assertEqual(result.trades.iloc[0].entry_time, frame.index[1])
        self.assertEqual(result.trades.iloc[0].exit_time, frame.index[2])
        self.assertAlmostEqual(result.equity.iloc[2], 1.05)
        self.assertAlmostEqual(result.trades.iloc[0].return_percent, (frame.open.iloc[2] / 200 - 1) * 100)

    def test_daily_equivalence_all_costs_and_liquidation(self):
        status = engine.daily_equivalence_gate(bars(700), progress=lambda _: None)
        self.assertEqual(set(status), {"synthetic_daily_gaps", "real_SPY"})
        self.assertTrue(all(value > 0 for value in status.values()))

    def test_gate_fails_closed_and_requires_sample(self):
        with self.assertRaisesRegex(ValueError, "real adjusted SPY"):
            engine.daily_equivalence_gate(None)
        with patch.object(engine, "_compare", side_effect=AssertionError("mismatch")), self.assertRaisesRegex(RuntimeError, "research stopped"):
            engine.daily_equivalence_gate(bars(), progress=lambda _: None)

    def test_existing_timeframes_identical_duration_and_returns(self):
        for tf, hours in (("1Hour", 1), ("2Hour", 2), ("4Hour", 4)):
            frame = bars(3)
            frame.index = pd.date_range("2024-01-01", periods=3, freq=f"{hours}h", tz="UTC")
            result = simulate(frame, [1, -1, 0], [4, 4, 4], timeframe=tf, fee_rate=0, slippage_rate=0, use_jit=False)
            self.assertEqual(result.equity.index[-1], frame.index[-1] + pd.Timedelta(hours=hours))
            self.assertEqual(result.trades.iloc[0].entry_time, frame.index[1])

    def test_sleeve_engine_equity_matches_fast_reinvestment(self):
        frame = bars()
        for rule in engine.RULES:
            curve = research.asset_curve("SPY", frame, rule, {"SPY": (0, .0002)})
            self.assertEqual(curve.equity.index[0], frame.index[221])
            self.assertGreater(curve.entries.sum(), 0)


class PortfolioAndMetricsTests(unittest.TestCase):
    def example(self):
        dates = pd.DatetimeIndex(["2024-01-30", "2024-01-31", "2024-02-01", "2024-02-02"])
        a, b = bars(4), bars(4)
        for frame, values in ((a, [100, 200, 200, 200]), (b, [100, 100, 100, 100])):
            frame.index = dates
            for name in ("open", "high", "low", "close"):
                frame[name] = values
        return dates, {"A": a, "B": b}

    def test_equal_weight_buy_hold_hand_math_monthly_turnover(self):
        dates, frames = self.example()
        curve = engine.portfolio(frames, {s: targets(dates) for s in frames}, {s: (0, 0) for s in frames})
        np.testing.assert_allclose(curve.equity, [1, 1.5, 1.5, 1.5])
        self.assertAlmostEqual(curve.turnover.iloc[2], .5)  # sell .25 A, buy .25 B
        self.assertEqual(curve.entries.sum(), 2)

    def test_monthly_cost_charged_on_actual_buy_and_sell_only(self):
        dates, frames = self.example()
        slip = .01
        curve = engine.portfolio(frames, {s: targets(dates) for s in frames}, {s: (0, slip) for s in frames})
        initial = 1 / 1.01
        pre = 1.5 * initial
        delta = .25 * initial
        sell_notional = delta / .99
        buy_notional = delta / 1.01
        expected_cost = sell_notional * .01 + buy_notional * .01
        self.assertAlmostEqual(curve.costs.iloc[2], expected_cost)
        self.assertAlmostEqual(curve.turnover.iloc[2], sell_notional + buy_notional)
        self.assertAlmostEqual(curve.equity.iloc[2], pre - expected_cost)
        self.assertEqual(curve.costs.iloc[3], 0)

    def test_cash_sleeve_retains_weight_and_cash_transfer_has_no_cost(self):
        dates, frames = self.example()
        curve = engine.portfolio(frames, {"A": targets(dates), "B": targets(dates, False)}, {s: (0, 0) for s in frames})
        self.assertEqual(curve.exposure.iloc[0], .5)
        self.assertAlmostEqual(curve.turnover.iloc[2], .25)
        both_cash = engine.portfolio(frames, {s: targets(dates, False) for s in frames}, {s: (.01, .01) for s in frames})
        np.testing.assert_allclose(both_cash.equity, 1)
        self.assertEqual(both_cash.costs.sum(), 0)

    def test_mixed_transfers_do_not_depend_on_future_etf_open(self):
        dates, frames = self.example()
        frames["B"].loc[dates[2], ["open", "high", "low", "close"]] = 200
        curve = engine.portfolio(frames, {s: targets(dates) for s in frames}, {s: (0, 0) for s in frames}, causal_transfers=True)
        # Fixed transfers .25 from A to B computed on Jan31 capital, before
        # either Feb1 execution; ETF opening gap cannot alter BTC order size.
        self.assertAlmostEqual(curve.turnover.iloc[2], .5)
        self.assertAlmostEqual(curve.equity.iloc[2], 2)

    def test_60_40_weights(self):
        dates, frames = self.example()
        curve = engine.portfolio(frames, {s: targets(dates) for s in frames}, {s: (0, 0) for s in frames}, weights=(.6, .4))
        self.assertAlmostEqual(curve.equity.iloc[1], 1.6)

    def test_metrics_hand_series_compounded_dd_sharpe_cagr_calmar(self):
        dates = pd.DatetimeIndex(["2023-01-01", "2023-06-01", "2024-01-01"])
        equity = pd.Series([1.1, .88, 1.21], index=dates)
        curve = engine.Curve(equity, pd.Series([1, 0, 1], index=dates), pd.Series([1, 0, 1], index=dates), targets(dates, 0.), targets(dates, 0.))
        result = engine.metrics(curve)
        returns = np.array([.1, -.2, .375])
        cagr = 1.21 ** (365.25 / 366) - 1
        self.assertAlmostEqual(result["total_return_pct"], 21)
        self.assertAlmostEqual(result["cagr_pct"], cagr * 100)
        self.assertAlmostEqual(result["max_drawdown_pct"], 20)
        self.assertAlmostEqual(result["calmar"], cagr / .2)
        self.assertAlmostEqual(result["sharpe"], returns.mean() / returns.std(ddof=1) * np.sqrt(252))
        self.assertAlmostEqual(result["time_invested_pct"], 200 / 3)
        self.assertEqual(result["worst_calendar_year"], 2023)
        self.assertAlmostEqual(result["worst_year_return_pct"], -12)
        yearly, periods = engine.period_tables(equity)
        self.assertAlmostEqual(yearly.loc[2023], -12)
        self.assertTrue(np.isnan(periods[0]["return_pct"]))
        self.assertAlmostEqual(periods[-1]["return_pct"], 21)

    def test_initial_entry_cost_counts_in_drawdown_and_yearly_return(self):
        dates = pd.bdate_range("2024-01-01", periods=2)
        curve = engine.Curve(pd.Series([.99, .99], index=dates), targets(dates), targets(dates, 0.), targets(dates, 0.), targets(dates, 0.))
        result = engine.metrics(curve)
        self.assertAlmostEqual(result["max_drawdown_pct"], 1)
        self.assertAlmostEqual(result["worst_year_return_pct"], -1)
        self.assertTrue(np.isnan(engine.metrics(engine.Curve(targets(dates, 1.), targets(dates, False), targets(dates, 0.), targets(dates, 0.), targets(dates, 0.)))["sharpe"]))

    def test_full_reports_spans_benchmarks_and_cost_sensitivity(self):
        datasets = {s: bars(300) for s in data.ETF_SYMBOLS}
        datasets["BTC"] = bars(260, start="2016-03-01")
        records, summary, years, periods = research.run_research(datasets, sensitivity=True, progress=lambda _: None)
        self.assertEqual(len(summary), 60)
        self.assertEqual(set(summary.cost), {"ETF_side_0.02pct", "ETF_side_0.05pct"})
        matched = summary[(summary.name == "ETF_EQUAL") & (summary.span == "BTC_matched")]
        mixed = summary[summary.name == "ETF_BTC_EQUAL"]
        self.assertEqual(set(matched.start), set(mixed.start))
        self.assertEqual(set(matched.end), set(mixed.end))
        self.assertTrue((summary.time_invested_pct <= 100).all())
        self.assertEqual(set(periods.period), {"2016-2019", "2020-2021", "2022", "2023-latest"})
        self.assertTrue({"metrics", "calendar_year", "fixed_period"}.issubset({r["section"] for r in records}))

    def test_cli_gate_failure_produces_no_reports(self):
        datasets = {s: bars() for s in data.ETF_SYMBOLS}
        output = io.StringIO()
        with patch.object(research, "load_etfs", return_value=datasets), patch.object(research, "daily_equivalence_gate", side_effect=RuntimeError("gate failed")), patch.object(research, "run_research") as run, redirect_stdout(output):
            self.assertEqual(research.main(["--no-btc"]), 1)
            run.assert_not_called()
        self.assertIn("RESEARCH STOPPED", output.getvalue())

    def test_cli_offline_csv_includes_quality_and_equivalence(self):
        datasets = {s: bars(260) for s in data.ETF_SYMBOLS}
        with tempfile.TemporaryDirectory() as tmp, patch.object(research, "load_etfs", return_value=datasets), patch.object(research, "daily_equivalence_gate", return_value={"real_SPY": 4}), redirect_stdout(io.StringIO()):
            path = Path(tmp) / "report.csv"
            self.assertEqual(research.main(["--no-btc", "--csv", str(path)]), 0)
            rows = pd.read_csv(path)
            self.assertTrue({"metrics", "calendar_year", "fixed_period", "data_quality", "equivalence"}.issubset(set(rows.section)))
