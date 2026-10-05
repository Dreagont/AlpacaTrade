import io
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import numpy as np
import pandas as pd

import config
import optimize_live_config as optimizer
import trade_config
from backtest import run_backtest
from strategy import (
    Decision,
    MA_RSI_CROSSOVER,
    StrategySpec,
    create_ma_rsi_strategy,
)


def _bars(closes, *, start="2026-01-01T00:00:00Z", freq="5min"):
    index = pd.date_range(start, periods=len(closes), freq=freq)
    opens = np.asarray(closes, dtype=float) + 0.25
    return pd.DataFrame(
        {
            "open": opens,
            "high": np.maximum(opens, closes) + 1,
            "low": np.minimum(opens, closes) - 1,
            "close": np.asarray(closes, dtype=float),
            "volume": 1.0,
        },
        index=index,
    )


def _candidate(**overrides):
    values = dict(
        timeframe="5Min", fast_ma=10, slow_ma=30, rsi_period=14,
        rsi_threshold=70, stop_loss=0.02, take_profit=0.04,
    )
    values.update(overrides)
    return optimizer.Candidate(**values)


class OptimizeLiveConfigTests(unittest.TestCase):
    def test_default_factory_matches_live_strategy_and_frozen_outputs(self):
        candidate = optimizer._baseline_candidate("5Min")
        factory = create_ma_rsi_strategy(
            candidate.fast_ma,
            candidate.slow_ma,
            candidate.rsi_period,
            candidate.rsi_threshold,
            candidate.stop_loss,
            candidate.take_profit,
        )
        rng = np.random.default_rng(813)
        closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 600)))
        bars = _bars(closes)
        live_prepared = MA_RSI_CROSSOVER.prepare_indicators(bars)
        factory_prepared = factory.prepare_indicators(bars)
        pd.testing.assert_frame_equal(live_prepared, factory_prepared)
        for index in range(len(bars)):
            self.assertEqual(
                MA_RSI_CROSSOVER.decide_at(live_prepared, index),
                factory.decide_at(factory_prepared, index),
            )
        live_result = run_backtest(
            bars, timeframe="5Min", fee_rate=config.BACKTEST_FEE_PERCENT,
            slippage=config.BACKTEST_SLIPPAGE_PERCENT, strategy=MA_RSI_CROSSOVER,
        )
        factory_result = run_backtest(
            bars, timeframe="5Min", fee_rate=config.BACKTEST_FEE_PERCENT,
            slippage=config.BACKTEST_SLIPPAGE_PERCENT, strategy=factory,
        )
        pd.testing.assert_frame_equal(live_result["trades"], factory_result["trades"])
        self.assertEqual(live_result["ending_equity"], factory_result["ending_equity"])

    def test_future_candles_do_not_change_previous_decisions(self):
        strategy = create_ma_rsi_strategy(3, 8, 5, 65, 0.02, 0.04)
        prefix = _bars([100, 99, 98, 97, 96, 97, 95, 98, 100, 103, 101, 99])
        extended = pd.concat([prefix, _bars(
            [400, 20, 500], start=(prefix.index[-1] + pd.Timedelta(minutes=5)).isoformat()
        )])
        old_prepared = strategy.prepare_indicators(prefix)
        extended_prepared = strategy.prepare_indicators(extended)
        for index in range(len(prefix)):
            self.assertEqual(
                strategy.decide_at(old_prepared, index),
                strategy.decide_at(extended_prepared, index),
            )

    def test_factory_candidate_entries_fill_at_next_candle_open(self):
        bars = _bars([5, 4, 3, 4, 3, 4, 5])
        strategy = create_ma_rsi_strategy(1, 2, 3, 55, 0.5, 0.5)
        prepared = strategy.prepare_indicators(bars)
        self.assertEqual(strategy.decide_at(prepared, 3).action, "BUY")
        result = run_backtest(
            bars, starting_capital=1000, timeframe="5Min", fee_rate=0,
            slippage=0, strategy=strategy, liquidate_at_end=False,
        )
        self.assertEqual(result["entry_events"].iloc[0]["entry_time"], bars.index[4])
        self.assertEqual(result["entry_events"].iloc[0]["entry_fill_price"], bars.iloc[4]["open"])

    def test_candidate_constraints_and_default_grid_size(self):
        with self.assertRaisesRegex(ValueError, "slow_ma"):
            optimizer.validate_candidate(_candidate(fast_ma=30, slow_ma=30))
        candidates = optimizer.generate_candidates()
        self.assertEqual(len(candidates), 24000)
        self.assertEqual(len(optimizer.group_candidates(candidates)), 300)
        self.assertTrue(all(candidate.slow_ma > candidate.fast_ma for candidate in candidates))
        self.assertEqual(optimizer.candidate_count(), 24000)

    def test_candidates_with_shared_indicators_reuse_one_prepared_frame(self):
        bars = _bars(np.linspace(90, 110, 80))
        cache = {}
        first = optimizer._candidate_strategy(_candidate(fast_ma=5, slow_ma=20), bars, cache)
        second = optimizer._candidate_strategy(
            _candidate(fast_ma=5, slow_ma=20, rsi_threshold=60, stop_loss=0.01),
            bars,
            cache,
        )
        self.assertEqual(len(cache), 1)
        pd.testing.assert_frame_equal(
            first.prepare_indicators(bars), second.prepare_indicators(bars)
        )
        segment_bars = bars.iloc[20:].copy()
        self.assertEqual(len(second.prepare_indicators(segment_bars)), len(segment_bars))

    def test_segments_are_exact_chronological_and_non_overlapping(self):
        end = pd.Timestamp("2026-10-05T00:00:00Z")
        segments = optimizer.build_segments(end)
        self.assertEqual(segments[0].start, end - pd.Timedelta(days=30))
        self.assertEqual(segments[-1].end, end)
        for earlier, later in zip(segments, segments[1:]):
            self.assertEqual(earlier.end, later.start)
            self.assertEqual(earlier.end - earlier.start, pd.Timedelta(days=10))
        self.assertTrue(all(a.start < b.start for a, b in zip(segments, segments[1:])))

    def test_segment_runs_ignore_presegment_signal_and_start_flat(self):
        end = pd.Timestamp("2026-10-05T00:00:00Z")
        segments = optimizer.build_segments(end)
        index = pd.date_range(segments[0].start - pd.Timedelta(days=1), end, freq="1h")
        bars = pd.DataFrame(
            {name: 100.0 for name in ("open", "high", "low", "close", "volume")},
            index=index,
        )
        for segment in segments:
            strategy = StrategySpec(
                name="presegment_only",
                _prepare_indicators=lambda frame: frame.copy(),
                _decide_at=lambda frame, i, boundary=segment.start: Decision(
                    "BUY" if frame.index[i] < boundary else "HOLD", "presegment"
                ),
                _warmup_lookback=lambda: 0,
                _parameters=lambda: {},
                _stop_loss=lambda: None,
                _take_profit=lambda: None,
            )
            segment_bars = optimizer.slice_segment(bars, segment)
            result = run_backtest(
                segment_bars, starting_capital=1000, timeframe="1Hour",
                fee_rate=0, slippage=0, test_start=segment.start,
                strategy=strategy,
            )
            self.assertEqual(result["starting_capital"], 1000)
            self.assertEqual(len(result["entry_events"]), 0)
            self.assertEqual(result["total_trades"], 0)

    def test_minimum_sample_eligibility_and_reason(self):
        self.assertEqual(optimizer.minimum_sample_status(10, [2, 2, 0]), (True, ""))
        eligible, reason = optimizer.minimum_sample_status(9, [2, 2, 0])
        self.assertFalse(eligible)
        self.assertIn("fewer than 10", reason)
        eligible, reason = optimizer.minimum_sample_status(10, [3, 1, 1])
        self.assertFalse(eligible)
        self.assertIn("fewer than 2 segments", reason)

    def test_classification_rules(self):
        args = dict(
            eligible=True, net_return=2, same_path_return=3, net_profit_factor=1.2,
        )
        self.assertEqual(
            optimizer.classify_candidate(**args, profitable_segment_count=3), "STRONG"
        )
        self.assertEqual(
            optimizer.classify_candidate(**args, profitable_segment_count=2), "PROMISING"
        )
        self.assertEqual(
            optimizer.classify_candidate(**{**args, "same_path_return": -1},
                                         profitable_segment_count=3), "WEAK"
        )
        self.assertEqual(
            optimizer.classify_candidate(**{**args, "eligible": False},
                                         profitable_segment_count=3), "INELIGIBLE"
        )

    def test_ranking_prefers_stability_over_single_period_return(self):
        stable = dict(
            classification="STRONG", profitable_segment_count=3,
            median_segment_return=1, worst_segment_return=0.2,
            total_net_return_percent=4, net_profit_factor=1.5,
            max_drawdown_percent=3,
        )
        lucky = dict(
            classification="PROMISING", profitable_segment_count=2,
            median_segment_return=0.2, worst_segment_return=-1,
            total_net_return_percent=40, net_profit_factor=4,
            max_drawdown_percent=1,
        )
        self.assertLess(optimizer.ranking_key(stable), optimizer.ranking_key(lucky))

    def test_baseline_is_in_default_search_and_uses_live_values(self):
        baseline = optimizer._baseline_candidate("5Min")
        self.assertIn(baseline, optimizer.generate_candidates())
        self.assertTrue(optimizer._is_current_baseline(baseline))
        self.assertFalse(optimizer._is_current_baseline(_candidate(timeframe="15Min")))
        self.assertEqual(baseline.fast_ma, trade_config.FAST_MA)
        self.assertEqual(baseline.slow_ma, trade_config.SLOW_MA)
        self.assertEqual(baseline.rsi_period, trade_config.RSI_PERIOD)
        self.assertEqual(baseline.rsi_threshold, trade_config.RSI_BUY_THRESHOLD)
        self.assertEqual(baseline.stop_loss, trade_config.STOP_LOSS_PERCENT)
        self.assertEqual(baseline.take_profit, trade_config.TAKE_PROFIT_PERCENT)

    def test_optimizer_fetches_once_per_timeframe_and_never_mutates_trade_config(self):
        end = datetime(2026, 10, 5, tzinfo=timezone.utc)
        before = {
            key: getattr(trade_config, key)
            for key in (
                "FAST_MA", "SLOW_MA", "RSI_PERIOD", "RSI_BUY_THRESHOLD",
                "STOP_LOSS_PERCENT", "TAKE_PROFIT_PERCENT", "LIVE_STRATEGY",
                "TRADE_AMOUNT_USD", "MAX_POSITION_USD",
            )
        }
        candidates = [_candidate(), _candidate(timeframe="15Min")]
        rows = []
        for candidate in candidates:
            row = {
                **candidate.__dict__, "rank": 1, "classification": "WEAK",
                "eligible_for_ranking": True, "completed_trades": 10,
                "profitable_segment_count": 1, "total_net_return_percent": -1,
                "same_path_zero_cost_return_percent": 1, "net_profit_factor": 0.8,
                "max_drawdown_percent": 2, "segment_1_net_return_percent": -1,
                "segment_2_net_return_percent": -1, "segment_3_net_return_percent": 1,
                "median_segment_return": -1, "worst_segment_return": -1,
                "is_current_live_baseline": candidate.timeframe == "5Min",
            }
            rows.append(row)
        bars = pd.DataFrame(
            {name: [100.0] * 9000 for name in ("open", "high", "low", "close", "volume")},
            index=pd.date_range(end=pd.Timestamp(end), periods=9000, freq="5min"),
        )
        with (
            patch.object(optimizer, "generate_candidates", return_value=candidates),
            patch("optimize_live_config.backtest.fetch_history", return_value=bars) as fetch,
            patch.object(optimizer, "_evaluate_candidate", side_effect=rows),
            redirect_stdout(io.StringIO()),
        ):
            optimizer.run_optimizer(
                end_time=end,
                timeframes=("5Min", "15Min"),
                output=None,
                engine="reference",
            )
        self.assertEqual(fetch.call_count, 2)
        self.assertEqual([call.args[1] for call in fetch.call_args_list], ["5Min", "15Min"])
        self.assertTrue(
            all(call.kwargs["warmup_bars"] == 90 for call in fetch.call_args_list)
        )
        self.assertEqual(
            before,
            {key: getattr(trade_config, key) for key in before},
        )


if __name__ == "__main__":
    unittest.main()
