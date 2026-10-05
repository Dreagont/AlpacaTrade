import unittest
from dataclasses import replace

import numpy as np
import pandas as pd

import config
import optimize_live_config as optimizer
import optimizer_fast
import trade_config
from backtest import run_backtest
from strategy import create_ma_rsi_strategy
from timeframes import parse_timeframe


def make_bars(size=10):
    index = pd.date_range("2026-01-01T00:00:00Z", periods=size, freq="5min")
    opens = np.full(size, 100.0)
    highs = np.full(size, 101.0)
    lows = np.full(size, 99.0)
    closes = np.full(size, 100.0)
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": 1.0},
        index=index,
    )


def controlled_strategy(bars, *, buy=True, sell=False, threshold=70, stop=0.02, target=0.04):
    prepared = pd.DataFrame(index=bars.index)
    prepared["ma_fast"] = 0.0
    prepared["ma_slow"] = 1.0
    prepared["rsi"] = 50.0
    if buy:
        prepared.loc[bars.index[1], "ma_fast"] = 2.0
    if sell:
        prepared.loc[bars.index[2], "ma_fast"] = 2.0
        prepared.loc[bars.index[3], "ma_fast"] = 0.0
    strategy = create_ma_rsi_strategy(1, 2, 1, threshold, stop, target)
    return replace(strategy, _prepare_indicators=lambda _bars: prepared.copy()), prepared


def assert_fast_matches_reference(testcase, bars, strategy, prepared, *, fee=0.0025, slip=0.0005):
    timeframe = "5Min"
    _, minutes = parse_timeframe(timeframe)
    reference = run_backtest(
        bars,
        starting_capital=config.BACKTEST_STARTING_CAPITAL,
        timeframe=timeframe,
        fee_rate=fee,
        slippage=slip,
        test_start=bars.index[0],
        strategy=strategy,
    )
    fast = optimizer_fast.simulate(
        optimizer_fast.FastPrepared.from_frame(
            optimizer_fast.FastBars.from_frame(bars), prepared
        ),
        start_time=bars.index[0],
        end_time=bars.index[-1] + pd.Timedelta(minutes=minutes),
        bar_minutes=minutes,
        rsi_threshold=strategy.parameters["rsi_buy_threshold"],
        stop_loss_percent=strategy.stop_loss_percent,
        take_profit_percent=strategy.take_profit_percent,
        fee_rate=fee,
        slippage=slip,
        starting_capital=config.BACKTEST_STARTING_CAPITAL,
        trade_amount=trade_config.TRADE_AMOUNT_USD,
        max_position=trade_config.MAX_POSITION_USD,
        use_numba=False,
    )
    expected_trades = reference["trades"]
    actual_trades = fast["trades"]
    testcase.assertEqual(fast["total_trades"], reference["total_trades"])
    testcase.assertEqual(list(actual_trades["entry_time"]), list(expected_trades["entry_time"]))
    testcase.assertEqual(list(actual_trades["exit_time"]), list(expected_trades["exit_time"]))
    testcase.assertEqual(list(actual_trades["exit_reason"]), list(expected_trades["exit_reason"]))
    for column in (
        "requested_notional", "gross_quantity_before_fee", "quantity_after_buy_fee",
        "net_pnl", "gross_pnl", "fees", "slippage_cost",
    ):
        np.testing.assert_allclose(
            actual_trades[column].to_numpy(dtype=float),
            expected_trades[column].to_numpy(dtype=float),
            rtol=1e-10,
            atol=1e-10,
        )
    for fast_key, reference_key in (
        ("ending_equity", "ending_equity"),
        ("strategy_return", "strategy_return"),
        ("total_fees", "total_fees"),
        ("total_slippage", "total_slippage"),
        ("max_drawdown", "max_drawdown"),
        ("win_rate", "win_rate"),
        ("net_profit_factor", "net_profit_factor"),
    ):
        actual = fast[fast_key]
        expected = reference[reference_key]
        if actual is None or expected is None:
            testcase.assertIs(actual, expected, fast_key)
        else:
            testcase.assertAlmostEqual(actual, expected, places=9, msg=fast_key)


class FastKernelEquivalenceTests(unittest.TestCase):
    def test_group_level_worker_pool_matches_serial_fast_engine(self):
        end = pd.Timestamp("2026-10-05T00:00:00Z")
        segments = optimizer.build_segments(end)
        index = pd.date_range(
            segments[0].start - pd.Timedelta(hours=100),
            end - pd.Timedelta(hours=1),
            freq="1h",
        )
        rng = np.random.default_rng(415)
        close = 50_000.0 * np.exp(np.cumsum(rng.normal(0, 0.003, len(index))))
        bars = pd.DataFrame(
            {
                "open": np.r_[close[0], close[:-1]],
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": 1.0,
            },
            index=index,
        )
        candidates = [
            optimizer.Candidate("1Hour", 5, 20, 7, 55, 0.01, 0.015),
            optimizer.Candidate("1Hour", 8, 30, 14, 65, 0.02, 0.04),
        ]
        groups = optimizer.group_candidates(candidates)
        serial = optimizer._run_fast_groups(
            groups, {"1Hour": bars}, segments, workers=1
        )
        parallel = optimizer._run_fast_groups(
            groups, {"1Hour": bars}, segments, workers=2
        )
        serial_by_candidate = {optimizer._candidate_from_row(row): row for row in serial}
        parallel_by_candidate = {optimizer._candidate_from_row(row): row for row in parallel}
        self.assertEqual(serial_by_candidate.keys(), parallel_by_candidate.keys())
        for candidate in serial_by_candidate:
            optimizer._assert_row_equivalence(
                serial_by_candidate[candidate], parallel_by_candidate[candidate]
            )

    def test_execution_priority_and_trade_scenarios_match_reference(self):
        scenarios = (
            "bearish_exit", "gap_sl", "gap_tp", "intrabar_sl", "intrabar_tp",
            "both_hit_stop_first", "no_trades", "open_at_end",
        )
        for scenario in scenarios:
            with self.subTest(scenario=scenario):
                bars = make_bars()
                buy = scenario != "no_trades"
                sell = scenario == "bearish_exit"
                if scenario == "gap_sl":
                    bars.iloc[3, bars.columns.get_loc("open")] = 97.0
                    bars.iloc[3, bars.columns.get_loc("high")] = 98.0
                    bars.iloc[3, bars.columns.get_loc("low")] = 96.0
                elif scenario == "gap_tp":
                    bars.iloc[3, bars.columns.get_loc("open")] = 105.0
                    bars.iloc[3, bars.columns.get_loc("high")] = 106.0
                    bars.iloc[3, bars.columns.get_loc("low")] = 104.0
                elif scenario == "intrabar_sl":
                    bars.iloc[3, bars.columns.get_loc("low")] = 97.0
                elif scenario == "intrabar_tp":
                    bars.iloc[3, bars.columns.get_loc("high")] = 105.0
                elif scenario == "both_hit_stop_first":
                    bars.iloc[3, bars.columns.get_loc("low")] = 97.0
                    bars.iloc[3, bars.columns.get_loc("high")] = 105.0
                strategy, prepared = controlled_strategy(
                    bars, buy=buy, sell=sell
                )
                assert_fast_matches_reference(self, bars, strategy, prepared)

    def test_fast_kernel_matches_multiple_seeded_price_paths(self):
        for seed in (4, 29, 1987):
            with self.subTest(seed=seed):
                rng = np.random.default_rng(seed)
                size = 400
                bars = make_bars(size)
                close = 100 * np.exp(np.cumsum(rng.normal(0.0, 0.012, size)))
                bars["close"] = close
                bars["open"] = np.r_[close[0], close[:-1]] * np.exp(
                    rng.normal(0.0, 0.002, size)
                )
                bars["high"] = np.maximum(bars["open"], close) * (1 + rng.uniform(0, 0.02, size))
                bars["low"] = np.minimum(bars["open"], close) * (1 - rng.uniform(0, 0.02, size))
                strategy = create_ma_rsi_strategy(5, 18, 7, 65, 0.02, 0.04)
                prepared = strategy.prepare_indicators(bars)
                assert_fast_matches_reference(self, bars, strategy, prepared)

    def test_all_optimizer_stop_and_take_values_match_reference(self):
        bars = make_bars(20)
        bars.loc[bars.index[12], "close"] = 102.0
        for stop in (0.01, 0.015, 0.02, 0.03):
            for target in (0.015, 0.02, 0.03, 0.04, 0.06):
                with self.subTest(stop=stop, target=target):
                    strategy, prepared = controlled_strategy(
                        bars, stop=stop, target=target
                    )
                    assert_fast_matches_reference(self, bars, strategy, prepared)


if __name__ == "__main__":
    unittest.main()
