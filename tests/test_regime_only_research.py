import unittest

import numpy as np
import pandas as pd

import backtest
import regime_filter_research as experiment
import strategy as strategy_module
from strategy import Decision, create_regime_only_strategy, get_strategy


def _bars_from_close(close, start="2020-01-01T00:00:00Z"):
    close = np.asarray(close, dtype=float)
    index = pd.date_range(start, periods=len(close), freq="4h")
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.ones(len(close)),
        },
        index=index,
    )


def _transition_bars():
    close = np.concatenate(
        (
            np.linspace(200.0, 80.0, 250),
            np.linspace(80.0, 300.0, 120),
            np.linspace(300.0, 50.0, 130),
        )
    )
    return _bars_from_close(close)


class RegimeOnlyStrategyTests(unittest.TestCase):
    def setUp(self):
        self.strategy = create_regime_only_strategy()

    def test_na_true_false_map_to_hold_buy_sell(self):
        prepared = pd.DataFrame(
            {"regime_bullish": pd.array([pd.NA, True, False], dtype="boolean")}
        )
        self.assertEqual(
            [self.strategy.decide_at(prepared, i) for i in range(3)],
            [
                Decision("HOLD", "regime_filter_not_ready"),
                Decision("BUY", "regime_on"),
                Decision("SELL", "regime_filter_off"),
            ],
        )

    def test_regime_columns_match_filtered_donchian_and_strategy_is_research_only(self):
        bars = _transition_bars()
        regime_prepared = self.strategy.prepare_indicators(bars)
        filtered_prepared = get_strategy(
            "donchian_regime_filter"
        ).prepare_indicators(bars)
        for column in ("regime_sma", "regime_sma_previous", "regime_bullish"):
            pd.testing.assert_series_equal(
                regime_prepared[column], filtered_prepared[column]
            )
        self.assertIs(
            strategy_module.STRATEGY_REGISTRY["regime_only_4h"],
            strategy_module.REGIME_ONLY_4H,
        )
        self.assertEqual(self.strategy.stop_loss_percent, None)
        self.assertEqual(self.strategy.take_profit_percent, None)
        self.assertEqual(self.strategy.max_holding_minutes, None)
        self.assertEqual(self.strategy.required_warmup_bars(), 230)

    def test_regime_transitions_enter_and_exit_at_the_next_4hour_open(self):
        bars = _transition_bars()
        prepared = self.strategy.prepare_indicators(bars)
        states = prepared["regime_bullish"]
        transitions = [
            index
            for index in range(1, len(states))
            if pd.notna(states.iloc[index])
            and pd.notna(states.iloc[index - 1])
            and states.iloc[index] != states.iloc[index - 1]
        ]
        self.assertEqual(transitions, [301, 426])
        test_start = bars.index[230]

        result = backtest.run_backtest(
            bars,
            starting_capital=1000.0,
            timeframe="4Hour",
            fee_rate=0.0,
            slippage=0.0,
            test_start=test_start,
            strategy=self.strategy,
            liquidate_at_end=False,
        )

        self.assertEqual(result["entry_events"].iloc[0]["entry_time"], bars.index[302])
        trade = result["trades"].iloc[0]
        self.assertEqual(trade["entry_time"], bars.index[302])
        self.assertEqual(trade["entry_reason"], "regime_on")
        self.assertEqual(trade["exit_time"], bars.index[427])
        self.assertEqual(trade["exit_reason"], "regime_filter_off")

    def test_future_bars_do_not_change_prior_regime_states_or_trades(self):
        bars = _transition_bars()
        cutoff_index = 430
        changed = bars.copy()
        changed.loc[changed.index[cutoff_index]:, "open"] *= 1.7
        changed.loc[changed.index[cutoff_index]:, "high"] *= 1.7
        changed.loc[changed.index[cutoff_index]:, "low"] *= 1.7
        changed.loc[changed.index[cutoff_index]:, "close"] *= 1.7

        original_prepared = self.strategy.prepare_indicators(bars)
        changed_prepared = self.strategy.prepare_indicators(changed)
        pd.testing.assert_series_equal(
            original_prepared["regime_bullish"].iloc[:cutoff_index],
            changed_prepared["regime_bullish"].iloc[:cutoff_index],
        )
        kwargs = {
            "starting_capital": 1000.0,
            "timeframe": "4Hour",
            "fee_rate": 0.0,
            "slippage": 0.0,
            "test_start": bars.index[230],
            "strategy": self.strategy,
            "liquidate_at_end": False,
        }
        original_result = backtest.run_backtest(bars, **kwargs)
        changed_result = backtest.run_backtest(changed, **kwargs)
        cutoff = bars.index[cutoff_index]
        original_trades = original_result["trades"]
        changed_trades = changed_result["trades"]
        original_prior = original_trades.loc[original_trades["exit_time"] <= cutoff]
        changed_prior = changed_trades.loc[changed_trades["exit_time"] <= cutoff]
        pd.testing.assert_frame_equal(
            original_prior.reset_index(drop=True), changed_prior.reset_index(drop=True)
        )


class NotionalResearchMetricTests(unittest.TestCase):
    def test_notional_pnl_and_candle_mark_drawdown(self):
        marks = pd.Series(
            [1000.0, 1010.0, 990.0, 1005.0],
            index=pd.date_range("2026-01-01", periods=4, freq="4h", tz="UTC"),
        )
        block = {
            "block_start": marks.index[0].to_pydatetime(),
            "block_end": marks.index[-1].to_pydatetime(),
        }
        result = {"equity_curve": marks}
        block_metrics = {
            "block_net_pnl": 5.0,
            "exits_in_block": 2,
            "charged_costs_in_block": 0.6,
            "time_invested_percent": 50.0,
        }

        metrics = experiment._block_notional_metrics(result, block_metrics, block)

        self.assertEqual(metrics["net_pnl_usd"], 5.0)
        self.assertEqual(metrics["pnl_percent_of_notional"], 25.0)
        self.assertEqual(metrics["max_drawdown_usd"], 20.0)
        self.assertEqual(metrics["max_drawdown_percent_of_notional"], 100.0)
        self.assertEqual(metrics["total_trades"], 2)
        self.assertEqual(metrics["total_costs_usd"], 0.6)
        self.assertEqual(metrics["time_invested_percent"], 50.0)

    def test_total_notional_metrics_use_net_pnl_and_dollar_drawdown(self):
        marks = pd.Series(
            [1000.0, 1010.0, 990.0, 1005.0],
            index=pd.date_range("2026-01-01", periods=4, freq="4h", tz="UTC"),
        )
        result = {
            "starting_capital": 1000.0,
            "ending_equity": 1005.0,
            "strategy_return": 0.5,
            "same_path_zero_cost_return_percent": 0.8,
            "total_trades": 1,
            "total_costs": 0.6,
            "candle_mark_max_drawdown_percent": 2.0,
            "equity_curve": marks,
            "exposure_curve": pd.DataFrame(
                {
                    "time_invested": [False, True, True, False],
                    "capital_exposure_percent": [0.0, 2.0, 2.0, 0.0],
                }
            ),
            "trades": pd.DataFrame(
                [{"requested_notional": 20.0, "net_pnl": 5.0}]
            ),
            "open_position": None,
        }

        metrics = experiment._total_summary(
            result, _bars_from_close([100.0, 101.0]), 0.0, 0.0
        )

        self.assertEqual(metrics["net_pnl_usd"], 5.0)
        self.assertEqual(metrics["pnl_percent_of_notional"], 25.0)
        self.assertEqual(metrics["max_drawdown_usd"], 20.0)
        self.assertEqual(metrics["max_drawdown_percent_of_notional"], 100.0)
        self.assertEqual(metrics["total_trades"], 1)
        self.assertEqual(metrics["total_costs_usd"], 0.6)
        self.assertEqual(metrics["time_invested_percent"], 50.0)

    def test_total_compounding_includes_open_position_marked_after_exit_costs(self):
        bars = _bars_from_close([100.0, 110.0])
        trades = pd.DataFrame([{"requested_notional": 20.0, "net_pnl": 2.0}])
        result = {
            "trades": trades,
            "open_position": {
                "quantity_after_buy_fee": 0.2,
                "requested_notional": 20.0,
            },
        }

        marked_pnl = experiment._marked_open_position_net_pnl(
            result, bars, fee_rate=0.01, slippage_rate=0.02
        )
        compounded = experiment._compounded_trade_return_percent(
            result, bars, fee_rate=0.01, slippage_rate=0.02
        )

        self.assertAlmostEqual(marked_pnl, 1.3444)
        self.assertAlmostEqual(compounded, ((1 + 2 / 20) * (1 + 1.3444 / 20) - 1) * 100)

    def test_buy_and_hold_metrics_use_close_drawdown_and_round_trip_costs(self):
        bars = pd.DataFrame(
            {
                "open": [100.0, 101.0, 119.0, 91.0],
                "high": [101.0, 121.0, 120.0, 111.0],
                "low": [99.0, 100.0, 89.0, 90.0],
                "close": [100.0, 120.0, 90.0, 110.0],
                "volume": [1.0] * 4,
            },
            index=pd.date_range("2026-01-01", periods=4, freq="4h", tz="UTC"),
        )

        metrics = experiment._btc_buy_hold_metrics(
            bars, fee_rate=0.01, slippage_rate=0.02
        )
        quantity_after_fee = 20.0 / (100.0 * 1.02) * 0.99
        expected_after_costs = (
            quantity_after_fee * (110.0 * 0.98) * 0.99 / 20.0 - 1
        ) * 100

        self.assertAlmostEqual(metrics["btc_buy_hold_return_percent"], 10.0)
        self.assertAlmostEqual(metrics["btc_buy_hold_max_drawdown_percent"], 25.0)
        self.assertAlmostEqual(
            metrics["btc_buy_hold_return_after_costs_percent"], expected_after_costs
        )
        self.assertGreater(metrics["btc_buy_hold_costs_usd"], 0.0)


if __name__ == "__main__":
    unittest.main()
