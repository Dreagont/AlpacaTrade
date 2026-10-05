import csv
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import config
import research
import trade_config


def history_frame(end_time):
    test_start = end_time - timedelta(days=2)
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
        self.assertIn("Failed timeframes: 15Min", output)

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
