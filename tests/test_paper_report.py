import csv
import hashlib
import io
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import backtest
import config
import paper_report
from test_fee_profiles import history


def order(identity, side, *, quantity=None, price=None, delta=None, timestamp=None, **extra):
    is_buy = side == "BUY"
    row = {
        "id": identity, "symbol": "BTC/USD", "side": side, "order_status": "filled",
        "timestamp": timestamp or f"2026-01-01T{identity * 4:02d}:05:00+00:00",
        "fill_price": price if price is not None else 100 if is_buy else 120,
        "quantity": quantity if quantity is not None else 1 if is_buy else 0.9975,
        "requested_notional": 100 if is_buy else None,
        "asset_quantity_delta": delta if delta is not None else 0.9975 if is_buy else -0.9975,
        "strategy_name": "regime_only_4h", "order_role": "strategy_entry" if is_buy else "strategy_exit",
    }
    row.update(extra)
    return row


def make_db(path, orders):
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY, symbol TEXT, side TEXT, "
                           "order_status TEXT, timestamp TEXT, fill_price REAL, quantity REAL, "
                           "requested_notional REAL, asset_quantity_delta REAL, strategy_name TEXT, "
                           "order_role TEXT, order_id TEXT, client_order_id TEXT)")
        for row in orders:
            keys = list(row)
            connection.execute(f"INSERT INTO orders ({','.join(keys)}) VALUES ({','.join('?' for _ in keys)})",
                               [row[key] for key in keys])


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class PaperReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.db = Path(self.temp.name) / "paper.db"

    def report(self, orders, **kwargs):
        make_db(self.db, orders)
        console = io.StringIO()
        with redirect_stdout(console):
            rows = paper_report.run_paper_report(db_path=self.db, **kwargs)
        return rows, console.getvalue()

    def test_pairing_and_hand_calculated_repricing_every_profile(self):
        rows, console = self.report([order(1, "BUY"), order(2, "SELL")])
        trips = [row for row in rows if row["row_type"] == "ROUND_TRIP"]
        self.assertEqual(len(trips), len(config.FEE_PROFILES))
        for row in trips:
            fee = config.get_fee_profile(row["fee_profile"])["fee_rate"]
            self.assertAlmostEqual(row["actual_recorded_pnl_usd"], 19.7)
            self.assertIsNone(row["actual_net_pnl_usd"])
            self.assertAlmostEqual(row["alpaca_net_estimate_usd"], 19.40075)
            self.assertAlmostEqual(row["repriced_pnl_usd"], 120 * (1 - fee) ** 2 - 100)
            self.assertEqual(row["applied_slippage_rate"], 0)
        self.assertIn("actual net PnL is unavailable", console)
        self.assertIn("applied slippage is zero", console)
        self.assertIn("fee_profile=binance_spot_bnb", console)
        totals = [row for row in rows if row["row_type"] == "TOTAL_CLOSED_AND_MARKED"]
        for total in totals:
            trip = next(row for row in trips if row["fee_profile"] == total["fee_profile"])
            self.assertAlmostEqual(total["repriced_pnl_usd"], trip["repriced_pnl_usd"])

    def test_fill_notional_is_actual_debit_requested_budget_preserved(self):
        rows, _ = self.report([order(1, "BUY", requested_notional=101), order(2, "SELL")])
        row = rows[0]
        self.assertEqual(row["requested_notional"], 101)
        self.assertEqual(row["actual_entry_debit_usd"], 100)
        self.assertAlmostEqual(row["actual_recorded_pnl_usd"], 19.7)

    def test_partial_sells_combined_fifo_and_repricing_math(self):
        orders = [order(1, "BUY"), order(2, "SELL", quantity=0.5, delta=-0.5, price=110),
                  order(3, "SELL", quantity=0.4975, delta=-0.4975, price=120)]
        pairs, opened, issues = paper_report.pair_orders(orders)
        self.assertIsNone(opened)
        self.assertFalse(issues)
        self.assertEqual(len(pairs), 1)
        rows, _ = self.report(orders)
        row = next(row for row in rows if row["row_type"] == "ROUND_TRIP" and row["fee_profile"] == "zero_cost")
        self.assertEqual(row["sell_order_ids"], "2,3")
        self.assertAlmostEqual(row["actual_recorded_pnl_usd"], 14.7)
        self.assertAlmostEqual(row["repriced_pnl_usd"], 114.7 / 0.9975 - 100)

    def test_unpaired_sell_and_overlapping_buys_reported(self):
        orders = [order(1, "SELL"), order(2, "BUY"), order(3, "BUY"), order(4, "SELL")]
        rows, console = self.report(orders, no_mark=True)
        self.assertFalse(any(row["row_type"] == "ROUND_TRIP" for row in rows))
        self.assertIn("unpaired SELL", console)
        self.assertIn("overlapping BUY", console)
        self.assertIn("ambiguous ownership", console)
        self.assertGreaterEqual(len([row for row in rows if row["row_type"] == "ISSUE"]), 4)

    def test_missing_delta_taints_ledger_instead_of_guessing(self):
        buy = order(1, "BUY")
        buy["asset_quantity_delta"] = None
        rows, console = self.report([buy, order(2, "SELL")], no_mark=True)
        self.assertIn("ambiguous", console)
        self.assertFalse(any(row["row_type"] == "ROUND_TRIP" for row in rows))

    def test_open_position_explicit_mark_math(self):
        rows, console = self.report([order(1, "BUY")], mark_price=120)
        marked = [row for row in rows if row["row_type"] == "OPEN_MARKED"]
        self.assertEqual(len(marked), len(config.FEE_PROFILES))
        self.assertIn("MARKED", console)
        for row in marked:
            fee = row["fee_rate"]
            self.assertAlmostEqual(row["actual_recorded_pnl_usd"], 19.7)
            self.assertAlmostEqual(row["repriced_marked_pnl_usd"], 120 * (1 - fee) ** 2 - 100)
            self.assertEqual(row["repriced_realized_pnl_usd"], 0)

    def test_open_position_latest_price_read_is_mocked(self):
        with patch.object(paper_report, "fetch_latest_price", return_value=120) as latest:
            rows, _ = self.report([order(1, "BUY")])
        latest.assert_called_once_with()
        self.assertEqual(rows[0]["mark_price"], 120)

    def test_no_mark_never_fetches_and_leaves_open_values_unknown(self):
        with patch.object(paper_report, "fetch_latest_price") as latest:
            rows, console = self.report([order(1, "BUY")], no_mark=True)
        latest.assert_not_called()
        self.assertIn("mark skipped", console)
        for row in rows:
            if row["row_type"] == "OPEN_UNMARKED":
                self.assertIsNone(row["mark_price"])
                self.assertIsNone(row["actual_recorded_pnl_usd"])
                self.assertIsNone(row["repriced_pnl_usd"])

    def test_partial_open_position_realized_and_marked_total(self):
        orders = [order(1, "BUY"), order(2, "SELL", quantity=0.5, delta=-0.5, price=110)]
        rows, _ = self.report(orders, mark_price=120)
        row = next(row for row in rows if row["row_type"] == "OPEN_MARKED" and row["fee_profile"] == "zero_cost")
        self.assertAlmostEqual(row["remaining_quantity"], 0.4975)
        self.assertAlmostEqual(row["actual_recorded_pnl_usd"], 14.7)
        self.assertAlmostEqual(row["repriced_pnl_usd"], 114.7 / 0.9975 - 100)
        self.assertAlmostEqual(row["repriced_realized_pnl_usd"] + row["repriced_marked_pnl_usd"], row["repriced_pnl_usd"])

    def test_non_btc_non_filled_orders_excluded_pending_fill_flagged(self):
        pending = order(3, "BUY", order_status="new", quantity=0, asset_quantity_delta=None)
        rows, _ = self.report([order(1, "BUY"), order(2, "SELL"), pending,
                               order(4, "BUY", symbol="ETH/USD")], no_mark=True)
        self.assertEqual(len([row for row in rows if row["row_type"] == "ROUND_TRIP"]), 4)
        self.assertFalse(any(row["row_type"] == "ISSUE" for row in rows))
        pairs, opened, issues = paper_report.pair_orders([order(1, "BUY", order_status="partial_pending")])
        self.assertFalse(pairs)
        self.assertIsNone(opened)
        self.assertTrue(issues)

    def test_bad_sell_delta_provenance_and_excess_quantity_flagged(self):
        bad_sells = [order(2, "SELL", asset_quantity_delta=None), order(2, "SELL", strategy_name="other"),
                     order(2, "SELL", order_role="unknown"), order(2, "SELL", quantity=2, delta=-2)]
        for sell in bad_sells:
            with self.subTest(sell=sell):
                pairs, opened, issues = paper_report.pair_orders([order(1, "BUY"), sell])
                self.assertFalse(pairs)
                self.assertIsNone(opened)
                self.assertTrue(issues)

    def test_tied_timestamps_duplicates_and_invalid_price_not_paired(self):
        scenarios = [
            [order(1, "BUY"), order(2, "SELL", timestamp=order(1, "BUY")["timestamp"])],
            [order(1, "BUY", order_id="duplicate"), order(2, "SELL", order_id="duplicate")],
            [order(1, "BUY", fill_price=float("nan")), order(2, "SELL")],
            [order(1, "BUY", timestamp="bad"), order(2, "SELL")],
        ]
        for orders in scenarios:
            pairs, opened, issues = paper_report.pair_orders(orders)
            self.assertFalse(pairs)
            self.assertIsNone(opened)
            self.assertTrue(issues)

    def test_uri_readonly_queryonly_and_byte_identical_db(self):
        make_db(self.db, [order(1, "BUY"), order(2, "SELL")])
        before = file_hash(self.db)
        original_connect = sqlite3.connect
        connections = []
        def connect(database_uri, **kwargs):
            self.assertIn("mode=ro", database_uri)
            self.assertTrue(kwargs["uri"])
            connection = original_connect(database_uri, **kwargs)
            statements = []
            connection.set_trace_callback(statements.append)
            connections.append(statements)
            return connection
        csv_path = Path(self.temp.name) / "paper_report.csv"
        with patch.object(paper_report.sqlite3, "connect", side_effect=connect), redirect_stdout(io.StringIO()):
            paper_report.run_paper_report(db_path=self.db, output=csv_path)
        self.assertEqual(before, file_hash(self.db))
        self.assertTrue(any("PRAGMA query_only=ON" in line for line in connections[0]))
        self.assertFalse(any(line.startswith(("INSERT", "UPDATE", "DELETE", "CREATE", "ALTER")) for line in connections[0]))
        with csv_path.open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 8)
        self.assertTrue(all(row["fee_profile"] in config.FEE_PROFILES for row in rows))

    def test_db_hash_unchanged_when_marking_skipping_and_comparing(self):
        make_db(self.db, [order(1, "BUY")])
        before = file_hash(self.db)
        with patch.object(paper_report, "compare_backtest", return_value=[]), redirect_stdout(io.StringIO()):
            paper_report.run_paper_report(db_path=self.db, mark_price=120, compare=True)
            paper_report.run_paper_report(db_path=self.db, no_mark=True)
        self.assertEqual(before, file_hash(self.db))

    def test_csv_refuses_database_hardlink_symlink_and_sidecar(self):
        make_db(self.db, [order(1, "BUY")])
        symlink = Path(self.temp.name) / "link.csv"
        symlink.symlink_to(self.db)
        hardlink = Path(self.temp.name) / "hard.csv"
        os.link(self.db, hardlink)
        before = file_hash(self.db)
        for output in (self.db, symlink, hardlink, str(self.db) + "-wal", config.DATABASE_PATH):
            with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "database"):
                paper_report.run_paper_report(db_path=self.db, no_mark=True, output=output)
        self.assertEqual(before, file_hash(self.db))

    def test_missing_database_not_created_and_schema_not_migrated(self):
        with self.assertRaises(sqlite3.OperationalError):
            paper_report.read_orders(self.db)
        self.assertFalse(self.db.exists())
        with sqlite3.connect(self.db) as connection:
            connection.execute("CREATE TABLE orders (id INTEGER)")
        before = file_hash(self.db)
        with self.assertRaisesRegex(ValueError, "no migration"):
            paper_report.read_orders(self.db)
        self.assertEqual(before, file_hash(self.db))

    def test_import_and_compare_never_import_live_modules(self):
        # Fresh subprocess isolates the assertion from other tests importing main.
        code = "import paper_report; paper_report.compare_backtest([], fee_profile='alpaca'); import sys; assert not {'main','broker','database'} & set(sys.modules)"
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_cli_marking_and_optional_csv_arguments(self):
        with patch.object(paper_report, "run_paper_report") as run:
            self.assertEqual(paper_report.main(["--db", "test.db", "--no-mark", "--fee-profile", "alpaca", "--compare-backtest", "--csv"]), 0)
        self.assertEqual(run.call_args.kwargs["output"], "paper_report.csv")
        self.assertTrue(run.call_args.kwargs["no_mark"])
        self.assertTrue(run.call_args.kwargs["compare"])
        self.assertEqual(run.call_args.kwargs["fee_profile"], "alpaca")
        with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
            paper_report._build_parser().parse_args(["--no-mark", "--mark-price", "120"])

    def test_compare_backtest_matching_and_mismatches_selected_profile(self):
        orders = [order(1, "BUY"), order(2, "SELL")]
        simulated = {
            "entry_events": pd.DataFrame({"entry_time": [pd.Timestamp("2026-01-01T04:00:00Z")]}),
            "trades": pd.DataFrame({"exit_time": [pd.Timestamp("2026-01-01T08:00:00Z")]}),
        }
        with patch.object(backtest, "fetch_history", side_effect=history) as fetch, \
             patch.object(backtest, "run_backtest", return_value=simulated) as run:
            rows = paper_report.compare_backtest(orders, fee_profile="alpaca", end_time=datetime(2026, 1, 1, 12, tzinfo=timezone.utc))
        self.assertEqual([row["status"] for row in rows], ["MATCH", "MATCH"])
        self.assertEqual(fetch.call_args.args[1], "4Hour")
        self.assertEqual(run.call_args.kwargs["strategy"].name, "regime_only_4h")
        self.assertEqual(run.call_args.kwargs["fee_rate"], 0.0025)
        self.assertFalse(run.call_args.kwargs["liquidate_at_end"])
        simulated["trades"]["exit_time"] = [pd.Timestamp("2026-01-01T00:00:00Z")]
        with patch.object(backtest, "fetch_history", side_effect=history), \
             patch.object(backtest, "run_backtest", return_value=simulated):
            rows = paper_report.compare_backtest(orders, fee_profile="zero_cost", end_time=datetime(2026, 1, 1, 12, tzinfo=timezone.utc))
        self.assertIn("PAPER_ONLY", [row["status"] for row in rows])

    def test_compare_ambiguous_candidates_flagged_not_arbitrarily_matched(self):
        orders = [order(1, "BUY"), order(2, "BUY", timestamp="2026-01-01T04:50:00Z")]
        result = {"entry_events": pd.DataFrame({"entry_time": [pd.Timestamp("2026-01-01T04:00:00Z")]}), "trades": pd.DataFrame()}
        with patch.object(backtest, "fetch_history", side_effect=history), \
             patch.object(backtest, "run_backtest", return_value=result):
            rows = paper_report.compare_backtest(orders, fee_profile="alpaca", end_time=datetime(2026, 1, 1, 12, tzinfo=timezone.utc))
        self.assertEqual([row["status"] for row in rows], ["AMBIGUOUS_MATCH", "AMBIGUOUS_MATCH", "BACKTEST_ONLY"])

    def test_compare_flags_strategy_mismatch(self):
        orders = [order(1, "BUY", strategy_name="other_strategy")]
        result = {"entry_events": pd.DataFrame({"entry_time": [pd.Timestamp("2026-01-01T04:00:00Z")]}), "trades": pd.DataFrame()}
        with patch.object(backtest, "fetch_history", side_effect=history), \
             patch.object(backtest, "run_backtest", return_value=result):
            rows = paper_report.compare_backtest(orders, fee_profile="alpaca", end_time=datetime(2026, 1, 1, 12, tzinfo=timezone.utc))
        self.assertEqual(rows[0]["status"], "STRATEGY_MISMATCH")
        self.assertEqual(rows[0]["paper_fill_price"], 100)

    def test_closed_ledger_comparison_ends_at_last_fill_candle_close(self):
        make_db(self.db, [order(1, "BUY"), order(2, "SELL")])
        with patch.object(paper_report, "compare_backtest", return_value=[]) as compare, redirect_stdout(io.StringIO()):
            paper_report.run_paper_report(db_path=self.db, compare=True)
        self.assertEqual(compare.call_args.kwargs["end_time"], pd.Timestamp("2026-01-01T12:00:00Z"))

    def test_actual_comparison_does_not_import_live_modules(self):
        # Exercise the nonempty comparison path in a fresh interpreter.
        code = """
import sys
from unittest.mock import patch
import pandas as pd
import paper_report
import backtest
orders = [{"id": 1, "side": "BUY", "order_status": "filled", "timestamp": "2026-01-01T04:05:00Z", "strategy_name": "regime_only_4h"}]
index = pd.date_range("2025-01-01T00:00:00Z", "2026-01-01T08:00:00Z", freq="4h")
bars = pd.DataFrame({"open": 100, "high": 101, "low": 99, "close": 100, "volume": 1}, index=index)
result = {"entry_events": pd.DataFrame(), "trades": pd.DataFrame()}
with patch.object(backtest, "fetch_history", return_value=bars), patch.object(backtest, "run_backtest", return_value=result):
    paper_report.compare_backtest(orders, fee_profile="alpaca", end_time=pd.Timestamp("2026-01-01T12:00:00Z"))
assert not {"main", "broker", "database"} & set(sys.modules)
"""
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
