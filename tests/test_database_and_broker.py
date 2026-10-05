import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import broker
import config
import database
import main


class DatabaseMigrationTests(unittest.TestCase):
    def test_old_realized_pnl_column_is_renamed_and_status_added(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "trading.sqlite"
            connection = sqlite3.connect(path)
            try:
                connection.execute(
                    """CREATE TABLE orders (
                    id INTEGER PRIMARY KEY, timestamp TEXT, order_id TEXT,
                    symbol TEXT, side TEXT, requested_notional REAL, quantity REAL,
                    fill_price REAL, reason TEXT, realized_pnl REAL)"""
                )
                connection.commit()
            finally:
                connection.close()
            with patch.object(config, "DATABASE_PATH", str(path)):
                database.init_db()
            connection = sqlite3.connect(path)
            try:
                columns = {row[1] for row in connection.execute("PRAGMA table_info(orders)")}
            finally:
                connection.close()
            self.assertIn("realized_gross_pnl", columns)
            self.assertIn("order_status", columns)
            self.assertNotIn("realized_pnl", columns)


class BrokerOrderTests(unittest.TestCase):
    def test_nonterminal_order_timeout_is_not_returned_as_reconciled(self):
        pending = SimpleNamespace(id="order-1", status=SimpleNamespace(value="new"))
        with patch.object(broker.client, "get_order_by_id", return_value=pending):
            with self.assertRaises(broker.OrderFillTimeoutError) as context:
                broker.wait_for_order_fill("order-1", timeout_seconds=0)
        self.assertIs(context.exception.order, pending)

    def test_pending_timeout_logs_no_fabricated_fill_or_realized_pnl(self):
        pending = SimpleNamespace(
            id="order-pending",
            status=SimpleNamespace(value="new"),
            filled_qty=None,
            filled_avg_price=None,
        )
        with patch.object(main, "log_order") as write_log:
            main._record_order(
                pending,
                "SELL",
                "stop_loss",
                20,
                position=SimpleNamespace(avg_entry_price=100),
                order_status="timeout_pending",
            )
        logged = write_log.call_args.kwargs
        self.assertIsNone(logged["quantity"])
        self.assertIsNone(logged["fill_price"])
        self.assertIsNone(logged["realized_gross_pnl"])
        self.assertEqual(logged["order_status"], "timeout_pending")


if __name__ == "__main__":
    unittest.main()
