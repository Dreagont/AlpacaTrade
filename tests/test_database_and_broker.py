import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import broker
import config
import database
import main
import trade_config


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
                evaluation_columns = {
                    row[1]
                    for row in connection.execute("PRAGMA table_info(strategy_evaluations)")
                }
            finally:
                connection.close()
            self.assertIn("realized_gross_pnl", columns)
            self.assertIn("order_status", columns)
            self.assertNotIn("realized_pnl", columns)
            self.assertIn("client_order_id", columns)
            self.assertIn("strategy_name", columns)
            self.assertIn("strategy_parameters_json", evaluation_columns)
            self.assertIn("asset_quantity_delta", columns)
            self.assertIn("submission_kind", columns)
            self.assertIn("created_at", columns)
            self.assertIn("reconcile_attempts", columns)
            self.assertIn("last_reconcile_at", columns)
            self.assertIn("position_before_quantity", columns)
            self.assertIn("order_role", columns)

    def test_legacy_gross_quantity_is_not_reliable_net_ownership(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "legacy.sqlite"
            with patch.object(config, "DATABASE_PATH", str(path)):
                database.upsert_order(
                    order_id="old-buy", symbol="BTC/USD", side="BUY",
                    requested_notional=20, quantity=0.0002, fill_price=40000, reason="legacy",
                    order_status="filled",
                )
                details = database.get_bot_owned_btc_quantity_details()
        self.assertFalse(details.reliable)
        self.assertEqual(details.quantity, 0.0)
        self.assertIn("legacy", details.evidence)

    def test_net_asset_deltas_sum_actual_credited_position_changes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "deltas.sqlite"
            with patch.object(config, "DATABASE_PATH", str(path)):
                database.upsert_order(
                    order_id="net-buy", symbol="BTC/USD", side="BUY",
                    requested_notional=20,
                    quantity=0.0002, asset_quantity_delta=0.0001995,
                    fill_price=40000, reason="entry", order_status="filled",
                )
                database.upsert_order(
                    order_id="net-sell", symbol="BTC/USD", side="SELL",
                    requested_notional=20,
                    quantity=0.0001995, asset_quantity_delta=-0.0001995,
                    fill_price=41000, reason="exit", order_status="filled",
                )
                details = database.get_bot_owned_btc_quantity_details()
        self.assertTrue(details.reliable)
        self.assertEqual(details.quantity, 0.0)
        self.assertEqual(details.evidence, "complete_asset_delta_ledger")

    def test_persisted_fee_deducted_position_provenance_survives_restart(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "restart.sqlite"
            position = SimpleNamespace(
                qty="0.0001995", symbol="BTCUSD", asset_id="btc-asset"
            )
            with patch.object(config, "DATABASE_PATH", str(path)):
                database.set_active_bot_position({
                    "asset_id": "btc-asset", "source_order_id": "buy-1",
                    "source_client_order_id": "buy-cid", "credited_quantity": 0.0001995,
                    "strategy_name": "ma_rsi_crossover", "entry_fill_price": 40000,
                    "timestamp": "2026-10-05T00:00:00+00:00", "source_confirmed": True,
                })
                self.assertTrue(database.init_db())
                owned, reason = main.bot_owns_position(position)
                persisted = database.get_active_bot_position()
        self.assertTrue(owned, reason)
        self.assertEqual(persisted["credited_quantity"], 0.0001995)
        self.assertEqual(persisted["source_order_id"], "buy-1")

    def test_existing_duplicate_order_ids_are_consolidated_before_unique_index(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "duplicates.sqlite"
            connection = sqlite3.connect(path)
            try:
                connection.execute(
                    """CREATE TABLE orders (
                    id INTEGER PRIMARY KEY, timestamp TEXT, order_id TEXT,
                    symbol TEXT, side TEXT, requested_notional REAL, quantity REAL,
                    fill_price REAL, reason TEXT, realized_gross_pnl REAL,
                    order_status TEXT)"""
                )
                connection.executemany(
                    "INSERT INTO orders(id, order_id, symbol, side, reason, order_status) "
                    "VALUES (?, ?, 'BTC/USD', 'BUY', 'test', ?)",
                    [(1, "same-order", "timeout_pending"), (2, "same-order", "filled")],
                )
                connection.commit()
            finally:
                connection.close()
            with patch.object(config, "DATABASE_PATH", str(path)):
                database.init_db()
            connection = sqlite3.connect(path)
            try:
                rows = connection.execute(
                    "SELECT id, order_id, order_status FROM orders ORDER BY id"
                ).fetchall()
                indexes = {
                    row[1] for row in connection.execute("PRAGMA index_list(orders)")
                }
            finally:
                connection.close()
        self.assertEqual(rows, [(1, None, "timeout_pending"), (2, "same-order", "filled")])
        self.assertIn("idx_orders_order_id", indexes)


class BrokerOrderTests(unittest.TestCase):
    def test_ambiguous_timeout_keeps_synthetic_order_pending_until_reconciled(self):
        pending = {}
        with (
            patch.object(main, "get_order_by_client_order_id", return_value=None),
            patch.object(main, "buy_btc", side_effect=TimeoutError("response lost")),
            patch.object(main, "upsert_order"),
        ):
            result = main._submit_and_log_buy("entry", pending, "candle-1")
        self.assertEqual(result, broker.OrderOutcome.PENDING)
        self.assertEqual(len(pending), 1)
        context = next(iter(pending.values()))
        self.assertEqual(context.submission_kind, "entry_intent")
        self.assertEqual(context.order_role, "strategy_entry")

    def test_ambiguous_submit_unknown_expires_after_old_confirmed_404_and_restart(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "unknown.sqlite"
            created = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
            client_id = "ambiguous-client-order"
            with patch.object(config, "DATABASE_PATH", str(path)):
                database.upsert_order(
                    order_id=f"client:{client_id}", client_order_id=client_id,
                    symbol="BTC/USD", side="BUY", requested_notional=20,
                    quantity=None, fill_price=None,
                    reason="entry", order_status="submit_unknown",
                    submission_kind="synthetic_ambiguous", created_at=created,
                    reconcile_attempts=0,
                )
                context = main.PendingReconciliation(
                    "BUY", "entry", 20, None, "candle", client_id,
                    "ma_rsi_crossover", "{}", None, "synthetic_ambiguous", created,
                )
                pending = {f"client:{client_id}": context}
                with (
                    patch.object(trade_config, "SUBMIT_UNKNOWN_MAX_AGE_SECONDS", 0),
                    patch.object(trade_config, "SUBMIT_UNKNOWN_MAX_RECONCILE_ATTEMPTS", 2),
                    patch.object(main, "reconcile_order", side_effect=broker.BrokerOrderNotFound("404")),
                ):
                    main.reconcile_pending_orders(pending)
                    self.assertIn(f"client:{client_id}", pending)
                    main.reconcile_pending_orders(pending)
                self.assertEqual(pending, {})
                self.assertEqual(database.get_order_records(pending_only=True), [])

                restored = {}
                with (
                    patch.object(main, "get_open_btc_orders", return_value=[]),
                    patch.object(main, "reconcile_order") as reconcile,
                ):
                    main.restore_pending_orders(restored)
                self.assertEqual(restored, {})
                reconcile.assert_not_called()

    def test_nonterminal_order_timeout_is_not_returned_as_reconciled(self):
        pending = SimpleNamespace(id="order-1", status=SimpleNamespace(value="new"))
        with patch.object(broker.client, "get_order_by_id", return_value=pending):
            with self.assertRaises(broker.OrderFillTimeoutError) as context:
                broker.wait_for_order_fill("order-1", timeout_seconds=0)
        self.assertIs(context.exception.order, pending)
        self.assertEqual(context.exception.outcome, broker.OrderOutcome.TIMEOUT_PENDING)

    def test_order_outcomes_distinguish_terminal_and_partial_states(self):
        def order(status, filled_qty=0):
            return SimpleNamespace(
                status=SimpleNamespace(value=status), filled_qty=filled_qty
            )

        self.assertEqual(
            broker.classify_order(order("filled", "0.1")),
            broker.OrderOutcome.FILLED,
        )
        for status in ("rejected", "canceled"):
            self.assertEqual(
                broker.classify_order(order(status)),
                broker.OrderOutcome.TERMINAL_NOT_FILLED,
            )
        self.assertEqual(
            broker.classify_order(order("partially_filled", "0.02")),
            broker.OrderOutcome.PARTIAL_PENDING,
        )
        self.assertEqual(
            broker.classify_order(order("new"), timed_out=True),
            broker.OrderOutcome.TIMEOUT_PENDING,
        )

    def test_pending_timeout_logs_no_fabricated_fill_or_realized_pnl(self):
        pending = SimpleNamespace(
            id="order-pending",
            status=SimpleNamespace(value="new"),
            filled_qty=None,
            filled_avg_price=None,
        )
        with patch.object(main, "upsert_order") as write_log:
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

    def test_partial_order_has_no_realized_gross_pnl_until_fully_filled(self):
        partial = SimpleNamespace(
            id="partial",
            status=SimpleNamespace(value="partially_filled"),
            filled_qty="0.02",
            filled_avg_price="100",
        )
        with patch.object(main, "upsert_order") as write_log:
            main._record_order(
                partial,
                "SELL",
                "test",
                20,
                position=SimpleNamespace(avg_entry_price=90),
                order_status=broker.OrderOutcome.PARTIAL_PENDING.value,
            )
        logged = write_log.call_args.kwargs
        self.assertEqual(logged["quantity"], 0.02)
        self.assertEqual(logged["fill_price"], 100.0)
        self.assertIsNone(logged["realized_gross_pnl"])

    def test_realized_gross_pnl_only_for_filled_sell(self):
        position = SimpleNamespace(avg_entry_price=100, market_value=10)
        rejected = SimpleNamespace(
            id="rejected",
            status=SimpleNamespace(value="rejected"),
            filled_qty=None,
            filled_avg_price=None,
        )
        filled = SimpleNamespace(
            id="filled",
            status=SimpleNamespace(value="filled"),
            filled_qty="0.1",
            filled_avg_price="110",
        )
        with patch.object(main, "upsert_order") as write_log:
            main._record_order(rejected, "SELL", "test", 10, position)
            self.assertIsNone(write_log.call_args.kwargs["realized_gross_pnl"])
            main._record_order(filled, "SELL", "test", 10, position)
            self.assertAlmostEqual(
                write_log.call_args.kwargs["realized_gross_pnl"], 1.0
            )

    def test_reconciliation_updates_and_removes_terminal_order(self):
        candle = "candle-1"
        pending = {
            "order-1": main.PendingReconciliation(
                "BUY", "test", 20, None, candle
            )
        }
        rejected = SimpleNamespace(
            id="order-1",
            status=SimpleNamespace(value="rejected"),
            filled_qty=None,
            filled_avg_price=None,
        )
        with patch.object(main, "reconcile_order", return_value=rejected), patch.object(
            main, "upsert_order"
        ) as write_log:
            retry_candles = main.reconcile_pending_orders(pending)
        self.assertEqual(retry_candles, [])
        self.assertEqual(pending, {})
        self.assertEqual(write_log.call_args.kwargs["order_status"], "rejected")

    def test_reconciliation_of_filled_order_does_not_retry_candle(self):
        pending = {
            "order-2": main.PendingReconciliation(
                "BUY", "test", 20, None, "candle-2"
            )
        }
        filled = SimpleNamespace(
            id="order-2",
            status=SimpleNamespace(value="filled"),
            filled_qty="0.2",
            filled_avg_price="100",
        )
        with patch.object(main, "reconcile_order", return_value=filled), patch.object(
            main, "upsert_order"
        ) as write_log:
            retry_candles = main.reconcile_pending_orders(pending)
        self.assertEqual(retry_candles, [])
        self.assertEqual(pending, {})
        self.assertEqual(write_log.call_args.kwargs["order_status"], "filled")

    def test_partial_reconciliation_stays_pending_and_logs_actual_fill(self):
        pending = {
            "order-3": main.PendingReconciliation(
                "SELL",
                "test",
                20,
                SimpleNamespace(avg_entry_price=90, market_value=20),
                "candle-3",
            )
        }
        partial = SimpleNamespace(
            id="order-3",
            status=SimpleNamespace(value="partially_filled"),
            filled_qty="0.05",
            filled_avg_price="100",
        )
        with patch.object(main, "reconcile_order", return_value=partial), patch.object(
            main, "upsert_order"
        ) as write_log:
            retry_candles = main.reconcile_pending_orders(pending)
        self.assertEqual(retry_candles, [])
        self.assertIn("order-3", pending)
        logged = write_log.call_args.kwargs
        self.assertEqual(logged["order_status"], "partial_pending")
        self.assertEqual(logged["quantity"], 0.05)
        self.assertIsNone(logged["realized_gross_pnl"])

    def test_terminal_not_filled_action_keeps_candle_processed(self):
        self.assertEqual(
            main.processed_candle_after_order(
                "candle", broker.OrderOutcome.TERMINAL_NOT_FILLED
            ), "candle"
        )
        self.assertEqual(
            main.processed_candle_after_order("candle", broker.OrderOutcome.TIMEOUT_PENDING),
            "candle",
        )

    def test_repeated_order_id_upserts_one_database_row(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "trading.sqlite"
            with patch.object(config, "DATABASE_PATH", str(path)):
                common = dict(
                    order_id="same-id",
                    symbol="BTC/USD",
                    side="BUY",
                    requested_notional=20,
                    quantity=None,
                    fill_price=None,
                    reason="test",
                    realized_gross_pnl=None,
                )
                database.upsert_order(**common, order_status="timeout_pending")
                database.upsert_order(
                    **{**common, "quantity": 0.2, "fill_price": 100},
                    order_status="filled",
                )
                connection = sqlite3.connect(path)
                try:
                    rows = connection.execute(
                        "SELECT order_status, quantity, fill_price FROM orders WHERE order_id=?",
                        ("same-id",),
                    ).fetchall()
                finally:
                    connection.close()
        self.assertEqual(rows, [("filled", 0.2, 100.0)])

    def test_client_order_id_upsert_persists_and_updates_one_database_row(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "trading.sqlite"
            with patch.object(config, "DATABASE_PATH", str(path)):
                common = dict(
                    order_id="broker-id",
                    client_order_id="bot-deterministic-id",
                    symbol="BTC/USD",
                    side="BUY",
                    requested_notional=20,
                    quantity=None,
                    fill_price=None,
                    reason="entry",
                    strategy_name="ma_rsi_crossover",
                    strategy_parameters_json='{"fast_ma":10}',
                )
                database.upsert_order(**common, order_status="pending")
                database.upsert_order(
                    **{**common, "quantity": 0.0005, "fill_price": 40000},
                    order_status="filled",
                )
                connection = sqlite3.connect(path)
                try:
                    rows = connection.execute(
                        "SELECT client_order_id, order_status, quantity, fill_price "
                        "FROM orders"
                    ).fetchall()
                finally:
                    connection.close()
        self.assertEqual(
            rows, [("bot-deterministic-id", "filled", 0.0005, 40000.0)]
        )


if __name__ == "__main__":
    unittest.main()
