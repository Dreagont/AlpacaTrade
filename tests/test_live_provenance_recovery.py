import io
import json
import tempfile
import unittest
import uuid
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

import broker
import config
import database
import main
import trade_config
from strategy import REGIME_ONLY_4H


class LiveProvenanceRecoveryTests(unittest.TestCase):
    quantity = 0.000228726
    order_id = uuid.UUID("f65a9129-6cd4-43a3-add6-89509951e704")
    candle = "2026-10-05T00:00:00+00:00"

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        database_path = patch.object(
            config, "DATABASE_PATH", str(Path(temporary.name) / "recovery.sqlite")
        )
        database_path.start()
        self.addCleanup(database_path.stop)
        self.position = SimpleNamespace(
            qty=str(self.quantity), symbol="BTCUSD", asset_id=uuid.uuid4(),
            avg_entry_price="87222", market_value="19.95", unrealized_pl="0",
        )
        with patch.object(trade_config, "LIVE_TIMEFRAME", "4Hour"):
            self.client_id = main._strategy_order_id(
                REGIME_ONLY_4H, self.candle, "BUY", role="strategy_entry"
            )

    def _order(self, *, order_id=None, client_id=None):
        return SimpleNamespace(
            id=self.order_id if order_id is None else order_id,
            client_order_id=self.client_id if client_id is None else client_id,
            status=SimpleNamespace(value="filled"), side=SimpleNamespace(value="buy"),
            symbol="BTC/USD", filled_qty=str(self.quantity / 0.9975),
            filled_avg_price="87222",
        )

    def _seed_buy(self, **overrides):
        values = dict(
            order_id=str(self.order_id), client_order_id=self.client_id,
            symbol="BTC/USD", side="BUY", requested_notional=20,
            quantity=self.quantity / 0.9975, fill_price=87222, reason="regime_on",
            order_status="filled", strategy_name="regime_only_4h",
            asset_quantity_delta=self.quantity, position_before_quantity=0,
            order_role="strategy_entry", candle_timestamp=self.candle,
        )
        values.update(overrides)
        database.upsert_order(**values)

    def _seed_ambiguous_buys(self, second_strategy="regime_only_4h"):
        self._seed_buy(
            quantity=self.quantity / 2, asset_quantity_delta=self.quantity / 2
        )
        self._seed_buy(
            order_id="second-buy", client_order_id="bot-second-buy",
            quantity=self.quantity / 2, asset_quantity_delta=self.quantity / 2,
            position_before_quantity=self.quantity / 2, strategy_name=second_strategy,
        )

    def _run_mocked_restart(self, live_strategy="regime_only_4h"):
        bars = pd.DataFrame(
            {"close": np.arange(87000.0, 87260.0)},
            index=pd.date_range("2026-01-01", periods=260, freq="4h", tz="UTC"),
        )
        output = io.StringIO()
        with (
            patch.object(trade_config, "LIVE_STRATEGY", live_strategy),
            patch.object(
                trade_config, "LIVE_TIMEFRAME",
                "4Hour" if live_strategy == "regime_only_4h" else "5Min",
            ),
            patch.object(main.client, "get_account", return_value=SimpleNamespace()),
            patch.object(main, "restore_pending_orders"),
            patch.object(main, "reconcile_pending_orders"),
            patch.object(main, "lookup_btc_position", return_value=broker.PositionLookup(
                broker.PositionLookupStatus.CONFIRMED_POSITION, self.position
            )),
            patch.object(main, "get_btc_market_price", return_value=87222),
            patch.object(main, "get_btc_bars", return_value=bars),
            patch.object(main, "_record_evaluation"),
            patch.object(main, "_load_risk_exit_state", return_value=None),
            patch.object(main, "buy_btc") as buy,
            patch.object(main, "sell_btc") as sell,
            patch.object(main, "_submit_and_log_buy") as submit_buy,
            patch.object(main, "_submit_and_log_sell") as submit_sell,
            patch.object(main.time, "sleep", side_effect=KeyboardInterrupt),
            redirect_stdout(output),
        ):
            try:
                main.run()
            except KeyboardInterrupt:
                pass
        buy.assert_not_called()
        sell.assert_not_called()
        submit_buy.assert_not_called()
        submit_sell.assert_not_called()
        return output.getvalue()

    def test_quantified_uuid_identifiers_are_normalized_before_persistence(self):
        runtime = main.LiveRuntime()
        client_uuid = uuid.uuid4()
        with patch.object(
            database, "set_active_bot_position", wraps=database.set_active_bot_position
        ) as persist:
            record = main._set_active_position(
                runtime, self.position, source_order=self._order(client_id=client_uuid),
                strategy=REGIME_ONLY_4H,
            )
        self.assertEqual(record["source_order_id"], str(self.order_id))
        self.assertEqual(record["source_client_order_id"], str(client_uuid))
        self.assertEqual(record["asset_id"], str(self.position.asset_id))
        json.dumps(record)
        persist.assert_called_once_with(record)
        self.assertEqual(database.get_active_bot_position(), record)
        self.assertFalse(runtime.accounting_degraded)

    def test_unquantified_uuid_identifiers_are_normalized_before_persistence(self):
        runtime = main.LiveRuntime()
        client_uuid = uuid.uuid4()
        with patch.object(
            database, "set_active_bot_position", wraps=database.set_active_bot_position
        ) as persist:
            main._record_unquantified_bot_position(
                runtime, self._order(client_id=client_uuid), REGIME_ONLY_4H
            )
        self.assertEqual(runtime.active_position["source_order_id"], str(self.order_id))
        self.assertEqual(runtime.active_position["source_client_order_id"], str(client_uuid))
        json.dumps(runtime.active_position)
        persist.assert_called_once_with(runtime.active_position)
        self.assertEqual(database.get_active_bot_position(), runtime.active_position)
        self.assertTrue(runtime.entries_disabled)
        self.assertFalse(runtime.accounting_degraded)

    def test_string_and_none_identifiers_remain_unchanged(self):
        for order_id, client_id in (("original-order", "bot-original-client"), (None, None)):
            with self.subTest(order_id=order_id):
                order = SimpleNamespace(id=order_id, client_order_id=client_id)
                runtime = main.LiveRuntime()
                record = main._set_active_position(
                    runtime, self.position, source_order=order, strategy=REGIME_ONLY_4H
                )
                self.assertEqual(record["source_order_id"], order_id)
                self.assertEqual(record["source_client_order_id"], client_id)
                main._record_unquantified_bot_position(runtime, order, REGIME_ONLY_4H)
                self.assertEqual(runtime.active_position["source_order_id"], order_id)
                self.assertEqual(runtime.active_position["source_client_order_id"], client_id)
                self.assertEqual(database.get_active_bot_position(), runtime.active_position)

    def test_fallback_identifiers_from_in_memory_provenance_are_normalized(self):
        client_uuid = uuid.uuid4()
        runtime = main.LiveRuntime(active_position={
            "source_order_id": self.order_id, "source_client_order_id": client_uuid,
        })
        record = main._set_active_position(runtime, self.position)
        self.assertEqual(record["source_order_id"], str(self.order_id))
        self.assertEqual(record["source_client_order_id"], str(client_uuid))
        json.dumps(record)

    def test_real_paper_quantity_is_owned_from_complete_filled_ledger(self):
        self._seed_buy()
        self.assertIsNone(database.get_active_bot_position())
        owned, reason = main.bot_owns_position(self.position, main.LiveRuntime())
        self.assertTrue(owned, reason)
        self.assertEqual(
            database.get_bot_owned_btc_quantity_details().evidence,
            "complete_asset_delta_ledger",
        )

    def test_unambiguous_ledger_reconstructs_original_source_and_strategy(self):
        self._seed_buy()
        runtime = main.LiveRuntime()
        self.assertTrue(main._reconstruct_active_position_from_ledger(self.position, runtime))
        record = database.get_active_bot_position()
        self.assertEqual(record, runtime.active_position)
        self.assertEqual(record["asset_id"], str(self.position.asset_id))
        self.assertEqual(record["source_order_id"], str(self.order_id))
        self.assertEqual(record["source_client_order_id"], self.client_id)
        self.assertEqual(record["credited_quantity"], self.quantity)
        self.assertEqual(record["strategy_name"], "regime_only_4h")
        self.assertEqual(record["entry_fill_price"], 87222)
        self.assertIs(record["source_confirmed"], True)
        self.assertTrue(main.bot_owns_position(self.position, runtime)[0])

    def test_two_source_buys_do_not_reconstruct(self):
        self._seed_ambiguous_buys()
        runtime = main.LiveRuntime()
        self.assertTrue(main.bot_owns_position(self.position, runtime)[0])
        self.assertFalse(main._reconstruct_active_position_from_ledger(self.position, runtime))
        self.assertIsNone(runtime.active_position)
        self.assertIsNone(database.get_active_bot_position())

    def test_prior_closed_episode_does_not_change_current_source_strategy(self):
        self._seed_buy(
            order_id="prior-buy", client_order_id="bot-prior-buy",
            quantity=0.001, asset_quantity_delta=0.001, strategy_name="ma_rsi_crossover",
        )
        self._seed_buy(
            order_id="prior-sell", client_order_id="bot-prior-sell", side="SELL",
            quantity=0.001, asset_quantity_delta=-0.001, position_before_quantity=0.001,
            order_role="strategy_exit", strategy_name="ma_rsi_crossover",
        )
        self._seed_buy()
        runtime = main.LiveRuntime()
        self.assertTrue(main._reconstruct_active_position_from_ledger(self.position, runtime))
        self.assertEqual(runtime.active_position["source_order_id"], str(self.order_id))
        self.assertEqual(runtime.active_position["strategy_name"], "regime_only_4h")

    def test_partial_reduction_preserves_unambiguous_original_buy(self):
        self._seed_buy(quantity=self.quantity * 2, asset_quantity_delta=self.quantity * 2)
        self._seed_buy(
            order_id="partial-reduction", client_order_id="bot-partial-reduction",
            side="SELL", quantity=self.quantity, asset_quantity_delta=-self.quantity,
            position_before_quantity=self.quantity * 2, order_role="strategy_exit",
        )
        runtime = main.LiveRuntime()
        self.assertTrue(main._reconstruct_active_position_from_ledger(self.position, runtime))
        self.assertEqual(runtime.active_position["source_order_id"], str(self.order_id))
        self.assertEqual(runtime.active_position["credited_quantity"], self.quantity)

    def test_persistence_failure_during_reconstruction_disables_entries(self):
        self._seed_buy()
        runtime = main.LiveRuntime()
        with patch.object(database, "set_active_bot_position", side_effect=OSError("disk full")):
            self.assertFalse(main._reconstruct_active_position_from_ledger(self.position, runtime))
        self.assertTrue(runtime.accounting_degraded)
        self.assertTrue(runtime.entries_disabled)
        self.assertIsNone(database.get_active_bot_position())

    def test_mixed_strategy_provenance_does_not_reconstruct(self):
        self._seed_ambiguous_buys(second_strategy="ma_rsi_crossover")
        runtime = main.LiveRuntime()
        self.assertFalse(main._reconstruct_active_position_from_ledger(self.position, runtime))
        self.assertIsNone(database.get_active_bot_position())

    def test_incomplete_source_context_does_not_reconstruct(self):
        self._seed_buy(client_order_id=None)
        runtime = main.LiveRuntime()
        self.assertFalse(main._reconstruct_active_position_from_ledger(self.position, runtime))
        self.assertIsNone(database.get_active_bot_position())

    def test_broker_quantity_mismatch_does_not_reconstruct(self):
        self._seed_buy()
        self.position.qty = str(self.quantity + 0.000001)
        runtime = main.LiveRuntime()
        self.assertFalse(main.bot_owns_position(self.position, runtime)[0])
        self.assertFalse(main._reconstruct_active_position_from_ledger(self.position, runtime))
        self.assertIsNone(database.get_active_bot_position())

    def test_restart_reconstructs_and_holds_existing_paper_position_without_buy(self):
        self._seed_buy()
        output = self._run_mocked_restart()
        self.assertIn("ACTION=HOLD REASON=position_already_open", output)
        self.assertEqual(database.get_active_bot_position()["strategy_name"], "regime_only_4h")

    def test_strategy_mismatch_safe_halts_after_reconstruction(self):
        self._seed_buy()
        output = self._run_mocked_restart(live_strategy="ma_rsi_crossover")
        self.assertIn("SAFE-HALT STRATEGY MISMATCH", output)
        self.assertNotIn("PAPER BOT STARTED", output)
        self.assertEqual(database.get_active_bot_position()["strategy_name"], "regime_only_4h")

    def test_ambiguous_restart_safe_halts_without_reconstruction_or_orders(self):
        self._seed_ambiguous_buys()
        output = self._run_mocked_restart()
        self.assertIn("SAFE-HALT ACTIVE POSITION PROVENANCE AMBIGUOUS", output)
        self.assertNotIn("PAPER BOT STARTED", output)
        self.assertIsNone(database.get_active_bot_position())

    def test_deterministic_buy_adopts_prior_uuid_order_without_resubmission(self):
        with patch.object(trade_config, "LIVE_TIMEFRAME", "4Hour"):
            self.client_id = main._strategy_order_id(
                REGIME_ONLY_4H, self.candle, "BUY", role="strategy_entry"
            )
            self._seed_buy()
            runtime = main.LiveRuntime()
            self.assertTrue(main.bot_owns_position(self.position, runtime)[0])
            prior = self._order()
            with (
                patch.object(main, "get_order_by_client_order_id", return_value=prior) as lookup,
                patch.object(main, "buy_btc") as buy,
                patch.object(main, "_persist_buy_submission_intent") as intent,
                patch.object(main, "wait_for_order_fill", return_value=prior) as wait,
                patch.object(main, "get_btc_position", return_value=self.position),
                redirect_stdout(io.StringIO()),
            ):
                outcome = main._submit_and_log_buy(
                    "regime_on", {}, self.candle, REGIME_ONLY_4H, runtime
                )
        self.assertEqual(outcome, broker.OrderOutcome.FILLED)
        lookup.assert_called_once_with(self.client_id)
        wait.assert_called_once_with(self.order_id)
        buy.assert_not_called()
        intent.assert_not_called()
        self.assertFalse(runtime.accounting_degraded)
        self.assertEqual(database.get_active_bot_position()["source_order_id"], str(self.order_id))
        self.assertEqual(len(database.get_order_records()), 1)


if __name__ == "__main__":
    unittest.main()
