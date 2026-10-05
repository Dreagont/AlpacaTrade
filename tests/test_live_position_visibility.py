import io
import json
import tempfile
import unittest
import uuid
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

import broker
import config
import database
import main
import trade_config
from strategy import REGIME_ONLY_4H


class PositionVisibilitySafetyTests(unittest.TestCase):
    quantity = 0.000228726
    candle = "2026-10-05T08:00:00+00:00"

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        for target, name, value in (
            (config, "DATABASE_PATH", str(Path(temporary.name) / "visibility.sqlite")),
            (trade_config, "LIVE_STRATEGY", "regime_only_4h"),
            (trade_config, "LIVE_TIMEFRAME", "4Hour"),
        ):
            setting = patch.object(target, name, value)
            setting.start()
            self.addCleanup(setting.stop)
        self.client_id = main._strategy_order_id(
            REGIME_ONLY_4H, self.candle, "BUY", role="strategy_entry"
        )
        self.order = SimpleNamespace(
            id=uuid.UUID("f65a9129-6cd4-43a3-add6-89509951e704"),
            client_order_id=self.client_id, status=SimpleNamespace(value="filled"),
            side=SimpleNamespace(value="buy"), symbol="BTC/USD",
            filled_qty="0.0002293", filled_avg_price="85539.8",
        )
        self.position = SimpleNamespace(
            qty=str(self.quantity), symbol="BTCUSD", asset_id="btc-asset",
            avg_entry_price="85539.8", market_value="19.5", unrealized_pl="0",
        )
        self.bars = pd.DataFrame(
            {"close": [85000 + i for i in range(260)]},
            index=pd.date_range("2026-08-01", periods=260, freq="4h", tz="UTC"),
        )

    def _seed_source(self, delta=None, status="filled"):
        database.upsert_order(
            order_id=str(self.order.id), client_order_id=self.client_id,
            symbol="BTC/USD", side="BUY", requested_notional=20,
            quantity=float(self.order.filled_qty), fill_price=float(self.order.filled_avg_price),
            reason="regime_on", order_status=status, strategy_name="regime_only_4h",
            asset_quantity_delta=delta, position_before_quantity=0,
            order_role="strategy_entry", candle_timestamp=self.candle,
        )

    def _unquantified_runtime(self):
        self._seed_source()
        runtime = main.LiveRuntime()
        main._record_unquantified_bot_position(runtime, self.order, REGIME_ONLY_4H)
        runtime.accounting_degraded = True
        return runtime

    def _quantified_runtime(self):
        self._seed_source(self.quantity)
        runtime = main.LiveRuntime()
        main._set_active_position(
            runtime, self.position, source_order=self.order, strategy=REGIME_ONLY_4H
        )
        return runtime

    def _run_cycles(self, runtime, positions):
        output = io.StringIO()
        snapshots = []
        states = [broker.PositionLookup(
            broker.PositionLookupStatus.CONFIRMED_FLAT if position is None
            else broker.PositionLookupStatus.CONFIRMED_POSITION, position
        ) for position in positions]

        def sleep(_seconds):
            snapshots.append({
                "active": json.loads(json.dumps(runtime.active_position)),
                "persisted": database.get_active_bot_position(),
                "degraded": runtime.accounting_degraded,
                "disabled": runtime.entries_disabled, "output": output.getvalue(),
            })
            if len(snapshots) == len(positions) - 1:
                raise KeyboardInterrupt

        with (
            patch.object(main, "LiveRuntime", return_value=runtime),
            patch.object(main.client, "get_account", return_value=SimpleNamespace()),
            patch.object(main, "restore_pending_orders"),
            patch.object(main, "reconcile_pending_orders"),
            patch.object(main, "lookup_btc_position", side_effect=states),
            patch.object(main, "get_btc_market_price", return_value=85539.8),
            patch.object(main, "get_btc_bars", return_value=self.bars) as bars,
            patch.object(main, "_record_evaluation"),
            patch.object(main, "_load_risk_exit_state", return_value=None),
            patch.object(main, "_submit_and_log_buy") as submit,
            patch.object(main, "buy_btc") as buy,
            patch.object(main, "sell_btc") as sell,
            patch.object(main, "_strategy_order_id", wraps=main._strategy_order_id) as identifier,
            patch.object(main.time, "sleep", side_effect=sleep),
            redirect_stdout(output),
        ):
            with self.assertRaises(KeyboardInterrupt):
                main.run()
        submit.assert_not_called()
        buy.assert_not_called()
        sell.assert_not_called()
        identifier.assert_not_called()
        return snapshots, bars

    def _assert_waiting(self, snapshot, quantity=None):
        self.assertEqual(snapshot["active"], snapshot["persisted"])
        self.assertEqual(snapshot["active"]["credited_quantity"], quantity)
        self.assertEqual(snapshot["active"]["source_order_id"], str(self.order.id))
        self.assertEqual(snapshot["active"]["source_client_order_id"], self.client_id)
        self.assertEqual(snapshot["active"]["strategy_name"], "regime_only_4h")
        self.assertTrue(snapshot["degraded"])
        self.assertTrue(snapshot["disabled"])
        self.assertNotIn("ACCOUNTING RECOVERY COMPLETE", snapshot["output"])

    def test_paper_fill_lag_retains_provenance_until_position_becomes_visible(self):
        self._seed_source()
        runtime = main.LiveRuntime()
        with (
            patch.object(main, "get_order_by_client_order_id", return_value=self.order),
            patch.object(main, "wait_for_order_fill", return_value=self.order),
            patch.object(main, "get_btc_position", return_value=None) as lookup,
            patch.object(main.time, "sleep"),
            patch.object(main, "buy_btc") as buy,
            redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(main._submit_and_log_buy(
                "regime_on", {}, self.candle, REGIME_ONLY_4H, runtime
            ), broker.OrderOutcome.FILLED)
        self.assertEqual(lookup.call_count, 5)
        buy.assert_not_called()
        self.assertIsNone(database.get_active_bot_position()["credited_quantity"])
        snapshots, _bars = self._run_cycles(runtime, [None, None, self.position])
        self._assert_waiting(snapshots[0])
        final = snapshots[1]
        self.assertEqual(final["active"], final["persisted"])
        self.assertEqual(final["active"]["credited_quantity"], self.quantity)
        self.assertEqual(final["active"]["source_order_id"], str(self.order.id))
        self.assertEqual(final["active"]["source_client_order_id"], self.client_id)
        self.assertEqual(final["active"]["strategy_name"], "regime_only_4h")
        self.assertEqual(database.get_order_records()[0]["asset_quantity_delta"], self.quantity)
        self.assertFalse(final["degraded"])
        self.assertFalse(final["disabled"])
        self.assertEqual(final["output"].count("ACCOUNTING RECOVERY COMPLETE"), 1)

    def test_new_four_hour_buy_candle_cannot_enter_during_visibility_lag(self):
        runtime = self._unquantified_runtime()
        for advance in (0, 1):
            bars = self.bars.copy()
            bars.index += pd.Timedelta(hours=4 * advance)
            prepared = REGIME_ONLY_4H.prepare_indicators(bars)
            self.assertEqual(REGIME_ONLY_4H.decide_at(prepared, len(prepared) - 1).action, "BUY")
        self.bars.index += pd.Timedelta(hours=4)
        snapshots, bars = self._run_cycles(runtime, [None, None, None])
        for snapshot in snapshots:
            self._assert_waiting(snapshot)
        bars.assert_not_called()

    def test_restart_loads_unquantified_provenance_and_recovers_only_after_visibility(self):
        self._unquantified_runtime()
        runtime = main.LiveRuntime()
        snapshots, _bars = self._run_cycles(runtime, [None, None, self.position])
        self._assert_waiting(snapshots[0])
        self.assertEqual(snapshots[1]["persisted"]["credited_quantity"], self.quantity)
        self.assertEqual(snapshots[1]["persisted"]["source_client_order_id"], self.client_id)
        self.assertFalse(snapshots[1]["degraded"])
        self.assertFalse(snapshots[1]["disabled"])

    def test_quantified_positive_exposure_is_retained_on_transient_flat(self):
        runtime = self._quantified_runtime()
        snapshots, bars = self._run_cycles(runtime, [None, None])
        self._assert_waiting(snapshots[0], self.quantity)
        bars.assert_not_called()

    def test_accounting_recovery_cannot_clear_unquantified_or_positive_exposure(self):
        for quantified in (False, True):
            with self.subTest(quantified=quantified):
                runtime = self._quantified_runtime() if quantified else self._unquantified_runtime()
                runtime.accounting_degraded = True
                output = io.StringIO()
                with redirect_stdout(output):
                    self.assertFalse(main._try_recover_accounting(runtime, {}, None))
                self.assertIsNotNone(database.get_active_bot_position())
                self.assertTrue(runtime.accounting_degraded)
                self.assertTrue(runtime.entries_disabled)
                self.assertNotIn("ACCOUNTING RECOVERY COMPLETE", output.getvalue())

    def test_missing_active_record_with_positive_ledger_still_blocks_flat_entry(self):
        self._seed_source(self.quantity)
        runtime = main.LiveRuntime()
        snapshots, bars = self._run_cycles(runtime, [None, None])
        self.assertIsNone(snapshots[0]["active"])
        self.assertTrue(snapshots[0]["degraded"])
        self.assertTrue(snapshots[0]["disabled"])
        bars.assert_not_called()

    def test_asset_identity_mismatch_does_not_quantify_or_recover(self):
        runtime = self._unquantified_runtime()
        runtime.active_position["asset_id"] = "different-btc-asset"
        database.set_active_bot_position(runtime.active_position)
        self.assertFalse(main.bot_owns_position(self.position, runtime)[0])
        self.assertFalse(main._try_recover_accounting(runtime, {}, self.position))
        self.assertIsNone(database.get_active_bot_position()["credited_quantity"])
        self.assertTrue(runtime.entries_disabled)

    def test_missing_filled_source_blocks_quantification_and_recovery(self):
        runtime = self._unquantified_runtime()
        self._seed_source(status="partially_filled")
        self.assertFalse(main.bot_owns_position(self.position, runtime)[0])
        self.assertFalse(main._try_recover_accounting(runtime, {}, self.position))
        self.assertIsNone(database.get_active_bot_position()["credited_quantity"])

    def test_quantification_persistence_failure_blocks_recovery(self):
        runtime = self._unquantified_runtime()
        with patch.object(database, "set_active_bot_position", side_effect=OSError("disk full")):
            self.assertFalse(main._try_recover_accounting(runtime, {}, self.position))
        self.assertIsNone(database.get_active_bot_position()["credited_quantity"])
        self.assertTrue(runtime.accounting_degraded)
        self.assertTrue(runtime.entries_disabled)

    def _seed_sell(self, status="filled", delta=None):
        database.upsert_order(
            order_id="closed-sell", client_order_id="bot-closed-sell", symbol="BTC/USD",
            side="SELL", requested_notional=19.5, quantity=self.quantity,
            fill_price=85539.8, reason="regime_filter_off", order_status=status,
            strategy_name="regime_only_4h", asset_quantity_delta=delta,
            position_before_quantity=self.quantity, order_role="strategy_exit",
        )

    def test_reconciled_full_sell_ledger_allows_clear_and_accounting_recovery(self):
        runtime = self._quantified_runtime()
        runtime.accounting_degraded = runtime.entries_disabled = True
        self._seed_sell(delta=-self.quantity)
        self.assertTrue(main._try_recover_accounting(runtime, {}, None))
        self.assertIsNone(runtime.active_position)
        self.assertIsNone(database.get_active_bot_position())
        self.assertFalse(runtime.entries_disabled)

    def test_partial_or_unquantified_sell_ledger_cannot_clear_provenance(self):
        runtime = self._quantified_runtime()
        self._seed_sell(status="partially_filled", delta=-self.quantity)
        self.assertFalse(main._flat_position_is_reconciled(runtime))
        self.assertIsNotNone(database.get_active_bot_position())
        self._seed_sell(status="filled", delta=-self.quantity / 2)
        self.assertFalse(main._flat_position_is_reconciled(runtime))
        self.assertIsNotNone(database.get_active_bot_position())

    def test_partial_sell_and_missing_broker_position_cannot_imply_zero_exposure(self):
        runtime = self._quantified_runtime()
        sell = SimpleNamespace(status=SimpleNamespace(value="filled"), filled_qty=str(self.quantity / 2))
        with patch.object(main, "get_btc_position", return_value=None):
            position, delta = main._refresh_position_after_order(
                runtime, sell, "SELL", self.quantity, REGIME_ONLY_4H
            )
        self.assertIsNone(position)
        self.assertIsNone(delta)
        self.assertEqual(database.get_active_bot_position()["credited_quantity"], self.quantity)
        self.assertTrue(runtime.entries_disabled)

    def test_unknown_filled_quantity_blocks_closed_position_proof(self):
        runtime = self._quantified_runtime()
        self._seed_sell(delta=-self.quantity)
        database.upsert_order(
            order_id="unknown-buy", client_order_id="bot-unknown-buy", symbol="BTC/USD",
            side="BUY", requested_notional=20, quantity=None,
            fill_price=None, reason="regime_on", order_status="filled",
            strategy_name="regime_only_4h", asset_quantity_delta=None,
            position_before_quantity=0, order_role="strategy_entry",
        )
        self.assertFalse(main._flat_position_is_reconciled(runtime))
        self.assertIsNotNone(database.get_active_bot_position())
        self.assertTrue(runtime.accounting_degraded)
        self.assertTrue(runtime.entries_disabled)

    def test_unknown_subsequent_fill_blocks_delayed_quantity_attribution(self):
        runtime = self._unquantified_runtime()
        database.upsert_order(
            order_id="unknown-sell", client_order_id="bot-unknown-sell", symbol="BTC/USD",
            side="SELL", requested_notional=20, quantity=None,
            fill_price=None, reason="regime_filter_off", order_status="filled",
            strategy_name="regime_only_4h", asset_quantity_delta=None,
            position_before_quantity=self.quantity, order_role="strategy_exit",
        )
        self.assertFalse(main._try_recover_accounting(runtime, {}, self.position))
        self.assertIsNone(database.get_active_bot_position()["credited_quantity"])
        self.assertTrue(runtime.accounting_degraded)
        self.assertTrue(runtime.entries_disabled)


if __name__ == "__main__":
    unittest.main()
