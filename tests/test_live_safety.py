import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import broker
import config
import database
import main
import market_data
import paper_position_smoke_test
import risk
import trade_config
from strategy import Decision, StrategySpec


def make_order(
    order_id="order-1",
    *,
    status="filled",
    side="buy",
    symbol="BTC/USD",
    client_order_id=None,
    filled_qty="0.25",
    filled_avg_price="40000",
):
    return SimpleNamespace(
        id=order_id,
        status=SimpleNamespace(value=status),
        side=SimpleNamespace(value=side),
        symbol=symbol,
        client_order_id=client_order_id,
        filled_qty=filled_qty,
        filled_avg_price=filled_avg_price,
    )


def make_position(*, qty="0.004", symbol="BTCUSD", asset_id="btc-asset"):
    return SimpleNamespace(
        qty=qty,
        symbol=symbol,
        asset_id=asset_id,
        avg_entry_price="40000",
        market_value="160",
        unrealized_pl="0",
    )


class BrokerPositionSafetyTests(unittest.TestCase):
    def test_market_symbol_normalizes_only_at_position_identifier_boundary(self):
        self.assertEqual(broker.broker_position_symbol("BTC/USD"), "BTCUSD")
        self.assertTrue(broker.btc_symbol_matches("BTC-USD"))
        self.assertTrue(broker.btc_symbol_matches("BTCUSD"))

    def test_close_uses_asset_id_or_central_normalized_symbol(self):
        position = make_position(asset_id="asset-uuid")
        with patch.object(broker.client, "close_position", return_value="closed") as close:
            self.assertEqual(broker.sell_btc(position), "closed")
        close.assert_called_once_with("asset-uuid")

        position.asset_id = None
        with patch.object(broker.client, "close_position", return_value="closed") as close:
            broker.sell_btc(position)
        close.assert_called_once_with("BTCUSD")

    def test_position_lookup_only_returns_none_for_confirmed_flat(self):
        with patch.object(broker.client, "get_all_positions", return_value=[]):
            self.assertIsNone(broker.get_btc_position())
        with patch.object(
            broker.client, "get_all_positions", return_value=[make_position()]
        ):
            self.assertEqual(broker.get_btc_position().symbol, "BTCUSD")

    def test_position_api_failure_is_unknown_not_flat(self):
        with patch.object(
            broker.client, "get_all_positions", side_effect=RuntimeError("offline")
        ):
            state = broker.lookup_btc_position()
            self.assertEqual(
                state.status, broker.PositionLookupStatus.POSITION_STATE_UNKNOWN
            )
            with self.assertRaises(broker.BrokerPositionStateUnknown):
                broker.get_btc_position()

    def test_malformed_position_response_is_unknown_not_flat(self):
        with patch.object(broker.client, "get_all_positions", return_value={}):
            state = broker.lookup_btc_position()
        self.assertEqual(
            state.status, broker.PositionLookupStatus.POSITION_STATE_UNKNOWN
        )

    def test_paper_mode_check_requires_the_exact_alpaca_paper_host(self):
        self.assertTrue(
            broker.is_paper_client(SimpleNamespace(_base_url="https://paper-api.alpaca.markets/v2"))
        )
        self.assertFalse(
            broker.is_paper_client(SimpleNamespace(_base_url="https://api.alpaca.markets"))
        )
        self.assertFalse(
            broker.is_paper_client(SimpleNamespace(_base_url="https://not-alpaca.test/paper"))
        )

    def test_unknown_position_at_startup_safe_halts_before_any_buy(self):
        unknown = broker.PositionLookup(
            broker.PositionLookupStatus.POSITION_STATE_UNKNOWN,
            error=RuntimeError("offline"),
        )
        output = io.StringIO()
        with (
            patch.object(main, "init_db"),
            patch.object(main.client, "get_account", return_value=SimpleNamespace()),
            patch.object(main, "restore_pending_orders"),
            patch.object(main, "lookup_btc_position", return_value=unknown),
            patch.object(main, "buy_btc") as buy,
            redirect_stdout(output),
        ):
            main.run()
        buy.assert_not_called()
        self.assertIn("SAFE-HALT BROKER POSITION STATE UNKNOWN", output.getvalue())

    def test_manual_position_at_startup_safe_halts_without_selling(self):
        output = io.StringIO()
        with (
            patch.object(main, "init_db"),
            patch.object(main.client, "get_account", return_value=SimpleNamespace()),
            patch.object(main, "restore_pending_orders"),
            patch.object(
                main,
                "lookup_btc_position",
                return_value=broker.PositionLookup(
                    broker.PositionLookupStatus.CONFIRMED_POSITION,
                    make_position(),
                ),
            ),
            patch("database.get_bot_owned_btc_quantity", return_value=(0.0, False)),
            patch.object(main, "sell_btc") as sell,
            redirect_stdout(output),
        ):
            main.run()
        sell.assert_not_called()
        self.assertIn("SAFE-HALT_MANUAL_OR_UNKNOWN_POSITION", output.getvalue())


class LiveRiskAndOrderSafetyTests(unittest.TestCase):
    def _run_one_risk_cycle(self, *, market_price=37000, latched=None, flat=False):
        position = make_position(qty="0.0001995")
        active = {
            "asset_id": "btc-asset", "credited_quantity": 0.0001995,
            "source_confirmed": True,
        }
        state = broker.PositionLookup(
            broker.PositionLookupStatus.CONFIRMED_POSITION, None if flat else position
        )
        stop = RuntimeError("cycle complete")
        with (
            patch.object(main, "init_db", return_value=True),
            patch.object(main.client, "get_account", return_value=SimpleNamespace(cash=1, portfolio_value=1)),
            patch("database.get_active_bot_position", return_value=active),
            patch("database.get_state", return_value=latched),
            patch("database.set_state"),
            patch.object(main, "restore_pending_orders"),
            patch.object(main, "lookup_btc_position", return_value=state),
            patch.object(main, "reconcile_pending_orders"),
            patch.object(main, "get_btc_market_price", side_effect=market_price if isinstance(market_price, Exception) else None if market_price is None else None) as price,
            patch.object(main, "get_btc_bars", side_effect=RuntimeError("historical unavailable")) as bars,
            patch.object(main, "execute_risk_exit", return_value=broker.OrderOutcome.PENDING) as sell,
            patch.object(main.time, "sleep", side_effect=KeyboardInterrupt),
            redirect_stdout(io.StringIO()),
        ):
            if not isinstance(market_price, Exception) and market_price is not None:
                price.return_value = market_price
            with self.assertRaises(KeyboardInterrupt):
                main.run()
        return sell, bars

    def test_stop_loss_is_submitted_before_historical_fetch(self):
        sell, bars = self._run_one_risk_cycle()
        sell.assert_called_once()
        self.assertEqual(sell.call_args.args[0], "stop_loss")
        bars.assert_not_called()

    def test_persisted_latch_reduces_risk_without_any_price_or_candle(self):
        import json

        latch = json.dumps({
            "episode_id": "episode-restart", "reason": "stop_loss",
            "attempt": 0, "next_attempt_at": 0,
        })
        sell, bars = self._run_one_risk_cycle(market_price=RuntimeError("price unavailable"), latched=latch)
        sell.assert_called_once()
        self.assertEqual(sell.call_args.args[0], "stop_loss")
        bars.assert_not_called()

    def test_risk_retry_continues_after_fast_attempts_at_escalated_cadence(self):
        runtime = main.LiveRuntime(risk_exit={
            "episode_id": "episode-1", "reason": "stop_loss", "attempt": 0,
            "next_attempt_at": 0,
        })
        with patch("database.set_state") as persist, patch.object(main.time, "time", side_effect=[100, 160, 220]):
            main._advance_risk_exit_retry(runtime)
            self.assertEqual(runtime.risk_exit["next_attempt_at"], 160)
            main._advance_risk_exit_retry(runtime)
            self.assertEqual(runtime.risk_exit["next_attempt_at"], 220)
            main._advance_risk_exit_retry(runtime)
        self.assertEqual(runtime.risk_exit["attempt"], 3)
        self.assertEqual(runtime.risk_exit["next_attempt_at"], 520)
        self.assertEqual(trade_config.RISK_EXIT_ESCALATED_COOLDOWN_SECONDS, 300)
        saved_latch = persist.call_args.args[1]
        restarted = main.LiveRuntime()
        with patch("database.get_state", return_value=saved_latch):
            loaded = main._load_risk_exit_state(restarted)
        self.assertEqual(loaded["attempt"], 3)
        self.assertEqual(loaded["next_attempt_at"], 520)

    def test_five_rejected_risk_sells_continue_through_escalated_cycles(self):
        class Rejected(Exception):
            status_code = 422

        position = make_position(qty="0.0001995")
        runtime = main.LiveRuntime(risk_exit={
            "episode_id": "episode-rejections", "reason": "stop_loss",
            "attempt": 0, "next_attempt_at": 0,
        })
        now = [100.0]
        with (
            patch.object(main, "get_open_btc_orders", return_value=[]),
            patch.object(main, "get_btc_position", return_value=position),
            patch.object(main, "has_open_order", return_value=False),
            patch.object(main, "get_order_by_client_order_id", return_value=None),
            patch.object(main, "sell_btc", side_effect=[Rejected("rejected")] * 5) as sell,
            patch.object(main, "upsert_order"),
            patch("database.set_state"),
            patch.object(main.time, "time", side_effect=lambda: now[0]),
        ):
            last_cooldown = None
            for _ in range(5):
                attempt = runtime.risk_exit["attempt"]
                self.assertGreaterEqual(now[0], runtime.risk_exit["next_attempt_at"])
                attempted_at = now[0]
                result = main._submit_and_log_sell(
                    "stop_loss", position, {}, None,
                    role="risk_exit_stop_loss", risk_episode="episode-rejections",
                    risk_attempt=attempt,
                )
                self.assertEqual(result, broker.OrderOutcome.TERMINAL_NOT_FILLED)
                main._advance_risk_exit_retry(runtime)
                last_cooldown = runtime.risk_exit["next_attempt_at"] - attempted_at
                now[0] = runtime.risk_exit["next_attempt_at"]
        self.assertEqual(sell.call_count, 5)
        self.assertEqual(runtime.risk_exit["attempt"], 5)
        identifiers = [call.kwargs["client_order_id"] for call in sell.call_args_list]
        self.assertEqual(len(set(identifiers)), 5)
        self.assertEqual(last_cooldown, trade_config.RISK_EXIT_ESCALATED_COOLDOWN_SECONDS)

    def test_escalated_retry_is_restored_after_restart_and_fourth_sell_is_attempted(self):
        latch = json.dumps({
            "episode_id": "episode-restart", "reason": "stop_loss",
            "attempt": 3, "next_attempt_at": 0,
        })
        restored = main.LiveRuntime()
        with patch("database.get_state", return_value=latch):
            state = main._load_risk_exit_state(restored)
        self.assertEqual(state["attempt"], 3)
        sell, bars = self._run_one_risk_cycle(latched=latch)
        sell.assert_called_once()
        self.assertEqual(sell.call_args.args[0], "stop_loss")
        bars.assert_not_called()

    def test_confirmed_broker_flat_clears_persisted_risk_episode(self):
        latch = json.dumps({
            "episode_id": "episode-flat", "reason": "stop_loss",
            "attempt": 4, "next_attempt_at": 0,
        })
        persisted = []
        with patch.object(
            main, "_persist_risk_exit_state",
            side_effect=lambda runtime: persisted.append(runtime.risk_exit),
        ):
            sell, _bars = self._run_one_risk_cycle(latched=latch, flat=True)
        sell.assert_not_called()
        self.assertIn(None, persisted)

    def test_rate_limit_is_retryable_and_does_not_create_terminal_safety_limit(self):
        class HttpError(Exception):
            status_code = 429

        self.assertEqual(
            broker.classify_submission_exception(HttpError("slow down")),
            broker.SubmissionFailureKind.RATE_LIMITED,
        )
        runtime = main.LiveRuntime(risk_exit={
            "episode_id": "episode-429", "reason": "stop_loss",
            "attempt": 2, "next_attempt_at": 0,
        })
        position = make_position(qty="0.0001995")
        with (
            patch.object(main, "get_open_btc_orders", return_value=[]),
            patch.object(main, "get_btc_position", return_value=position),
            patch.object(main, "has_open_order", return_value=False),
            patch.object(main, "get_order_by_client_order_id", return_value=None),
            patch.object(main, "sell_btc", side_effect=HttpError("slow down")),
            patch.object(main, "upsert_order"),
            patch("database.set_state"),
        ):
            result = main._submit_and_log_sell(
                "stop_loss", position, {}, None, runtime=runtime,
                role="risk_exit_stop_loss", risk_episode="episode-429", risk_attempt=2,
            )
        self.assertEqual(result, broker.OrderOutcome.RATE_LIMITED)
        with patch("database.set_state"), patch.object(main.time, "time", return_value=1000):
            main._advance_risk_exit_retry(runtime)
        self.assertEqual(runtime.risk_exit["attempt"], 3)

    def test_accounting_recovery_requires_provenance_and_no_risk_episode(self):
        position = make_position(qty="0.0001995")
        runtime = main.LiveRuntime(
            active_position={
                "asset_id": "btc-asset", "credited_quantity": 0.0001995,
                "source_confirmed": True, "source_order_id": "buy-order",
                "source_client_order_id": "bot-buy",
            },
            accounting_degraded=True, entries_disabled=True,
        )
        with (
            patch("database.probe_writable") as write_probe,
            patch("database.get_active_bot_position", return_value=runtime.active_position),
            patch("database.get_order_records", side_effect=lambda pending_only=False: [] if pending_only else [{
                "client_order_id": "bot-buy", "order_id": "buy-order", "side": "BUY",
                "order_role": "strategy_entry", "position_before_quantity": 0,
                "order_status": "filled",
            }]),
        ):
            self.assertTrue(main._try_recover_accounting(runtime, {}, position))
        write_probe.assert_called_once()
        self.assertFalse(runtime.accounting_degraded)
        self.assertFalse(runtime.entries_disabled)

        runtime.accounting_degraded = True
        runtime.entries_disabled = True
        runtime.risk_exit = {"episode_id": "risk", "reason": "stop_loss"}
        with patch("database.probe_writable") as blocked_probe:
            self.assertFalse(main._try_recover_accounting(runtime, {}, position))
        blocked_probe.assert_not_called()
        self.assertTrue(runtime.entries_disabled)

        runtime.risk_exit = None
        unresolved = {"sell": main.PendingReconciliation(
            "SELL", "strategy_exit", 20, None, "candle", "bot-sell",
            "ma_rsi_crossover", "{}", None, "broker_order", None, 0.0001995,
            "strategy_exit",
        )}
        with patch("database.probe_writable") as unresolved_probe:
            self.assertFalse(main._try_recover_accounting(runtime, unresolved, position))
        unresolved_probe.assert_not_called()
        self.assertTrue(runtime.entries_disabled)

    def test_stale_broker_quantity_risk_sell_is_capped_to_proven_bot_exposure(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            config, "DATABASE_PATH", str(Path(temp_dir) / "stale.sqlite")
        ):
            database.upsert_order(
                order_id="buy-source", client_order_id="bot-source-buy",
                symbol="BTC/USD", side="BUY", requested_notional=20,
                quantity=0.004, fill_price=40000, reason="entry",
                order_status="filled", position_before_quantity=0,
                order_role="strategy_entry", asset_quantity_delta=0.004,
            )
            runtime = main.LiveRuntime(
                active_position={
                    "asset_id": "btc-asset", "source_order_id": "buy-source",
                    "source_client_order_id": "bot-source-buy",
                    "credited_quantity": 0.004, "source_confirmed": True,
                },
                ownership_mismatch=True, proven_bot_quantity=0.004,
            )
            before = make_position(qty="0.005")
            after = make_position(qty="0.001")
            sold = make_order("capped-sale", side="sell", filled_qty="0.004")
            with (
                patch.object(main, "get_open_btc_orders", return_value=[]),
                patch.object(main, "get_btc_position", side_effect=[before, after]),
                patch.object(main, "has_open_order", return_value=False),
                patch.object(main, "get_order_by_client_order_id", return_value=None),
                patch.object(main, "sell_btc", return_value=sold) as sell,
                patch.object(main, "wait_for_order_fill", return_value=sold),
                patch("database.set_active_bot_position"),
                patch.object(main, "upsert_order"),
            ):
                result = main._submit_and_log_sell(
                    "stop_loss", before, {}, None, runtime=runtime,
                    role="risk_exit_stop_loss", risk_episode="stale", risk_attempt=0,
                )
            self.assertEqual(result, broker.OrderOutcome.FILLED)
            self.assertEqual(sell.call_args.kwargs["quantity"], 0.004)
            self.assertIsNone(runtime.active_position)

    def test_fee_deducted_broker_quantity_remains_owned_and_risk_check_runs(self):
        position = make_position(qty="0.000199500")
        runtime = main.LiveRuntime(active_position={
            "asset_id": "btc-asset",
            "credited_quantity": 0.0001995,
            "source_confirmed": True,
        })
        with patch("database.get_active_bot_position", return_value=runtime.active_position):
            owned, reason = main.bot_owns_position(position, runtime)
        self.assertTrue(owned, reason)
        strategy = StrategySpec(
            name="safety_test", _prepare_indicators=lambda bars: bars,
            _decide_at=lambda bars, index: Decision("HOLD", "test"),
            _warmup_lookback=lambda: 0, _parameters=lambda: {},
            _stop_loss=lambda: 0.05, _take_profit=lambda: 0.10,
        )
        self.assertEqual(main._risk_exit_reason(position, 37000, strategy), "stop_loss")

    def test_definitive_403_and_422_are_terminal_and_future_candle_can_submit(self):
        class HttpError(Exception):
            def __init__(self, status_code):
                self.status_code = status_code
                super().__init__(str(status_code))

        for code in (403, 422):
            with self.subTest(status_code=code):
                pending = {}
                filled = make_order("later-buy", client_order_id="later-id")
                with (
                    patch.object(main, "get_order_by_client_order_id", side_effect=[None, None, None]),
                    patch.object(main, "buy_btc", side_effect=[HttpError(code), filled]) as buy,
                    patch.object(main, "wait_for_order_fill", return_value=filled),
                    patch.object(main, "upsert_order"),
                ):
                    rejected = main._submit_and_log_buy("entry", pending, "candle-1")
                    accepted = main._submit_and_log_buy("entry", pending, "candle-2")
                self.assertEqual(rejected, broker.OrderOutcome.TERMINAL_NOT_FILLED)
                self.assertEqual(accepted, broker.OrderOutcome.FILLED)
                self.assertEqual(pending, {})
                self.assertEqual(buy.call_count, 2)

    def test_strategy_exit_and_risk_exit_have_distinct_order_ids_without_candle(self):
        strategy = main._live_strategy()
        strategy_exit = main._strategy_order_id(
            strategy, "same-candle", "SELL", role="strategy_exit"
        )
        risk_exit = main._strategy_order_id(
            strategy, "episode-abc:0:stop_loss", "SELL", role="risk_exit_stop_loss"
        )
        self.assertNotEqual(strategy_exit, risk_exit)
        self.assertNotEqual(
            risk_exit,
            main._strategy_order_id(
                strategy, "episode-abc:1:stop_loss", "SELL", role="risk_exit_stop_loss"
            ),
        )

    def test_rejected_strategy_sell_does_not_suppress_same_candle_risk_sell(self):
        class HttpError(Exception):
            status_code = 422

        position = make_position(qty="0.004")
        risk_order = make_order("risk-sale", side="sell", filled_qty="0.004")
        with (
            patch.object(main, "get_open_btc_orders", return_value=[]),
            patch.object(main, "get_btc_position", side_effect=[position, position, position, position, None]),
            patch.object(main, "has_open_order", return_value=False),
            patch.object(main, "get_order_by_client_order_id", return_value=None),
            patch.object(main, "sell_btc", side_effect=[HttpError("rejected"), risk_order]) as sell,
            patch.object(main, "wait_for_order_fill", return_value=risk_order),
            patch.object(main, "upsert_order"),
            redirect_stdout(io.StringIO()),
        ):
            strategy_result = main._submit_and_log_sell(
                "bearish_ma_crossover", position, {}, "same-candle", role="strategy_exit"
            )
            risk_result = main._submit_and_log_sell(
                "stop_loss", position, {}, None, role="risk_exit_stop_loss",
                risk_episode="episode-1", risk_attempt=0,
            )
        self.assertEqual(strategy_result, broker.OrderOutcome.TERMINAL_NOT_FILLED)
        self.assertEqual(risk_result, broker.OrderOutcome.FILLED)
        ids = [call.kwargs["client_order_id"] for call in sell.call_args_list]
        self.assertNotEqual(ids[0], ids[1])

    def test_post_fill_db_failure_keeps_in_memory_ownership_and_disables_entries(self):
        order = make_order("filled-buy", filled_qty="0.0002")
        position = make_position(qty="0.0001995")
        runtime = main.LiveRuntime()
        with (
            patch.object(main, "get_order_by_client_order_id", return_value=None),
            patch.object(main, "buy_btc", return_value=order) as buy,
            patch.object(main, "wait_for_order_fill", return_value=order),
            patch.object(main, "get_btc_position", return_value=position),
            patch("database.set_active_bot_position", side_effect=OSError("disk full")),
            patch.object(main, "upsert_order", side_effect=[None, OSError("disk full"), OSError("disk full")]),
            redirect_stdout(io.StringIO()),
        ):
            outcome = main._submit_and_log_buy("entry", {}, "candle-1", runtime=runtime)
            owned, reason = main.bot_owns_position(position, runtime)
        self.assertEqual(outcome, broker.OrderOutcome.FILLED)
        buy.assert_called_once()
        self.assertTrue(runtime.entries_disabled)
        self.assertTrue(runtime.accounting_degraded)
        self.assertTrue(owned, reason)
        self.assertAlmostEqual(runtime.active_position["credited_quantity"], 0.0001995)

    def test_db_failure_during_risk_sell_does_not_prevent_broker_reduction(self):
        position = make_position(qty="0.0001995")
        sold = make_order("risk-sale", side="sell", filled_qty="0.0001995")
        runtime = main.LiveRuntime(active_position={
            "asset_id": "btc-asset", "credited_quantity": 0.0001995,
            "source_confirmed": True, "strategy_name": "ma_rsi_crossover",
        })
        pending = {}
        with (
            patch.object(main, "get_open_btc_orders", return_value=[]),
            patch.object(main, "get_btc_position", side_effect=[position, None]),
            patch.object(main, "has_open_order", return_value=False),
            patch.object(main, "get_order_by_client_order_id", return_value=None),
            patch.object(main, "sell_btc", return_value=sold) as sell,
            patch.object(main, "wait_for_order_fill", return_value=sold),
            patch.object(main, "upsert_order", side_effect=OSError("disk full")),
            patch("database.set_active_bot_position"),
            redirect_stdout(io.StringIO()),
        ):
            result = main._submit_and_log_sell(
                "stop_loss", position, pending, None, runtime=runtime,
                role="risk_exit_stop_loss", risk_episode="episode-1", risk_attempt=0,
            )
        self.assertEqual(result, broker.OrderOutcome.FILLED)
        sell.assert_called_once()
        self.assertTrue(runtime.entries_disabled)
        self.assertIsNone(runtime.active_position)

    def test_protective_fill_records_actual_delta_and_clears_flat_provenance(self):
        filled = make_order(
            "protect-1", side="sell", filled_qty="0.0001995",
            client_order_id="protect-cid",
        )
        runtime = main.LiveRuntime(active_position={
            "asset_id": "btc-asset", "credited_quantity": 0.0001995,
            "source_confirmed": True,
        })
        context = main.PendingReconciliation(
            "SELL", "protective_stop_limit", None,
            main.PositionSnapshot(40000, 20, 0.0001995), None,
            "protect-cid", "ma_rsi_crossover", "{}", None,
            "broker_order", "2026-10-05T00:00:00+00:00", 0.0001995,
            "protective_stop",
        )
        pending = {"protect-1": context}
        with (
            patch.object(main, "reconcile_order", return_value=filled),
            patch.object(main, "get_btc_position", return_value=None),
            patch("database.set_active_bot_position"),
            patch.object(main, "upsert_order") as write,
        ):
            main.reconcile_pending_orders(pending, runtime=runtime)
        self.assertEqual(pending, {})
        self.assertIsNone(runtime.active_position)
        self.assertAlmostEqual(write.call_args.kwargs["asset_quantity_delta"], -0.0001995)

    def test_position_cap_counts_existing_market_value_plus_new_notional(self):
        self.assertTrue(risk.within_max_position_exposure(80, 20, cap=100))
        self.assertFalse(risk.within_max_position_exposure(-80, 20.01, cap=100))
        with patch.object(
            risk,
            "get_btc_position",
            side_effect=broker.BrokerPositionStateUnknown("unknown"),
        ), patch.object(risk.client, "get_account") as account:
            with self.assertRaises(broker.BrokerPositionStateUnknown):
                risk.can_buy(20)
        account.assert_not_called()

    def test_position_value_fallback_uses_quantity_times_market_price(self):
        position = SimpleNamespace(qty="0.01", market_value=None)
        self.assertFalse(
            risk.can_buy(20, position=position, market_price=10000)
        )

    def test_pending_partial_buy_is_canceled_then_actual_exposure_is_exited(self):
        pending = {
            "buy-1": main.PendingReconciliation(
                "BUY", "entry", 20, None, "candle-1", "client-buy"
            )
        }
        partially_filled = make_order(
            "buy-1", status="partially_filled", filled_qty="0.004", side="buy"
        )
        canceled = make_order(
            "buy-1", status="canceled", filled_qty="0.004", side="buy"
        )
        actual_position = make_position(qty="0.004")
        with (
            patch.object(main, "reconcile_order", return_value=partially_filled),
            patch.object(main, "cancel_order") as cancel,
            patch.object(main, "wait_for_order_fill", return_value=canceled),
            patch.object(main, "upsert_order"),
            patch.object(main, "get_btc_position", return_value=actual_position),
            patch.object(main, "has_open_order", return_value=False),
            patch.object(main, "_submit_and_log_sell", return_value=broker.OrderOutcome.FILLED) as sell,
        ):
            outcome = main.execute_risk_exit("stop_loss", pending, "candle-2")
        cancel.assert_called_once_with("buy-1")
        sell.assert_called_once()
        self.assertIs(sell.call_args.args[1], actual_position)
        self.assertEqual(outcome, broker.OrderOutcome.FILLED)
        self.assertEqual(pending, {})

    def test_unconfirmed_buy_cancel_does_not_hide_known_risk_exposure(self):
        pending = {
            "buy-2": main.PendingReconciliation(
                "BUY", "entry", 20, None, "candle-1", "client-buy"
            )
        }
        partial = make_order(
            "buy-2", status="partially_filled", filled_qty="0.002", side="buy"
        )
        position = make_position(qty="0.002")
        with (
            patch.object(main, "reconcile_order", return_value=partial),
            patch.object(main, "cancel_order"),
            patch.object(
                main,
                "wait_for_order_fill",
                side_effect=broker.OrderFillTimeoutError(partial),
            ),
            patch.object(main, "upsert_order"),
            patch.object(main, "get_btc_position", return_value=position),
            patch.object(main, "has_open_order", return_value=False),
            patch.object(main, "_submit_and_log_sell", return_value=broker.OrderOutcome.PENDING) as sell,
        ):
            result = main.execute_risk_exit("stop_loss", pending, "candle-2")
        sell.assert_called_once()
        self.assertIn("buy-2", pending)
        self.assertEqual(result, broker.OrderOutcome.PENDING)

    def test_pending_sell_blocks_duplicate_risk_sell(self):
        pending = {
            "sell-1": main.PendingReconciliation(
                "SELL", "stop_loss", 160, None, "candle-1"
            )
        }
        with (
            patch.object(main, "get_btc_position", return_value=make_position()),
            patch.object(main, "has_open_order", return_value=True),
            patch.object(main, "_submit_and_log_sell") as sell,
        ):
            result = main.execute_risk_exit("stop_loss", pending, "candle-2")
        self.assertEqual(result, broker.OrderOutcome.PENDING)
        sell.assert_not_called()

    def test_protective_pending_does_not_block_risk_sell_and_is_canceled_first(self):
        protective = make_order(
            "protect-risk", status="new", side="sell", client_order_id="protect-risk-cid", filled_qty="0"
        )
        canceled = make_order(
            "protect-risk", status="canceled", side="sell", client_order_id="protect-risk-cid", filled_qty="0"
        )
        sale = make_order("risk-market-sale", side="sell", filled_qty="0.004")
        events = []
        pending = {
            "protect-risk": main.PendingReconciliation(
                "SELL", "protective_stop_limit", None, None, None,
                "protect-risk-cid", "ma_rsi_crossover", "{}", None,
                "broker_order", "2026-10-05T00:00:00+00:00", 0.004,
                "protective_stop",
            )
        }

        def open_orders():
            events.append("list")
            return [protective] if events.count("list") == 1 else []

        def wait(order_id, **kwargs):
            events.append(("reconcile", order_id))
            return canceled if order_id == "protect-risk" else sale

        with (
            patch.object(main, "get_open_btc_orders", side_effect=open_orders),
            patch.object(main, "get_btc_position", return_value=make_position()),
            patch.object(main, "cancel_order", side_effect=lambda oid: events.append(("cancel", oid))),
            patch.object(main, "wait_for_order_fill", side_effect=wait),
            patch.object(main, "has_open_order", return_value=False),
            patch.object(main, "get_order_by_client_order_id", return_value=None),
            patch.object(main, "sell_btc", side_effect=lambda *a, **k: events.append("sell") or sale),
            patch.object(main, "upsert_order"),
        ):
            outcome = main.execute_risk_exit("take_profit", pending, None)
        self.assertEqual(outcome, broker.OrderOutcome.FILLED)
        self.assertNotIn("protect-risk", pending)
        self.assertLess(events.index(("cancel", "protect-risk")), events.index("sell"))
        self.assertLess(events.index(("reconcile", "protect-risk")), events.index("sell"))

    def test_rejected_order_keeps_signal_candle_processed_across_repeated_polls(self):
        candle = "2026-10-05T01:00:00+00:00"
        last_processed = main.processed_candle_after_order(
            candle, broker.OrderOutcome.TERMINAL_NOT_FILLED
        )
        self.assertEqual(last_processed, candle)
        self.assertTrue(all(
            not main.should_process_candle(candle, last_processed)
            for _ in range(100)
        ))

    def test_deterministic_client_order_ids_track_logical_attempt(self):
        identity = ("ma_rsi_crossover", "1Hour", "BTC/USD", "2026-10-05T00:00", "BUY")
        first = broker.deterministic_client_order_id(*identity)
        self.assertEqual(first, broker.deterministic_client_order_id(*identity))
        self.assertNotEqual(first, broker.deterministic_client_order_id(*identity[:-2], "01:00", "BUY"))
        self.assertNotEqual(first, broker.deterministic_client_order_id(*identity[:-1], "SELL"))
        self.assertLessEqual(len(first), 48)

    def test_ambiguous_buy_submit_reconciles_client_id_without_second_submission(self):
        found_order = make_order("broker-order", client_order_id="client-id")
        pending = {}
        with (
            patch.object(main, "get_order_by_client_order_id", side_effect=[None, found_order]),
            patch.object(main, "buy_btc", side_effect=TimeoutError("response lost")) as buy,
            patch.object(main, "wait_for_order_fill", return_value=found_order),
            patch.object(main, "upsert_order"),
        ):
            outcome = main._submit_and_log_buy("entry", pending, "candle-1")
        self.assertEqual(outcome, broker.OrderOutcome.FILLED)
        buy.assert_called_once()
        self.assertEqual(pending, {})

    def test_startup_reconstructs_unmatched_open_broker_orders(self):
        order = make_order(
            "open-buy", status="new", side="buy", client_order_id="bot-open"
        )
        pending = {}
        with (
            patch("database.get_order_records", return_value=[]),
            patch.object(main, "get_open_btc_orders", return_value=[order]),
            patch.object(main, "upsert_order"),
            patch("sys.stdout", new_callable=io.StringIO) as output,
        ):
            main.restore_pending_orders(pending)
        self.assertEqual(pending["open-buy"].side, "BUY")
        self.assertIn("incomplete DB context", output.getvalue())


class SubmissionIntentRecoveryTests(unittest.TestCase):
    def test_buy_post_422_or_429_adopts_filled_order_found_by_client_id(self):
        class HttpError(Exception):
            def __init__(self, status_code):
                self.status_code = status_code
                super().__init__(str(status_code))

        for status_code in (422, 429):
            with self.subTest(status_code=status_code):
                candle = f"candle-post-error-{status_code}"
                strategy = main._live_strategy()
                client_id = main._strategy_order_id(
                    strategy, candle, "BUY", role="strategy_entry"
                )
                filled = make_order(
                    f"filled-after-{status_code}", client_order_id=client_id,
                    filled_qty="0.0002",
                )
                position = make_position(qty="0.0001995")
                runtime = main.LiveRuntime()
                pending = {}
                with (
                    patch.object(
                        main, "get_order_by_client_order_id",
                        side_effect=[None, filled],
                    ) as lookup,
                    patch.object(main, "buy_btc", side_effect=HttpError(status_code)) as buy,
                    patch.object(main, "wait_for_order_fill", return_value=filled),
                    patch.object(main, "get_btc_position", return_value=position),
                    patch.object(main, "upsert_order") as write,
                    patch("database.set_active_bot_position"),
                    patch.object(main, "_place_protective_stop"),
                ):
                    outcome = main._submit_and_log_buy(
                        "entry", pending, candle, strategy=strategy, runtime=runtime
                    )

                self.assertEqual(outcome, broker.OrderOutcome.FILLED)
                self.assertEqual(lookup.call_count, 2)
                buy.assert_called_once()
                self.assertAlmostEqual(
                    runtime.active_position["credited_quantity"], 0.0001995
                )
                owned, reason = main.bot_owns_position(position, runtime)
                self.assertTrue(owned, reason)
                self.assertFalse(
                    any(
                        call.kwargs.get("order_status") in {"rejected", "rate_limited"}
                        for call in write.call_args_list
                    )
                )
                self.assertEqual(pending, {})

    def test_buy_post_422_with_confirmed_not_found_is_terminal_rejection(self):
        class HttpError(Exception):
            status_code = 422

        candle = "candle-post-422-not-found"
        pending = {}
        with (
            patch.object(
                main, "get_order_by_client_order_id", side_effect=[None, None]
            ) as lookup,
            patch.object(main, "buy_btc", side_effect=HttpError("422")) as buy,
            patch.object(main, "upsert_order") as write,
        ):
            outcome = main._submit_and_log_buy("entry", pending, candle)

        self.assertEqual(outcome, broker.OrderOutcome.TERMINAL_NOT_FILLED)
        self.assertEqual(lookup.call_count, 2)
        buy.assert_called_once()
        self.assertEqual(pending, {})
        self.assertTrue(
            any(call.kwargs.get("order_status") == "rejected" for call in write.call_args_list)
        )

    def test_preflight_lookup_failure_skips_buy_without_synthetic_pending_order(self):
        pending = {}
        with (
            patch.object(main, "get_order_by_client_order_id", side_effect=RuntimeError("broker unavailable")),
            patch.object(main, "buy_btc") as buy,
            patch.object(main, "upsert_order") as write,
            redirect_stdout(io.StringIO()),
        ):
            outcome = main._submit_and_log_buy("entry", pending, "candle-preflight")
        self.assertEqual(outcome, broker.OrderOutcome.TERMINAL_NOT_FILLED)
        buy.assert_not_called()
        self.assertEqual(pending, {})
        self.assertEqual(write.call_args.kwargs["order_status"], "entry_preflight_unavailable")
        self.assertEqual(write.call_args.kwargs["submission_kind"], "preflight_unavailable")
        self.assertIsNone(write.call_args.kwargs["client_order_id"])

    def test_buy_intent_is_persisted_before_broker_post(self):
        order = make_order("intent-order", client_order_id="will-be-attached")
        events = []

        def persist(**kwargs):
            events.append(("persist", kwargs.get("order_status")))

        def submit(*args, **kwargs):
            self.assertEqual(events[0], ("persist", "submission_intent"))
            events.append(("post", kwargs.get("client_order_id")))
            return order

        with (
            patch.object(main, "get_order_by_client_order_id", return_value=None),
            patch.object(main, "upsert_order", side_effect=persist),
            patch.object(main, "buy_btc", side_effect=submit),
            patch.object(main, "wait_for_order_fill", return_value=order) as wait,
            patch.object(main, "_place_protective_stop"),
        ):
            result = main._submit_and_log_buy("entry", {}, "candle-intent")
        self.assertEqual(result, broker.OrderOutcome.FILLED)
        self.assertEqual(events[0], ("persist", "submission_intent"))
        self.assertLess(events.index(("post", events[1][1])), events.index(("persist", "filled")))
        wait.assert_called_once_with("intent-order")

    def test_failed_buy_intent_write_prevents_broker_post(self):
        runtime = main.LiveRuntime()
        with (
            patch.object(main, "get_order_by_client_order_id", return_value=None),
            patch.object(main, "upsert_order", side_effect=OSError("database locked")),
            patch.object(main, "buy_btc") as buy,
            redirect_stdout(io.StringIO()),
        ):
            outcome = main._submit_and_log_buy("entry", {}, "candle-db-fail", runtime=runtime)
        self.assertEqual(outcome, broker.OrderOutcome.TERMINAL_NOT_FILLED)
        buy.assert_not_called()
        self.assertTrue(runtime.entries_disabled)

    def test_restart_recovers_crash_after_buy_submit_using_net_broker_quantity(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            config, "DATABASE_PATH", str(Path(temp_dir) / "intent.sqlite")
        ):
            candle = "2026-10-05T00:00:00+00:00"
            strategy = main._live_strategy()
            client_id = main._strategy_order_id(strategy, candle, "BUY", role="strategy_entry")
            accepted = make_order("accepted-buy", status="new", side="buy", client_order_id=client_id, filled_qty="0")
            with (
                patch.object(main, "get_order_by_client_order_id", return_value=None),
                patch.object(main, "buy_btc", return_value=accepted),
                patch.object(main, "wait_for_order_fill", side_effect=KeyboardInterrupt("simulated crash")),
            ):
                with self.assertRaises(KeyboardInterrupt):
                    main._submit_and_log_buy("entry", {}, candle)
            saved = database.get_order_records(pending_only=True)
            self.assertEqual(len(saved), 1)
            self.assertEqual(saved[0]["order_status"], "new")
            self.assertEqual(saved[0]["client_order_id"], client_id)
            self.assertEqual(saved[0]["candle_timestamp"], candle)

            filled = make_order("accepted-buy", status="filled", side="buy", client_order_id=client_id, filled_qty="0.003")
            credited = make_position(qty="0.0029925")
            runtime = main.LiveRuntime()
            pending = {}
            with (
                patch.object(main, "get_open_btc_orders", return_value=[]),
                patch.object(main, "reconcile_order", return_value=filled),
                patch.object(main, "get_btc_position", return_value=credited),
                patch.object(main, "_place_protective_stop"),
            ):
                main.restore_pending_orders(pending, strategy, runtime=runtime)
            self.assertEqual(pending, {})
            self.assertAlmostEqual(runtime.active_position["credited_quantity"], 0.0029925)
            self.assertAlmostEqual(database.get_order_records()[0]["asset_quantity_delta"], 0.0029925)
            owned, reason = main.bot_owns_position(credited, runtime)
            self.assertTrue(owned, reason)

    def test_restart_marks_pre_post_intent_terminal_after_confirmed_not_found(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            config, "DATABASE_PATH", str(Path(temp_dir) / "not-created.sqlite")
        ):
            client_id = "bot-not-created"
            created = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
            database.upsert_order(
                order_id=f"client:{client_id}", client_order_id=client_id,
                symbol="BTC/USD", side="BUY", requested_notional=20,
                quantity=None, fill_price=None, reason="entry",
                order_status="submission_intent", submission_kind="entry_intent",
                created_at=created, reconcile_attempts=0,
                position_before_quantity=0, order_role="strategy_entry",
            )
            pending = {}
            with (
                patch.object(main, "get_open_btc_orders", return_value=[]),
                patch.object(main, "reconcile_order", side_effect=broker.BrokerOrderNotFound("404")),
                patch.object(trade_config, "SUBMIT_UNKNOWN_MAX_AGE_SECONDS", 0),
                patch.object(trade_config, "SUBMIT_UNKNOWN_MAX_RECONCILE_ATTEMPTS", 1),
            ):
                main.restore_pending_orders(pending)
            self.assertEqual(pending, {})
            self.assertEqual(database.get_order_records()[0]["order_status"], "terminal_not_created")

    def test_restart_recovers_buy_accepted_before_broker_response(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            config, "DATABASE_PATH", str(Path(temp_dir) / "response-lost.sqlite")
        ):
            candle = "2026-10-05T00:05:00+00:00"
            strategy = main._live_strategy()
            client_id = main._strategy_order_id(strategy, candle, "BUY", role="strategy_entry")
            with (
                patch.object(main, "get_order_by_client_order_id", return_value=None),
                patch.object(main, "buy_btc", side_effect=KeyboardInterrupt("response lost")) as buy,
            ):
                with self.assertRaises(KeyboardInterrupt):
                    main._submit_and_log_buy("entry", {}, candle)
            buy.assert_called_once()
            self.assertEqual(database.get_order_records(pending_only=True)[0]["order_status"], "submission_intent")

            filled = make_order("response-lost-order", status="filled", side="buy", client_order_id=client_id, filled_qty="0.002")
            credited = make_position(qty="0.001995")
            runtime = main.LiveRuntime()
            with (
                patch.object(main, "get_open_btc_orders", return_value=[]),
                patch.object(main, "reconcile_order", return_value=filled) as reconcile,
                patch.object(main, "get_btc_position", return_value=credited),
                patch.object(main, "_place_protective_stop"),
            ):
                main.restore_pending_orders({}, strategy, runtime=runtime)
            reconcile.assert_called_once()
            self.assertAlmostEqual(runtime.active_position["credited_quantity"], 0.001995)

    def test_recovered_filled_sell_clears_active_provenance_when_broker_flat(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            config, "DATABASE_PATH", str(Path(temp_dir) / "sell.sqlite")
        ):
            source_buy = "bot-source-buy"
            database.set_active_bot_position({
                "asset_id": "btc-asset", "source_order_id": "source-order",
                "source_client_order_id": source_buy, "credited_quantity": 0.004,
                "source_confirmed": True, "strategy_name": "ma_rsi_crossover",
            })
            sell_cid = "bot-recovered-sell"
            database.upsert_order(
                order_id="recovered-sell", client_order_id=sell_cid,
                symbol="BTC/USD", side="SELL", requested_notional=160,
                quantity=None, fill_price=None, reason="stop_loss",
                order_status="timeout_pending", submission_kind="broker_order",
                position_before_quantity=0.004, order_role="risk_exit_stop_loss",
                strategy_name="ma_rsi_crossover",
            )
            filled = make_order("recovered-sell", status="filled", side="sell", client_order_id=sell_cid, filled_qty="0.004")
            runtime = main.LiveRuntime(active_position=database.get_active_bot_position())
            with (
                patch.object(main, "get_open_btc_orders", return_value=[]),
                patch.object(main, "reconcile_order", return_value=filled),
                patch.object(main, "get_btc_position", return_value=None),
            ):
                main.restore_pending_orders({}, runtime=runtime)
            self.assertIsNone(runtime.active_position)
            self.assertIsNone(database.get_active_bot_position())
            self.assertAlmostEqual(database.get_order_records()[0]["asset_quantity_delta"], -0.004)


class LiveStrategyAndProtectionTests(unittest.TestCase):
    def _run_until_first_sleep(self, live_strategy, timeframe, active_position, position):
        state = broker.PositionLookup(
            broker.PositionLookupStatus.CONFIRMED_POSITION, position
        )
        output = io.StringIO()
        with (
            patch.object(trade_config, "LIVE_STRATEGY", live_strategy),
            patch.object(trade_config, "LIVE_TIMEFRAME", timeframe),
            patch.object(main, "init_db", return_value=True),
            patch.object(main.client, "get_account", return_value=SimpleNamespace()),
            patch("database.get_active_bot_position", return_value=active_position),
            patch("database.get_state", return_value=None),
            patch("database.set_state"),
            patch("database.set_active_bot_position") as save_active,
            patch.object(main, "restore_pending_orders") as restore,
            patch.object(main, "reconcile_pending_orders"),
            patch.object(main, "lookup_btc_position", return_value=state),
            patch.object(main, "get_btc_market_price", return_value=None),
            patch.object(main, "get_btc_bars", return_value=None),
            patch.object(main.time, "sleep", side_effect=KeyboardInterrupt),
            redirect_stdout(output),
        ):
            try:
                main.run()
            except KeyboardInterrupt:
                pass
        return output.getvalue(), restore, save_active

    def test_startup_safe_halts_for_ma_position_when_regime_strategy_selected(self):
        active = {
            "asset_id": "btc-asset", "credited_quantity": 0.004,
            "source_confirmed": True, "strategy_name": "ma_rsi_crossover",
        }
        state = broker.PositionLookup(
            broker.PositionLookupStatus.CONFIRMED_POSITION, make_position()
        )
        output = io.StringIO()
        with (
            patch.object(trade_config, "LIVE_STRATEGY", "regime_only_4h"),
            patch.object(trade_config, "LIVE_TIMEFRAME", "4Hour"),
            patch.object(main, "init_db", return_value=True),
            patch.object(main.client, "get_account", return_value=SimpleNamespace()),
            patch("database.get_active_bot_position", return_value=active),
            patch.object(main, "lookup_btc_position", return_value=state),
            patch.object(main, "restore_pending_orders") as restore,
            patch.object(main, "buy_btc") as buy,
            patch.object(main, "sell_btc") as sell,
            redirect_stdout(output),
        ):
            main.run()
        restore.assert_not_called()
        buy.assert_not_called()
        sell.assert_not_called()
        self.assertIn("SAFE-HALT STRATEGY MISMATCH", output.getvalue())

    def test_startup_safe_halts_for_regime_position_when_ma_strategy_selected(self):
        active = {
            "asset_id": "btc-asset", "credited_quantity": 0.004,
            "source_confirmed": True, "strategy_name": "regime_only_4h",
        }
        state = broker.PositionLookup(
            broker.PositionLookupStatus.CONFIRMED_POSITION, make_position()
        )
        output = io.StringIO()
        with (
            patch.object(trade_config, "LIVE_STRATEGY", "ma_rsi_crossover"),
            patch.object(trade_config, "LIVE_TIMEFRAME", "5Min"),
            patch.object(main, "init_db", return_value=True),
            patch.object(main.client, "get_account", return_value=SimpleNamespace()),
            patch("database.get_active_bot_position", return_value=active),
            patch.object(main, "lookup_btc_position", return_value=state),
            patch.object(main, "restore_pending_orders") as restore,
            patch.object(main, "buy_btc") as buy,
            patch.object(main, "sell_btc") as sell,
            redirect_stdout(output),
        ):
            main.run()
        restore.assert_not_called()
        buy.assert_not_called()
        sell.assert_not_called()
        self.assertIn("SAFE-HALT STRATEGY MISMATCH", output.getvalue())

    def test_matching_active_strategy_provenance_starts_normally(self):
        active = {
            "asset_id": "btc-asset", "credited_quantity": 0.004,
            "source_confirmed": True, "strategy_name": "ma_rsi_crossover",
        }
        output, restore, _save_active = self._run_until_first_sleep(
            "ma_rsi_crossover", "5Min", active, make_position()
        )
        restore.assert_called_once()
        self.assertIn("PAPER BOT STARTED", output)
        self.assertNotIn("SAFE-HALT STRATEGY MISMATCH", output)

    def test_flat_account_clears_stale_mismatched_provenance_and_starts(self):
        active = {
            "asset_id": "btc-asset", "credited_quantity": 0.004,
            "source_confirmed": True, "strategy_name": "ma_rsi_crossover",
        }
        output, restore, save_active = self._run_until_first_sleep(
            "regime_only_4h", "4Hour", active, None
        )
        restore.assert_called_once()
        save_active.assert_called_once_with(None)
        self.assertIn("PAPER BOT STARTED", output)
        self.assertNotIn("SAFE-HALT STRATEGY MISMATCH", output)

    def test_persisted_regime_order_restores_with_registered_regime_strategy(self):
        import strategy

        persisted = main._context_from_record({
            "side": "BUY", "reason": "regime_on", "requested_notional": 20,
            "order_id": "pending-regime-buy", "client_order_id": "regime-buy-client",
            "strategy_name": "regime_only_4h", "strategy_parameters_json": "{}",
            "submission_kind": "broker_order", "created_at": "2026-10-05T00:00:00Z",
            "position_before_quantity": 0, "order_role": "strategy_entry",
        })
        with (
            patch.object(trade_config, "LIVE_STRATEGY", "ma_rsi_crossover"),
            patch.object(main, "reconcile_order", return_value=make_order(
                "pending-regime-buy", status="canceled", client_order_id="regime-buy-client"
            )),
            patch.object(main, "_safe_record_order") as record,
        ):
            main.reconcile_pending_orders(
                {"pending-regime-buy": persisted}, main._live_strategy()
            )
        self.assertIs(record.call_args.kwargs["strategy"], strategy.REGIME_ONLY_4H)
        self.assertIs(strategy.get_strategy("regime_only_4h"), strategy.REGIME_ONLY_4H)

    def test_default_live_strategy_timeframe_and_ma_behavior_are_preserved(self):
        selected = main._live_strategy()
        self.assertEqual(trade_config.LIVE_STRATEGY, "ma_rsi_crossover")
        self.assertEqual(trade_config.LIVE_TIMEFRAME, "5Min")
        import pandas as pd
        import strategy

        bars = pd.DataFrame({"close": [100 + ((index % 9) - 4) for index in range(80)]})
        prepared = selected.prepare_indicators(bars)
        pd.testing.assert_frame_equal(prepared, strategy.calculate_indicators(bars))
        self.assertEqual(
            selected.decide_at(prepared, len(prepared) - 1),
            strategy.decide(prepared),
        )

    def test_live_allow_list_keeps_ma_rsi_and_uses_frozen_regime_factory(self):
        import strategy

        with patch.object(trade_config, "LIVE_STRATEGY", "ma_rsi_crossover"):
            self.assertIs(main._live_strategy(), strategy.MA_RSI_CROSSOVER)

        with (
            patch.object(trade_config, "LIVE_STRATEGY", "regime_only_4h"),
            patch.object(trade_config, "LIVE_TIMEFRAME", "4Hour"),
        ):
            selected = main._live_strategy()
        self.assertIs(selected, strategy.REGIME_ONLY_4H)
        self.assertEqual(selected.parameters["regime_sma_period"], 200)
        self.assertEqual(selected.parameters["regime_slope_lookback"], 20)
        self.assertIsNone(selected.stop_loss_percent)
        self.assertIsNone(selected.take_profit_percent)
        self.assertIsNone(selected.max_holding_minutes)

    def test_regime_only_live_strategy_rejects_non_four_hour_timeframe(self):
        with (
            patch.object(trade_config, "LIVE_STRATEGY", "regime_only_4h"),
            patch.object(trade_config, "LIVE_TIMEFRAME", "1Hour"),
        ):
            with self.assertRaisesRegex(RuntimeError, "requires LIVE_TIMEFRAME"):
                main._live_strategy()

    def test_unknown_live_strategy_is_rejected(self):
        with patch.object(trade_config, "LIVE_STRATEGY", "not_a_strategy"):
            with self.assertRaisesRegex(RuntimeError, "restricted"):
                main._live_strategy()

    def test_regime_only_live_signals_and_not_ready_hold_match_frozen_factory(self):
        import numpy as np
        import pandas as pd
        from strategy import Decision, create_regime_only_strategy

        frozen = create_regime_only_strategy()
        index = pd.date_range("2024-01-01", periods=260, freq="4h", tz="UTC")
        bullish = pd.DataFrame({"close": np.arange(100.0, 360.0)}, index=index)
        bearish = pd.DataFrame({"close": np.arange(360.0, 100.0, -1.0)}, index=index)
        self.assertEqual(
            frozen.decide_at(frozen.prepare_indicators(bullish), len(bullish) - 1),
            Decision("BUY", "regime_on"),
        )
        self.assertEqual(
            frozen.decide_at(frozen.prepare_indicators(bearish), len(bearish) - 1),
            Decision("SELL", "regime_filter_off"),
        )
        short_history = bullish.iloc[:219]
        self.assertEqual(
            frozen.decide_at(
                frozen.prepare_indicators(short_history), len(short_history) - 1
            ),
            Decision("HOLD", "regime_filter_not_ready"),
        )

    def test_regime_only_without_fixed_stops_is_safe_in_live_risk_path(self):
        import strategy

        selected = strategy.REGIME_ONLY_4H
        position = make_position()
        self.assertIsNone(main._risk_exit_reason(position, 1.0, selected))
        self.assertIsNone(main._risk_exit_reason(position, 1_000_000.0, selected))
        with patch.object(trade_config, "ENABLE_BROKER_STOP_LIMIT", True):
            self.assertIsNone(
                main._place_protective_stop(
                    make_order(), "candle-1", {}, selected,
                    position=position,
                    runtime=main.LiveRuntime(active_position={"source_confirmed": True}),
                )
            )

    def test_live_strategy_rejects_donchian_and_uses_spec_risk_values(self):
        with patch.object(trade_config, "LIVE_STRATEGY", "donchian_breakout"):
            with self.assertRaisesRegex(RuntimeError, "restricted"):
                main._live_strategy()
        spec = StrategySpec(
            name="test",
            _prepare_indicators=lambda bars: bars,
            _decide_at=lambda bars, index: Decision("HOLD", "test"),
            _warmup_lookback=lambda: 70,
            _parameters=lambda: {},
            _stop_loss=lambda: 0.10,
            _take_profit=lambda: 0.20,
        )
        position = make_position()
        self.assertEqual(spec.required_warmup_bars(), 80)
        self.assertEqual(main._risk_exit_reason(position, 35500, spec), "stop_loss")
        self.assertEqual(main._risk_exit_reason(position, 48500, spec), "take_profit")

    def test_live_market_data_warmup_is_requested_from_strategy_spec(self):
        import pandas as pd

        selected = Mock()
        selected.required_warmup_bars.return_value = 77
        request_factory = Mock(side_effect=lambda **kwargs: SimpleNamespace(**kwargs))
        with (
            patch.object(market_data, "CryptoBarsRequest", request_factory),
            patch.object(
                market_data.data_client,
                "get_crypto_bars",
                return_value=SimpleNamespace(df=pd.DataFrame()),
            ),
        ):
            self.assertIsNone(market_data.get_btc_bars(limit_bars=10, strategy=selected))
        selected.required_warmup_bars.assert_called_once_with()
        request = request_factory.call_args.kwargs
        self.assertEqual(request["start"], request["end"] - timedelta(minutes=5 * 81))

    def test_broker_stop_limit_uses_actual_broker_position_quantity(self):
        filled_buy = make_order(filled_qty="0.0037", filled_avg_price="42000")
        protective = make_order(
            "protect-1", status="new", side="sell", client_order_id="protect-id",
            filled_qty="0",
        )
        actual_position = make_position(qty="0.00369075")
        runtime = main.LiveRuntime(active_position={"source_confirmed": True})
        with (
            patch.object(trade_config, "ENABLE_BROKER_STOP_LIMIT", True),
            patch.object(main, "get_open_btc_orders", return_value=[]),
            patch.object(main, "get_order_by_client_order_id", return_value=None),
            patch.object(main, "submit_protective_stop_limit", return_value=protective) as submit,
            patch.object(main, "upsert_order"),
        ):
            result = main._place_protective_stop(
                filled_buy, "candle-1", {}, main._live_strategy(),
                position=actual_position, runtime=runtime,
            )
        self.assertIs(result, protective)
        self.assertAlmostEqual(submit.call_args.args[0], 0.00369075)

    def test_protective_sell_is_canceled_and_reconciled_before_normal_close(self):
        protective = make_order(
            "protect-1", status="new", side="sell", client_order_id="protect-id"
        )
        canceled = make_order(
            "protect-1", status="canceled", side="sell", client_order_id="protect-id", filled_qty="0"
        )
        sale = make_order("normal-sale", side="sell", filled_qty="0.004")
        events = []

        def open_orders():
            events.append("list_open")
            return [protective] if events.count("list_open") == 1 else []

        def wait(order_id, **kwargs):
            events.append(("reconcile", order_id))
            return canceled if order_id == "protect-1" else sale

        position = make_position()
        pending = {"protect-1": main.PendingReconciliation(
            "SELL", "protective_stop_limit", None, None, "candle-1",
            "protect-id", "ma_rsi_crossover", "{}", None,
            "broker_order", "2026-10-05T00:00:00+00:00", 0.004,
            "protective_stop",
        )}
        self.assertFalse(main._has_pending_nonprotective_sell(pending))
        self.assertFalse(main._has_pending_entry_conflict(pending))
        with (
            patch.object(main, "get_open_btc_orders", side_effect=open_orders),
            patch.object(main, "cancel_order", side_effect=lambda order_id: events.append(("cancel", order_id))),
            patch.object(main, "wait_for_order_fill", side_effect=wait),
            patch.object(main, "get_btc_position", return_value=position),
            patch.object(main, "has_open_order", return_value=False),
            patch.object(main, "get_order_by_client_order_id", return_value=None),
            patch.object(main, "sell_btc", side_effect=lambda *args, **kwargs: events.append("sell") or sale),
            patch.object(main, "upsert_order"),
        ):
            main._submit_and_log_sell("signal_exit", position, pending, "candle-1")
        self.assertLess(events.index(("cancel", "protect-1")), events.index("sell"))
        self.assertLess(events.index(("reconcile", "protect-1")), events.index("sell"))
        self.assertEqual(pending, {})

    def test_protective_fill_during_cancel_race_refreshes_position_and_persists_delta(self):
        protective = make_order(
            "protect-race", status="new", side="sell", client_order_id="protect-race-cid", filled_qty="0"
        )
        filled = make_order(
            "protect-race", status="filled", side="sell", client_order_id="protect-race-cid", filled_qty="0.004"
        )
        runtime = main.LiveRuntime(active_position={
            "asset_id": "btc-asset", "source_order_id": "buy-order",
            "source_client_order_id": "bot-buy", "credited_quantity": 0.004,
            "source_confirmed": True,
        })
        rows = {}
        pending = {"protect-race": main.PendingReconciliation(
            "SELL", "protective_stop_limit", None, None, None,
            "protect-race-cid", "ma_rsi_crossover", "{}", None,
            "broker_order", "2026-10-05T00:00:00+00:00", 0.004,
            "protective_stop",
        )}
        with (
            patch.object(main, "get_open_btc_orders", side_effect=[[protective], []]),
            patch.object(main, "cancel_order"),
            patch.object(main, "wait_for_order_fill", return_value=filled),
            patch.object(main, "get_btc_position", return_value=None) as refresh,
            patch.object(main, "upsert_order", side_effect=lambda **kwargs: rows.update(kwargs)),
            patch("database.set_active_bot_position"),
        ):
            main._cancel_protective_stops(
                pending, main._live_strategy(), runtime,
                position_before_quantity=0.004,
            )
        refresh.assert_called_once()
        self.assertIsNone(runtime.active_position)
        self.assertEqual(rows["order_role"], "protective_stop")
        self.assertAlmostEqual(rows["asset_quantity_delta"], -0.004)
        self.assertNotIn("protect-race", pending)

    def test_flat_position_cleanup_verifies_no_stale_protective_sell(self):
        protective = make_order(
            "protect-1", status="new", side="sell", client_order_id="protect-id"
        )
        canceled = make_order(
            "protect-1", status="canceled", side="sell", client_order_id="protect-id", filled_qty="0"
        )
        with (
            patch.object(main, "get_open_btc_orders", side_effect=[[protective], [protective], []]) as listing,
            patch.object(main, "cancel_order") as cancel,
            patch.object(main, "wait_for_order_fill", return_value=canceled),
            patch.object(main, "upsert_order"),
        ):
            main._cancel_stale_protective_sells_if_flat(None, {}, main._live_strategy())
        cancel.assert_called_once_with("protect-1")
        self.assertEqual(listing.call_count, 3)

    def test_flat_cleanup_halts_if_protective_sell_remains_open(self):
        protective = make_order(
            "protect-1", status="new", side="sell", client_order_id="protect-id"
        )
        with (
            patch.object(main, "get_open_btc_orders", return_value=[protective]),
            patch.object(main, "cancel_order"),
            patch.object(main, "wait_for_order_fill", return_value=make_order(
                "protect-1", status="canceled", side="sell", client_order_id="protect-id", filled_qty="0"
            )),
            patch.object(main, "upsert_order"),
        ):
            with self.assertRaisesRegex(RuntimeError, "remain open"):
                main._cancel_stale_protective_sells_if_flat(None, {}, main._live_strategy())

    def test_paper_smoke_dry_run_never_contacts_broker_or_submits(self):
        with (
            patch.object(paper_position_smoke_test.broker, "is_paper_client") as is_paper,
            patch.object(paper_position_smoke_test.broker.client, "get_account") as account,
            patch.object(paper_position_smoke_test.broker, "lookup_btc_position") as lookup,
            patch.object(paper_position_smoke_test, "get_open_btc_orders") as orders,
            patch.object(
                paper_position_smoke_test.broker, "get_order_by_client_order_id"
            ) as lookup_order,
            patch.object(paper_position_smoke_test.broker, "buy_btc") as buy,
            patch.object(paper_position_smoke_test.broker, "sell_btc") as sell,
            redirect_stdout(io.StringIO()) as output,
        ):
            self.assertEqual(paper_position_smoke_test.main([]), 0)
        is_paper.assert_not_called()
        account.assert_not_called()
        lookup.assert_not_called()
        orders.assert_not_called()
        lookup_order.assert_not_called()
        buy.assert_not_called()
        sell.assert_not_called()
        self.assertIn("DRY RUN", output.getvalue())


if __name__ == "__main__":
    unittest.main()
