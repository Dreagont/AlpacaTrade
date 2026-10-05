import io
import unittest
from contextlib import redirect_stdout
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock, patch

import broker
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


class LiveStrategyAndProtectionTests(unittest.TestCase):
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

    def test_broker_stop_limit_uses_confirmed_filled_quantity(self):
        filled_buy = make_order(filled_qty="0.0037", filled_avg_price="42000")
        protective = make_order(
            "protect-1", status="new", side="sell", client_order_id="protect-id"
        )
        with (
            patch.object(trade_config, "ENABLE_BROKER_STOP_LIMIT", True),
            patch.object(main, "get_open_btc_orders", return_value=[]),
            patch.object(main, "get_order_by_client_order_id", return_value=None),
            patch.object(main, "submit_protective_stop_limit", return_value=protective) as submit,
            patch.object(main, "upsert_order"),
        ):
            result = main._place_protective_stop(
                filled_buy, "candle-1", {}, main._live_strategy()
            )
        self.assertIs(result, protective)
        self.assertAlmostEqual(submit.call_args.args[0], 0.0037)

    def test_protective_sell_is_canceled_and_reconciled_before_normal_close(self):
        protective = make_order(
            "protect-1", status="new", side="sell", client_order_id="protect-id"
        )
        canceled = make_order(
            "protect-1", status="canceled", side="sell", client_order_id="protect-id"
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
            main._submit_and_log_sell("stop_loss", position, {}, "candle-1")
        self.assertLess(events.index(("cancel", "protect-1")), events.index("sell"))
        self.assertLess(events.index(("reconcile", "protect-1")), events.index("sell"))

    def test_flat_position_cleanup_verifies_no_stale_protective_sell(self):
        protective = make_order(
            "protect-1", status="new", side="sell", client_order_id="protect-id"
        )
        canceled = make_order(
            "protect-1", status="canceled", side="sell", client_order_id="protect-id"
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
                "protect-1", status="canceled", side="sell", client_order_id="protect-id"
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
