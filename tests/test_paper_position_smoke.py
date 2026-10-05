import unittest
from types import SimpleNamespace
from unittest.mock import patch

import broker
import paper_position_smoke_test as smoke_test
import trade_config


def position(qty, symbol="BTCUSD"):
    return SimpleNamespace(
        qty=str(qty),
        symbol=symbol,
        asset_id="btc-asset",
        market_value="40000",
    )


class PaperPositionQuantityTests(unittest.TestCase):
    def test_equal_filled_and_position_quantities_are_accepted(self):
        self.assertEqual(smoke_test._validate_position_quantity(1.0, 1.0), 0.0)

    def test_crypto_fee_quantity_reduction_of_point_25_percent_is_accepted(self):
        filled = 0.5
        credited = filled * (1 - 0.0025)
        self.assertAlmostEqual(
            smoke_test._validate_position_quantity(filled, credited), 0.0025
        )

    def test_reasonable_fee_within_configured_limit_is_accepted(self):
        filled = 0.25
        configured_fee = 0.0065
        with patch.object(
            trade_config, "PAPER_SMOKE_MAX_BUY_FEE_RATE", 0.008
        ):
            inferred = smoke_test._validate_position_quantity(
                filled, filled * (1 - configured_fee)
            )
        self.assertAlmostEqual(inferred, configured_fee)

    def test_position_quantity_above_fill_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "exceeds this smoke BUY fill"):
            smoke_test._validate_position_quantity(1.0, 1.000001)

    def test_fee_above_configured_maximum_is_rejected(self):
        filled = 1.0
        above_limit = filled * (1 - trade_config.PAPER_SMOKE_MAX_BUY_FEE_RATE - 0.001)
        with self.assertRaisesRegex(RuntimeError, "position will not be closed"):
            smoke_test._validate_position_quantity(filled, above_limit)

    def test_fee_above_limit_aborts_execute_without_closing_position(self):
        filled_order = SimpleNamespace(
            id="paper-buy",
            status=SimpleNamespace(value="filled"),
            filled_qty="1.0",
            filled_avg_price="40000",
        )
        broker_position = position(1.0 * (1 - trade_config.PAPER_SMOKE_MAX_BUY_FEE_RATE - 0.001))
        with (
            patch.object(smoke_test.broker, "is_paper_client", return_value=True),
            patch.object(
                smoke_test.broker.client,
                "get_account",
                return_value=SimpleNamespace(status="ACTIVE"),
            ),
            patch.object(
                smoke_test.broker,
                "lookup_btc_position",
                return_value=broker.PositionLookup(
                    broker.PositionLookupStatus.CONFIRMED_FLAT
                ),
            ),
            patch.object(smoke_test, "get_open_btc_orders", return_value=[]),
            patch.object(smoke_test.broker, "get_order_by_client_order_id", return_value=None),
            patch.object(smoke_test.broker, "buy_btc", return_value=filled_order) as buy,
            patch.object(smoke_test, "wait_for_order_fill", return_value=filled_order),
            patch.object(smoke_test, "_wait_for_position", return_value=broker_position),
            patch.object(smoke_test.broker, "btc_symbol_matches", return_value=True),
            patch.object(smoke_test, "sell_btc") as sell,
        ):
            with self.assertRaisesRegex(RuntimeError, "position will not be closed"):
                smoke_test._execute_diagnostic(20.0)
        buy.assert_called_once()
        sell.assert_not_called()


if __name__ == "__main__":
    unittest.main()
