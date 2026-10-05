"""Manual paper-only position lookup/close diagnostic; never runs automatically."""

import argparse
import math
import time
from datetime import datetime, timezone

import broker
import config
import trade_config
from broker import (
    BrokerPositionStateUnknown,
    OrderFillTimeoutError,
    deterministic_client_order_id,
    get_btc_position,
    get_open_btc_orders,
    order_status_value,
    sell_btc,
    wait_for_order_fill,
)


def _validate_position_quantity(filled_quantity, position_quantity):
    """Validate a smoke BUY position while allowing plausible crypto-asset fees."""
    try:
        filled_quantity = float(filled_quantity)
        position_quantity = float(position_quantity)
        max_fee_rate = float(trade_config.PAPER_SMOKE_MAX_BUY_FEE_RATE)
    except (TypeError, ValueError) as error:
        raise RuntimeError("ABORT: BUY/position quantity or fee limit is invalid") from error

    if not math.isfinite(filled_quantity) or filled_quantity <= 0:
        raise RuntimeError("ABORT: smoke BUY has no valid positive filled quantity")
    if not math.isfinite(position_quantity) or position_quantity <= 0:
        raise RuntimeError(
            "ABORT: account BTC quantity is not positive; position will not be closed"
        )
    if not math.isfinite(max_fee_rate) or not 0 <= max_fee_rate < 1:
        raise RuntimeError("ABORT: PAPER_SMOKE_MAX_BUY_FEE_RATE must be in [0, 1)")

    quantity_tolerance = max(1e-12, filled_quantity * 1e-9)
    if position_quantity > filled_quantity + quantity_tolerance:
        raise RuntimeError(
            "ABORT: account BTC quantity exceeds this smoke BUY fill beyond numeric "
            "tolerance; possible pre-existing/manual BTC exposure, position will not be closed"
        )

    inferred_buy_fee_rate = 1 - (position_quantity / filled_quantity)
    fee_numeric_tolerance = quantity_tolerance / filled_quantity
    if (
        inferred_buy_fee_rate < -fee_numeric_tolerance
        or inferred_buy_fee_rate > max_fee_rate
    ):
        raise RuntimeError(
            f"ABORT: inferred BUY fee {inferred_buy_fee_rate:.4%} is outside the "
            f"allowed range (maximum {max_fee_rate:.2%}); position will not be closed"
        )
    # A tiny negative fee can result from accepted floating-point quantity noise.
    return max(0.0, inferred_buy_fee_rate)


def _wait_for_position(timeout_seconds=30, interval_seconds=1):
    deadline = time.monotonic() + timeout_seconds
    while True:
        position = get_btc_position()
        if position is not None:
            return position
        if time.monotonic() >= deadline:
            return None
        time.sleep(interval_seconds)


def _wait_for_flat(timeout_seconds=30, interval_seconds=1):
    deadline = time.monotonic() + timeout_seconds
    while True:
        position = get_btc_position()
        if position is None:
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(interval_seconds)


def _execute_diagnostic(amount_usd):
    if not broker.is_paper_client():
        raise RuntimeError("ABORT: broker client is not connected to Alpaca PAPER")
    if amount_usd <= 0 or amount_usd > trade_config.MAX_POSITION_USD:
        raise RuntimeError(
            f"ABORT: amount must be > 0 and <= MAX_POSITION_USD "
            f"(${trade_config.MAX_POSITION_USD:g})"
        )
    account = broker.client.get_account()
    if str(getattr(account, "status", "")).lower().endswith("inactive"):
        raise RuntimeError("ABORT: paper account is inactive")
    state = broker.lookup_btc_position()
    if state.status == broker.PositionLookupStatus.POSITION_STATE_UNKNOWN:
        raise BrokerPositionStateUnknown(f"ABORT: BTC position state unknown: {state.error}")
    if state.position is not None:
        raise RuntimeError("ABORT: pre-existing BTC position detected; it will not be managed")
    open_orders = get_open_btc_orders()
    if open_orders:
        raise RuntimeError(
            "ABORT: pre-existing open BTC order(s): "
            + ", ".join(str(order.id) for order in open_orders)
        )

    timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    buy_client_id = deterministic_client_order_id(
        "paper_position_smoke_test",
        "manual",
        config.SYMBOL,
        timestamp,
        "BUY",
    )
    if broker.get_order_by_client_order_id(buy_client_id) is not None:
        raise RuntimeError("ABORT: this diagnostic order identity already exists")
    print(f"1/6 PAPER confirmed; submitting one BTC/USD BUY for ${amount_usd:.2f}")
    submitted = broker.buy_btc(amount_usd, client_order_id=buy_client_id)
    try:
        buy_order = wait_for_order_fill(submitted.id, timeout_seconds=30)
    except OrderFillTimeoutError as timeout:
        print(f"BUY remains {order_status_value(timeout.order)}; requesting cancel and reconciling")
        broker.cancel_order(timeout.order.id)
        try:
            buy_order = wait_for_order_fill(timeout.order.id, timeout_seconds=15)
        except OrderFillTimeoutError as still_pending:
            raise RuntimeError(
                "ABORT: BUY cancellation is not confirmed; inspect broker order before proceeding"
            ) from still_pending
    filled_quantity = float(getattr(buy_order, "filled_qty", 0) or 0)
    if filled_quantity <= 0:
        raise RuntimeError(
            f"ABORT: smoke BUY did not fill (status={order_status_value(buy_order)})"
        )
    print(
        f"2/6 BUY reconciled: id={buy_order.id} status={order_status_value(buy_order)} "
        f"filled_qty={filled_quantity:g}"
    )

    position = _wait_for_position()
    if position is None:
        raise RuntimeError("ABORT: filled BUY has not appeared in account positions")
    if not broker.btc_symbol_matches(getattr(position, "symbol", None)):
        raise RuntimeError("ABORT: broker position identity is not BTC/USD")
    position_quantity = float(position.qty)
    inferred_buy_fee_rate = _validate_position_quantity(
        filled_quantity, position_quantity
    )
    print(
        "3/6 BTC position confirmed:\n"
        f"    symbol={position.symbol} asset_id={getattr(position, 'asset_id', None)}\n"
        f"    filled_qty={filled_quantity:.12g}\n"
        f"    position_qty={position_quantity:.12g}\n"
        f"    inferred_buy_fee={inferred_buy_fee_rate:.4%}\n"
        f"    market_value={getattr(position, 'market_value', None)}"
    )

    print("4/6 Closing only the validated smoke-test BTC position")
    close_order = sell_btc(position)
    if close_order is None:
        raise RuntimeError("ABORT: broker returned no close order for the validated position")
    try:
        close_order = wait_for_order_fill(close_order.id, timeout_seconds=30)
    except OrderFillTimeoutError as timeout:
        raise RuntimeError(
            "ABORT: BTC close remains pending; inspect the broker before retrying"
        ) from timeout
    if order_status_value(close_order) != "filled":
        raise RuntimeError(
            f"ABORT: close order did not fill (status={order_status_value(close_order)})"
        )
    print(f"5/6 Close reconciled: id={close_order.id} status=filled")
    if not _wait_for_flat():
        raise RuntimeError("ABORT: broker still reports BTC exposure after close")

    remaining_orders = get_open_btc_orders()
    stale_sells = [
        order for order in remaining_orders
        if str(getattr(getattr(order, "side", None), "value", getattr(order, "side", ""))).lower()
        == "sell"
    ]
    if stale_sells:
        for order in stale_sells:
            broker.cancel_order(order.id)
            try:
                wait_for_order_fill(order.id, timeout_seconds=15)
            except OrderFillTimeoutError as timeout:
                raise RuntimeError(
                    f"ABORT: stale SELL {order.id} is not confirmed canceled"
                ) from timeout
        if any(
            str(getattr(getattr(order, "side", None), "value", getattr(order, "side", ""))).lower()
            == "sell"
            for order in get_open_btc_orders()
        ):
            raise RuntimeError("ABORT: stale BTC SELL order remains open")
    print("6/6 Verified BTC is flat and no stale BTC SELL/protective order remains")


def _build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Manual Alpaca PAPER position lookup/close diagnostic. Use a dedicated "
            "paper account; an existing BTC position or BTC order causes an abort."
        )
    )
    parser.add_argument("--execute", action="store_true", help="Submit one small PAPER BUY and close it")
    parser.add_argument(
        "--amount-usd",
        type=float,
        default=trade_config.PAPER_SMOKE_TEST_AMOUNT_USD,
        help="Test notional in USD (default: configured small test amount)",
    )
    return parser


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not args.execute:
        print("DRY RUN: no broker requests or orders will be made.")
        print(
            f"Would verify Alpaca PAPER, require a flat BTC/USD account with no BTC orders, "
            f"then BUY and close ${args.amount_usd:.2f}."
        )
        print("Run with --execute only from a dedicated paper account after reviewing the checks.")
        return 0
    try:
        _execute_diagnostic(args.amount_usd)
    except Exception as error:
        print(f"PAPER POSITION DIAGNOSTIC ABORTED: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
