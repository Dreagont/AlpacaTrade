import os
import time
from enum import Enum

from dotenv import load_dotenv

from alpaca.common.exceptions import APIError
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest

import config


class OrderOutcome(str, Enum):
    FILLED = "filled"
    TERMINAL_NOT_FILLED = "terminal_not_filled"
    TIMEOUT_PENDING = "timeout_pending"
    PARTIAL_PENDING = "partial_pending"
    PENDING = "pending"


TERMINAL_ORDER_STATUSES = {
    "filled",
    "canceled",
    "expired",
    "replaced",
    "done_for_day",
    "rejected",
    "stopped",
}


def order_status_value(order):
    value = getattr(order, "status", None)
    return str(getattr(value, "value", value)).lower()


def classify_order(order, *, timed_out=False):
    status = order_status_value(order)
    if status == "filled":
        return OrderOutcome.FILLED
    if timed_out:
        return OrderOutcome.TIMEOUT_PENDING
    if status in TERMINAL_ORDER_STATUSES:
        return OrderOutcome.TERMINAL_NOT_FILLED
    try:
        filled_quantity = float(getattr(order, "filled_qty", 0) or 0)
    except (TypeError, ValueError):
        filled_quantity = 0.0
    if filled_quantity > 0:
        return OrderOutcome.PARTIAL_PENDING
    return OrderOutcome.PENDING


class OrderFillTimeoutError(TimeoutError):
    """Raised when an order remains non-terminal after the reconciliation timeout."""

    def __init__(self, order):
        self.order = order
        self.outcome = OrderOutcome.TIMEOUT_PENDING
        status = order_status_value(order)
        super().__init__(f"Order {order.id} is still non-terminal (status={status})")

load_dotenv()

client = TradingClient(
    os.getenv("ALPACA_API_KEY"),
    os.getenv("ALPACA_SECRET_KEY"),
    paper=True,
)


def buy_btc(amount_usd):
    order = MarketOrderRequest(
        symbol=config.SYMBOL,
        notional=amount_usd,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.GTC,
    )

    return client.submit_order(order_data=order)


def sell_btc():
    try:
        return client.close_position(config.SYMBOL)
    except APIError as error:
        if error.status_code == 404:
            return None
        raise


def has_open_order(side: OrderSide) -> bool:
    orders = client.get_orders(
        filter=GetOrdersRequest(
            status=QueryOrderStatus.OPEN,
            symbols=[config.SYMBOL],
            side=side,
        )
    )
    return bool(orders)


def wait_for_order_fill(order_id, timeout_seconds=10, poll_interval_seconds=1):
    deadline = time.monotonic() + max(timeout_seconds, 0)

    while True:
        order = client.get_order_by_id(order_id)
        if order_status_value(order) in TERMINAL_ORDER_STATUSES:
            return order
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise OrderFillTimeoutError(order)
        time.sleep(min(poll_interval_seconds, remaining))


def reconcile_order(order_id):
    """Fetch the latest broker state for an order that may still be pending."""
    return client.get_order_by_id(order_id)


def get_btc_position():
    try:
        return client.get_open_position(config.SYMBOL)
    except APIError as error:
        if error.status_code == 404:
            return None
        raise
