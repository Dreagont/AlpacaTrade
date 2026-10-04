import os
import time

from dotenv import load_dotenv

from alpaca.common.exceptions import APIError
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import OrderSide, QueryOrderStatus, TimeInForce

import config

load_dotenv()

client = TradingClient(
    os.getenv("ALPACA_API_KEY"),
    os.getenv("ALPACA_SECRET_KEY"),
    paper=True,
)

def buy_btc(amount_usd):
    order = MarketOrderRequest(
        symbol="BTC/USD",
        notional=amount_usd,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.GTC,
    )

    return client.submit_order(order_data=order)


def sell_btc():
    try:
        return client.close_position("BTC/USD")
    except APIError as error:
        if (
            error.status_code == 404
            and error.message.lower() in {"position does not exist", "position not found"}
        ):
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
    terminal_statuses = {
        "filled",
        "canceled",
        "expired",
        "replaced",
        "done_for_day",
        "rejected",
        "stopped",
    }
    deadline = time.monotonic() + max(timeout_seconds, 0)

    while True:
        order = client.get_order_by_id(order_id)
        status = getattr(order.status, "value", order.status)
        if status in terminal_statuses or time.monotonic() >= deadline:
            return order
        time.sleep(min(poll_interval_seconds, max(0, deadline - time.monotonic())))


def get_btc_position():
    try:
        return client.get_open_position("BTC/USD")
    except APIError as error:
        message = error.message.lower()
        if error.status_code == 404 and message in {
            "position does not exist",
            "position not found",
        }:
            return None
        raise
