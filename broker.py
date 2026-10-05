"""Broker operations with explicit symbol and unknown-state handling."""

import hashlib
import os
import time
from dataclasses import dataclass
from enum import Enum
from urllib.parse import urlsplit

from dotenv import load_dotenv

from alpaca.common.exceptions import APIError
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, OrderType, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import (
    GetOrdersRequest,
    MarketOrderRequest,
    StopLimitOrderRequest,
)

import config


class OrderOutcome(str, Enum):
    FILLED = "filled"
    TERMINAL_NOT_FILLED = "terminal_not_filled"
    TIMEOUT_PENDING = "timeout_pending"
    PARTIAL_PENDING = "partial_pending"
    PENDING = "pending"
    RATE_LIMITED = "rate_limited"


class SubmissionFailureKind(str, Enum):
    DEFINITIVE_REJECTION = "definitive_rejection"
    AMBIGUOUS_SUBMISSION = "ambiguous_submission"
    RETRYABLE_TRANSPORT_ERROR = "retryable_transport_error"
    RATE_LIMITED = "rate_limited"


class PositionLookupStatus(str, Enum):
    CONFIRMED_FLAT = "confirmed_flat"
    CONFIRMED_POSITION = "confirmed_position"
    POSITION_STATE_UNKNOWN = "position_state_unknown"


@dataclass(frozen=True)
class PositionLookup:
    status: PositionLookupStatus
    position: object | None = None
    error: Exception | None = None


class BrokerPositionStateUnknown(RuntimeError):
    """Raised when Alpaca could not confirm whether the BTC account is flat."""


class BrokerOrderNotFound(LookupError):
    """The broker definitively returned 404 for a client/order identifier."""


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


def classify_submission_exception(error: Exception) -> SubmissionFailureKind:
    """Classify whether a failed submit definitively created no broker order.

    Alpaca validation/auth 4xx responses are definitive rejections. HTTP 429 is
    retryable after a cooldown; HTTP 408 is ambiguous because the server may have
    accepted the request before timing out. 5xx and transport failures are
    reconciled by client_order_id before any retry.
    """
    status_code = getattr(error, "status_code", None)
    try:
        status_code = int(status_code) if status_code is not None else None
    except (TypeError, ValueError):
        status_code = None
    if status_code is not None and 400 <= status_code < 500 and status_code != 408:
        if status_code == 429:
            return SubmissionFailureKind.RATE_LIMITED
        return SubmissionFailureKind.DEFINITIVE_REJECTION
    if isinstance(error, (TimeoutError, ConnectionError, OSError)):
        return SubmissionFailureKind.RETRYABLE_TRANSPORT_ERROR
    return SubmissionFailureKind.AMBIGUOUS_SUBMISSION


load_dotenv()

client = TradingClient(
    os.getenv("ALPACA_API_KEY"),
    os.getenv("ALPACA_SECRET_KEY"),
    paper=True,
)


def broker_position_symbol(symbol: str) -> str:
    """Normalize a crypto pair only at the broker position/order identity boundary."""
    return str(symbol or "").strip().upper().replace("/", "").replace("-", "")


def btc_symbol_matches(symbol: str | None) -> bool:
    return bool(symbol) and broker_position_symbol(symbol) == broker_position_symbol(config.SYMBOL)


def broker_position_identifier(position):
    asset_id = getattr(position, "asset_id", None)
    if asset_id:
        return asset_id
    symbol = getattr(position, "symbol", None) or config.SYMBOL
    normalized = broker_position_symbol(symbol)
    if not normalized:
        raise BrokerPositionStateUnknown("BTC position has no usable broker identifier")
    return normalized


def lookup_btc_position() -> PositionLookup:
    """Return explicit flat/position/unknown state using the account positions list."""
    try:
        positions = client.get_all_positions()
        if not isinstance(positions, (list, tuple)):
            raise ValueError("Unexpected response from get_all_positions")
        matches = []
        for position in positions:
            symbol = getattr(position, "symbol", None)
            if not symbol:
                raise ValueError("Broker position response omitted a symbol")
            if btc_symbol_matches(symbol):
                matches.append(position)
        if len(matches) > 1:
            raise ValueError("Broker returned multiple positions matching BTC/USD")
        if matches:
            return PositionLookup(PositionLookupStatus.CONFIRMED_POSITION, matches[0])
        return PositionLookup(PositionLookupStatus.CONFIRMED_FLAT)
    except Exception as error:
        return PositionLookup(PositionLookupStatus.POSITION_STATE_UNKNOWN, error=error)


def get_btc_position():
    """Return None only for a broker-confirmed flat state; raise on uncertainty."""
    state = lookup_btc_position()
    if state.status == PositionLookupStatus.CONFIRMED_FLAT:
        return None
    if state.status == PositionLookupStatus.CONFIRMED_POSITION:
        return state.position
    raise BrokerPositionStateUnknown(
        f"BTC position state is unknown: {state.error}"
    ) from state.error


def deterministic_client_order_id(
    strategy_name: str,
    timeframe: str,
    symbol: str,
    candle_timestamp,
    side: str,
    *,
    role: str = "strategy",
) -> str:
    identity = "|".join(
        (
            role.strip().lower(),
            str(strategy_name).strip().lower(),
            str(timeframe).strip().lower(),
            broker_position_symbol(symbol),
            str(candle_timestamp),
            str(side).strip().upper(),
        )
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:40]
    # Alpaca client_order_id is limited to 48 characters.
    prefix = "protect-" if role == "protective_stop" else "bot-"
    return prefix + digest


def buy_btc(amount_usd, *, client_order_id=None):
    order = MarketOrderRequest(
        symbol=config.SYMBOL,
        notional=amount_usd,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.GTC,
        client_order_id=client_order_id,
    )
    return client.submit_order(order_data=order)


def sell_btc(position=None, *, quantity=None, client_order_id=None):
    """Sell known BTC exposure; use an identified market order when deterministic IDs are needed."""
    if position is None:
        position = get_btc_position()
    if position is None:
        return None
    if client_order_id is not None:
        if quantity is None:
            try:
                quantity = abs(float(position.qty))
            except (TypeError, ValueError, AttributeError) as error:
                raise BrokerPositionStateUnknown(
                    "Cannot confirm BTC quantity for deterministic sell"
                ) from error
        if quantity <= 0:
            return None
        order = MarketOrderRequest(
            symbol=config.SYMBOL,
            qty=quantity,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.GTC,
            client_order_id=client_order_id,
        )
        return client.submit_order(order_data=order)
    return client.close_position(broker_position_identifier(position))


def submit_protective_stop_limit(
    quantity: float,
    stop_price: float,
    limit_price: float,
    *,
    client_order_id: str,
):
    request = StopLimitOrderRequest(
        symbol=config.SYMBOL,
        qty=quantity,
        side=OrderSide.SELL,
        type=OrderType.STOP_LIMIT,
        time_in_force=TimeInForce.GTC,
        stop_price=stop_price,
        limit_price=limit_price,
        client_order_id=client_order_id,
    )
    return client.submit_order(order_data=request)


def get_open_btc_orders():
    orders = client.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN))
    if orders is None or isinstance(orders, (str, bytes)):
        raise RuntimeError("Unexpected response from open orders endpoint")
    result = []
    for order in orders:
        symbol = getattr(order, "symbol", None)
        if not symbol:
            raise RuntimeError("Open broker order response omitted a symbol; state is unknown")
        if btc_symbol_matches(symbol):
            result.append(order)
    return result


def has_open_order(side: OrderSide, *, include_protective=True) -> bool:
    side_value = str(getattr(side, "value", side)).lower()
    return any(
        str(getattr(getattr(order, "side", None), "value", getattr(order, "side", ""))).lower()
        == side_value
        and (include_protective or not is_protective_order(order))
        for order in get_open_btc_orders()
    )


def is_protective_order(order) -> bool:
    client_order_id = str(getattr(order, "client_order_id", "") or "")
    return client_order_id.startswith("protect-")


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


def get_order_by_client_order_id(client_order_id):
    try:
        return client.get_order_by_client_id(client_order_id)
    except APIError as error:
        if error.status_code == 404:
            return None
        raise


def reconcile_order(order_id=None, *, client_order_id=None):
    """Fetch latest broker state by the strongest persisted identifier available."""
    if client_order_id:
        order = get_order_by_client_order_id(client_order_id)
        if order is None:
            raise BrokerOrderNotFound(
                f"Broker has no order for client_order_id={client_order_id}"
            )
        return order
    if not order_id:
        raise ValueError("order_id or client_order_id is required")
    return client.get_order_by_id(order_id)


def cancel_order(order_id):
    return client.cancel_order_by_id(order_id)


def is_paper_client(client_obj=None) -> bool:
    selected = client if client_obj is None else client_obj
    base_url = getattr(selected, "_base_url", None)
    parsed = urlsplit(str(getattr(base_url, "value", base_url)).strip().lower())
    return parsed.scheme == "https" and parsed.hostname == "paper-api.alpaca.markets"
