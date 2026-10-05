import trade_config
from alpaca.trading.enums import OrderSide

from broker import client, get_btc_position, has_open_order


_POSITION_NOT_PROVIDED = object()


def within_max_position_exposure(current_market_value, requested_notional, cap=None):
    cap = trade_config.MAX_POSITION_USD if cap is None else float(cap)
    current_market_value = abs(float(current_market_value or 0.0))
    requested_notional = float(requested_notional)
    return (
        requested_notional > 0
        and current_market_value + requested_notional <= cap
    )


def can_buy(
    amount_usd=trade_config.TRADE_AMOUNT_USD,
    *,
    position=_POSITION_NOT_PROVIDED,
    market_price=None,
):
    if amount_usd <= 0 or amount_usd > trade_config.MAX_POSITION_USD:
        return False

    if position is _POSITION_NOT_PROVIDED:
        position = get_btc_position()
    current_value = 0.0
    if position is not None:
        position_value = getattr(position, "market_value", None)
        if position_value is not None:
            current_value = abs(float(position_value))
        else:
            if market_price is None:
                return False
            current_value = abs(float(position.qty)) * float(market_price)
    if not within_max_position_exposure(current_value, amount_usd):
        return False
    # The current live policy allows one position and does not pyramid.
    if position is not None:
        return False

    cash_available = float(client.get_account().cash)
    return (
        cash_available >= amount_usd
        and not has_open_order(OrderSide.BUY)
        and not has_open_order(OrderSide.SELL)
    )
