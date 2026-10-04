import config
from alpaca.trading.enums import OrderSide, QueryOrderStatus
from alpaca.trading.requests import GetOrdersRequest

from broker import client, get_btc_position


def can_buy(amount_usd=config.TRADE_AMOUNT_USD):
    if amount_usd <= 0:
        return False

    position = get_btc_position()
    position_value = 0.0 if position is None else abs(float(position.market_value))

    # Do not stack another buy while one is still waiting to fill.
    open_buy_orders = client.get_orders(
        filter=GetOrdersRequest(
            status=QueryOrderStatus.OPEN,
            symbols=[config.SYMBOL],
            side=OrderSide.BUY,
        )
    )
    if open_buy_orders:
        return False

    return position_value + amount_usd <= config.MAX_POSITION_USD
