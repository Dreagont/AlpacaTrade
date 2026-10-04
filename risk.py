import config
from alpaca.trading.enums import OrderSide

from broker import client, get_btc_position, has_open_order


def can_buy(amount_usd=config.TRADE_AMOUNT_USD):
    if amount_usd <= 0 or amount_usd > config.MAX_POSITION_USD:
        return False

    position = get_btc_position()
    cash_available = float(client.get_account().cash)
    return (
        position is None
        and cash_available >= amount_usd
        and not has_open_order(OrderSide.BUY)
        and not has_open_order(OrderSide.SELL)
    )
