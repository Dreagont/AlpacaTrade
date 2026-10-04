import os
from dotenv import load_dotenv

from alpaca.common.exceptions import APIError
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

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


def get_btc_position():
    try:
        return client.get_open_position("BTC/USD")
    except APIError as error:
        # Alpaca returns 404 when the account has no position for this symbol.
        # Other API failures must reach the risk check instead of looking like zero exposure.
        if error.status_code == 404:
            return None
        raise
