from datetime import datetime, timedelta, timezone

from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest, CryptoLatestTradeRequest
from alpaca.data.timeframe import TimeFrame

data_client = CryptoHistoricalDataClient()


def get_btc_bars(limit_minutes=200):
    request = CryptoBarsRequest(
        symbol_or_symbols=["BTC/USD"],
        timeframe=TimeFrame.Minute,
        start=datetime.now(timezone.utc) - timedelta(minutes=limit_minutes),
    )

    bars = data_client.get_crypto_bars(request).df

    if bars.empty:
        return None

    return bars.loc["BTC/USD"]


def get_btc_market_price():
    request = CryptoLatestTradeRequest(symbol_or_symbols="BTC/USD")
    trades = data_client.get_crypto_latest_trade(request)
    return float(trades["BTC/USD"].price)
