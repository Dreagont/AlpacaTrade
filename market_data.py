from datetime import datetime, timedelta, timezone

import pandas as pd
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest, CryptoLatestTradeRequest

import config
from timeframes import parse_timeframe

data_client = CryptoHistoricalDataClient()


def get_btc_bars(limit_bars=200):
    timeframe, minutes_per_bar = parse_timeframe(config.LIVE_TIMEFRAME)
    limit_bars = max(
        limit_bars,
        max(config.FAST_MA, config.SLOW_MA, config.RSI_PERIOD) + 5,
    )
    now = datetime.now(timezone.utc)
    request = CryptoBarsRequest(
        symbol_or_symbols=[config.SYMBOL],
        timeframe=timeframe,
        start=now - timedelta(minutes=minutes_per_bar * (limit_bars + 2)),
        end=now,
    )

    bars = data_client.get_crypto_bars(request).df

    if bars.empty:
        return None

    bars = bars.loc[config.SYMBOL].sort_index()
    completed_before = pd.Timestamp(now)
    candle_closes = bars.index + pd.to_timedelta(minutes_per_bar, unit="m")
    return bars.loc[candle_closes <= completed_before]


def get_btc_market_price():
    request = CryptoLatestTradeRequest(symbol_or_symbols=config.SYMBOL)
    trades = data_client.get_crypto_latest_trade(request)
    return float(trades[config.SYMBOL].price)
