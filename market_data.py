from datetime import datetime, timedelta, timezone

import pandas as pd
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest, CryptoLatestTradeRequest

import config
import trade_config
from strategy import StrategySpec, get_strategy
from timeframes import parse_timeframe

data_client = CryptoHistoricalDataClient()


def get_btc_bars(limit_bars=200, *, strategy: StrategySpec | None = None):
    timeframe, minutes_per_bar = parse_timeframe(trade_config.LIVE_TIMEFRAME)
    selected_strategy = strategy or get_strategy(trade_config.LIVE_STRATEGY)
    limit_bars = max(limit_bars, selected_strategy.required_warmup_bars() + 2)
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
