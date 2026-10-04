from dataclasses import dataclass
from math import isfinite

import config


@dataclass(frozen=True)
class Decision:
    action: str
    reason: str


def calculate_indicators(df):
    df = df.copy()

    df["ma_fast"] = df["close"].rolling(config.FAST_MA).mean()
    df["ma_slow"] = df["close"].rolling(config.SLOW_MA).mean()

    delta = df["close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(config.RSI_PERIOD).mean()
    avg_loss = loss.rolling(config.RSI_PERIOD).mean()

    rs = avg_gain / avg_loss

    df["rsi"] = 100 - (100 / (1 + rs))

    return df


def decide_at(df, index):
    if index < 1 or index >= len(df):
        return Decision("HOLD", "not_enough_data")

    row = df.iloc[index]
    prev = df.iloc[index - 1]

    indicator_values = (
        prev["ma_fast"],
        prev["ma_slow"],
        row["ma_fast"],
        row["ma_slow"],
        row["rsi"],
    )
    if not all(isfinite(float(value)) for value in indicator_values):
        return Decision("HOLD", "indicators_not_ready")

    bullish_cross = (
        prev["ma_fast"] <= prev["ma_slow"]
        and row["ma_fast"] > row["ma_slow"]
    )

    bearish_cross = (
        prev["ma_fast"] >= prev["ma_slow"]
        and row["ma_fast"] < row["ma_slow"]
    )

    if bullish_cross:
        if row["rsi"] < config.RSI_BUY_THRESHOLD:
            return Decision("BUY", "bullish_ma_crossover_rsi_below_70")
        return Decision("HOLD", "bullish_ma_crossover_rsi_filter_failed")

    if bearish_cross:
        return Decision("SELL", "bearish_ma_crossover")

    return Decision("HOLD", "no_crossover")


def decide(df):
    return decide_at(df, len(df) - 1)
