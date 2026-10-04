import config


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


def decide(df):
    row = df.iloc[-1]
    prev = df.iloc[-2]

    bullish_cross = (
        prev["ma_fast"] <= prev["ma_slow"]
        and row["ma_fast"] > row["ma_slow"]
    )

    bearish_cross = (
        prev["ma_fast"] >= prev["ma_slow"]
        and row["ma_fast"] < row["ma_slow"]
    )

    if bullish_cross and row["rsi"] < 70:
        return "BUY"

    if bearish_cross:
        return "SELL"

    return "HOLD"
