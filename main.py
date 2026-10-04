import time
from datetime import datetime

import config

from market_data import get_btc_bars, get_btc_market_price
from strategy import calculate_indicators, decide
from broker import buy_btc, get_btc_position, sell_btc, client
from risk import can_buy


def run():
    account = client.get_account()

    print("=== BOT STARTED ===")
    print("Cash:", account.cash)
    print("Portfolio:", account.portfolio_value)

    while True:
        try:
            df = get_btc_bars()

            required_bars = max(config.FAST_MA, config.SLOW_MA, config.RSI_PERIOD) + 1
            if df is None or len(df) < required_bars:
                print("Not enough data")
                time.sleep(config.CHECK_INTERVAL_SECONDS)
                continue

            df = calculate_indicators(df)

            current = df.iloc[-1]
            market_price = get_btc_market_price()
            position = get_btc_position()

            action = decide(df)
            exit_reason = None

            if position is not None:
                average_entry_price = float(position.avg_entry_price)

                if average_entry_price <= 0:
                    raise ValueError("BTC position has an invalid average entry price")

                if market_price <= average_entry_price * (1 - config.STOP_LOSS_PERCENT):
                    action = "SELL"
                    exit_reason = "STOP_LOSS"
                elif market_price >= average_entry_price * (1 + config.TAKE_PROFIT_PERCENT):
                    action = "SELL"
                    exit_reason = "TAKE_PROFIT"

            print(
                datetime.now(),
                "BTC:",
                round(market_price, 2),
                "RSI:",
                round(current["rsi"], 2),
                "ACTION:",
                action,
            )

            if action == "BUY":
                if can_buy(config.TRADE_AMOUNT_USD):
                    order = buy_btc(config.TRADE_AMOUNT_USD)
                    print("BUY ORDER:", order.id)
                else:
                    print("BUY SKIPPED: position cap reached or a BUY order is still open")

            elif action == "SELL":
                if position is None:
                    print("SELL SKIPPED: no open BTC position")
                else:
                    result = sell_btc()

                    if result:
                        print("SELL:", exit_reason or "STRATEGY", "ORDER:", result.id)

        except Exception as e:
            print("ERROR:", e)

        time.sleep(config.CHECK_INTERVAL_SECONDS)


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        print("\n=== BOT STOPPED ===")
