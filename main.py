import sys
import time
from datetime import datetime, timezone
from math import isfinite

from alpaca.trading.enums import OrderSide

import config
import trade_config
from broker import (
    buy_btc,
    client,
    get_btc_position,
    has_open_order,
    sell_btc,
    OrderFillTimeoutError,
    wait_for_order_fill,
)
from database import init_db, log_evaluation, log_order
from market_data import get_btc_bars, get_btc_market_price
from risk import can_buy
from strategy import calculate_indicators, decide


def should_process_candle(candle_timestamp, last_processed_candle):
    return candle_timestamp is not None and candle_timestamp != last_processed_candle


def _optional_float(value):
    if value is None:
        return None
    number = float(value)
    return number if isfinite(number) else None


def _record_evaluation(bars, market_price, position, action, reason):
    indicators = None
    if bars is not None and len(bars) >= 2:
        indicators = calculate_indicators(bars).iloc[-1]

    try:
        log_evaluation(
            timestamp=datetime.now(timezone.utc).isoformat(),
            symbol=config.SYMBOL,
            timeframe=trade_config.LIVE_TIMEFRAME,
            current_price=_optional_float(market_price),
            ma_fast=_optional_float(indicators["ma_fast"]) if indicators is not None else None,
            ma_slow=_optional_float(indicators["ma_slow"]) if indicators is not None else None,
            rsi=_optional_float(indicators["rsi"]) if indicators is not None else None,
            action=action,
            reason=reason,
            position_value=(
                _optional_float(position.market_value) if position is not None else 0.0
            ),
            average_entry_price=(
                _optional_float(position.avg_entry_price) if position is not None else None
            ),
            unrealized_pnl=(
                _optional_float(position.unrealized_pl) if position is not None else 0.0
            ),
        )
    except Exception as error:
        print(f"DATABASE LOG ERROR: {error}", file=sys.stderr)


def _record_order(
    order, side, reason, requested_notional, position=None, order_status=None
):
    try:
        filled_quantity = _optional_float(getattr(order, "filled_qty", None))
        fill_price = _optional_float(getattr(order, "filled_avg_price", None))
        if filled_quantity is None or filled_quantity <= 0 or fill_price == 0:
            fill_price = None
        if (
            order_status != "timeout_pending"
            and side == "SELL"
            and position is not None
            and fill_price is not None
            and filled_quantity is not None
            and filled_quantity > 0
        ):
            realized_gross_pnl = (
                fill_price - float(position.avg_entry_price)
            ) * filled_quantity
        else:
            realized_gross_pnl = None

        status_value = getattr(getattr(order, "status", None), "value", None)
        final_status = order_status or status_value

        log_order(
            timestamp=datetime.now(timezone.utc).isoformat(),
            order_id=str(order.id),
            symbol=config.SYMBOL,
            side=side,
            requested_notional=_optional_float(requested_notional),
            quantity=filled_quantity,
            fill_price=fill_price,
            reason=reason,
            realized_gross_pnl=realized_gross_pnl,
            order_status=final_status,
        )
    except Exception as error:
        print(f"DATABASE LOG ERROR: {error}", file=sys.stderr)


def _risk_exit_reason(position, market_price):
    if position is None:
        return None

    average_entry_price = float(position.avg_entry_price)
    if average_entry_price <= 0:
        raise ValueError("BTC position has an invalid average entry price")
    if market_price <= average_entry_price * (1 - trade_config.STOP_LOSS_PERCENT):
        return "stop_loss"
    if market_price >= average_entry_price * (1 + trade_config.TAKE_PROFIT_PERCENT):
        return "take_profit"
    return None


def _submit_and_log_buy(reason):
    submitted = buy_btc(trade_config.TRADE_AMOUNT_USD)
    try:
        order = wait_for_order_fill(submitted.id)
    except OrderFillTimeoutError as error:
        _record_order(
            error.order,
            "BUY",
            reason,
            trade_config.TRADE_AMOUNT_USD,
            order_status="timeout_pending",
        )
        print(f"BUY ORDER PENDING RECONCILIATION: {error}")
        return
    except Exception:
        _record_order(
            submitted,
            "BUY",
            reason,
            trade_config.TRADE_AMOUNT_USD,
        )
        raise

    _record_order(order, "BUY", reason, trade_config.TRADE_AMOUNT_USD)
    print(
        "BUY ORDER:",
        order.id,
        "STATUS:",
        getattr(order.status, "value", order.status),
        "FILLED_QTY:",
        getattr(order, "filled_qty", None),
        "FILL_PRICE:",
        getattr(order, "filled_avg_price", None),
    )


def _submit_and_log_sell(reason, position):
    if has_open_order(OrderSide.SELL):
        print(f"EXIT ORDER PENDING: reason={reason}")
        return

    submitted = sell_btc()
    if submitted is None:
        print("SELL SKIPPED: position no longer exists")
        return

    try:
        order = wait_for_order_fill(submitted.id)
    except OrderFillTimeoutError as error:
        _record_order(
            error.order,
            "SELL",
            reason,
            _optional_float(position.market_value),
            position,
            order_status="timeout_pending",
        )
        print(f"SELL ORDER PENDING RECONCILIATION: {error}")
        return
    except Exception:
        _record_order(
            submitted,
            "SELL",
            reason,
            _optional_float(position.market_value),
            position,
        )
        raise

    _record_order(
        order,
        "SELL",
        reason,
        _optional_float(position.market_value),
        position,
    )
    print(
        "SELL ORDER:",
        order.id,
        "STATUS:",
        getattr(order.status, "value", order.status),
        "FILLED_QTY:",
        getattr(order, "filled_qty", None),
        "FILL_PRICE:",
        getattr(order, "filled_avg_price", None),
        "REASON:",
        reason,
    )


def _print_candle_status(timestamp, market_price, indicators, action, reason):
    current = indicators.iloc[-1]
    print(
        f"{datetime.now().strftime('%H:%M:%S')} "
        f"BTC={market_price:.2f} "
        f"MA_FAST={current['ma_fast']:.2f} "
        f"MA_SLOW={current['ma_slow']:.2f} "
        f"RSI={current['rsi']:.2f} "
        f"ACTION={action} "
        f"REASON={reason} "
        f"CANDLE={timestamp}"
    )


def run():
    init_db()
    account = client.get_account()
    print("=== PAPER BOT STARTED ===")
    print("Cash:", account.cash)
    print("Portfolio:", account.portfolio_value)

    last_processed_candle = None
    required_bars = max(
        trade_config.FAST_MA, trade_config.SLOW_MA, trade_config.RSI_PERIOD
    ) + 1

    while True:
        try:
            bars = get_btc_bars()
            market_price = get_btc_market_price()
            position = get_btc_position()
            latest_candle = bars.index[-1] if bars is not None and not bars.empty else None
            risk_reason = _risk_exit_reason(position, market_price)

            if risk_reason is not None:
                if latest_candle is not None:
                    last_processed_candle = latest_candle
                if has_open_order(OrderSide.SELL):
                    print(f"EXIT ORDER PENDING: reason={risk_reason}")
                else:
                    _record_evaluation(bars, market_price, position, "SELL", risk_reason)
                    print(f"RISK EXIT ACTION=SELL REASON={risk_reason} PRICE={market_price:.2f}")
                    _submit_and_log_sell(risk_reason, position)
            elif bars is None or len(bars) < required_bars:
                print("Waiting for enough completed candles")
            elif should_process_candle(latest_candle, last_processed_candle):
                indicators = calculate_indicators(bars)
                decision = decide(indicators)
                action = decision.action
                reason = decision.reason

                if action == "BUY" and position is not None:
                    action = "HOLD"
                    reason = "position_already_open"
                elif action == "BUY" and not can_buy():
                    action = "HOLD"
                    reason = "buy_blocked_by_position_cash_or_pending_order"
                elif action == "SELL" and position is None:
                    action = "HOLD"
                    reason = "no_open_position"
                elif action == "SELL" and has_open_order(OrderSide.SELL):
                    action = "HOLD"
                    reason = "sell_order_pending"

                _print_candle_status(latest_candle, market_price, indicators, action, reason)

                # Mark before submitting so one candle cannot place duplicate orders.
                last_processed_candle = latest_candle
                _record_evaluation(bars, market_price, position, action, reason)

                if action == "BUY":
                    _submit_and_log_buy(reason)
                elif action == "SELL":
                    _submit_and_log_sell(reason, position)
            else:
                print(
                    f"STATUS BTC={market_price:.2f} "
                    f"CANDLE={latest_candle} STRATEGY=already_processed"
                )
        except Exception as error:
            print("ERROR:", error)

        time.sleep(trade_config.CHECK_INTERVAL_SECONDS)


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        print("\n=== BOT STOPPED ===")
