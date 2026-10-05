import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from math import isfinite

from alpaca.trading.enums import OrderSide

import config
import trade_config
from broker import (
    OrderFillTimeoutError,
    OrderOutcome,
    buy_btc,
    classify_order,
    client,
    get_btc_position,
    has_open_order,
    order_status_value,
    reconcile_order,
    sell_btc,
    wait_for_order_fill,
)
from database import init_db, log_evaluation, upsert_order
from market_data import get_btc_bars, get_btc_market_price
from risk import can_buy
from strategy import calculate_indicators, decide


def should_process_candle(candle_timestamp, last_processed_candle):
    return candle_timestamp is not None and candle_timestamp != last_processed_candle


def processed_candle_after_order(candle_timestamp, outcome):
    if outcome == OrderOutcome.TERMINAL_NOT_FILLED:
        return None
    return candle_timestamp


@dataclass
class PendingReconciliation:
    side: str
    reason: str
    requested_notional: float | None
    position: object | None
    candle_timestamp: object | None


@dataclass(frozen=True)
class PositionSnapshot:
    avg_entry_price: float
    market_value: float | None


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
            order_status is None
            and order_status_value(order) == "filled"
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

        status_value = order_status_value(order)
        final_status = order_status or status_value

        upsert_order(
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


def _submit_and_log_buy(reason, pending_reconciliations, candle_timestamp):
    submitted = buy_btc(trade_config.TRADE_AMOUNT_USD)
    try:
        order = wait_for_order_fill(submitted.id)
    except OrderFillTimeoutError as error:
        pending_status = (
            OrderOutcome.PARTIAL_PENDING.value
            if classify_order(error.order) == OrderOutcome.PARTIAL_PENDING
            else OrderOutcome.TIMEOUT_PENDING.value
        )
        pending_reconciliations[str(error.order.id)] = PendingReconciliation(
            "BUY", reason, trade_config.TRADE_AMOUNT_USD, None, candle_timestamp
        )
        _record_order(
            error.order,
            "BUY",
            reason,
            trade_config.TRADE_AMOUNT_USD,
            order_status=pending_status,
        )
        state = "PARTIAL/PENDING" if pending_status == OrderOutcome.PARTIAL_PENDING.value else "PENDING"
        print(
            f"BUY ORDER {state} RECONCILIATION: id={error.order.id} "
            f"status={order_status_value(error.order)} "
            f"filled_qty={getattr(error.order, 'filled_qty', None)} "
            f"fill_price={getattr(error.order, 'filled_avg_price', None)}"
        )
        return OrderOutcome.TIMEOUT_PENDING
    except Exception:
        pending_reconciliations[str(submitted.id)] = PendingReconciliation(
            "BUY", reason, trade_config.TRADE_AMOUNT_USD, None, candle_timestamp
        )
        _record_order(
            submitted,
            "BUY",
            reason,
            trade_config.TRADE_AMOUNT_USD,
            order_status=OrderOutcome.TIMEOUT_PENDING.value,
        )
        raise

    outcome = classify_order(order)
    _record_order(order, "BUY", reason, trade_config.TRADE_AMOUNT_USD)
    if outcome == OrderOutcome.FILLED:
        print(
            f"BUY ORDER FILLED: id={order.id} status={order_status_value(order)} "
            f"filled_qty={getattr(order, 'filled_qty', None)} "
            f"fill_price={getattr(order, 'filled_avg_price', None)}"
        )
    else:
        partial = _optional_float(getattr(order, "filled_qty", None)) or 0.0
        print(
            f"BUY ORDER NOT FILLED: id={order.id} status={order_status_value(order)} "
            f"partial_filled_qty={partial:g} reason={reason}"
        )
    return outcome


def _submit_and_log_sell(reason, position, pending_reconciliations, candle_timestamp):
    if has_open_order(OrderSide.SELL):
        print(f"EXIT ORDER PENDING: reason={reason}")
        return OrderOutcome.PENDING

    submitted = sell_btc()
    if submitted is None:
        print("SELL SKIPPED: position no longer exists")
        return

    try:
        order = wait_for_order_fill(submitted.id)
    except OrderFillTimeoutError as error:
        snapshot = _position_snapshot(position)
        pending_status = (
            OrderOutcome.PARTIAL_PENDING.value
            if classify_order(error.order) == OrderOutcome.PARTIAL_PENDING
            else OrderOutcome.TIMEOUT_PENDING.value
        )
        pending_reconciliations[str(error.order.id)] = PendingReconciliation(
            "SELL",
            reason,
            _optional_float(position.market_value),
            snapshot,
            candle_timestamp,
        )
        _record_order(
            error.order,
            "SELL",
            reason,
            _optional_float(position.market_value),
            position,
            order_status=pending_status,
        )
        state = "PARTIAL/PENDING" if pending_status == OrderOutcome.PARTIAL_PENDING.value else "PENDING"
        print(
            f"SELL ORDER {state} RECONCILIATION: id={error.order.id} "
            f"status={order_status_value(error.order)} "
            f"filled_qty={getattr(error.order, 'filled_qty', None)} "
            f"fill_price={getattr(error.order, 'filled_avg_price', None)}"
        )
        return OrderOutcome.TIMEOUT_PENDING
    except Exception:
        snapshot = _position_snapshot(position)
        pending_reconciliations[str(submitted.id)] = PendingReconciliation(
            "SELL",
            reason,
            _optional_float(position.market_value),
            snapshot,
            candle_timestamp,
        )
        _record_order(
            submitted,
            "SELL",
            reason,
            _optional_float(position.market_value),
            position,
            order_status=OrderOutcome.TIMEOUT_PENDING.value,
        )
        raise

    _record_order(
        order,
        "SELL",
        reason,
        _optional_float(position.market_value),
        position,
    )
    outcome = classify_order(order)
    if outcome == OrderOutcome.FILLED:
        print(
            f"SELL ORDER FILLED: id={order.id} status={order_status_value(order)} "
            f"filled_qty={getattr(order, 'filled_qty', None)} "
            f"fill_price={getattr(order, 'filled_avg_price', None)} reason={reason}"
        )
    else:
        partial = _optional_float(getattr(order, "filled_qty", None)) or 0.0
        print(
            f"SELL ORDER NOT FILLED: id={order.id} status={order_status_value(order)} "
            f"partial_filled_qty={partial:g} reason={reason}"
        )
    return outcome


def _position_snapshot(position):
    if position is None:
        return None
    return PositionSnapshot(
        avg_entry_price=float(position.avg_entry_price),
        market_value=_optional_float(position.market_value),
    )


def reconcile_pending_orders(pending_reconciliations):
    retry_candles = []
    for order_id, context in list(pending_reconciliations.items()):
        try:
            order = reconcile_order(order_id)
        except Exception as error:
            print(f"ORDER RECONCILIATION ERROR: id={order_id} error={error}")
            continue

        outcome = classify_order(order)
        status = order_status_value(order)
        if outcome in {OrderOutcome.FILLED, OrderOutcome.TERMINAL_NOT_FILLED}:
            _record_order(
                order,
                context.side,
                context.reason,
                context.requested_notional,
                context.position,
            )
            print(
                f"ORDER RECONCILED: id={order_id} outcome={outcome.value} status={status} "
                f"filled_qty={getattr(order, 'filled_qty', None)} "
                f"fill_price={getattr(order, 'filled_avg_price', None)}"
            )
            pending_reconciliations.pop(order_id, None)
            if (
                outcome == OrderOutcome.TERMINAL_NOT_FILLED
                and context.candle_timestamp is not None
            ):
                retry_candles.append(context.candle_timestamp)
        else:
            pending_status = (
                OrderOutcome.PARTIAL_PENDING.value
                if outcome == OrderOutcome.PARTIAL_PENDING
                else OrderOutcome.TIMEOUT_PENDING.value
            )
            _record_order(
                order,
                context.side,
                context.reason,
                context.requested_notional,
                context.position,
                order_status=pending_status,
            )
    return retry_candles


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
    pending_reconciliations = {}
    required_bars = max(
        trade_config.FAST_MA, trade_config.SLOW_MA, trade_config.RSI_PERIOD
    ) + 1

    while True:
        try:
            retry_candles = reconcile_pending_orders(pending_reconciliations)
            bars = get_btc_bars()
            market_price = get_btc_market_price()
            position = get_btc_position()
            latest_candle = bars.index[-1] if bars is not None and not bars.empty else None
            if latest_candle in retry_candles:
                last_processed_candle = None
            risk_reason = _risk_exit_reason(position, market_price)

            if risk_reason is not None:
                if latest_candle is not None:
                    last_processed_candle = latest_candle
                if pending_reconciliations:
                    print(f"RISK EXIT WAITING FOR ORDER RECONCILIATION: reason={risk_reason}")
                elif has_open_order(OrderSide.SELL):
                    print(f"EXIT ORDER PENDING: reason={risk_reason}")
                else:
                    _record_evaluation(bars, market_price, position, "SELL", risk_reason)
                    print(f"RISK EXIT ACTION=SELL REASON={risk_reason} PRICE={market_price:.2f}")
                    _submit_and_log_sell(
                        risk_reason,
                        position,
                        pending_reconciliations,
                        latest_candle,
                    )
            elif bars is None or len(bars) < required_bars:
                print("Waiting for enough completed candles")
            elif should_process_candle(latest_candle, last_processed_candle):
                indicators = calculate_indicators(bars)
                decision = decide(indicators)
                action = decision.action
                reason = decision.reason

                if action in {"BUY", "SELL"} and pending_reconciliations:
                    action = "HOLD"
                    reason = "order_pending_reconciliation"
                elif action == "BUY" and position is not None:
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
                    outcome = _submit_and_log_buy(
                        reason, pending_reconciliations, latest_candle
                    )
                    if outcome == OrderOutcome.TERMINAL_NOT_FILLED:
                        last_processed_candle = processed_candle_after_order(
                            latest_candle, outcome
                        )
                elif action == "SELL":
                    outcome = _submit_and_log_sell(
                        reason,
                        position,
                        pending_reconciliations,
                        latest_candle,
                    )
                    if outcome == OrderOutcome.TERMINAL_NOT_FILLED:
                        last_processed_candle = processed_candle_after_order(
                            latest_candle, outcome
                        )
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
