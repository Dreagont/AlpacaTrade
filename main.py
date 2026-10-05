"""Live MA/RSI bot entry point with conservative broker-state handling."""

import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from math import floor, isfinite
from types import SimpleNamespace

from alpaca.trading.enums import OrderSide

import config
import database
from database import init_db, log_evaluation, upsert_order
import trade_config
from broker import (
    BrokerPositionStateUnknown,
    OrderFillTimeoutError,
    OrderOutcome,
    PositionLookupStatus,
    buy_btc,
    cancel_order,
    classify_order,
    client,
    deterministic_client_order_id,
    get_btc_position,
    get_order_by_client_order_id,
    get_open_btc_orders,
    has_open_order,
    is_protective_order,
    lookup_btc_position,
    order_status_value,
    reconcile_order,
    sell_btc,
    submit_protective_stop_limit,
    wait_for_order_fill,
)
from market_data import get_btc_bars, get_btc_market_price
from risk import can_buy
from strategy import StrategySpec, get_strategy


RISK_EXIT_STATE_KEY = "pending_risk_exit"


def should_process_candle(candle_timestamp, last_processed_candle):
    return candle_timestamp is not None and candle_timestamp != last_processed_candle


def processed_candle_after_order(candle_timestamp, outcome):
    """A terminal order does not make its signal candle eligible for retry."""
    return candle_timestamp


@dataclass
class PendingReconciliation:
    side: str
    reason: str
    requested_notional: float | None
    position: object | None
    candle_timestamp: object | None
    client_order_id: str | None = None
    strategy_name: str | None = None
    strategy_parameters_json: str | None = None
    risk_exit_reason: str | None = None


@dataclass(frozen=True)
class PositionSnapshot:
    avg_entry_price: float
    market_value: float | None
    quantity: float | None = None


def _optional_float(value):
    if value is None:
        return None
    number = float(value)
    return number if isfinite(number) else None


def _live_strategy() -> StrategySpec:
    strategy = get_strategy(trade_config.LIVE_STRATEGY)
    if strategy.name != "ma_rsi_crossover":
        raise RuntimeError(
            "SAFE-HALT: live execution is restricted to ma_rsi_crossover in this release"
        )
    return strategy


def _strategy_parameters_json(strategy: StrategySpec) -> str:
    return json.dumps(strategy.parameters, sort_keys=True, separators=(",", ":"))


def _record_evaluation(bars, market_price, position, action, reason, strategy=None):
    strategy = strategy or _live_strategy()
    indicators = None
    if bars is not None and len(bars) >= 2:
        indicators = strategy.prepare_indicators(bars).iloc[-1]
    is_ma = strategy.name == "ma_rsi_crossover"
    try:
        log_evaluation(
            timestamp=datetime.now(timezone.utc).isoformat(),
            symbol=config.SYMBOL,
            timeframe=trade_config.LIVE_TIMEFRAME,
            current_price=_optional_float(market_price),
            ma_fast=(
                _optional_float(indicators.get("ma_fast"))
                if is_ma and indicators is not None
                else None
            ),
            ma_slow=(
                _optional_float(indicators.get("ma_slow"))
                if is_ma and indicators is not None
                else None
            ),
            rsi=(
                _optional_float(indicators.get("rsi"))
                if is_ma and indicators is not None
                else None
            ),
            action=action,
            reason=reason,
            position_value=(
                _optional_float(position.market_value) if position is not None else 0.0
            ),
            average_entry_price=(
                _optional_float(position.avg_entry_price) if position is not None else None
            ),
            unrealized_pnl=(
                _optional_float(getattr(position, "unrealized_pl", None))
                if position is not None
                else 0.0
            ),
            strategy_name=strategy.name,
            strategy_parameters_json=_strategy_parameters_json(strategy),
        )
    except Exception as error:
        print(f"DATABASE LOG ERROR: {error}", file=sys.stderr)


def _record_order(
    order,
    side,
    reason,
    requested_notional,
    position=None,
    order_status=None,
    *,
    client_order_id=None,
    strategy=None,
):
    strategy = strategy or _live_strategy()
    filled_quantity = _optional_float(getattr(order, "filled_qty", None))
    fill_price = _optional_float(getattr(order, "filled_avg_price", None))
    if filled_quantity is None or filled_quantity <= 0 or fill_price == 0:
        fill_price = None
    status_value = order_status_value(order)
    final_status = order_status or status_value
    if (
        final_status == "filled"
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
    selected_client_id = (
        client_order_id or getattr(order, "client_order_id", None)
    )
    upsert_order(
        timestamp=datetime.now(timezone.utc).isoformat(),
        order_id=str(getattr(order, "id", f"client:{selected_client_id or 'unknown'}")),
        client_order_id=selected_client_id,
        symbol=config.SYMBOL,
        side=side,
        requested_notional=_optional_float(requested_notional),
        quantity=filled_quantity,
        fill_price=fill_price,
        reason=reason,
        realized_gross_pnl=realized_gross_pnl,
        order_status=final_status,
        strategy_name=strategy.name,
        strategy_parameters_json=_strategy_parameters_json(strategy),
    )


def _risk_exit_reason(position, market_price, strategy=None):
    if position is None:
        return None
    strategy = strategy or _live_strategy()
    average_entry_price = float(position.avg_entry_price)
    if average_entry_price <= 0:
        raise ValueError("BTC position has an invalid average entry price")
    stop = strategy.stop_loss_percent
    target = strategy.take_profit_percent
    if stop is not None and market_price <= average_entry_price * (1 - stop):
        return "stop_loss"
    if target is not None and market_price >= average_entry_price * (1 + target):
        return "take_profit"
    return None


def _strategy_order_id(strategy, candle_timestamp, side, *, role="strategy"):
    return deterministic_client_order_id(
        strategy.name,
        trade_config.LIVE_TIMEFRAME,
        config.SYMBOL,
        candle_timestamp,
        side,
        role=role,
    )


def _unknown_order_stub(client_order_id):
    return SimpleNamespace(
        id=f"client:{client_order_id}",
        client_order_id=client_order_id,
        status=SimpleNamespace(value="new"),
        filled_qty=None,
        filled_avg_price=None,
    )


def _persist_ambiguous_submission(
    *, side, reason, requested_notional, candle_timestamp, client_order_id,
    pending_reconciliations, strategy,
):
    order = _unknown_order_stub(client_order_id)
    context = PendingReconciliation(
        side,
        reason,
        requested_notional,
        None,
        candle_timestamp,
        client_order_id,
        strategy.name,
        _strategy_parameters_json(strategy),
    )
    pending_reconciliations[str(order.id)] = context
    _record_order(
        order,
        side,
        reason,
        requested_notional,
        order_status="submit_unknown",
        client_order_id=client_order_id,
        strategy=strategy,
    )
    print(
        f"SAFE-HALT ORDER STATE UNKNOWN: side={side} "
        f"client_order_id={client_order_id}; will reconcile before any retry"
    )
    return OrderOutcome.PENDING


def _submit_and_log_buy(
    reason, pending_reconciliations, candle_timestamp, strategy=None
):
    strategy = strategy or _live_strategy()
    amount = trade_config.TRADE_AMOUNT_USD
    client_order_id = _strategy_order_id(strategy, candle_timestamp, "BUY")
    try:
        order = get_order_by_client_order_id(client_order_id)
    except Exception:
        return _persist_ambiguous_submission(
            side="BUY",
            reason=reason,
            requested_notional=amount,
            candle_timestamp=candle_timestamp,
            client_order_id=client_order_id,
            pending_reconciliations=pending_reconciliations,
            strategy=strategy,
        )
    if order is None:
        try:
            order = buy_btc(amount, client_order_id=client_order_id)
        except Exception:
            try:
                order = get_order_by_client_order_id(client_order_id)
            except Exception:
                order = None
            if order is None:
                return _persist_ambiguous_submission(
                    side="BUY",
                    reason=reason,
                    requested_notional=amount,
                    candle_timestamp=candle_timestamp,
                    client_order_id=client_order_id,
                    pending_reconciliations=pending_reconciliations,
                    strategy=strategy,
                )
    try:
        order = wait_for_order_fill(order.id)
    except OrderFillTimeoutError as error:
        order = error.order
        outcome = classify_order(order, timed_out=True)
        context = PendingReconciliation(
            "BUY", reason, amount, None, candle_timestamp, client_order_id,
            strategy.name, _strategy_parameters_json(strategy)
        )
        pending_reconciliations[str(order.id)] = context
        _record_order(
            order, "BUY", reason, amount, order_status=outcome.value,
            client_order_id=client_order_id, strategy=strategy
        )
        print(
            f"BUY ORDER PENDING: id={order.id} status={order_status_value(order)} "
            f"filled_qty={getattr(order, 'filled_qty', None)}"
        )
        return outcome
    outcome = classify_order(order)
    _record_order(
        order, "BUY", reason, amount, client_order_id=client_order_id, strategy=strategy
    )
    if outcome == OrderOutcome.FILLED:
        print(
            f"BUY ORDER FILLED: id={order.id} "
            f"filled_qty={getattr(order, 'filled_qty', None)}"
        )
        _place_protective_stop(order, candle_timestamp, pending_reconciliations, strategy)
    elif outcome in {OrderOutcome.PENDING, OrderOutcome.PARTIAL_PENDING}:
        pending_reconciliations[str(order.id)] = PendingReconciliation(
            "BUY", reason, amount, None, candle_timestamp, client_order_id,
            strategy.name, _strategy_parameters_json(strategy)
        )
    else:
        print(f"BUY ORDER NOT FILLED: id={order.id} status={order_status_value(order)}")
    return outcome


def _price_to_tick(price, tick, *, round_down=True):
    if tick <= 0:
        raise ValueError("Broker stop-limit price increment must be positive")
    steps = price / tick
    return (floor(steps) if round_down else round(steps)) * tick


def _place_protective_stop(order, candle_timestamp, pending_reconciliations, strategy):
    if not trade_config.ENABLE_BROKER_STOP_LIMIT:
        return None
    stop_loss = strategy.stop_loss_percent
    if stop_loss is None:
        return None
    quantity = _optional_float(getattr(order, "filled_qty", None))
    entry_fill = _optional_float(getattr(order, "filled_avg_price", None))
    if quantity is None or quantity <= 0 or entry_fill is None or entry_fill <= 0:
        print("PROTECTIVE STOP SKIPPED: confirmed fill quantity/price is unavailable")
        return None
    if any(is_protective_order(open_order) for open_order in get_open_btc_orders()):
        return None
    tick = float(trade_config.BROKER_STOP_LIMIT_PRICE_INCREMENT)
    stop_price = _price_to_tick(entry_fill * (1 - stop_loss), tick)
    limit_price = _price_to_tick(
        stop_price * (1 - trade_config.BROKER_STOP_LIMIT_OFFSET_PERCENT), tick
    )
    if stop_price <= 0 or limit_price <= 0 or limit_price >= stop_price:
        raise ValueError("Calculated broker stop-limit prices are invalid")
    client_order_id = deterministic_client_order_id(
        strategy.name,
        trade_config.LIVE_TIMEFRAME,
        config.SYMBOL,
        candle_timestamp,
        "SELL",
        role="protective_stop",
    )
    try:
        existing = get_order_by_client_order_id(client_order_id)
        protective = existing or submit_protective_stop_limit(
            quantity,
            stop_price,
            limit_price,
            client_order_id=client_order_id,
        )
    except Exception:
        try:
            protective = get_order_by_client_order_id(client_order_id)
        except Exception:
            protective = None
        if protective is None:
            return _persist_ambiguous_submission(
                side="SELL",
                reason="protective_stop_limit",
                requested_notional=None,
                candle_timestamp=candle_timestamp,
                client_order_id=client_order_id,
                pending_reconciliations=pending_reconciliations,
                strategy=strategy,
            )
    _record_order(
        protective,
        "SELL",
        "protective_stop_limit",
        None,
        client_order_id=client_order_id,
        strategy=strategy,
    )
    return protective


def _position_snapshot(position):
    if position is None:
        return None
    try:
        quantity = abs(float(position.qty))
    except (TypeError, ValueError, AttributeError):
        quantity = None
    return PositionSnapshot(
        avg_entry_price=float(position.avg_entry_price),
        market_value=_optional_float(getattr(position, "market_value", None)),
        quantity=quantity,
    )


def _cancel_protective_stops(pending_reconciliations, strategy):
    for protective in get_open_btc_orders():
        if not is_protective_order(protective):
            continue
        order_id = str(protective.id)
        try:
            cancel_order(order_id)
        except Exception:
            # The order may have filled between listing and cancellation; reconcile below.
            pass
        try:
            current = wait_for_order_fill(order_id, timeout_seconds=10)
        except OrderFillTimeoutError as error:
            current = error.order
            pending_reconciliations[order_id] = PendingReconciliation(
                "SELL", "protective_stop_limit", None, None, None,
                getattr(current, "client_order_id", None), strategy.name,
                _strategy_parameters_json(strategy)
            )
            _record_order(
                current,
                "SELL",
                "protective_stop_limit",
                None,
                order_status=OrderOutcome.TIMEOUT_PENDING.value,
                strategy=strategy,
            )
            raise RuntimeError(
                f"Protective SELL {order_id} cancellation is not confirmed; normal SELL held"
            ) from error
        _record_order(
            current,
            "SELL",
            "protective_stop_limit",
            None,
            strategy=strategy,
        )
        pending_reconciliations.pop(order_id, None)
    remaining = [
        order for order in get_open_btc_orders() if is_protective_order(order)
    ]
    if remaining:
        identifiers = ", ".join(str(order.id) for order in remaining)
        raise RuntimeError(
            f"Protective SELL order(s) remain open after cancellation: {identifiers}"
        )


def _submit_and_log_sell(
    reason, position, pending_reconciliations, candle_timestamp, strategy=None
):
    strategy = strategy or _live_strategy()
    _cancel_protective_stops(pending_reconciliations, strategy)
    position = get_btc_position()
    if position is None:
        print("SELL SKIPPED: broker confirms position is flat")
        return None
    if has_open_order(OrderSide.SELL, include_protective=False):
        print(f"EXIT ORDER PENDING: reason={reason}")
        return OrderOutcome.PENDING
    quantity = abs(float(position.qty))
    client_order_id = _strategy_order_id(strategy, candle_timestamp, "SELL")
    existing = get_order_by_client_order_id(client_order_id)
    if existing is not None:
        order = existing
    else:
        try:
            order = sell_btc(
                position,
                quantity=quantity,
                client_order_id=client_order_id,
            )
        except Exception:
            try:
                order = get_order_by_client_order_id(client_order_id)
            except Exception:
                order = None
            if order is None:
                return _persist_ambiguous_submission(
                    side="SELL",
                    reason=reason,
                    requested_notional=_optional_float(position.market_value),
                    candle_timestamp=candle_timestamp,
                    client_order_id=client_order_id,
                    pending_reconciliations=pending_reconciliations,
                    strategy=strategy,
                )
    try:
        order = wait_for_order_fill(order.id)
    except OrderFillTimeoutError as error:
        order = error.order
        outcome = classify_order(order, timed_out=True)
        pending_reconciliations[str(order.id)] = PendingReconciliation(
            "SELL", reason, _optional_float(position.market_value),
            _position_snapshot(position), candle_timestamp, client_order_id,
            strategy.name, _strategy_parameters_json(strategy)
        )
        _record_order(
            order,
            "SELL",
            reason,
            _optional_float(position.market_value),
            position,
            order_status=outcome.value,
            client_order_id=client_order_id,
            strategy=strategy,
        )
        print(f"SELL ORDER PENDING: id={order.id} status={order_status_value(order)}")
        return outcome
    _record_order(
        order,
        "SELL",
        reason,
        _optional_float(position.market_value),
        position,
        client_order_id=client_order_id,
        strategy=strategy,
    )
    outcome = classify_order(order)
    if outcome in {OrderOutcome.PENDING, OrderOutcome.PARTIAL_PENDING}:
        pending_reconciliations[str(order.id)] = PendingReconciliation(
            "SELL", reason, _optional_float(position.market_value),
            _position_snapshot(position), candle_timestamp, client_order_id,
            strategy.name, _strategy_parameters_json(strategy)
        )
    print(f"SELL ORDER {outcome.value.upper()}: id={order.id} reason={reason}")
    return outcome


def reconcile_pending_orders(pending_reconciliations, strategy=None):
    strategy = strategy or _live_strategy()
    for order_id, context in list(pending_reconciliations.items()):
        try:
            order = reconcile_order(
                order_id,
                client_order_id=context.client_order_id,
            )
        except Exception as error:
            print(f"ORDER RECONCILIATION ERROR: id={order_id} error={error}")
            continue
        outcome = classify_order(order)
        status = order_status_value(order)
        selected_strategy = strategy
        if context.strategy_name:
            try:
                selected_strategy = get_strategy(context.strategy_name)
            except Exception:
                selected_strategy = strategy
        _record_order(
            order,
            context.side,
            context.reason,
            context.requested_notional,
            context.position,
            order_status=(
                status if outcome in {OrderOutcome.FILLED, OrderOutcome.TERMINAL_NOT_FILLED}
                else outcome.value
            ),
            client_order_id=context.client_order_id,
            strategy=selected_strategy,
        )
        if outcome in {OrderOutcome.FILLED, OrderOutcome.TERMINAL_NOT_FILLED}:
            print(f"ORDER RECONCILED: id={order_id} outcome={outcome.value} status={status}")
            pending_reconciliations.pop(order_id, None)
            if (
                context.side == "BUY"
                and outcome == OrderOutcome.FILLED
                and context.risk_exit_reason is None
            ):
                _place_protective_stop(
                    order,
                    context.candle_timestamp,
                    pending_reconciliations,
                    selected_strategy,
                )
        else:
            print(f"ORDER STILL PENDING: id={order_id} outcome={outcome.value} status={status}")
    # Terminal rejections/cancellations never reopen their original strategy candle.
    return []


def cancel_pending_buys_for_risk_exit(pending_reconciliations, risk_reason, strategy):
    for order_id, context in list(pending_reconciliations.items()):
        if context.side != "BUY":
            continue
        context.risk_exit_reason = risk_reason
        try:
            order = reconcile_order(order_id, client_order_id=context.client_order_id)
            if classify_order(order) not in {
                OrderOutcome.FILLED,
                OrderOutcome.TERMINAL_NOT_FILLED,
            }:
                cancel_order(order.id)
                order = wait_for_order_fill(order.id, timeout_seconds=10)
        except OrderFillTimeoutError as error:
            order = error.order
        except Exception as error:
            print(f"RISK EXIT BUY-CANCEL/RECONCILE ERROR: id={order_id} error={error}")
            continue
        outcome = classify_order(order)
        _record_order(
            order,
            "BUY",
            context.reason,
            context.requested_notional,
            order_status=(order_status_value(order) if outcome in {
                OrderOutcome.FILLED, OrderOutcome.TERMINAL_NOT_FILLED
            } else outcome.value),
            client_order_id=context.client_order_id,
            strategy=strategy,
        )
        if outcome in {OrderOutcome.FILLED, OrderOutcome.TERMINAL_NOT_FILLED}:
            pending_reconciliations.pop(order_id, None)
        else:
            context.risk_exit_reason = risk_reason
            print(
                f"RISK EXIT: BUY remaining quantity is still pending; "
                f"actual filled BTC will be reduced and rechecked"
            )


def execute_risk_exit(
    risk_reason, pending_reconciliations, candle_timestamp, strategy=None
):
    """Cancel/reconcile BUYs, refresh confirmed exposure, then reduce known BTC."""
    strategy = strategy or _live_strategy()
    cancel_pending_buys_for_risk_exit(
        pending_reconciliations, risk_reason, strategy
    )
    position = get_btc_position()
    if position is None:
        if any(ctx.side == "BUY" for ctx in pending_reconciliations.values()):
            print("RISK EXIT WAITING: pending BUY cancellation/fill is not terminal")
            return OrderOutcome.PENDING
        return None
    if has_open_order(OrderSide.SELL, include_protective=False):
        print(f"RISK EXIT SELL PENDING: reason={risk_reason}")
        return OrderOutcome.PENDING
    return _submit_and_log_sell(
        risk_reason,
        position,
        pending_reconciliations,
        candle_timestamp,
        strategy,
    )


def _context_from_record(record):
    side = str(record.get("side") or "UNKNOWN").upper()
    return PendingReconciliation(
        side,
        record.get("reason") or "startup_reconciliation",
        _optional_float(record.get("requested_notional")),
        None,
        None,
        record.get("client_order_id"),
        record.get("strategy_name"),
        record.get("strategy_parameters_json"),
    )


def restore_pending_orders(pending_reconciliations, strategy=None):
    """Rebuild pending state from current broker orders and the persistent ledger."""
    strategy = strategy or _live_strategy()
    records = database.get_order_records(pending_only=True)
    records_by_order_id = {str(row["order_id"]): row for row in records if row.get("order_id")}
    records_by_client_id = {
        row["client_order_id"]: row for row in records if row.get("client_order_id")
    }
    matched_ids = set()
    open_orders = get_open_btc_orders()
    for order in open_orders:
        order_id = str(order.id)
        client_id = getattr(order, "client_order_id", None)
        record = records_by_order_id.get(order_id) or records_by_client_id.get(client_id)
        if record is None:
            side_value = getattr(getattr(order, "side", None), "value", getattr(order, "side", None))
            context = PendingReconciliation(
                str(side_value or "UNKNOWN").upper(),
                "startup_unmatched_broker_order",
                None,
                None,
                None,
                client_id,
                strategy.name,
                _strategy_parameters_json(strategy),
            )
            print(
                f"SAFE-HALT WARNING: open BTC order {order_id} has incomplete DB context; "
                "conflicting trades remain blocked"
            )
            _record_order(
                order,
                context.side,
                context.reason,
                None,
                order_status=OrderOutcome.PENDING.value,
                client_order_id=client_id,
                strategy=strategy,
            )
        else:
            context = _context_from_record(record)
            matched_ids.add(record["id"])
            _record_order(
                order,
                context.side,
                context.reason,
                context.requested_notional,
                order_status=OrderOutcome.PENDING.value,
                client_order_id=client_id,
                strategy=strategy,
            )
        pending_reconciliations[order_id] = context

    for record in records:
        if record["id"] in matched_ids:
            continue
        context = _context_from_record(record)
        order_id = str(record.get("order_id") or f"client:{record.get('client_order_id')}")
        try:
            order = reconcile_order(
                record.get("order_id"),
                client_order_id=record.get("client_order_id"),
            )
        except Exception as error:
            print(
                f"SAFE-HALT WARNING: persisted order {order_id} is absent from open-order list "
                f"and could not be reconciled: {error}"
            )
            pending_reconciliations[order_id] = context
            continue
        outcome = classify_order(order)
        if outcome in {OrderOutcome.FILLED, OrderOutcome.TERMINAL_NOT_FILLED}:
            _record_order(
                order,
                context.side,
                context.reason,
                context.requested_notional,
                order_status=order_status_value(order),
                client_order_id=context.client_order_id,
                strategy=strategy,
            )
        else:
            pending_reconciliations[str(order.id)] = context
            _record_order(
                order,
                context.side,
                context.reason,
                context.requested_notional,
                order_status=outcome.value,
                client_order_id=context.client_order_id,
                strategy=strategy,
            )
    return pending_reconciliations


def bot_owns_position(position):
    """Require the broker quantity to reconcile with recorded bot fills."""
    if position is None:
        return True, ""
    try:
        actual = abs(float(position.qty))
        expected, reliable = database.get_bot_owned_btc_quantity()
    except Exception as error:
        return False, f"position ownership ledger unavailable: {error}"
    if not reliable or expected <= 0:
        return False, "no reliable bot-owned BTC fill record exists"
    tolerance = max(1e-8, actual * 1e-5)
    if abs(actual - expected) > tolerance:
        return False, f"broker BTC quantity {actual:g} differs from bot ledger {expected:g}"
    return True, ""


def _print_candle_status(timestamp, market_price, indicators, action, reason, strategy):
    values = indicators.iloc[-1]
    details = []
    for name in ("ma_fast", "ma_slow", "rsi"):
        value = _optional_float(values.get(name))
        if value is not None:
            details.append(f"{name.upper()}={value:.2f}")
    detail_text = " ".join(details)
    print(
        f"{datetime.now().strftime('%H:%M:%S')} STRATEGY={strategy.name} "
        f"BTC={market_price:.2f} {detail_text} ACTION={action} REASON={reason} "
        f"CANDLE={timestamp}"
    )


def _cancel_stale_protective_sells_if_flat(position, pending, strategy):
    if position is None and any(is_protective_order(order) for order in get_open_btc_orders()):
        _cancel_protective_stops(pending, strategy)


def run():
    strategy = _live_strategy()
    init_db(strict=True)
    account = client.get_account()
    pending_reconciliations = {}
    try:
        restore_pending_orders(pending_reconciliations, strategy)
        state = lookup_btc_position()
        if state.status == PositionLookupStatus.POSITION_STATE_UNKNOWN:
            print(f"SAFE-HALT BROKER POSITION STATE UNKNOWN: {state.error}")
            return
        position = state.position
        owned, ownership_reason = bot_owns_position(position)
        if not owned:
            print(
                "SAFE-HALT_MANUAL_OR_UNKNOWN_POSITION: "
                f"{ownership_reason}. Use a dedicated Alpaca account for this bot's BTC."
            )
            return
    except Exception as error:
        print(f"SAFE-HALT STARTUP RECONCILIATION FAILED: {error}")
        return

    print("=== PAPER BOT STARTED ===")
    print(f"Strategy: {strategy.name} | Timeframe: {trade_config.LIVE_TIMEFRAME}")
    print("Cash:", account.cash)
    print("Portfolio:", account.portfolio_value)
    last_processed_candle = None
    risk_exit_latch = database.get_state(RISK_EXIT_STATE_KEY)
    required_bars = strategy.required_warmup_bars()

    while True:
        try:
            reconcile_pending_orders(pending_reconciliations, strategy)
            bars = get_btc_bars(strategy=strategy)
            market_price = get_btc_market_price()
            state = lookup_btc_position()
            if state.status == PositionLookupStatus.POSITION_STATE_UNKNOWN:
                print(f"SAFE-HALT BROKER POSITION STATE UNKNOWN: {state.error}")
                time.sleep(trade_config.CHECK_INTERVAL_SECONDS)
                continue
            position = state.position
            owned, ownership_reason = bot_owns_position(position)
            if not owned:
                print(f"SAFE-HALT_MANUAL_OR_UNKNOWN_POSITION: {ownership_reason}")
                time.sleep(trade_config.CHECK_INTERVAL_SECONDS)
                continue
            _cancel_stale_protective_sells_if_flat(
                position, pending_reconciliations, strategy
            )
            latest_candle = bars.index[-1] if bars is not None and not bars.empty else None
            risk_reason = _risk_exit_reason(position, market_price, strategy)
            if risk_reason is not None:
                risk_exit_latch = risk_reason
                database.set_state(RISK_EXIT_STATE_KEY, risk_reason)

            # A latched risk exit survives partial/late BUY fills and process restarts.
            if risk_exit_latch:
                if latest_candle is not None:
                    last_processed_candle = latest_candle
                position = get_btc_position()
                if position is not None:
                    _record_evaluation(
                        bars, market_price, position, "SELL", risk_exit_latch, strategy
                    )
                execute_risk_exit(
                    risk_exit_latch,
                    pending_reconciliations,
                    latest_candle,
                    strategy,
                )
                position = get_btc_position()
                if position is None and not any(
                    ctx.side == "BUY" for ctx in pending_reconciliations.values()
                ):
                    risk_exit_latch = None
                    database.set_state(RISK_EXIT_STATE_KEY, None)
                time.sleep(trade_config.CHECK_INTERVAL_SECONDS)
                continue

            if bars is None or len(bars) < required_bars:
                print("Waiting for enough completed candles")
            elif should_process_candle(latest_candle, last_processed_candle):
                indicators = strategy.prepare_indicators(bars)
                decision = strategy.decide_at(indicators, len(indicators) - 1)
                action, reason = decision.action, decision.reason
                if pending_reconciliations:
                    action, reason = "HOLD", "order_pending_reconciliation"
                elif action == "BUY" and position is not None:
                    action, reason = "HOLD", "position_already_open"
                elif action == "BUY" and not can_buy(
                    trade_config.TRADE_AMOUNT_USD,
                    position=position,
                    market_price=market_price,
                ):
                    action, reason = "HOLD", "buy_blocked_by_position_cap_cash_or_order"
                elif action == "SELL" and position is None:
                    action, reason = "HOLD", "no_open_position"
                elif action == "SELL" and has_open_order(
                    OrderSide.SELL, include_protective=False
                ):
                    action, reason = "HOLD", "sell_order_pending"

                _print_candle_status(
                    latest_candle, market_price, indicators, action, reason, strategy
                )
                # One strategy order attempt per candle, including rejected/canceled orders.
                last_processed_candle = latest_candle
                _record_evaluation(bars, market_price, position, action, reason, strategy)
                if action == "BUY":
                    _submit_and_log_buy(
                        reason, pending_reconciliations, latest_candle, strategy
                    )
                elif action == "SELL":
                    _submit_and_log_sell(
                        reason,
                        position,
                        pending_reconciliations,
                        latest_candle,
                        strategy,
                    )
            else:
                print(
                    f"STATUS BTC={market_price:.2f} CANDLE={latest_candle} "
                    "STRATEGY=already_processed"
                )
        except BrokerPositionStateUnknown as error:
            print(f"SAFE-HALT BROKER POSITION STATE UNKNOWN: {error}")
        except Exception as error:
            print("ERROR:", error)
        time.sleep(trade_config.CHECK_INTERVAL_SECONDS)


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        print("\n=== BOT STOPPED ===")
