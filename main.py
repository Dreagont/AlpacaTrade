"""Live MA/RSI bot entry point with conservative broker-state handling."""

import json
import sys
import time
import uuid
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
    BrokerOrderNotFound,
    BrokerPositionStateUnknown,
    OrderFillTimeoutError,
    OrderOutcome,
    PositionLookupStatus,
    SubmissionFailureKind,
    broker_position_identifier,
    buy_btc,
    cancel_order,
    classify_order,
    classify_submission_exception,
    client,
    deterministic_client_order_id,
    get_btc_position,
    get_order_by_client_order_id,
    get_open_btc_orders,
    has_open_order,
    btc_symbol_matches,
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
    submission_kind: str | None = None
    created_at: str | None = None
    position_before_quantity: float | None = None
    order_role: str | None = None


@dataclass
class LiveRuntime:
    active_position: dict | None = None
    accounting_degraded: bool = False
    entries_disabled: bool = False
    risk_exit: dict | None = None
    last_risk_price_source: str | None = None


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


def _mark_accounting_degraded(runtime, message):
    if runtime is not None:
        runtime.accounting_degraded = True
        runtime.entries_disabled = True
    print(f"ACCOUNTING DEGRADED: {message}", file=sys.stderr)


def _position_quantity(position):
    if position is None:
        return 0.0
    quantity = abs(float(position.qty))
    if not isfinite(quantity):
        raise ValueError("Broker BTC position quantity is not finite")
    return quantity


def _set_active_position(runtime, position, *, source_order=None, strategy=None):
    if runtime is None:
        return None
    if position is None:
        runtime.active_position = None
        try:
            database.set_active_bot_position(None)
        except Exception as error:
            _mark_accounting_degraded(runtime, f"could not clear active BTC provenance: {error}")
        return None

    previous = runtime.active_position or {}
    quantity = _position_quantity(position)
    source_order = source_order or SimpleNamespace()
    record = {
        "asset_id": str(getattr(position, "asset_id", None) or broker_position_identifier(position)),
        "source_order_id": getattr(source_order, "id", None) or previous.get("source_order_id"),
        "source_client_order_id": (
            getattr(source_order, "client_order_id", None)
            or previous.get("source_client_order_id")
        ),
        "credited_quantity": quantity,
        "strategy_name": (
            strategy.name if strategy is not None else previous.get("strategy_name")
        ),
        "entry_fill_price": (
            _optional_float(getattr(source_order, "filled_avg_price", None))
            or previous.get("entry_fill_price")
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_confirmed": True,
    }
    if not btc_symbol_matches(getattr(position, "symbol", None)):
        raise ValueError("Cannot attribute a non-BTC broker position to this bot")
    runtime.active_position = record
    try:
        database.set_active_bot_position(record)
    except Exception as error:
        _mark_accounting_degraded(runtime, f"could not persist active BTC provenance: {error}")
    return record


def _record_unquantified_bot_position(runtime, order, strategy):
    if runtime is None:
        return
    runtime.active_position = {
        "asset_id": None,
        "source_order_id": getattr(order, "id", None),
        "source_client_order_id": getattr(order, "client_order_id", None),
        "credited_quantity": None,
        "strategy_name": strategy.name,
        "entry_fill_price": _optional_float(getattr(order, "filled_avg_price", None)),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_confirmed": True,
    }
    runtime.entries_disabled = True
    try:
        database.set_active_bot_position(runtime.active_position)
    except Exception as error:
        _mark_accounting_degraded(runtime, f"could not persist pending BTC provenance: {error}")


def _safe_record_order(runtime, *args, **kwargs):
    try:
        _record_order(*args, **kwargs)
        return True
    except Exception as error:
        _mark_accounting_degraded(runtime, f"broker order is not fully recorded: {error}")
        return False


def _refresh_position_after_order(runtime, order, side, before_quantity, strategy):
    """Read broker position and return (position, actual asset delta) after an order."""
    try:
        position = get_btc_position()
    except Exception as error:
        _mark_accounting_degraded(runtime, f"could not confirm post-order BTC quantity: {error}")
        return None, None
    after_quantity = _position_quantity(position)
    if before_quantity is None and runtime is not None and runtime.active_position:
        before_quantity = runtime.active_position.get("credited_quantity")
    before_quantity = float(before_quantity or 0.0)
    asset_delta = after_quantity - before_quantity
    if position is None:
        _set_active_position(runtime, None)
    elif side == "BUY":
        _set_active_position(runtime, position, source_order=order, strategy=strategy)
    elif runtime is not None and runtime.active_position is not None:
        _set_active_position(runtime, position, strategy=strategy)
    return position, asset_delta


def _load_runtime_provenance(runtime):
    try:
        runtime.active_position = database.get_active_bot_position()
    except Exception as error:
        _mark_accounting_degraded(runtime, f"active BTC provenance unavailable: {error}")


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
    asset_quantity_delta=None,
    submission_kind=None,
    created_at=None,
    reconcile_attempts=None,
    last_reconcile_at=None,
    position_before_quantity=None,
    order_role=None,
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
        asset_quantity_delta=asset_quantity_delta,
        submission_kind=submission_kind,
        created_at=created_at,
        reconcile_attempts=reconcile_attempts,
        last_reconcile_at=last_reconcile_at,
        position_before_quantity=position_before_quantity,
        order_role=order_role,
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
    pending_reconciliations, strategy, runtime=None, position_before_quantity=None,
    order_role=None,
):
    created_at = datetime.now(timezone.utc).isoformat()
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
        None,
        "synthetic_ambiguous",
        created_at,
        position_before_quantity,
        order_role,
    )
    pending_reconciliations[str(order.id)] = context
    _safe_record_order(
        runtime,
        order,
        side,
        reason,
        requested_notional,
        order_status="submit_unknown",
        client_order_id=client_order_id,
        strategy=strategy,
        submission_kind="synthetic_ambiguous",
        created_at=created_at,
        reconcile_attempts=0,
        position_before_quantity=position_before_quantity,
        order_role=order_role,
    )
    print(
        f"SAFE-HALT ORDER STATE UNKNOWN: side={side} "
        f"client_order_id={client_order_id}; will reconcile before any retry"
    )
    return OrderOutcome.PENDING


def _record_definitive_rejection(
    *, side, reason, requested_notional, client_order_id, strategy, runtime,
    order_role,
):
    rejected = SimpleNamespace(
        id=f"rejected:{client_order_id}",
        client_order_id=client_order_id,
        status=SimpleNamespace(value="rejected"),
        filled_qty=None,
        filled_avg_price=None,
    )
    _safe_record_order(
        runtime,
        rejected,
        side,
        reason,
        requested_notional,
        order_status="rejected",
        client_order_id=client_order_id,
        strategy=strategy,
        order_role=order_role,
    )
    print(
        f"ORDER DEFINITIVELY REJECTED: side={side} role={order_role} "
        f"client_order_id={client_order_id}; same-candle retry suppressed"
    )
    return OrderOutcome.TERMINAL_NOT_FILLED


def _confirmed_position_after_buy(order, strategy, runtime, *, before_quantity=0.0):
    position = None
    last_error = None
    for attempt in range(5):
        try:
            position = get_btc_position()
            if position is not None:
                break
        except Exception as error:
            last_error = error
        if attempt < 4:
            time.sleep(1)
    if position is None:
        _record_unquantified_bot_position(runtime, order, strategy)
        _mark_accounting_degraded(
            runtime,
            "BUY is broker-confirmed but credited BTC quantity is not yet available"
            + (f": {last_error}" if last_error else ""),
        )
        return None, None
    if not btc_symbol_matches(getattr(position, "symbol", None)):
        _record_unquantified_bot_position(runtime, order, strategy)
        _mark_accounting_degraded(runtime, "post-BUY position identity is not BTC/USD")
        return None, None
    quantity = _position_quantity(position)
    if quantity <= 0:
        _record_unquantified_bot_position(runtime, order, strategy)
        _mark_accounting_degraded(runtime, "post-BUY broker BTC quantity is not positive")
        return None, None
    _set_active_position(runtime, position, source_order=order, strategy=strategy)
    return position, quantity - before_quantity


def _submit_and_log_buy(
    reason, pending_reconciliations, candle_timestamp, strategy=None, runtime=None
):
    strategy = strategy or _live_strategy()
    amount = trade_config.TRADE_AMOUNT_USD
    order_role = "strategy_entry"
    client_order_id = _strategy_order_id(
        strategy, candle_timestamp, "BUY", role=order_role
    )
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
            runtime=runtime,
            position_before_quantity=0.0,
            order_role=order_role,
        )
    if order is None:
        try:
            order = buy_btc(amount, client_order_id=client_order_id)
        except Exception as error:
            try:
                order = get_order_by_client_order_id(client_order_id)
            except Exception:
                order = None
            if order is None:
                failure_kind = classify_submission_exception(error)
                if failure_kind == SubmissionFailureKind.DEFINITIVE_REJECTION:
                    return _record_definitive_rejection(
                        side="BUY",
                        reason=reason,
                        requested_notional=amount,
                        client_order_id=client_order_id,
                        strategy=strategy,
                        runtime=runtime,
                        order_role=order_role,
                    )
                return _persist_ambiguous_submission(
                    side="BUY",
                    reason=reason,
                    requested_notional=amount,
                    candle_timestamp=candle_timestamp,
                    client_order_id=client_order_id,
                    pending_reconciliations=pending_reconciliations,
                    strategy=strategy,
                    runtime=runtime,
                    position_before_quantity=0.0,
                    order_role=order_role,
                )
    try:
        order = wait_for_order_fill(order.id)
    except OrderFillTimeoutError as error:
        order = error.order
        outcome = classify_order(order, timed_out=True)
        context = PendingReconciliation(
            "BUY", reason, amount, None, candle_timestamp, client_order_id,
            strategy.name, _strategy_parameters_json(strategy), None,
            "broker_order", datetime.now(timezone.utc).isoformat(), 0.0,
            order_role,
        )
        pending_reconciliations[str(order.id)] = context
        _safe_record_order(
            runtime,
            order, "BUY", reason, amount, order_status=outcome.value,
            client_order_id=client_order_id, strategy=strategy,
            submission_kind="broker_order", position_before_quantity=0.0,
            order_role=order_role,
        )
        print(
            f"BUY ORDER PENDING: id={order.id} status={order_status_value(order)} "
            f"filled_qty={getattr(order, 'filled_qty', None)}"
        )
        return outcome
    outcome = classify_order(order)
    credited_delta = None
    position = None
    if runtime is not None and _optional_float(getattr(order, "filled_qty", None)):
        position, credited_delta = _confirmed_position_after_buy(
            order, strategy, runtime, before_quantity=0.0
        )
    _safe_record_order(
        runtime,
        order,
        "BUY",
        reason,
        amount,
        client_order_id=client_order_id,
        strategy=strategy,
        asset_quantity_delta=credited_delta,
        submission_kind="broker_order",
        position_before_quantity=0.0,
        order_role=order_role,
    )
    if outcome == OrderOutcome.FILLED:
        print(
            f"BUY ORDER FILLED: id={order.id} "
            f"filled_qty={getattr(order, 'filled_qty', None)}"
        )
        _place_protective_stop(
            order, candle_timestamp, pending_reconciliations, strategy,
            position=position, runtime=runtime,
        )
    elif outcome in {OrderOutcome.PENDING, OrderOutcome.PARTIAL_PENDING}:
        pending_reconciliations[str(order.id)] = PendingReconciliation(
            "BUY", reason, amount, None, candle_timestamp, client_order_id,
            strategy.name, _strategy_parameters_json(strategy), None,
            "broker_order", datetime.now(timezone.utc).isoformat(), 0.0,
            order_role,
        )
    else:
        print(f"BUY ORDER NOT FILLED: id={order.id} status={order_status_value(order)}")
    return outcome


def _price_to_tick(price, tick, *, round_down=True):
    if tick <= 0:
        raise ValueError("Broker stop-limit price increment must be positive")
    steps = price / tick
    return (floor(steps) if round_down else round(steps)) * tick


def _place_protective_stop(
    order, candle_timestamp, pending_reconciliations, strategy, *, position=None,
    runtime=None,
):
    if not trade_config.ENABLE_BROKER_STOP_LIMIT:
        return None
    stop_loss = strategy.stop_loss_percent
    if stop_loss is None:
        return None
    active = getattr(runtime, "active_position", None) if runtime is not None else None
    if not active or not active.get("source_confirmed"):
        print("PROTECTION WARNING: position provenance unavailable; protective stop skipped", file=sys.stderr)
        return None
    if position is None:
        for attempt in range(5):
            try:
                position = get_btc_position()
                if position is not None:
                    break
            except Exception:
                position = None
            if attempt < 4:
                time.sleep(1)
    if position is None or not btc_symbol_matches(getattr(position, "symbol", None)):
        print("PROTECTION WARNING: BTC position identity/quantity unavailable; client-side risk management remains active", file=sys.stderr)
        return None
    buy_order_id = str(getattr(order, "id", ""))
    buy_client_id = getattr(order, "client_order_id", None)
    if (
        (active.get("source_order_id") and str(active["source_order_id"]) != buy_order_id)
        or (
            active.get("source_client_order_id")
            and active["source_client_order_id"] != buy_client_id
        )
    ):
        print("PROTECTION WARNING: active BTC provenance does not match the confirmed BUY; protective stop skipped", file=sys.stderr)
        return None
    quantity = _position_quantity(position)
    entry_fill = _optional_float(getattr(order, "filled_avg_price", None))
    if quantity <= 0 or entry_fill is None or entry_fill <= 0:
        print("PROTECTIVE STOP SKIPPED: confirmed position quantity/price is unavailable")
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
    except Exception as error:
        try:
            protective = get_order_by_client_order_id(client_order_id)
        except Exception:
            protective = None
        if protective is None:
            if classify_submission_exception(error) == SubmissionFailureKind.DEFINITIVE_REJECTION:
                return _record_definitive_rejection(
                    side="SELL", reason="protective_stop_limit", requested_notional=None,
                    client_order_id=client_order_id, strategy=strategy, runtime=runtime,
                    order_role="protective_stop",
                )
            return _persist_ambiguous_submission(
                side="SELL",
                reason="protective_stop_limit",
                requested_notional=None,
                candle_timestamp=candle_timestamp,
                client_order_id=client_order_id,
                pending_reconciliations=pending_reconciliations,
                strategy=strategy,
                runtime=runtime,
                position_before_quantity=quantity,
                order_role="protective_stop",
            )
    outcome = classify_order(protective)
    asset_delta = None
    if runtime is not None and _optional_float(getattr(protective, "filled_qty", None)):
        _position, asset_delta = _refresh_position_after_order(
            runtime, protective, "SELL", quantity, strategy
        )
    if outcome in {OrderOutcome.PENDING, OrderOutcome.PARTIAL_PENDING, OrderOutcome.TIMEOUT_PENDING}:
        pending_reconciliations[str(protective.id)] = PendingReconciliation(
            "SELL", "protective_stop_limit", None, _position_snapshot(position),
            candle_timestamp, client_order_id, strategy.name,
            _strategy_parameters_json(strategy), None, "broker_order",
            datetime.now(timezone.utc).isoformat(), quantity, "protective_stop",
        )
    _safe_record_order(
        runtime,
        protective,
        "SELL",
        "protective_stop_limit",
        None,
        client_order_id=client_order_id,
        strategy=strategy,
        asset_quantity_delta=asset_delta,
        submission_kind="broker_order",
        position_before_quantity=quantity,
        order_role="protective_stop",
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


def _cancel_protective_stops(pending_reconciliations, strategy, runtime=None):
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
                _strategy_parameters_json(strategy), None, "broker_order",
                datetime.now(timezone.utc).isoformat(), None, "protective_stop"
            )
            _safe_record_order(
                runtime,
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
        _safe_record_order(
            runtime,
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
    reason, position, pending_reconciliations, candle_timestamp, strategy=None,
    *, runtime=None, role="strategy_exit", risk_episode=None, risk_attempt=0,
):
    strategy = strategy or _live_strategy()
    _cancel_protective_stops(pending_reconciliations, strategy, runtime)
    position = get_btc_position()
    if position is None:
        print("SELL SKIPPED: broker confirms position is flat")
        return None
    if has_open_order(OrderSide.SELL, include_protective=False):
        print(f"EXIT ORDER PENDING: reason={reason}")
        return OrderOutcome.PENDING
    quantity = abs(float(position.qty))
    if role.startswith("risk_exit"):
        identity = f"{risk_episode or uuid.uuid4().hex}:{risk_attempt}:{reason}"
        client_order_id = _strategy_order_id(strategy, identity, "SELL", role=role)
    else:
        client_order_id = _strategy_order_id(strategy, candle_timestamp, "SELL", role=role)
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
        except Exception as error:
            try:
                order = get_order_by_client_order_id(client_order_id)
            except Exception:
                order = None
            if order is None:
                if classify_submission_exception(error) == SubmissionFailureKind.DEFINITIVE_REJECTION:
                    return _record_definitive_rejection(
                        side="SELL", reason=reason,
                        requested_notional=_optional_float(position.market_value),
                        client_order_id=client_order_id, strategy=strategy,
                        runtime=runtime, order_role=role,
                    )
                return _persist_ambiguous_submission(
                    side="SELL",
                    reason=reason,
                    requested_notional=_optional_float(position.market_value),
                    candle_timestamp=candle_timestamp,
                    client_order_id=client_order_id,
                    pending_reconciliations=pending_reconciliations,
                    strategy=strategy,
                    runtime=runtime,
                    position_before_quantity=quantity,
                    order_role=role,
                )
    try:
        order = wait_for_order_fill(order.id)
    except OrderFillTimeoutError as error:
        order = error.order
        outcome = classify_order(order, timed_out=True)
        pending_reconciliations[str(order.id)] = PendingReconciliation(
            "SELL", reason, _optional_float(position.market_value),
            _position_snapshot(position), candle_timestamp, client_order_id,
            strategy.name, _strategy_parameters_json(strategy), None,
            "broker_order", datetime.now(timezone.utc).isoformat(), quantity, role
        )
        _safe_record_order(
            runtime,
            order,
            "SELL",
            reason,
            _optional_float(position.market_value),
            position,
            order_status=outcome.value,
            client_order_id=client_order_id,
            strategy=strategy,
            submission_kind="broker_order", position_before_quantity=quantity,
            order_role=role,
        )
        print(f"SELL ORDER PENDING: id={order.id} status={order_status_value(order)}")
        return outcome
    outcome = classify_order(order)
    asset_delta = None
    if runtime is not None and _optional_float(getattr(order, "filled_qty", None)):
        _post_position, asset_delta = _refresh_position_after_order(
            runtime, order, "SELL", quantity, strategy
        )
    _safe_record_order(
        runtime,
        order,
        "SELL",
        reason,
        _optional_float(position.market_value),
        position,
        client_order_id=client_order_id,
        strategy=strategy,
        asset_quantity_delta=asset_delta,
        submission_kind="broker_order", position_before_quantity=quantity,
        order_role=role,
    )
    if outcome in {OrderOutcome.PENDING, OrderOutcome.PARTIAL_PENDING}:
        pending_reconciliations[str(order.id)] = PendingReconciliation(
            "SELL", reason, _optional_float(position.market_value),
            _position_snapshot(position), candle_timestamp, client_order_id,
            strategy.name, _strategy_parameters_json(strategy), None,
            "broker_order", datetime.now(timezone.utc).isoformat(), quantity, role
        )
    print(f"SELL ORDER {outcome.value.upper()}: id={order.id} reason={reason}")
    return outcome


def _expire_synthetic_if_confirmed_absent(order_id, context):
    if context.submission_kind != "synthetic_ambiguous":
        return False
    try:
        state = database.note_synthetic_order_reconciliation(context.client_order_id)
    except Exception as error:
        print(f"ACCOUNTING DEGRADED: cannot record submit_unknown reconciliation: {error}", file=sys.stderr)
        return False
    if not state:
        return False
    context.created_at = state["created_at"] or context.created_at
    age_seconds = 0.0
    try:
        created = datetime.fromisoformat(context.created_at)
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        age_seconds = (datetime.now(timezone.utc) - created.astimezone(timezone.utc)).total_seconds()
    except (TypeError, ValueError):
        return False
    attempts = int(state["reconcile_attempts"] or 0)
    if (age_seconds >= trade_config.SUBMIT_UNKNOWN_MAX_AGE_SECONDS
            and attempts >= trade_config.SUBMIT_UNKNOWN_MAX_RECONCILE_ATTEMPTS):
        try:
            return database.mark_synthetic_order_not_created(context.client_order_id)
        except Exception as error:
            print(f"ACCOUNTING DEGRADED: cannot expire submit_unknown: {error}", file=sys.stderr)
    return False


def reconcile_pending_orders(pending_reconciliations, strategy=None, *, runtime=None):
    strategy = strategy or _live_strategy()
    for order_id, context in list(pending_reconciliations.items()):
        try:
            order = reconcile_order(
                order_id,
                client_order_id=context.client_order_id,
            )
        except BrokerOrderNotFound as error:
            if _expire_synthetic_if_confirmed_absent(order_id, context):
                pending_reconciliations.pop(order_id, None)
                print(f"SYNTHETIC ORDER EXPIRED AFTER CONFIRMED NOT-FOUND: {context.client_order_id}")
            else:
                print(f"ORDER RECONCILIATION CONFIRMED NOT-FOUND: id={order_id} error={error}")
            continue
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
        asset_delta = None
        has_filled_quantity = bool(_optional_float(getattr(order, "filled_qty", None)))
        if runtime is not None and has_filled_quantity and context.side == "SELL":
            before = context.position_before_quantity
            if before is None and context.position is not None:
                before = getattr(context.position, "quantity", None)
            _position, asset_delta = _refresh_position_after_order(
                runtime, order, "SELL", before, selected_strategy
            )
        elif has_filled_quantity and context.side == "BUY" and runtime is not None:
            _position, asset_delta = _confirmed_position_after_buy(
                order, selected_strategy, runtime,
                before_quantity=context.position_before_quantity or 0.0,
            )
        _safe_record_order(
            runtime,
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
            asset_quantity_delta=asset_delta,
            submission_kind=context.submission_kind,
            created_at=context.created_at,
            position_before_quantity=context.position_before_quantity,
            order_role=context.order_role,
        )
        if outcome in {OrderOutcome.FILLED, OrderOutcome.TERMINAL_NOT_FILLED}:
            print(f"ORDER RECONCILED: id={order_id} outcome={outcome.value} status={status}")
            pending_reconciliations.pop(order_id, None)
            if runtime is not None and context.order_role and context.order_role.startswith("risk_exit"):
                _advance_risk_exit_retry(runtime)
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
                    position=get_btc_position() if runtime is not None else None,
                    runtime=runtime,
                )
        else:
            print(f"ORDER STILL PENDING: id={order_id} outcome={outcome.value} status={status}")
    # Terminal rejections/cancellations never reopen their original strategy candle.
    return []


def cancel_pending_buys_for_risk_exit(pending_reconciliations, risk_reason, strategy, runtime=None):
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
        asset_delta = None
        if runtime is not None and _optional_float(getattr(order, "filled_qty", None)):
            _position, asset_delta = _confirmed_position_after_buy(
                order, strategy, runtime,
                before_quantity=context.position_before_quantity or 0.0,
            )
        _safe_record_order(
            runtime,
            order,
            "BUY",
            context.reason,
            context.requested_notional,
            order_status=(order_status_value(order) if outcome in {
                OrderOutcome.FILLED, OrderOutcome.TERMINAL_NOT_FILLED
            } else outcome.value),
            client_order_id=context.client_order_id,
            strategy=strategy,
            asset_quantity_delta=asset_delta,
            submission_kind=context.submission_kind or "broker_order",
            created_at=context.created_at,
            position_before_quantity=context.position_before_quantity or 0.0,
            order_role=context.order_role or "strategy_entry",
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
    risk_reason, pending_reconciliations, candle_timestamp, strategy=None, *,
    runtime=None, risk_state=None,
):
    """Cancel/reconcile BUYs, refresh confirmed exposure, then reduce known BTC."""
    strategy = strategy or _live_strategy()
    cancel_pending_buys_for_risk_exit(
        pending_reconciliations, risk_reason, strategy, runtime
    )
    position = get_btc_position()
    if position is None:
        if any(ctx.side == "BUY" for ctx in pending_reconciliations.values()):
            print("RISK EXIT WAITING: pending BUY cancellation/fill is not terminal")
            return OrderOutcome.PENDING
        return None
    if any(ctx.side == "SELL" for ctx in pending_reconciliations.values()):
        print("RISK EXIT WAITING: a SELL order is still under broker reconciliation")
        return OrderOutcome.PENDING
    if has_open_order(OrderSide.SELL, include_protective=False):
        print(f"RISK EXIT SELL PENDING: reason={risk_reason}")
        return OrderOutcome.PENDING
    return _submit_and_log_sell(
        risk_reason,
        position,
        pending_reconciliations,
        candle_timestamp,
        strategy,
        runtime=runtime,
        role=f"risk_exit_{risk_reason}",
        risk_episode=(risk_state or {}).get("episode_id"),
        risk_attempt=(risk_state or {}).get("attempt", 0),
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
        None,
        record.get("submission_kind"),
        record.get("created_at"),
        _optional_float(record.get("position_before_quantity")),
        record.get("order_role"),
    )


def restore_pending_orders(pending_reconciliations, strategy=None, *, runtime=None):
    """Rebuild pending state from current broker orders and the persistent ledger."""
    strategy = strategy or _live_strategy()
    try:
        records = database.get_order_records(pending_only=True)
    except Exception as error:
        records = []
        _mark_accounting_degraded(runtime, f"persistent order ledger unavailable at startup: {error}")
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
            protective = is_protective_order(order)
            context = PendingReconciliation(
                str(side_value or "UNKNOWN").upper(),
                "protective_stop_limit" if protective else "startup_unmatched_broker_order",
                None,
                None,
                None,
                client_id,
                strategy.name,
                _strategy_parameters_json(strategy),
                None,
                "broker_order",
                datetime.now(timezone.utc).isoformat(),
                (
                    _optional_float((runtime.active_position or {}).get("credited_quantity"))
                    if runtime else None
                ),
                "protective_stop" if protective else "broker_order",
            )
            print(
                f"SAFE-HALT WARNING: open BTC order {order_id} has incomplete DB context; "
                "conflicting trades remain blocked"
            )
            _safe_record_order(
                runtime,
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
            _safe_record_order(
                runtime,
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
        except BrokerOrderNotFound as error:
            if _expire_synthetic_if_confirmed_absent(order_id, context):
                print(f"SYNTHETIC ORDER EXPIRED AFTER CONFIRMED NOT-FOUND: {context.client_order_id}")
            else:
                pending_reconciliations[order_id] = context
                print(f"SAFE-HALT WARNING: persisted order {order_id} confirmed absent but retained: {error}")
            continue
        except Exception as error:
            print(
                f"SAFE-HALT WARNING: persisted order {order_id} is absent from open-order list "
                f"and could not be reconciled: {error}"
            )
            pending_reconciliations[order_id] = context
            continue
        outcome = classify_order(order)
        if outcome in {OrderOutcome.FILLED, OrderOutcome.TERMINAL_NOT_FILLED}:
            _safe_record_order(
                runtime,
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
            _safe_record_order(
                runtime,
                order,
                context.side,
                context.reason,
                context.requested_notional,
                order_status=outcome.value,
                client_order_id=context.client_order_id,
                strategy=strategy,
            )
    return pending_reconciliations


def bot_owns_position(position, runtime=None):
    """Accept only broker-confirmed active provenance or a complete net-delta ledger."""
    if position is None:
        return True, ""
    try:
        actual = abs(float(position.qty))
        if not btc_symbol_matches(getattr(position, "symbol", None)):
            return False, "broker position identity is not BTC/USD"
        active = runtime.active_position if runtime and runtime.active_position else database.get_active_bot_position()
        if active is not None:
            expected = _optional_float(active.get("credited_quantity"))
            if not active.get("source_confirmed"):
                return False, "active bot position provenance is incomplete"
            if expected is None and runtime is not None:
                # This process has a confirmed bot BUY but the exchange position
                # became visible after the bounded post-fill lookup window.
                expected = actual
                runtime.active_position = dict(active)
                runtime.active_position.update({
                    "asset_id": str(getattr(position, "asset_id", None) or broker_position_identifier(position)),
                    "credited_quantity": actual,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
                try:
                    database.set_active_bot_position(runtime.active_position)
                except Exception as error:
                    _mark_accounting_degraded(runtime, f"could not persist delayed BTC quantity: {error}")
            if expected is None or expected <= 0:
                return False, "active bot position provenance is incomplete"
            expected_asset = active.get("asset_id")
            actual_asset = str(getattr(position, "asset_id", None) or broker_position_identifier(position))
            if expected_asset and str(expected_asset) != actual_asset:
                return False, "active bot position asset identity does not match broker position"
            reliable = True
        else:
            details = database.get_bot_owned_btc_quantity_details()
            expected, reliable = details.quantity, details.reliable
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


def _load_risk_exit_state(runtime):
    try:
        raw = database.get_state(RISK_EXIT_STATE_KEY)
        if not raw:
            return None
        normalized_legacy = False
        try:
            value = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            value = {"reason": raw}
            normalized_legacy = True
        if isinstance(value, str):
            value = {"reason": value}
            normalized_legacy = True
        if not isinstance(value, dict) or not value.get("reason"):
            return None
        if "episode_id" not in value or "attempt" not in value or "next_attempt_at" not in value:
            normalized_legacy = True
        value.setdefault("episode_id", uuid.uuid4().hex)
        value.setdefault("attempt", 0)
        value.setdefault("next_attempt_at", 0.0)
        runtime.risk_exit = value
        if normalized_legacy:
            _persist_risk_exit_state(runtime)
        return value
    except Exception as error:
        _mark_accounting_degraded(runtime, f"risk-exit latch unavailable: {error}")
        return None


def _persist_risk_exit_state(runtime):
    try:
        database.set_state(
            RISK_EXIT_STATE_KEY,
            json.dumps(runtime.risk_exit, sort_keys=True, separators=(",", ":"))
            if runtime.risk_exit else None,
        )
    except Exception as error:
        _mark_accounting_degraded(runtime, f"could not persist risk-exit latch: {error}")


def _latch_risk_exit(runtime, reason):
    if runtime.risk_exit is None:
        runtime.risk_exit = {
            "episode_id": uuid.uuid4().hex,
            "reason": reason,
            "attempt": 0,
            "next_attempt_at": 0.0,
        }
        _persist_risk_exit_state(runtime)
    return runtime.risk_exit


def _advance_risk_exit_retry(runtime):
    state = runtime.risk_exit
    if state is None:
        return
    state["attempt"] = int(state.get("attempt", 0)) + 1
    state["next_attempt_at"] = time.time() + trade_config.RISK_EXIT_RETRY_COOLDOWN_SECONDS
    _persist_risk_exit_state(runtime)


def run():
    strategy = _live_strategy()
    runtime = LiveRuntime()
    if not init_db(strict=False):
        _mark_accounting_degraded(runtime, "database initialization failed; entries disabled")
    try:
        account = client.get_account()
    except Exception as error:
        account = None
        _mark_accounting_degraded(runtime, f"account summary unavailable: {error}")
    pending_reconciliations = {}
    try:
        _load_runtime_provenance(runtime)
        try:
            restore_pending_orders(pending_reconciliations, strategy, runtime=runtime)
        except Exception as error:
            _mark_accounting_degraded(runtime, f"startup order reconciliation unavailable: {error}")
            if runtime.active_position is None:
                print("SAFE-HALT STARTUP: database order state unavailable and no active position provenance is loaded")
                return
        state = lookup_btc_position()
        if state.status == PositionLookupStatus.POSITION_STATE_UNKNOWN:
            print(f"SAFE-HALT BROKER POSITION STATE UNKNOWN: {state.error}")
            return
        position = state.position
        owned, ownership_reason = bot_owns_position(position, runtime)
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
    print("Cash:", getattr(account, "cash", "unavailable"))
    print("Portfolio:", getattr(account, "portfolio_value", "unavailable"))
    last_processed_candle = None
    risk_exit_state = _load_risk_exit_state(runtime)
    required_bars = strategy.required_warmup_bars()

    while True:
        try:
            reconcile_pending_orders(pending_reconciliations, strategy, runtime=runtime)
            state = lookup_btc_position()
            if state.status == PositionLookupStatus.POSITION_STATE_UNKNOWN:
                print(f"SAFE-HALT BROKER POSITION STATE UNKNOWN: {state.error}")
                time.sleep(trade_config.CHECK_INTERVAL_SECONDS)
                continue
            position = state.position
            owned, ownership_reason = bot_owns_position(position, runtime)
            if not owned:
                print(f"SAFE-HALT_MANUAL_OR_UNKNOWN_POSITION: {ownership_reason}")
                time.sleep(trade_config.CHECK_INTERVAL_SECONDS)
                continue
            market_price = None
            price_source = "market_data"
            try:
                market_price = _optional_float(get_btc_market_price())
            except Exception as error:
                print(f"MARKET PRICE ERROR: {error}", file=sys.stderr)
            if market_price is None and position is not None:
                market_price = _optional_float(getattr(position, "current_price", None))
                price_source = "broker_position.current_price"
            if market_price is not None and position is not None and price_source != runtime.last_risk_price_source:
                print(f"Risk price source: {price_source}")
                runtime.last_risk_price_source = price_source
            if risk_exit_state is None and position is not None and market_price is not None:
                new_risk_reason = _risk_exit_reason(position, market_price, strategy)
                if new_risk_reason:
                    risk_exit_state = _latch_risk_exit(runtime, new_risk_reason)
            if risk_exit_state is not None:
                if position is None and not any(ctx.side == "BUY" for ctx in pending_reconciliations.values()):
                    runtime.risk_exit = None
                    risk_exit_state = None
                    _persist_risk_exit_state(runtime)
                elif time.time() >= float(risk_exit_state.get("next_attempt_at", 0)):
                    attempt = int(risk_exit_state.get("attempt", 0))
                    if attempt >= trade_config.RISK_EXIT_MAX_ATTEMPTS:
                        print(
                            f"RISK EXIT RETRY LIMIT REACHED: reason={risk_exit_state['reason']} "
                            f"attempts={attempt}; no further automatic retries in this episode",
                            file=sys.stderr,
                        )
                    else:
                        _record_evaluation(
                            None, market_price, position, "SELL", risk_exit_state["reason"], strategy
                        )
                        outcome = execute_risk_exit(
                            risk_exit_state["reason"], pending_reconciliations, None,
                            strategy, runtime=runtime, risk_state=risk_exit_state,
                        )
                        if outcome == OrderOutcome.TERMINAL_NOT_FILLED:
                            _advance_risk_exit_retry(runtime)
                        elif outcome == OrderOutcome.FILLED:
                            fresh = get_btc_position()
                            if fresh is None:
                                runtime.risk_exit = None
                                risk_exit_state = None
                                _persist_risk_exit_state(runtime)
                            else:
                                _advance_risk_exit_retry(runtime)
                time.sleep(trade_config.CHECK_INTERVAL_SECONDS)
                continue

            try:
                bars = get_btc_bars(strategy=strategy)
            except Exception as error:
                print(f"HISTORICAL BARS UNAVAILABLE; strategy evaluation skipped: {error}", file=sys.stderr)
                time.sleep(trade_config.CHECK_INTERVAL_SECONDS)
                continue
            if market_price is None:
                print("Strategy evaluation skipped: no valid current-price source", file=sys.stderr)
                time.sleep(trade_config.CHECK_INTERVAL_SECONDS)
                continue
            _cancel_stale_protective_sells_if_flat(
                position, pending_reconciliations, strategy
            )
            latest_candle = bars.index[-1] if bars is not None and not bars.empty else None

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
                elif action == "BUY" and runtime.entries_disabled:
                    action, reason = "HOLD", "new_entries_disabled_accounting_degraded"
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
                        reason, pending_reconciliations, latest_candle, strategy, runtime
                    )
                elif action == "SELL":
                    _submit_and_log_sell(
                        reason,
                        position,
                        pending_reconciliations,
                        latest_candle,
                        strategy,
                        runtime=runtime,
                        role="strategy_exit",
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
