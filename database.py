import json
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from math import isfinite
from typing import Optional

import config


ACTIVE_BOT_POSITION_KEY = "active_bot_position"


@dataclass(frozen=True)
class BotOwnedQuantityDetails:
    quantity: float
    reliable: bool
    evidence: str


def _create_tables(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS strategy_evaluations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            current_price REAL,
            ma_fast REAL,
            ma_slow REAL,
            rsi REAL,
            action TEXT NOT NULL,
            reason TEXT NOT NULL,
            position_value REAL,
            average_entry_price REAL,
            unrealized_pnl REAL,
            strategy_name TEXT,
            strategy_parameters_json TEXT
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            order_id TEXT,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            requested_notional REAL,
            quantity REAL,
            fill_price REAL,
            reason TEXT NOT NULL,
            realized_gross_pnl REAL,
            order_status TEXT,
            client_order_id TEXT,
            strategy_name TEXT,
            strategy_parameters_json TEXT,
            asset_quantity_delta REAL,
            submission_kind TEXT,
            created_at TEXT,
            reconcile_attempts INTEGER DEFAULT 0,
            last_reconcile_at TEXT,
            position_before_quantity REAL,
            order_role TEXT
        )
        """
    )
    columns = {
        row[1] for row in connection.execute("PRAGMA table_info(orders)").fetchall()
    }
    if "realized_gross_pnl" not in columns and "realized_pnl" in columns:
        connection.execute(
            "ALTER TABLE orders RENAME COLUMN realized_pnl TO realized_gross_pnl"
        )
        columns.remove("realized_pnl")
        columns.add("realized_gross_pnl")
    if "order_status" not in columns:
        connection.execute("ALTER TABLE orders ADD COLUMN order_status TEXT")
        columns.add("order_status")
    for name in ("client_order_id", "strategy_name", "strategy_parameters_json"):
        if name not in columns:
            connection.execute(f"ALTER TABLE orders ADD COLUMN {name} TEXT")
            columns.add(name)
    for name, declaration in (
        ("asset_quantity_delta", "REAL"),
        ("submission_kind", "TEXT"),
        ("created_at", "TEXT"),
        ("reconcile_attempts", "INTEGER DEFAULT 0"),
        ("last_reconcile_at", "TEXT"),
        ("position_before_quantity", "REAL"),
        ("order_role", "TEXT"),
    ):
        if name not in columns:
            connection.execute(f"ALTER TABLE orders ADD COLUMN {name} {declaration}")
            columns.add(name)
    evaluation_columns = {
        row[1]
        for row in connection.execute("PRAGMA table_info(strategy_evaluations)").fetchall()
    }
    for name in ("strategy_name", "strategy_parameters_json"):
        if name not in evaluation_columns:
            connection.execute(
                f"ALTER TABLE strategy_evaluations ADD COLUMN {name} TEXT"
            )
    indexes = {
        row[1] for row in connection.execute("PRAGMA index_list(orders)").fetchall()
    }
    if "idx_orders_order_id" not in indexes:
        connection.execute(
            """
            UPDATE orders SET order_id=NULL
            WHERE order_id IS NOT NULL
              AND id NOT IN (
                  SELECT MAX(id) FROM orders
                  WHERE order_id IS NOT NULL
                  GROUP BY order_id
              )
            """
        )
        connection.execute(
            "CREATE UNIQUE INDEX idx_orders_order_id "
            "ON orders(order_id) WHERE order_id IS NOT NULL"
        )
    indexes = {
        row[1] for row in connection.execute("PRAGMA index_list(orders)").fetchall()
    }
    if "idx_orders_client_order_id" not in indexes:
        connection.execute(
            "CREATE UNIQUE INDEX idx_orders_client_order_id "
            "ON orders(client_order_id) WHERE client_order_id IS NOT NULL"
        )
    connection.execute(
        "CREATE TABLE IF NOT EXISTS bot_state (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )


def _report_failure(error: Exception) -> None:
    print(f"DATABASE LOG ERROR: {error}", file=sys.stderr)


def _close_safely(connection: Optional[sqlite3.Connection]) -> None:
    if connection is not None:
        try:
            connection.close()
        except Exception as error:
            _report_failure(error)


def init_db(*, strict: bool = False) -> bool:
    connection = None
    try:
        connection = sqlite3.connect(config.DATABASE_PATH, timeout=5)
        with connection:
            _create_tables(connection)
    except Exception as error:
        _report_failure(error)
        if strict:
            raise
        return False
    finally:
        _close_safely(connection)
    return True


def log_evaluation(
    *,
    timestamp: Optional[str] = None,
    symbol: str,
    timeframe: str,
    current_price: Optional[float],
    ma_fast: Optional[float],
    ma_slow: Optional[float],
    rsi: Optional[float],
    action: str,
    reason: str,
    position_value: Optional[float],
    average_entry_price: Optional[float],
    unrealized_pnl: Optional[float],
    strategy_name: Optional[str] = None,
    strategy_parameters_json: Optional[str] = None,
) -> None:
    timestamp = timestamp or datetime.now(timezone.utc).isoformat()
    connection = None
    try:
        connection = sqlite3.connect(config.DATABASE_PATH, timeout=5)
        with connection:
            _create_tables(connection)
            connection.execute(
                """
                INSERT INTO strategy_evaluations (
                    timestamp, symbol, timeframe, current_price, ma_fast, ma_slow,
                    rsi, action, reason, position_value, average_entry_price,
                    unrealized_pnl, strategy_name, strategy_parameters_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp,
                    symbol,
                    timeframe,
                    current_price,
                    ma_fast,
                    ma_slow,
                    rsi,
                    action,
                    reason,
                    position_value,
                    average_entry_price,
                    unrealized_pnl,
                    strategy_name,
                    strategy_parameters_json,
                ),
            )
    except Exception as error:
        _report_failure(error)
    finally:
        _close_safely(connection)


def upsert_order(
    *,
    timestamp: Optional[str] = None,
    order_id: Optional[str],
    symbol: str,
    side: str,
    requested_notional: Optional[float],
    quantity: Optional[float],
    fill_price: Optional[float],
    reason: str,
    realized_gross_pnl: Optional[float] = None,
    order_status: Optional[str] = None,
    client_order_id: Optional[str] = None,
    strategy_name: Optional[str] = None,
    strategy_parameters_json: Optional[str] = None,
    asset_quantity_delta: Optional[float] = None,
    submission_kind: Optional[str] = None,
    created_at: Optional[str] = None,
    reconcile_attempts: Optional[int] = None,
    last_reconcile_at: Optional[str] = None,
    position_before_quantity: Optional[float] = None,
    order_role: Optional[str] = None,
) -> None:
    timestamp = timestamp or datetime.now(timezone.utc).isoformat()
    created_at = created_at or timestamp
    connection = None
    try:
        connection = sqlite3.connect(config.DATABASE_PATH, timeout=5)
        with connection:
            _create_tables(connection)
            values = (
                timestamp,
                order_id,
                symbol,
                side,
                requested_notional,
                quantity,
                fill_price,
                reason,
                realized_gross_pnl,
                order_status,
                client_order_id,
                strategy_name,
                strategy_parameters_json,
                asset_quantity_delta,
                submission_kind,
                created_at,
                reconcile_attempts,
                last_reconcile_at,
                position_before_quantity,
                order_role,
            )
            if client_order_id is not None:
                connection.execute(
                    """
                    UPDATE orders SET timestamp=?, order_id=?, symbol=?, side=?,
                        requested_notional=?, quantity=?, fill_price=?, reason=?,
                        realized_gross_pnl=?, order_status=?, strategy_name=?,
                        strategy_parameters_json=?,
                        asset_quantity_delta=COALESCE(?, asset_quantity_delta),
                        submission_kind=COALESCE(?, submission_kind),
                        created_at=COALESCE(orders.created_at, ?),
                        reconcile_attempts=COALESCE(?, reconcile_attempts),
                        last_reconcile_at=COALESCE(?, last_reconcile_at),
                        position_before_quantity=COALESCE(?, position_before_quantity),
                        order_role=COALESCE(?, order_role)
                    WHERE client_order_id=?
                    """,
                    (
                        *values[:10],
                        strategy_name,
                        strategy_parameters_json,
                        asset_quantity_delta,
                        submission_kind,
                        created_at,
                        reconcile_attempts,
                        last_reconcile_at,
                        position_before_quantity,
                        order_role,
                        client_order_id,
                    ),
                )
                if connection.execute("SELECT changes()").fetchone()[0]:
                    return
            connection.execute(
                """
                INSERT INTO orders (
                    timestamp, order_id, symbol, side, requested_notional,
                    quantity, fill_price, reason, realized_gross_pnl, order_status,
                    client_order_id, strategy_name, strategy_parameters_json,
                    asset_quantity_delta, submission_kind, created_at,
                    reconcile_attempts, last_reconcile_at,
                    position_before_quantity, order_role
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(order_id) WHERE order_id IS NOT NULL DO UPDATE SET
                    timestamp=excluded.timestamp,
                    symbol=excluded.symbol,
                    side=excluded.side,
                    requested_notional=excluded.requested_notional,
                    quantity=excluded.quantity,
                    fill_price=excluded.fill_price,
                    reason=excluded.reason,
                    realized_gross_pnl=excluded.realized_gross_pnl,
                    order_status=excluded.order_status,
                    client_order_id=excluded.client_order_id,
                    strategy_name=excluded.strategy_name,
                    strategy_parameters_json=excluded.strategy_parameters_json,
                    asset_quantity_delta=COALESCE(excluded.asset_quantity_delta, orders.asset_quantity_delta),
                    submission_kind=COALESCE(excluded.submission_kind, orders.submission_kind),
                    created_at=COALESCE(orders.created_at, excluded.created_at),
                    reconcile_attempts=COALESCE(excluded.reconcile_attempts, orders.reconcile_attempts),
                    last_reconcile_at=COALESCE(excluded.last_reconcile_at, orders.last_reconcile_at),
                    position_before_quantity=COALESCE(excluded.position_before_quantity, orders.position_before_quantity),
                    order_role=COALESCE(excluded.order_role, orders.order_role)
                """,
                values,
            )
    except Exception as error:
        _report_failure(error)
        raise
    finally:
        _close_safely(connection)


def log_order(**kwargs) -> None:
    """Backward-compatible name for the order upsert operation."""
    upsert_order(**kwargs)


def get_order_records(*, pending_only: bool = False) -> list[dict]:
    connection = sqlite3.connect(config.DATABASE_PATH, timeout=5)
    connection.row_factory = sqlite3.Row
    try:
        _create_tables(connection)
        if pending_only:
            statuses = (
                "timeout_pending",
                "partial_pending",
                "pending",
                "submit_unknown",
                "new",
                "accepted",
                "pending_new",
                "pending_cancel",
            )
            placeholders = ",".join("?" for _ in statuses)
            rows = connection.execute(
                f"SELECT * FROM orders WHERE order_status IN ({placeholders}) ORDER BY id",
                statuses,
            ).fetchall()
        else:
            rows = connection.execute("SELECT * FROM orders ORDER BY id").fetchall()
        return [dict(row) for row in rows]
    finally:
        connection.close()


def get_active_bot_position() -> dict | None:
    raw = get_state(ACTIVE_BOT_POSITION_KEY)
    if raw is None:
        return None
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("Persisted active bot position is not an object")
    return value


def set_active_bot_position(position: dict | None) -> None:
    if position is None:
        set_state(ACTIVE_BOT_POSITION_KEY, None)
        return
    set_state(
        ACTIVE_BOT_POSITION_KEY,
        json.dumps(position, sort_keys=True, separators=(",", ":")),
    )


def get_bot_owned_btc_quantity_details() -> BotOwnedQuantityDetails:
    """Use active position provenance first; never infer net BTC from gross fills."""
    try:
        active = get_active_bot_position()
    except Exception:
        return BotOwnedQuantityDetails(0.0, False, "active_position_record_unavailable")
    if active is not None:
        try:
            active_quantity = float(active["credited_quantity"])
        except (KeyError, TypeError, ValueError):
            return BotOwnedQuantityDetails(0.0, False, "active_position_record_invalid")
        if not isfinite(active_quantity) or active_quantity <= 0:
            return BotOwnedQuantityDetails(0.0, False, "active_position_record_invalid")
        return BotOwnedQuantityDetails(active_quantity, True, "active_position_provenance")

    records = get_order_records()
    relevant = [row for row in records if row["symbol"] in {"BTC/USD", "BTCUSD"}]
    if not relevant:
        return BotOwnedQuantityDetails(0.0, False, "no_bot_order_evidence")
    executed = [
        row for row in relevant
        if row.get("quantity") is not None and abs(float(row.get("quantity") or 0)) > 0
    ]
    if not executed:
        return BotOwnedQuantityDetails(0.0, False, "no_filled_asset_delta_evidence")
    if any(row.get("asset_quantity_delta") is None for row in executed):
        return BotOwnedQuantityDetails(0.0, False, "legacy_or_incomplete_asset_delta_ledger")
    quantity = sum(float(row["asset_quantity_delta"]) for row in executed)
    if not isfinite(quantity):
        return BotOwnedQuantityDetails(0.0, False, "invalid_asset_delta_ledger")
    return BotOwnedQuantityDetails(quantity, True, "complete_asset_delta_ledger")


def get_bot_owned_btc_quantity() -> tuple[float, bool]:
    """Backward-compatible pair; quantity uses net broker-confirmed deltas only."""
    details = get_bot_owned_btc_quantity_details()
    return details.quantity, details.reliable


def note_synthetic_order_reconciliation(client_order_id: str) -> dict | None:
    """Record one confirmed not-found lookup for an ambiguous synthetic order."""
    now = datetime.now(timezone.utc).isoformat()
    connection = sqlite3.connect(config.DATABASE_PATH, timeout=5)
    try:
        with connection:
            _create_tables(connection)
            connection.execute(
                "UPDATE orders SET reconcile_attempts=COALESCE(reconcile_attempts, 0)+1, "
                "last_reconcile_at=? WHERE client_order_id=? "
                "AND submission_kind='synthetic_ambiguous' "
                "AND order_status='submit_unknown'",
                (now, client_order_id),
            )
            row = connection.execute(
                "SELECT created_at, reconcile_attempts, last_reconcile_at "
                "FROM orders WHERE client_order_id=? "
                "AND submission_kind='synthetic_ambiguous' "
                "AND order_status='submit_unknown'",
                (client_order_id,),
            ).fetchone()
        return (
            {
                "created_at": row[0],
                "reconcile_attempts": row[1] or 0,
                "last_reconcile_at": row[2],
            }
            if row
            else None
        )
    finally:
        connection.close()


def mark_synthetic_order_not_created(client_order_id: str) -> bool:
    connection = sqlite3.connect(config.DATABASE_PATH, timeout=5)
    try:
        with connection:
            _create_tables(connection)
            connection.execute(
                "UPDATE orders SET order_status='terminal_not_created' "
                "WHERE client_order_id=? AND submission_kind='synthetic_ambiguous' "
                "AND order_status='submit_unknown'",
                (client_order_id,),
            )
            return connection.execute("SELECT changes()").fetchone()[0] > 0
    finally:
        connection.close()


def get_state(key: str) -> str | None:
    connection = sqlite3.connect(config.DATABASE_PATH, timeout=5)
    try:
        _create_tables(connection)
        row = connection.execute(
            "SELECT value FROM bot_state WHERE key=?", (key,)
        ).fetchone()
        return row[0] if row else None
    finally:
        connection.close()


def set_state(key: str, value: str | None) -> None:
    connection = sqlite3.connect(config.DATABASE_PATH, timeout=5)
    try:
        with connection:
            _create_tables(connection)
            if value is None:
                connection.execute("DELETE FROM bot_state WHERE key=?", (key,))
            else:
                connection.execute(
                    "INSERT INTO bot_state(key, value) VALUES (?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (key, value),
                )
    finally:
        connection.close()
