import sqlite3
import sys
from datetime import datetime, timezone
from typing import Optional

import config


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
            strategy_parameters_json TEXT
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
) -> None:
    timestamp = timestamp or datetime.now(timezone.utc).isoformat()
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
            )
            if client_order_id is not None:
                connection.execute(
                    """
                    UPDATE orders SET timestamp=?, order_id=?, symbol=?, side=?,
                        requested_notional=?, quantity=?, fill_price=?, reason=?,
                        realized_gross_pnl=?, order_status=?, strategy_name=?,
                        strategy_parameters_json=?
                    WHERE client_order_id=?
                    """,
                    (*values[:10], strategy_name, strategy_parameters_json, client_order_id),
                )
                if connection.execute("SELECT changes()").fetchone()[0]:
                    return
            connection.execute(
                """
                INSERT INTO orders (
                    timestamp, order_id, symbol, side, requested_notional,
                    quantity, fill_price, reason, realized_gross_pnl, order_status,
                    client_order_id, strategy_name, strategy_parameters_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    strategy_parameters_json=excluded.strategy_parameters_json
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


def get_bot_owned_btc_quantity() -> tuple[float, bool]:
    """Return filled BTC quantity from bot records and whether evidence exists."""
    records = get_order_records()
    relevant = [row for row in records if row["symbol"] in {"BTC/USD", "BTCUSD"}]
    known_fills = [row for row in relevant if row["quantity"] is not None]
    if not known_fills:
        return 0.0, False
    quantity = 0.0
    for row in known_fills:
        filled = abs(float(row["quantity"] or 0.0))
        quantity += filled if row["side"].upper() == "BUY" else -filled
    return quantity, True


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
