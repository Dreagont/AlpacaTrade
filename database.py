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
            unrealized_pnl REAL
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
            order_status TEXT
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
    indexes = {
        row[1] for row in connection.execute("PRAGMA index_list(orders)").fetchall()
    }
    if "idx_orders_order_id" not in indexes:
        connection.execute(
            """
            DELETE FROM orders
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


def _report_failure(error: Exception) -> None:
    print(f"DATABASE LOG ERROR: {error}", file=sys.stderr)


def _close_safely(connection: Optional[sqlite3.Connection]) -> None:
    if connection is not None:
        try:
            connection.close()
        except Exception as error:
            _report_failure(error)


def init_db() -> None:
    connection = None
    try:
        connection = sqlite3.connect(config.DATABASE_PATH, timeout=5)
        with connection:
            _create_tables(connection)
    except Exception as error:
        _report_failure(error)
    finally:
        _close_safely(connection)


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
                    unrealized_pnl
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
) -> None:
    timestamp = timestamp or datetime.now(timezone.utc).isoformat()
    connection = None
    try:
        connection = sqlite3.connect(config.DATABASE_PATH, timeout=5)
        with connection:
            _create_tables(connection)
            connection.execute(
                """
                INSERT INTO orders (
                    timestamp, order_id, symbol, side, requested_notional,
                    quantity, fill_price, reason, realized_gross_pnl, order_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(order_id) WHERE order_id IS NOT NULL DO UPDATE SET
                    timestamp=excluded.timestamp,
                    symbol=excluded.symbol,
                    side=excluded.side,
                    requested_notional=excluded.requested_notional,
                    quantity=excluded.quantity,
                    fill_price=excluded.fill_price,
                    reason=excluded.reason,
                    realized_gross_pnl=excluded.realized_gross_pnl,
                    order_status=excluded.order_status
                """,
                (
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
                ),
            )
    except Exception as error:
        _report_failure(error)
    finally:
        _close_safely(connection)


def log_order(**kwargs) -> None:
    """Backward-compatible name for the order upsert operation."""
    upsert_order(**kwargs)
