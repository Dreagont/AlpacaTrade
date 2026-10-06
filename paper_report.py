"""Read-only BTC paper ledger report and hypothetical fee re-pricing.

No broker, database helpers, or live runner imports. Recorded BTC deltas determine
ownership; missing or inconsistent deltas are reported rather than reconstructed.
"""

import argparse
import csv
from collections import Counter
import math
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

import config
import trade_config
from report_output import csv_output_path

BTC_SYMBOLS = {"BTC/USD", "BTCUSD"}
EXIT_ROLES = {"strategy_exit", "risk_exit_stop_loss", "risk_exit_take_profit", "protective_stop"}
REQUIRED_COLUMNS = {
    "id", "symbol", "side", "order_status", "timestamp", "fill_price", "quantity",
    "requested_notional", "asset_quantity_delta", "strategy_name", "order_role",
}


def read_orders(db_path):
    """Use SQLite's read-only URI; never initialize or migrate a trading database."""
    uri = Path(db_path).resolve().as_uri() + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        connection.execute("PRAGMA query_only=ON")
        connection.row_factory = sqlite3.Row
        columns = {row[1] for row in connection.execute("PRAGMA table_info(orders)")}
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(f"Orders schema missing fields: {', '.join(sorted(missing))}; no migration performed")
        return [dict(row) for row in connection.execute(
            "SELECT * FROM orders WHERE symbol IN ('BTC/USD', 'BTCUSD') "
            "AND UPPER(side) IN ('BUY', 'SELL') ORDER BY id"
        )]
    finally:
        connection.close()


def _positive(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError("expected a finite positive number")
    return number


def _timestamp(value):
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError("missing/invalid timestamp or UTC offset")
    return stamp.tz_convert("UTC")


def pair_orders(rows):
    """Chronological FIFO, one BUY position with zero or more partial SELL fills.

    An invalid execution or overlapping BUY taints the remaining ledger: no later
    ownership inference is made across that gap. A leading SELL is unpaired.
    """
    issues, prepared, pairs = [], [], []
    for row in rows:
        row = dict(row)
        status = str(row.get("order_status") or "").lower()
        if status != "filled":
            if row.get("quantity") or row.get("asset_quantity_delta"):
                issues.append({"order_id": row["id"], "issue": "ambiguous non-filled execution; pairing stopped"})
                row["_invalid"] = "non-filled execution"
            else:
                continue
        try:
            row["_time"] = _timestamp(row["timestamp"])
        except (ValueError, TypeError, OverflowError):
            issues.append({"order_id": row["id"], "issue": "ambiguous timestamp; cannot establish chronology"})
            # Unknown chronological placement makes the whole ledger unsafe to pair.
            return [], None, issues + [
                {"order_id": other["id"], "issue": "ambiguous ledger chronology"}
                for other in rows if other["id"] != row["id"]
            ]
        prepared.append(row)
    prepared.sort(key=lambda row: (row["_time"], row["id"]))
    duplicate_ids = set()
    seen = {}
    for row in prepared:
        for key in ("order_id", "client_order_id"):
            value = row.get(key)
            if value:
                identity = (key, value)
                if identity in seen:
                    duplicate_ids.update((row["id"], seen[identity]))
                seen[identity] = row["id"]
    time_counts = Counter(row["_time"] for row in prepared)
    same_times = {stamp for stamp, count in time_counts.items() if count > 1}
    active, tainted = None, False
    for row in prepared:
        if tainted:
            issues.append({"order_id": row["id"], "issue": "ambiguous ownership after earlier ledger gap"})
            continue
        try:
            if row.get("_invalid"):
                raise ValueError(row["_invalid"])
            if row["id"] in duplicate_ids or row["_time"] in same_times:
                raise ValueError("duplicate order identity or tied execution timestamps")
            row["quantity"] = _positive(row["quantity"])
            row["fill_price"] = _positive(row["fill_price"])
            delta = float(row["asset_quantity_delta"])
            if not math.isfinite(delta):
                raise ValueError("invalid asset_quantity_delta")
            side = str(row["side"]).upper()
            row["side"] = side
            if side == "BUY":
                if active:
                    raise ValueError("overlapping BUY while a position is open")
                if delta <= 0 or delta > row["quantity"] + 1e-10:
                    raise ValueError("missing/inconsistent credited BUY quantity")
                if row["order_role"] != "strategy_entry" or not row["strategy_name"]:
                    raise ValueError("BUY has no strategy-entry provenance")
                active = {"buy": row, "sells": [], "remaining_quantity": delta,
                          "credited_quantity": delta, "entry_debit": row["quantity"] * row["fill_price"]}
            else:
                if delta >= 0 or not math.isclose(-delta, row["quantity"], rel_tol=1e-5, abs_tol=1e-10):
                    raise ValueError("missing/inconsistent SELL asset delta")
                if row["order_role"] not in EXIT_ROLES:
                    raise ValueError("SELL has no exit provenance")
                if active is None:
                    issues.append({"order_id": row["id"], "issue": "unpaired SELL without preceding BUY"})
                    continue
                if (row["strategy_name"] != active["buy"]["strategy_name"]
                        or not row["strategy_name"]):
                    raise ValueError("missing/mismatched strategy provenance")
                if -delta > active["remaining_quantity"] + max(1e-10, active["credited_quantity"] * 1e-5):
                    raise ValueError("SELL exceeds known position")
                active["sells"].append(row)
                active["remaining_quantity"] = max(0.0, active["remaining_quantity"] + delta)
                if active["remaining_quantity"] <= max(1e-10, active["credited_quantity"] * 1e-5):
                    pairs.append(active)
                    active = None
        except (ValueError, TypeError, OverflowError) as error:
            issues.append({"order_id": row["id"], "issue": f"ambiguous: {error}; pairing stopped"})
            if active:
                for affected in [active["buy"], *active["sells"]]:
                    issues.append({"order_id": affected["id"], "issue": "unpaired position affected by ambiguous execution"})
            active, tainted = None, True
    return pairs, active, issues


def fetch_latest_price():
    """Public market-data read; no account or trading client."""
    from alpaca.data.historical import CryptoHistoricalDataClient
    from alpaca.data.requests import CryptoLatestTradeRequest
    latest = CryptoHistoricalDataClient().get_crypto_latest_trade(
        CryptoLatestTradeRequest(symbol_or_symbols=[config.SYMBOL])
    )
    return _positive(latest[config.SYMBOL].price)


def _position_rows(position, number, mark_price=None):
    buy, sells = position["buy"], position["sells"]
    credited, debit = position["credited_quantity"], position["entry_debit"]
    actual_proceeds = sum(sell["quantity"] * sell["fill_price"] for sell in sells)
    remaining = position["remaining_quantity"]
    open_position = remaining > max(1e-10, credited * 1e-5)
    mark_value = remaining * mark_price if open_position and mark_price is not None else 0.0
    actual = actual_proceeds + mark_value - debit if not open_position or mark_price is not None else None
    # Only the closed portion is realized when marking is skipped.
    realized_basis = debit * (credited - remaining) / credited
    realized_recorded = actual_proceeds - realized_basis
    weighted_exit = sum(sell["quantity"] * sell["fill_price"] / credited for sell in sells)
    fraction_open = remaining / credited
    rows = []
    for name in config.FEE_PROFILES:
        rates = config.get_fee_profile(name)
        fee = rates["fee_rate"]
        hypothetical_quantity = debit / buy["fill_price"] * (1 - fee)
        repriced_realized = hypothetical_quantity * weighted_exit * (1 - fee) - realized_basis
        repriced_marked = (
            hypothetical_quantity * fraction_open * mark_price * (1 - fee) - debit * fraction_open
            if open_position and mark_price is not None else None
        )
        # SELL quote fees are absent from this DB, so actual net is unavailable.
        alpaca_fee = config.get_fee_profile("alpaca")["fee_rate"]
        alpaca_estimate = actual - (actual_proceeds + mark_value) * alpaca_fee if actual is not None else None
        rows.append({
            "row_type": "OPEN_MARKED" if open_position and mark_price is not None else "OPEN_UNMARKED" if open_position else "ROUND_TRIP",
            "position_number": number, "buy_order_id": buy["id"],
            "sell_order_ids": ",".join(str(sell["id"]) for sell in sells),
            "entry_time": buy["timestamp"], "exit_time": sells[-1]["timestamp"] if sells else "",
            "strategy_name": buy["strategy_name"], "entry_role": buy["order_role"],
            "exit_roles": ",".join(str(sell["order_role"]) for sell in sells),
            "entry_fill_price": buy["fill_price"], "requested_notional": buy["requested_notional"],
            "actual_entry_debit_usd": debit, "actual_credited_quantity": credited,
            "remaining_quantity": remaining, "mark_price": mark_price if open_position else None,
            "fee_profile": name, **rates, "applied_slippage_rate": 0.0,
            "actual_recorded_pnl_usd": actual, "actual_net_pnl_usd": None,
            "actual_recorded_realized_pnl_usd": realized_recorded,
            "alpaca_net_estimate_usd": alpaca_estimate,
            "repriced_realized_pnl_usd": repriced_realized,
            "repriced_marked_pnl_usd": repriced_marked,
            "repriced_pnl_usd": (repriced_realized + (repriced_marked or 0.0)
                                 if not open_position or mark_price is not None else None),
            "issue": "SELL quote fee not recorded; actual net PnL unavailable",
        })
    return rows


def compare_backtest(orders, *, fee_profile, end_time=None):
    """Compare individual execution events; unique same-side matches within 4H."""
    import backtest
    from strategy import get_strategy
    executed = [row for row in orders if str(row.get("order_status") or "").lower() == "filled"]
    if not executed:
        return []
    events, issues = [], []
    for order in executed:
        try:
            events.append((str(order["side"]).upper(), _timestamp(order["timestamp"]), order["id"]))
        except (ValueError, TypeError) as error:
            issues.append({"row_type": "COMPARISON", "status": "INVALID_PAPER_TIMESTAMP",
                           "paper_order_id": order["id"], "issue": str(error)})
    if issues:
        for row in issues:
            row.update(fee_profile=fee_profile, **config.get_fee_profile(fee_profile))
        return issues
    start = min(event[1] for event in events)
    end = pd.Timestamp(end_time or datetime.now(timezone.utc))
    end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
    if end <= max(event[1] for event in events):
        raise ValueError("Comparison end must be after the last paper fill")
    if trade_config.LIVE_TIMEFRAME != "4Hour":
        raise ValueError("Paper comparison requires the configured live timeframe to be 4Hour")
    strategy = get_strategy(trade_config.LIVE_STRATEGY)
    days = max(1, math.ceil((end - start).total_seconds() / 86400))
    bars = backtest.fetch_history(days, "4Hour", warmup_bars=backtest.required_warmup_bars(strategy=strategy) + 2,
                                 end_time=end.to_pydatetime())
    if bars.empty or int((bars.index < start).sum()) < backtest.required_warmup_bars(strategy=strategy):
        raise ValueError("Insufficient historical warm-up for paper comparison")
    # Start flat at the first fill's 4H candle. This makes an entry inside that
    # candle eligible; a preceding flat candle supplies the engine's initial bar.
    first_candle = start.floor("4h")
    test_start = first_candle - pd.Timedelta(hours=4)
    if bars.index.min() > test_start or bars.index.max() + pd.Timedelta(hours=4) < end.floor("4h"):
        raise ValueError("History does not cover paper comparison boundaries")
    if int((bars.index < test_start).sum()) < backtest.required_warmup_bars(strategy=strategy):
        raise ValueError("Insufficient warm-up before comparison seed candle")
    rates = config.get_fee_profile(fee_profile)
    result = backtest.run_backtest(bars, timeframe="4Hour", test_start=test_start,
                                  strategy=strategy, liquidate_at_end=False,
                                  fee_profile=fee_profile, fee_rate=rates["fee_rate"],
                                  slippage=rates["slippage_rate"])
    simulated = []
    simulated_prices = {}
    for _, entry in result["entry_events"].iterrows():
        event = ("BUY", _timestamp(entry["entry_time"]))
        simulated.append(event)
        simulated_prices[event] = entry.get("entry_fill_price")
    for _, trade in result["trades"].iterrows():
        event = ("SELL", _timestamp(trade["exit_time"]))
        simulated.append(event)
        simulated_prices[event] = trade.get("exit_fill_price")
    # Exclude the seed candle; compare precisely the execution window onward.
    simulated = [(side, stamp) for side, stamp in simulated if stamp >= first_candle]
    rows, used = [], set()
    orders_by_id = {order["id"]: order for order in executed}
    for side, stamp, identity in events:
        candidates = [i for i, (candidate_side, candidate_time) in enumerate(simulated)
                      if candidate_side == side and abs(candidate_time - stamp) <= pd.Timedelta(hours=4)]
        # Require a unique match in both directions to avoid arbitrary assignments.
        unique = len(candidates) == 1 and candidates[0] not in used
        if unique:
            candidate_time = simulated[candidates[0]][1]
            competitors = [event for event in events
                           if event[0] == side and abs(event[1] - candidate_time) <= pd.Timedelta(hours=4)]
            unique = len(competitors) == 1
        if unique:
            used.add(candidates[0])
        row = {"row_type": "COMPARISON", "side": side, "paper_order_id": identity,
               "paper_time": stamp.isoformat(),
               "backtest_time": simulated[candidates[0]][1].isoformat() if unique else "",
               "status": "MATCH" if unique else "AMBIGUOUS_MATCH" if candidates else "PAPER_ONLY",
               "strategy_name": strategy.name, "period_start": start.isoformat(), "period_end": end.isoformat(),
               "issue": "" if unique else "entry/exit mismatch"}
        paper_order = orders_by_id[identity]
        row.update(paper_strategy_name=paper_order.get("strategy_name"),
                   paper_order_role=paper_order.get("order_role"),
                   paper_fill_price=paper_order.get("fill_price"),
                   backtest_fill_price=simulated_prices.get(simulated[candidates[0]]) if unique else None,
                   time_difference_minutes=(stamp - simulated[candidates[0]][1]).total_seconds() / 60 if unique else None)
        if unique and paper_order.get("strategy_name") != strategy.name:
            row.update(status="STRATEGY_MISMATCH", issue="paper strategy differs from configured live strategy")
        rows.append(row)
    for i, (side, stamp) in enumerate(simulated):
        if i not in used:
            rows.append({"row_type": "COMPARISON", "side": side, "paper_order_id": "",
                         "paper_time": "", "backtest_time": stamp.isoformat(), "status": "BACKTEST_ONLY",
                         "strategy_name": strategy.name, "issue": "entry/exit mismatch"})
    for row in rows:
        row.update(fee_profile=fee_profile, **rates)
    return rows


def run_paper_report(*, db_path=config.DATABASE_PATH, mark_price=None, no_mark=False,
                     fee_profile=config.BACKTEST_FEE_PROFILE, compare=False, output=None,
                     end_time=None):
    rates = config.get_fee_profile(fee_profile)
    if output is not None:
        csv_output_path(output, db_path=db_path)
    if mark_price is not None:
        mark_price = _positive(mark_price)
    orders = read_orders(db_path)
    pairs, open_position, issues = pair_orders(orders)
    print("\n=== READ-ONLY BTC PAPER REPORT ===")
    config.print_fee_profile(fee_profile)
    print("Re-pricing uses ACTUAL Alpaca execution prices and each profile's fee_rate only. "
          "Profile slippage_rate is metadata; applied slippage is zero on recorded fills.")
    print("Recorded PnL = gross SELL proceeds + marked BTC value - actual BUY fill debit. "
          "BUY debit = fill_price * gross filled quantity; credited BTC uses asset_quantity_delta. "
          "SELL quote fees are not recorded: actual net PnL is unavailable. "
          "Alpaca net estimates assume its configured taker rate on SELL proceeds.")
    print("FIFO; one BUY position at a time, partial SELLs combined. "
          "Timestamps are recorded order/reconciliation times, not guaranteed broker fill times.")
    rows = []
    for number, position in enumerate(pairs, 1):
        rows.extend(_position_rows(position, number))
    if open_position:
        if not no_mark and mark_price is None:
            mark_price = fetch_latest_price()
        rows.extend(_position_rows(open_position, len(pairs) + 1, None if no_mark else mark_price))
        print(f"Open position: {open_position['remaining_quantity']:.10g} BTC; "
              + ("mark skipped (--no-mark)" if no_mark else f"MARKED at ${mark_price:,.2f} (hypothetical exit fee for re-pricing)"))
    for name in config.FEE_PROFILES:
        selected = [row for row in rows if row["fee_profile"] == name]
        totals = {"row_type": "TOTAL_CLOSED_AND_MARKED", "fee_profile": name,
                  **config.get_fee_profile(name), "applied_slippage_rate": 0.0,
                  "actual_net_pnl_usd": None,
                  "issue": "Only resolved positions included; SELL quote fee unrecorded"}
        for field in ("actual_recorded_pnl_usd", "alpaca_net_estimate_usd", "repriced_pnl_usd"):
            values = [row[field] for row in selected if row[field] is not None]
            totals[field] = sum(values) if values or not selected else None
        for field in ("actual_recorded_realized_pnl_usd", "repriced_realized_pnl_usd"):
            totals[field] = sum(row[field] for row in selected)
        rows.append(totals)
    for issue in issues:
        rows.append({"row_type": "ISSUE", **issue, "fee_profile": fee_profile, **rates})
    if compare:
        print("Comparison: configured live strategy on 4H bars; unique same-side events within "
              "one 4H bar. Flat seed one candle before first fill; end is the last fill candle close for closed ledgers, report time for open positions. "
              "Reconciliation timestamps and pre-existing holdings can cause mismatches.")
        if end_time is None and not open_position:
            fill_times = [_timestamp(order["timestamp"]) for order in orders
                          if str(order.get("order_status") or "").lower() == "filled"]
            if fill_times:
                end_time = max(fill_times).ceil("4h")
                if end_time == max(fill_times):
                    end_time += pd.Timedelta(hours=4)
        rows.extend(compare_backtest(orders, fee_profile=fee_profile, end_time=end_time))
    print("Type / position       Profile             Recorded PnL*  Alpaca est.  Re-priced PnL  Recorded realized*  Re-priced realized")
    fmt = lambda value: "N/A" if value is None else f"{value:.4f}"
    for row in rows:
        if row["row_type"] == "ISSUE":
            print(f"ISSUE order {row['order_id']}: {row['issue']}")
        elif row["row_type"] == "COMPARISON":
            print(f"{row['status']}: {row.get('side', '')} | paper {row.get('paper_time', '')} "
                  f"| backtest {row.get('backtest_time', '')} | order {row.get('paper_order_id', '')}")
        else:
            if row.get("position_number") and row["fee_profile"] == "alpaca":
                print(f"Position {row['position_number']}: BUY order {row['buy_order_id']} "
                      f"at {row['entry_time']} (${row['entry_fill_price']:.4f}); "
                      f"SELL orders {row['sell_order_ids'] or 'none'}; last exit {row['exit_time'] or 'open'}; "
                      f"strategy={row['strategy_name']}")
            print(f"{row['row_type']}/{row.get('position_number', '')} | {row['fee_profile']} | "
                  f"{fmt(row.get('actual_recorded_pnl_usd'))} | {fmt(row.get('alpaca_net_estimate_usd'))} | "
                  f"{fmt(row.get('repriced_pnl_usd'))} | {fmt(row.get('actual_recorded_realized_pnl_usd'))} | "
                  f"{fmt(row.get('repriced_realized_pnl_usd'))}")
    print("* Recorded PnL includes recorded BUY base fee, excludes unrecorded SELL quote fee. "
          "OPEN_MARKED and TOTAL_CLOSED_AND_MARKED can include unrealized values.")
    if output is not None:
        target = csv_output_path(output, db_path=db_path)
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with target.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"CSV saved to {target}")
    return rows


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=config.DATABASE_PATH)
    marking = parser.add_mutually_exclusive_group()
    marking.add_argument("--mark-price", type=float)
    marking.add_argument("--no-mark", action="store_true")
    parser.add_argument("--compare-backtest", action="store_true")
    config.add_fee_profile_argument(parser)
    parser.add_argument("--csv", nargs="?", const="paper_report.csv", help="Optional CSV path")
    return parser


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        run_paper_report(db_path=args.db, mark_price=args.mark_price, no_mark=args.no_mark,
                         fee_profile=args.fee_profile, compare=args.compare_backtest, output=args.csv)
    except (ValueError, sqlite3.Error, OSError, RuntimeError) as error:
        parser.exit(1, f"PAPER REPORT ERROR: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
