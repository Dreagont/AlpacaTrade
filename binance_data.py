"""Public BTCUSDT hourly klines and a bounded, read-only research cache loader."""

import json
import math
import sqlite3
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd

from report_output import csv_output_path

ENDPOINT = "https://api.binance.com/api/v3/klines"
CACHE_PATH = Path("data_cache/binance_btcusdt_1h.sqlite")
HISTORY_START = pd.Timestamp("2018-01-01", tz="UTC")
HOUR_MS = 3_600_000
COLUMNS = ["open", "high", "low", "close", "volume"]


def utc(value):
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def milliseconds(value):
    return utc(value).value // 1_000_000


def parse_klines(payload, *, end_time):
    """Drop uncompleted/off-grid bars; preserve gaps instead of filling prices."""
    rows = []
    cutoff = milliseconds(end_time)
    for item in payload:
        if len(item) < 7:
            raise ValueError("Malformed Binance kline: at least seven fields required")
        start, close_time = int(item[0]), int(item[6])
        values = [float(value) for value in item[1:6]]
        if start % HOUR_MS or close_time > start + HOUR_MS - 1:
            raise ValueError("Binance kline is not a UTC hourly candle")
        if close_time < start + HOUR_MS - 1:
            # Historical maintenance can leave a truncated candle. Keep it missing.
            continue
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Non-finite Binance OHLCV")
        o, h, lo, c, volume = values
        if min(o, h, lo, c) <= 0 or volume < 0 or h < max(o, c, lo) or lo > min(o, c, h):
            raise ValueError("Invalid Binance OHLCV")
        if start + HOUR_MS <= cutoff and close_time < cutoff:
            rows.append((start, *values))
    frame = pd.DataFrame(rows, columns=["timestamp_ms", *COLUMNS])
    frame = frame.drop_duplicates("timestamp_ms", keep="last").sort_values("timestamp_ms")
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.pop("timestamp_ms"), unit="ms", utc=True))
    frame.index.name = "timestamp"
    return frame


def request_page(start_ms, end_ms, *, retries=5, opener=None, sleep=None):
    opener, sleep = opener or urlopen, sleep or time.sleep
    query = urlencode({"symbol": "BTCUSDT", "interval": "1h", "limit": 1000,
                       "startTime": int(start_ms), "endTime": int(end_ms) - 1})
    for attempt in range(retries + 1):
        try:
            with opener(f"{ENDPOINT}?{query}", timeout=20) as response:
                payload = json.load(response)
            if not isinstance(payload, list):
                raise ValueError(f"Binance response is not a kline list: {payload}")
            return payload
        except (URLError, TimeoutError, OSError) as error:
            if isinstance(error, HTTPError):
                error.close()
            if isinstance(error, HTTPError) and error.code not in {418, 429, 500, 502, 503, 504}:
                raise RuntimeError(f"Binance public data unavailable (HTTP {error.code}); no endpoint substitution") from error
            if attempt == retries:
                raise RuntimeError(f"Binance download failed after {retries + 1} attempts: {error}") from error
            delay = min(2 ** attempt, 10)
            if isinstance(error, HTTPError) and error.headers:
                try:
                    delay = min(max(delay, float(error.headers.get("Retry-After", delay))), 30)
                except ValueError:
                    pass
                error.close()
            print(f"Download retry {attempt + 1}/{retries} in {delay}s: {error}", flush=True)
            sleep(delay)


def download_history(*, cache_path=CACHE_PATH, end_time=None, fetch_page=None):
    """Explicit ingestion command may download holdout; search reads are bounded."""
    path = csv_output_path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    end = utc(end_time or pd.Timestamp.now(tz="UTC")).floor("h")
    connection = sqlite3.connect(path)
    try:
        connection.execute("CREATE TABLE IF NOT EXISTS klines (timestamp_ms INTEGER PRIMARY KEY, "
                           "open REAL NOT NULL, high REAL NOT NULL, low REAL NOT NULL, "
                           "close REAL NOT NULL, volume REAL NOT NULL)")
        existing = connection.execute("SELECT MAX(timestamp_ms) FROM klines").fetchone()[0]
        cursor = max(milliseconds(HISTORY_START), existing + HOUR_MS if existing is not None else 0)
        print(f"Public Binance BTCUSDT 1h download: {utc(pd.to_datetime(cursor, unit='ms'))} -> {end}", flush=True)
        pages, count = 0, 0
        while cursor < milliseconds(end):
            payload = (fetch_page or request_page)(cursor, milliseconds(end))
            if not payload:
                raise RuntimeError(f"Empty Binance page before requested end at {pd.to_datetime(cursor, unit='ms', utc=True)}")
            bars = parse_klines(payload, end_time=end)
            bars = bars.loc[(bars.index >= pd.to_datetime(cursor, unit="ms", utc=True)) & (bars.index < end)]
            if bars.empty:
                raise RuntimeError("Binance pagination made no progress")
            values = [(milliseconds(stamp), *row) for stamp, row in bars[COLUMNS].iterrows()]
            connection.executemany("INSERT OR REPLACE INTO klines VALUES (?, ?, ?, ?, ?, ?)", values)
            connection.commit()  # Resume safely after an interrupted download.
            cursor = milliseconds(bars.index[-1]) + HOUR_MS
            pages += 1
            count += len(bars)
            print(f"  page {pages}: {count:,} new bars, completed through {pd.to_datetime(cursor, unit='ms', utc=True)}", flush=True)
        return count
    finally:
        connection.close()


def load_history(*, start=HISTORY_START, end, cache_path=CACHE_PATH):
    """SQL predicate excludes every row whose close lies after the permitted end."""
    start, end = utc(start), utc(end)
    if start >= end:
        raise ValueError("Cache window must have start < end")
    uri = Path(cache_path).resolve().as_uri() + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        connection.execute("PRAGMA query_only=ON")
        rows = connection.execute(
            "SELECT timestamp_ms, open, high, low, close, volume FROM klines "
            "WHERE timestamp_ms >= ? AND timestamp_ms + ? <= ? ORDER BY timestamp_ms",
            (milliseconds(start), HOUR_MS, milliseconds(end)),
        ).fetchall()
    finally:
        connection.close()
    frame = pd.DataFrame(rows, columns=["timestamp_ms", *COLUMNS])
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.pop("timestamp_ms"), unit="ms", utc=True))
    frame.index.name = "timestamp"
    frame.attrs.update(requested_start_time=start, requested_end_time=end)
    return frame


def resample_bars(hourly, timeframe):
    hours = {"1Hour": 1, "2Hour": 2, "4Hour": 4}[timeframe]
    frame = hourly[COLUMNS].sort_index()
    frame = frame.loc[~frame.index.duplicated(keep="last")]
    if hours == 1:
        result = frame.copy()
    else:
        group = frame.resample(f"{hours}h", origin="epoch", label="left", closed="left")
        result = group.agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        counts = group["close"].count()
        result = result.loc[counts == hours].dropna(subset=COLUMNS)
    result.attrs.update(hourly.attrs)
    return result


def data_quality(bars, timeframe, start, end):
    hours = {"1Hour": 1, "2Hour": 2, "4Hour": 4}[timeframe]
    start, end = utc(start), utc(end)
    expected = pd.date_range(start.ceil(f"{hours}h"), end - pd.Timedelta(hours=hours), freq=f"{hours}h")
    actual = bars.loc[bars.index.isin(expected)]
    return {"expected": len(expected), "actual": len(actual),
            "missing": len(expected.difference(actual.index)), "zero_volume": int((actual["volume"] == 0).sum())}


def parity_check(binance_4h, *, end, days=30, fetch=None):
    """Informational USD/USDT venue comparison; never fetch beyond allowed cutoff."""
    import backtest
    end = utc(end)
    start = max(binance_4h.index.min(), end - pd.Timedelta(days=days))
    other = (fetch or backtest.fetch_history)(days, "4Hour", warmup_bars=0, end_time=end.to_pydatetime())
    joined = pd.concat([binance_4h["close"].rename("binance"), other["close"].rename("alpaca")], axis=1).dropna()
    joined = joined.loc[(joined.index >= start) & (joined.index + pd.Timedelta(hours=4) <= end)]
    if joined.empty:
        return {"matched_bars": 0, "median_abs_percent": None, "max_abs_percent": None}
    diff = abs(joined["binance"] / joined["alpaca"] - 1) * 100
    return {"matched_bars": len(diff), "median_abs_percent": float(diff.median()), "max_abs_percent": float(diff.max())}
