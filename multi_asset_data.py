"""Read-only daily research ingestion. No broker or trading client is imported."""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from dotenv import load_dotenv

import binance_data

ETF_SYMBOLS = ("SPY", "TLT", "GLD", "UUP")
START = pd.Timestamp("2016-01-01")
CACHE = Path("data_cache/multi_asset")
COLUMNS = ["open", "high", "low", "close", "volume"]


def session_time(dates, hour=16):
    """US session date -> DST-aware scheduled close (fixed 16:00 by registration)."""
    days = pd.DatetimeIndex(dates).tz_localize(None).normalize()
    return (days + pd.Timedelta(hours=hour)).tz_localize("America/New_York").tz_convert("UTC")


def completed_end(end_date=None, now=None):
    now = binance_data.utc(now if now is not None else pd.Timestamp.now(tz="UTC"))
    local = now.tz_convert("America/New_York")
    latest = local.tz_localize(None).normalize()
    if local.hour < 16:
        latest -= pd.Timedelta(days=1)
    requested = pd.Timestamp(end_date).normalize() if end_date else latest
    if requested.tzinfo is not None:
        raise ValueError("--end-date must be a calendar date")
    return min(requested, latest)


def validate_bars(frame):
    frame = frame[COLUMNS].copy().sort_index()
    if frame.index.has_duplicates:
        raise ValueError("Duplicate daily dates")
    values = frame.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values[:, :4] <= 0).any() or (values[:, 4] < 0).any():
        raise ValueError("Daily OHLCV must be finite, with positive prices and nonnegative volume")
    if (frame.high < frame[["open", "close", "low"]].max(axis=1)).any() or (frame.low > frame[["open", "close", "high"]].min(axis=1)).any():
        raise ValueError("Invalid OHLC ordering")
    return frame


def normalize_etf(raw, symbol, end):
    if isinstance(raw.index, pd.MultiIndex):
        raw = raw.xs(symbol, level=0)
    frame = raw.copy()
    stamps = pd.DatetimeIndex(frame.index)
    if stamps.tz is not None:
        stamps = stamps.tz_convert("America/New_York").tz_localize(None)
    frame.index = stamps.normalize()
    # Alpaca adjustment=all already adjusts ALL OHLC. Do not adjust twice.
    return validate_bars(frame.loc[(frame.index >= START) & (frame.index <= end)])


def sip_denied(error):
    message = str(error).lower()
    status = getattr(error, "status_code", None)
    return (status in (403, 422) or "subscription" in message or "permission" in message) and any(
        token in message for token in ("sip", "subscription", "entitlement", "feed"))


def fetch_etf(client, symbol, end, progress=print):
    params = dict(symbol_or_symbols=[symbol], timeframe=TimeFrame.Day,
                  start=START.tz_localize("America/New_York").to_pydatetime(),
                  end=(end + pd.Timedelta(days=1)).tz_localize("America/New_York").to_pydatetime(),
                  adjustment=Adjustment.ALL)
    feed = "sip"
    try:
        raw = client.get_stock_bars(StockBarsRequest(**params, feed=DataFeed.SIP)).df
    except Exception as error:
        if not sip_denied(error):
            # Do not echo network exceptions, which can contain authenticated URLs.
            raise RuntimeError(f"{symbol}: historical SIP request failed; no feed substitution") from None
        feed = "iex"
        progress(f"WARNING: {symbol}: SIP entitlement refused; FALLING BACK TO IEX (single exchange).")
        try:
            raw = client.get_stock_bars(StockBarsRequest(**params, feed=DataFeed.IEX)).df
        except Exception:
            raise RuntimeError(f"{symbol}: historical IEX request failed") from None
    frame = normalize_etf(raw, symbol, end)
    if frame.empty:
        raise ValueError(f"{symbol}: no completed daily bars")
    frame.attrs.update(symbol=symbol, feed=feed, adjustment="all", downloaded_end=str(end.date()))
    progress(f"{symbol}: feed={feed}, adjustment=all")
    return frame


def load_etfs(end, *, download=False, cache_dir=CACHE, client=None, progress=print):
    path = Path(cache_dir)
    if download:
        path.mkdir(parents=True, exist_ok=True)
        if client is None:
            load_dotenv()
            key, secret = os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY")
            if not key or not secret:
                raise RuntimeError("Paper data keys missing from .env")
            client = StockHistoricalDataClient(key, secret)
    result = {}
    for symbol in ETF_SYMBOLS:
        csv, meta = path / f"{symbol}_1day_all.csv", path / f"{symbol}_1day_all.json"
        if download:
            frame = fetch_etf(client, symbol, end, progress)
            frame.to_csv(csv, index_label="date")
            meta.write_text(json.dumps(frame.attrs), encoding="utf-8")
        else:
            if not csv.exists() or not meta.exists():
                raise RuntimeError(f"{symbol}: adjusted daily cache missing; run --download")
            metadata = json.loads(meta.read_text(encoding="utf-8"))
            if metadata.get("adjustment") != "all" or metadata.get("feed") not in ("sip", "iex") or metadata.get("symbol") != symbol:
                raise ValueError(f"{symbol}: invalid adjusted-price cache metadata; run --download")
            frame = pd.read_csv(csv, index_col="date", parse_dates=True)
            frame = validate_bars(frame.loc[(frame.index >= START) & (frame.index <= end)])
            frame.attrs.update(metadata)
            progress(f"{symbol}: feed={metadata['feed']} (cache), adjustment=all")
            if metadata["feed"] == "iex":
                progress(f"WARNING: {symbol}: cached IEX prices represent a single exchange.")
            if pd.Timestamp(metadata["downloaded_end"]) < end:
                progress(f"WARNING: {symbol}: cache requested only through {metadata['downloaded_end']}; run --download to refresh.")
        result[symbol] = frame
    return result


def btc_daily(hourly):
    """Derive UTC 1Day candles from the existing Binance cache; require all 24 hours."""
    hourly = hourly.sort_index()
    if hourly.index.has_duplicates or (hourly.index != hourly.index.floor("h")).any():
        raise ValueError("BTC hourly source must be unique and on the UTC hourly grid")
    grouped = hourly[COLUMNS].resample("1D")
    frame = grouped.agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    frame = frame.loc[grouped.close.count() == 24].dropna()
    frame = validate_bars(frame)
    frame["close_time"] = frame.index + pd.Timedelta(days=1)
    return frame


def align_btc(daily, us_dates, execution_opens=None):
    """As-of CLOSE alignment, never a later daily candle, with causal execution opens.

    At US date D, close marks the latest UTC daily close <= D 16:00 ET.
    At the next US date, open executes at the first UTC open AFTER D 16:00 ET.
    Mondays therefore capture the weekend after a Saturday UTC opening fill.
    A missing UTC close leaves the latest older completed bar eligible, exactly
    as registered. Actual midnight opens can come from the hourly cache even if
    another hour of that UTC day is missing. An absent execution open stays missing.
    """
    dates = pd.DatetimeIndex(us_dates)
    cutoffs = session_time(dates)
    source = daily.sort_index()
    openings = source.open if execution_opens is None else execution_opens.sort_index()
    if openings.index.has_duplicates:
        raise ValueError("BTC execution opens must be unique")
    close_times = pd.DatetimeIndex(source["close_time"])
    if close_times.has_duplicates or not close_times.is_monotonic_increasing:
        raise ValueError("BTC close times must be unique and increasing")
    positions = close_times.searchsorted(cutoffs, side="right") - 1
    rows = []
    for i, (day, cutoff, pos) in enumerate(zip(dates, cutoffs, positions)):
        if pos < 0:
            continue
        execution = (cutoffs[i - 1].floor("D") + pd.Timedelta(days=1)) if i else cutoff.floor("D")
        if execution not in openings.index:
            continue
        row = source.iloc[pos]
        opening = float(openings.loc[execution])
        rows.append(dict(date=day, open=opening, high=max(opening, row.high),
                         low=min(opening, row.low), close=row.close, volume=row.volume,
                         source_close_time=close_times[pos], execution_time=execution))
    frame = pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame(columns=COLUMNS + ["source_close_time", "execution_time"], index=pd.DatetimeIndex([]))
    frame.attrs.update(feed="binance_cache_1h_to_1day", adjustment="none", symbol="BTC")
    return frame


def load_btc(us_dates, end, *, download=False):
    cutoff = session_time([end])[0]
    if download:
        binance_data.download_history(end_time=cutoff)
    if not binance_data.CACHE_PATH.exists():
        raise RuntimeError("BTC cache missing; run --download or --no-btc")
    hourly = binance_data.load_history(start=binance_data.HISTORY_START, end=cutoff)
    return align_btc(btc_daily(hourly), us_dates, execution_opens=hourly.open)


def quality_report(symbol, bars, reference):
    reference = pd.DatetimeIndex(reference)
    missing = reference.difference(bars.index)
    in_span = missing[(missing >= bars.index.min()) & (missing <= bars.index.max())] if len(bars) else missing
    return dict(symbol=symbol, feed=bars.attrs.get("feed", "unknown"),
                first_date=str(bars.index.min().date()) if len(bars) else "",
                last_date=str(bars.index.max().date()) if len(bars) else "",
                rows=len(bars), missing_us_days=len(missing), missing_within_span=len(in_span),
                missing_dates=",".join(str(day.date()) for day in missing))
