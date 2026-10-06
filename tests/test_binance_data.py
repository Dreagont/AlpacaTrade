import io
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError, URLError

import pandas as pd

import binance_data as data


def kline(stamp, *, close=102, volume=4, closing=None):
    start = data.milliseconds(stamp)
    return [start, "100", str(max(105, float(close))), "95", str(close), str(volume), closing if closing is not None else start + data.HOUR_MS - 1]


class BinanceDataTests(unittest.TestCase):
    def test_parse_completed_deduped_utc_bar_starts_and_gaps(self):
        start = data.HISTORY_START
        payload = [kline(start + pd.Timedelta(hours=2)), kline(start), kline(start, close=103),
                   kline(start + pd.Timedelta(hours=3))]
        bars = data.parse_klines(payload, end_time=start + pd.Timedelta(hours=3, minutes=30))
        self.assertEqual(list(bars.index), [start, start + pd.Timedelta(hours=2)])
        self.assertEqual(bars.iloc[0]["close"], 103)
        quality = data.data_quality(bars, "1Hour", start, start + pd.Timedelta(hours=3))
        self.assertEqual(quality, {"expected": 3, "actual": 2, "missing": 1, "zero_volume": 0})

    def test_truncated_historical_candle_stays_missing(self):
        start = data.HISTORY_START
        payload = [kline(start, closing=data.milliseconds(start) + 10000), kline(start + pd.Timedelta(hours=1))]
        bars = data.parse_klines(payload, end_time=start + pd.Timedelta(hours=2))
        self.assertEqual(list(bars.index), [start + pd.Timedelta(hours=1)])

    def test_malformed_or_invalid_prices_rejected(self):
        for row in ([0], kline(data.HISTORY_START, close=float("nan")), kline(data.HISTORY_START, close=0)):
            with self.assertRaises(ValueError):
                data.parse_klines([row], end_time="2026-01-01Z".replace("Z", "T00:00:00Z"))

    def test_resampling_utc_boundaries_and_incomplete_buckets_removed(self):
        start = data.HISTORY_START
        payload = [kline(start + pd.Timedelta(hours=i), close=100 + i, volume=i) for i in range(8)]
        bars = data.parse_klines(payload, end_time=start + pd.Timedelta(hours=8))
        four = data.resample_bars(bars, "4Hour")
        self.assertEqual(list(four.index), [start, start + pd.Timedelta(hours=4)])
        self.assertEqual(four.iloc[0].to_dict(), {"open": 100, "high": 105, "low": 95, "close": 103, "volume": 6})
        gapped = bars.drop(start + pd.Timedelta(hours=1))
        self.assertEqual(list(data.resample_bars(gapped, "2Hour").index), list(bars.index[[2, 4, 6]]))
        self.assertEqual(list(data.resample_bars(gapped, "4Hour").index), [start + pd.Timedelta(hours=4)])
        self.assertEqual(data.data_quality(data.resample_bars(gapped, "4Hour"), "4Hour", start, start + pd.Timedelta(hours=8))["missing"], 1)

    def test_zero_volume_quality_and_duplicate_source_resample(self):
        bars = data.parse_klines([kline(data.HISTORY_START, volume=0)], end_time=data.HISTORY_START + pd.Timedelta(hours=1))
        bars = pd.concat([bars, bars])
        result = data.resample_bars(bars, "1Hour")
        self.assertEqual(data.data_quality(result, "1Hour", data.HISTORY_START, data.HISTORY_START + pd.Timedelta(hours=1))["zero_volume"], 1)

    def test_retry_backoff_and_response_is_closed(self):
        response = io.StringIO(json.dumps([kline(data.HISTORY_START)]))
        calls, sleeps = [], []
        def opener(url, timeout):
            calls.append(url)
            if len(calls) < 3:
                raise URLError("temporary")
            self.assertEqual(timeout, 20)
            return response
        payload = data.request_page(0, 3600000, opener=opener, sleep=sleeps.append)
        self.assertEqual(len(payload), 1)
        self.assertEqual(sleeps, [1, 2])
        self.assertTrue(response.closed)
        self.assertIn("symbol=BTCUSDT", calls[0])
        self.assertIn("limit=1000", calls[0])

    def test_retry_limit_and_nonretryable_error(self):
        with self.assertRaises(RuntimeError):
            data.request_page(0, 1, retries=1, opener=lambda *args, **kwargs: (_ for _ in ()).throw(URLError("offline")), sleep=lambda _: None)
        error = HTTPError(data.ENDPOINT, 451, "unavailable", {}, io.BytesIO())
        with self.assertRaisesRegex(RuntimeError, "451"):
            data.request_page(0, 1, opener=lambda *args, **kwargs: (_ for _ in ()).throw(error), sleep=lambda _: None)
        self.assertTrue(error.fp.closed)

    def test_pagination_resume_and_cache_bounds_close_all_handles(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.sqlite"
            start = data.HISTORY_START
            calls = []
            def fetch(cursor, end):
                calls.append(cursor)
                return [kline(pd.to_datetime(cursor, unit="ms", utc=True) + pd.Timedelta(hours=i)) for i in range(2)]
            with patch("sys.stdout", io.StringIO()):
                self.assertEqual(data.download_history(cache_path=path, end_time=start + pd.Timedelta(hours=4), fetch_page=fetch), 4)
                self.assertEqual(data.download_history(cache_path=path, end_time=start + pd.Timedelta(hours=4), fetch_page=fetch), 0)
            self.assertEqual(calls, [data.milliseconds(start), data.milliseconds(start + pd.Timedelta(hours=2))])
            statements = []
            original = sqlite3.connect
            def connect(database, **kwargs):
                self.assertTrue(kwargs["uri"])
                self.assertIn("mode=ro", database)
                conn = original(database, **kwargs)
                conn.set_trace_callback(statements.append)
                return conn
            with patch.object(data.sqlite3, "connect", side_effect=connect):
                allowed = data.load_history(end=start + pd.Timedelta(hours=2), cache_path=path)
            self.assertEqual(len(allowed), 2)
            self.assertTrue(any("timestamp_ms +" in statement for statement in statements))
            self.assertTrue((allowed.index + pd.Timedelta(hours=1) <= start + pd.Timedelta(hours=2)).all())
            path.unlink()  # Windows would fail here if either cache handle stayed open.

    def test_empty_page_stops_without_spinning_or_creating_live_db(self):
        with tempfile.TemporaryDirectory() as directory, patch("sys.stdout", io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "Empty"):
                data.download_history(cache_path=Path(directory) / "empty.sqlite", end_time=data.HISTORY_START + pd.Timedelta(hours=2), fetch_page=lambda *_: [])
        with self.assertRaisesRegex(ValueError, "database"):
            data.download_history(cache_path="trading_bot.db")

    def test_parity_is_informational_and_cutoff_bounded(self):
        index = pd.date_range("2020-01-01", periods=2, freq="4h", tz="UTC")
        bars = pd.DataFrame({"close": [100, 110]}, index=index)
        other = pd.DataFrame({"close": [100, 100]}, index=index)
        end = index[-1] + pd.Timedelta(hours=4)
        with patch("backtest.fetch_history", return_value=other) as fetch:
            metrics = data.parity_check(bars, end=end)
        self.assertEqual(metrics["matched_bars"], 2)
        self.assertAlmostEqual(metrics["median_abs_percent"], 5)
        self.assertAlmostEqual(metrics["max_abs_percent"], 10)
        self.assertEqual(fetch.call_args.kwargs["end_time"], end.to_pydatetime())


if __name__ == "__main__":
    unittest.main()
