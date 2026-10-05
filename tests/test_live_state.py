import unittest

from main import should_process_candle


class LiveCandleStateTests(unittest.TestCase):
    def test_same_candle_is_not_processed_twice(self):
        candle = "2026-10-05T00:00:00Z"
        self.assertFalse(should_process_candle(candle, candle))
        self.assertTrue(should_process_candle("2026-10-05T00:05:00Z", candle))
        self.assertFalse(should_process_candle(None, candle))


if __name__ == "__main__":
    unittest.main()
