import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import backtest


class BacktestOutputTests(unittest.TestCase):
    def test_stdout_and_stderr_are_captured_and_saved(self):
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)
            console = StringIO()

            def fake_run():
                print("backtest report")
                print("warning", file=sys.stderr)
                return 0

            try:
                with patch("backtest._run_main", side_effect=fake_run), redirect_stdout(console):
                    result = backtest.main()
                saved = Path("last_backtest_output.txt").read_text(encoding="utf-8")
            finally:
                os.chdir(original_cwd)

        self.assertEqual(result, 0)
        self.assertIn("backtest report", saved)
        self.assertIn("warning", saved)
        self.assertEqual(console.getvalue(), saved)


if __name__ == "__main__":
    unittest.main()
