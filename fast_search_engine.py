"""Long/flat array simulator with reference per-trade accounting and reinvestment."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from search_space import REASONS

try:
    from numba import njit
except Exception:
    njit = None


def _kernel(stamps, opens, highs, lows, closes, actions, reasons, start, bar_ns,
            fee, slip, stop, take, max_hold_ns, liquidate):
    count = len(opens)
    equity = np.ones(count - start + 1)
    invested = np.zeros(count - start, dtype=np.bool_)
    trades = np.empty((count, 4))  # entry index, exit index, reason, return percent
    used, held = 0, False
    cash, quantity, basis, entry_fill, entry_market, entry_index = 1.0, 0.0, 0.0, 0.0, 0.0, 0
    for i in range(start, count):
        signal = 0 if i == start else actions[i - 1]
        exit_price, reason = 0.0, 0
        exited_at_open = False
        if held:
            if stop > 0 and opens[i] <= entry_fill * (1 - stop):
                exit_price, reason = opens[i], 1
            elif take > 0 and opens[i] >= entry_fill * (1 + take):
                exit_price, reason = opens[i], 2
            elif max_hold_ns > 0 and stamps[i] - stamps[entry_index] >= max_hold_ns:
                exit_price, reason = opens[i], 3
            elif signal == -1:
                exit_price, reason = opens[i], reasons[i - 1]
            exited_at_open = reason != 0
        elif signal == 1:
            basis, entry_market = cash, opens[i]
            entry_fill = opens[i] * (1 + slip)
            quantity = basis / entry_fill * (1 - fee)
            cash, held, entry_index = 0.0, True, i
        # Exposure is sampled after any open exit, before an intrabar exit.
        if held and not exited_at_open:
            invested[i - start] = True
            if stop > 0 and lows[i] <= entry_fill * (1 - stop):
                exit_price, reason = min(opens[i], entry_fill * (1 - stop)), 1
            elif take > 0 and highs[i] >= entry_fill * (1 + take):
                exit_price, reason = max(opens[i], entry_fill * (1 + take)), 2
        exit_index = float(i)
        if held and reason == 0 and liquidate and i == count - 1:
            exit_price, reason, exit_index = closes[i], 7, float(i + 1)
        if held and reason != 0:
            fill = exit_price * (1 - slip)
            proceeds = quantity * fill * (1 - fee)
            # Same cash-flow formula as reference, normalized to invested notional.
            net_return = (proceeds - basis) / basis * 100
            trades[used, 0], trades[used, 1] = entry_index, exit_index
            trades[used, 2], trades[used, 3] = reason, net_return
            used += 1
            cash, held, quantity = proceeds, False, 0.0
        equity[i - start + 1] = cash + (quantity * closes[i] if held else 0.0)
    return equity, invested, trades[:used]


# Optional acceleration uses the exact same function; NumPy/Python always works.
_kernel_jit = njit(cache=True)(_kernel) if njit is not None else None


@dataclass
class Simulation:
    equity: pd.Series
    invested: pd.Series
    trades: pd.DataFrame


def simulate(bars, actions, reasons, *, timeframe, fee_rate, slippage_rate,
             test_start=None, stop_loss=None, take_profit=None, max_hold_hours=None,
             liquidate_at_end=False, use_jit=True):
    if not 0 <= fee_rate < 1 or not 0 <= slippage_rate < 1:
        raise ValueError("Costs must be in [0, 1)")
    if len(actions) != len(bars) or len(reasons) != len(bars):
        raise ValueError("Signal arrays must match bars")
    if len(bars) < 2:
        raise ValueError("At least two bars required")
    start = int(bars.index.searchsorted(test_start or bars.index[0]))
    if start >= len(bars) - 1:
        raise ValueError("At least two simulation bars required")
    hours = {"1Hour": 1, "2Hour": 2, "4Hour": 4, "1Day": 24}[timeframe]
    duration = pd.Timedelta(hours=hours)
    stamps = bars.index.as_unit("ns").asi8
    values = [np.ascontiguousarray(bars[name].to_numpy(dtype=float)) for name in ("open", "high", "low", "close")]
    if not all(np.isfinite(value).all() for value in values) or any((value <= 0).any() for value in values):
        raise ValueError("Simulation requires finite positive OHLC; gaps must remain absent")
    kernel = _kernel_jit if use_jit and _kernel_jit is not None else _kernel
    equity, invested, trades = kernel(stamps, *values, np.asarray(actions, dtype=np.int8), np.asarray(reasons, dtype=np.int8),
                                     start, duration.value, fee_rate, slippage_rate,
                                     stop_loss or 0.0, take_profit or 0.0,
                                     int((max_hold_hours or 0) * 3_600_000_000_000), liquidate_at_end)
    times = bars.index[start:] + duration
    equity_times = times.insert(0, bars.index[start])
    rows = []
    for entry, exit_at, reason, net_return in trades:
        exit_idx = int(exit_at)
        exit_time = bars.index[exit_idx] if exit_idx < len(bars) else bars.index[-1] + duration
        rows.append({"entry_time": bars.index[int(entry)], "exit_time": exit_time,
                     "exit_reason": REASONS[int(reason)], "return_percent": float(net_return)})
    return Simulation(pd.Series(equity, index=equity_times), pd.Series(invested, index=bars.index[start:]),
                      pd.DataFrame(rows, columns=["entry_time", "exit_time", "exit_reason", "return_percent"]))
