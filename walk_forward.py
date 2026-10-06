"""Pre-registered calendar windows, selection, random control, and PASS criteria."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from binance_data import utc

RANDOM_DRAWS = 1000
RANDOM_SEED = 20261006
FIRST_TRAIN = pd.Timestamp("2019-01-01", tz="UTC")


@dataclass(frozen=True)
class Fold:
    number: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def holdout_bounds(as_of):
    end = utc(as_of).floor("4h")
    return end - pd.DateOffset(months=6), end


def build_folds(holdout_start):
    folds, start = [], FIRST_TRAIN
    while True:
        train_end = start + pd.DateOffset(months=12)
        test_end = train_end + pd.DateOffset(months=3)
        if test_end > utc(holdout_start):
            break
        folds.append(Fold(len(folds) + 1, start, train_end, train_end, test_end))
        start += pd.DateOffset(months=3)
    return tuple(folds)


def window_metrics(result, start, end):
    """Carry continuous positions; count completed trades by exit in [start,end)."""
    start, end = utc(start), utc(end)
    equity = result.equity
    left = int(equity.index.searchsorted(start, side="right")) - 1
    right = int(equity.index.searchsorted(end, side="right"))
    if left < 0 or right <= left + 1:
        raise ValueError(f"No marked equity covering window {start} -> {end}")
    values = equity.iloc[left:right].to_numpy(dtype=float)
    initial = float(values[0])
    factor = float(values[-1] / initial)
    drawdown = float(np.max(1 - values / np.maximum.accumulate(values))) * 100
    trades = result.trades
    completed = trades.loc[(trades["exit_time"] >= start) & (trades["exit_time"] < end)] if not trades.empty else trades
    count = len(completed)
    returns = completed["return_percent"].to_numpy(dtype=float) if count else np.empty(0)
    invested = result.invested
    exposure = invested.iloc[invested.index.searchsorted(start):invested.index.searchsorted(end)]
    days = (end - start).total_seconds() / 86400
    return {"return_percent": (factor - 1) * 100, "max_drawdown_percent": drawdown,
            "trades": count, "trades_per_week": count / days * 7, "trades_per_day": count / days,
            "time_invested_percent": float(exposure.mean() * 100) if len(exposure) else 0.0,
            "trade_return_sum": float(returns.sum()), "average_trade_return_percent": float(returns.mean()) if count else None}


def eligible(metrics):
    return (metrics["trades"] >= 20 and metrics["trades_per_week"] >= 1
            and metrics["trades_per_day"] <= 5 and metrics["time_invested_percent"] >= 5)


def selection_score(metrics):
    profit, drawdown = metrics["return_percent"], metrics["max_drawdown_percent"]
    if drawdown == 0:
        return float("inf") if profit > 0 else float("-inf") if profit < 0 else 0.0
    return profit / drawdown


def select_top(metrics_by_id, count=5):
    names = [name for name, metrics in metrics_by_id.items() if eligible(metrics)]
    names.sort(key=lambda name: (-selection_score(metrics_by_id[name]), -metrics_by_id[name]["trades"], name))
    return names[:count], names


def random_baseline(eligible_sets, returns_by_fold, *, seed=RANDOM_SEED, draws=RANDOM_DRAWS):
    """Each draw is an independent path; sample five distinct configs per fold."""
    if len(eligible_sets) != len(returns_by_fold):
        raise ValueError("Eligible sets must align with test windows")
    rng, factors, sampled = np.random.default_rng(seed), np.ones(draws), []
    for names, test_returns in zip(eligible_sets, returns_by_fold):
        names = sorted(names)
        if len(names) < 5:
            raise ValueError("Random control requires at least five eligible configs in every fold")
        choices = [rng.choice(names, 5, replace=False).tolist() for _ in range(draws)]
        factors *= 1 + np.array([np.mean([test_returns[name] for name in chosen]) for chosen in choices]) / 100
        sampled.append(choices)
    return (factors - 1) * 100, sampled


def compound(returns):
    return float((np.prod(1 + np.asarray(returns, dtype=float) / 100) - 1) * 100)


def pass_rule(*, compounded_return, random_p95, average_trade_return, positive_fold_percent, double_cost_return):
    rows = [
        {"criterion": "positive_oos_return", "value": compounded_return, "threshold": 0.0,
         "passed": bool(np.isfinite(compounded_return) and compounded_return > 0)},
        {"criterion": "random_selection_p95", "value": compounded_return, "threshold": random_p95,
         "passed": bool(np.isfinite(compounded_return) and np.isfinite(random_p95) and compounded_return >= random_p95)},
        {"criterion": "positive_average_completed_trade", "value": average_trade_return, "threshold": 0.0,
         "passed": bool(average_trade_return is not None and np.isfinite(average_trade_return) and average_trade_return > 0)},
        {"criterion": "positive_folds_at_least_55_percent", "value": positive_fold_percent, "threshold": 55.0,
         "passed": bool(np.isfinite(positive_fold_percent) and positive_fold_percent >= 55)},
        {"criterion": "positive_at_double_binance_costs", "value": double_cost_return, "threshold": 0.0,
         "passed": bool(np.isfinite(double_cost_return) and double_cost_return > 0)},
    ]
    return all(row["passed"] for row in rows), rows
