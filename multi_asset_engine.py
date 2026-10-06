"""Fixed daily signals, a fail-closed reference gate, and sleeve accounting."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

import backtest
from fast_search_engine import simulate
from search_space import REASONS
from strategy import Decision, StrategySpec

RULES = ("sma200_slope20", "sma200")
WARMUP = 220


def trend_actions(bars, rule="sma200_slope20"):
    if rule not in RULES:
        raise ValueError("Only the two pre-registered rules are allowed")
    average = bars.close.rolling(200, min_periods=200).mean()
    bullish = bars.close > average
    if rule == RULES[0]:
        bullish &= average > average.shift(20)
    actions = np.where(bullish, 1, -1).astype(np.int8)
    # 220 completed observations BEFORE the first decision (index 220).
    actions[:WARMUP] = 0
    return actions


def _compare(bars, actions, *, fee, slip, liquidate):
    reasons = np.full(len(bars), 4, dtype=np.int8)
    spec = StrategySpec("daily_research_gate", lambda df: df,
                        lambda df, i: Decision({1: "BUY", -1: "SELL", 0: "HOLD"}[int(actions[i])], REASONS[4]),
                        lambda: 210, lambda: {}, lambda: None, lambda: None)
    expected = backtest.run_backtest(bars, starting_capital=1e9, timeframe="1Day",
                                    fee_rate=fee, slippage=slip, test_start=bars.index[WARMUP],
                                    strategy=spec, liquidate_at_end=liquidate)["trades"]
    actual = simulate(bars, actions, reasons, timeframe="1Day", fee_rate=fee,
                      slippage_rate=slip, test_start=bars.index[WARMUP],
                      liquidate_at_end=liquidate).trades
    if len(expected) != len(actual):
        raise AssertionError("Daily trade counts differ")
    for field in ("entry_time", "exit_time", "exit_reason"):
        if list(expected[field]) != list(actual[field]):
            raise AssertionError(f"Daily {field} differs")
    if not np.allclose(expected.return_percent.astype(float), actual.return_percent.astype(float), rtol=0, atol=1e-9):
        raise AssertionError("Daily per-trade returns differ beyond 1e-9")
    return len(actual)


def daily_equivalence_gate(spy, progress=print):
    if spy is None or len(spy) < WARMUP + 30:
        raise ValueError("Daily equivalence requires a real adjusted SPY sample (>=250 bars)")
    rng = np.random.default_rng(20261006)
    count = 900
    close = 100 * np.exp(.28 * np.sin(np.arange(count) / 60) + rng.normal(0, .01, count))
    opens = close * np.exp(rng.normal(0, .01, count))
    dates = pd.bdate_range("2016-01-04", periods=count).delete([25, 55, 202, 430])
    synthetic = pd.DataFrame({"open": opens[:len(dates)], "close": close[:len(dates)], "volume": 1.0}, index=dates)
    synthetic["high"] = synthetic[["open", "close"]].max(axis=1) * 1.02
    synthetic["low"] = synthetic[["open", "close"]].min(axis=1) * .98
    status = {}
    try:
        for label, bars in (("synthetic_daily_gaps", synthetic), ("real_SPY", spy.iloc[:1400])):
            checked = 0
            schedules = [trend_actions(bars, rule) for rule in RULES]
            # An additional deterministic schedule exercises costs/times even when
            # a real sample trends continuously and the rule never completes a trade.
            schedules.append(np.where((np.arange(len(bars)) // 17) % 2, 1, -1).astype(np.int8))
            for actions in schedules:
                for fee, slip in ((0, .0002), (.00075, .0002), (0, .0005)):
                    for liquidate in (False, True):
                        checked += _compare(bars, actions, fee=fee, slip=slip, liquidate=liquidate)
            if checked == 0:
                raise AssertionError("No completed trades; daily gate inconclusive")
            status[label] = checked
            progress(f"EQUIVALENCE PASS {label}: 18 cases, {checked} trades; times exact, return tolerance 1e-9")
    except AssertionError as error:
        progress(f"EQUIVALENCE FAIL: {error}")
        raise RuntimeError("Daily equivalence gate failed; research stopped") from error
    return status


@dataclass
class Curve:
    # One observation per session; baseline 1 is implicit before the first open.
    equity: pd.Series
    exposure: pd.Series
    entries: pd.Series
    turnover: pd.Series
    costs: pd.Series


def portfolio(bars, targets, rates, *, weights=None, causal_transfers=False):
    """Long/flat sleeves. Missing sessions must be removed BEFORE calling this.

    ETF portfolios equalize capital at simultaneous session opens. For a mixed
    BTC/ETF portfolio, fixed USD transfers use preceding session closing capital:
    BTC's earlier UTC fill must never be sized from later ETF opening prices.
    Rebalance purchases/sales incur costs; moving an entirely cash sleeve is free.
    Costs use the reference engine's buy fee withholding and sell cash deduction.
    Endpoints mark holdings at the final close without artificial liquidation.
    """
    symbols = tuple(bars)
    dates = bars[symbols[0]].index
    if len(dates) < 2 or any(not bars[s].index.equals(dates) for s in symbols):
        raise ValueError("Sleeves require at least two identical session dates")
    n = len(symbols)
    weights = np.array(weights if weights is not None else [1 / n] * n, dtype=float)
    if len(weights) != n or (weights <= 0).any() or not np.isclose(weights.sum(), 1):
        raise ValueError("Weights must be positive and sum to one")
    opens = np.column_stack([bars[s].open for s in symbols])
    closes = np.column_stack([bars[s].close for s in symbols])
    if any(targets[s].reindex(dates).isna().any() for s in symbols):
        raise ValueError("Missing sleeve target on an execution date")
    if any(not 0 <= value < 1 for s in symbols for value in rates[s]):
        raise ValueError("Costs must be in [0, 1)")
    desired = np.column_stack([targets[s].reindex(dates).to_numpy(dtype=bool) for s in symbols])
    cash, qty = weights.copy(), np.zeros(n)
    held = np.zeros(n, dtype=bool)
    previous_capital = weights.copy()
    equities, exposures, entries, turnovers, costs = [], [], [], [], []
    for i, date in enumerate(dates):
        monthly = i == 0 or date.to_period("M") != dates[i - 1].to_period("M")
        capital = cash + qty * opens[i]
        if monthly:
            basis = previous_capital if causal_transfers and i else capital
            transfers = basis.sum() * weights - basis
            cash += transfers  # Fixed net-zero sleeve funding; charged only if assets trade.
            capital += transfers
            if (capital <= 0).any():
                raise ValueError("Opening gap exhausted a sleeve during fixed cash transfer")
        day_cost, day_turnover, day_entries = 0., 0., 0
        for j, symbol in enumerate(symbols):
            fee, slip = rates[symbol]
            buying = (1 - fee) / (1 + slip)
            selling = (1 - fee) * (1 - slip)
            old_notional = qty[j] * opens[i, j]
            if desired[i, j]:
                # Any sleeve cash (including incoming capital) buys; negative cash
                # is funded by selling exactly enough to pay the outgoing transfer.
                amount = cash[j]
                if amount >= 0:
                    market_turnover = amount / (1 + slip)
                    qty[j] += amount * buying / opens[i, j]
                    day_cost += amount * (1 - buying)
                else:
                    market_turnover = -amount / selling
                    qty[j] -= market_turnover / opens[i, j]
                    day_cost += market_turnover * (1 - selling)
                cash[j] = 0
                if not held[j]:
                    day_entries += 1
            else:
                market_turnover = old_notional
                cash[j] += old_notional * selling
                qty[j] = 0
                day_cost += old_notional * (1 - selling)
            day_turnover += market_turnover
            held[j] = desired[i, j]
        if (cash < -1e-10).any() or (qty < -1e-10).any():
            raise AssertionError("Sleeve cash or quantity became negative")
        previous_capital = cash + qty * closes[i]
        equity = previous_capital.sum()
        equities.append(equity)
        exposures.append(float(np.dot(weights, held)))
        entries.append(day_entries)
        turnovers.append(day_turnover)
        costs.append(day_cost)
    return Curve(*(pd.Series(values, index=dates, dtype=float) for values in (equities, exposures, entries, turnovers, costs)))


def daily_returns(equity):
    previous = equity.shift(1)
    previous.iloc[0] = 1.0
    return equity / previous - 1


def metrics(curve):
    equity, returns = curve.equity, daily_returns(curve.equity)
    years = ((equity.index[-1] - equity.index[0]).days + 1) / 365.25
    total = float(equity.iloc[-1] - 1)
    cagr = float(equity.iloc[-1] ** (1 / years) - 1)
    std = returns.std(ddof=1)
    sharpe = float(returns.mean() / std * np.sqrt(252)) if std > 0 else np.nan
    peaks = equity.cummax().clip(lower=1)
    dd = float((1 - equity / peaks).max())
    annual = (1 + returns).groupby(returns.index.year).prod() - 1
    return dict(start=str(equity.index[0].date()), end=str(equity.index[-1].date()),
                total_return_pct=total * 100, cagr_pct=cagr * 100,
                volatility_pct=float(std * np.sqrt(252) * 100), sharpe=sharpe,
                max_drawdown_pct=dd * 100, calmar=cagr / dd if dd > 0 else np.nan,
                time_invested_pct=float(curve.exposure.mean() * 100),
                trades_per_year=float(curve.entries.sum() / years),
                worst_calendar_year=int(annual.idxmin()), worst_year_return_pct=float(annual.min() * 100),
                entries=int(curve.entries.sum()), rebalance_and_signal_turnover=float(curve.turnover.sum()),
                cost_initial_capital_pct=float(curve.costs.sum() * 100))


def period_tables(equity):
    returns = daily_returns(equity)
    yearly = (1 + returns).groupby(returns.index.year).prod() - 1
    rows = []
    for label, first, last in (("2016-2019", "2016-01-01", "2019-12-31"),
                               ("2020-2021", "2020-01-01", "2021-12-31"),
                               ("2022", "2022-01-01", "2022-12-31"),
                               ("2023-latest", "2023-01-01", "2099-12-31")):
        selected = returns.loc[first:last]
        rows.append(dict(period=label, start=str(selected.index[0].date()) if len(selected) else "",
                         end=str(selected.index[-1].date()) if len(selected) else "",
                         return_pct=float((1 + selected).prod() - 1) * 100 if len(selected) else np.nan))
    return yearly * 100, rows
