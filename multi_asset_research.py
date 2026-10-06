"""Pre-registered daily trend research. This module cannot place orders."""
import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

import config
from fast_search_engine import simulate
from multi_asset_data import ETF_SYMBOLS, START, completed_end, load_btc, load_etfs, quality_report
from multi_asset_engine import (RULES, WARMUP, Curve, daily_equivalence_gate, metrics,
                                period_tables, portfolio, trend_actions)
from report_output import csv_output_path


def build_parser():
    parser = argparse.ArgumentParser(description="Research only: fixed daily SMA200/slope20 across SPY, TLT, GLD, UUP and optional BTC; no parameter search.")
    parser.add_argument("--download", action="store_true", help="Download adjusted ETF bars and update the existing public Binance hourly cache")
    parser.add_argument("--end-date", type=lambda value: pd.Timestamp(date.fromisoformat(value)), metavar="YYYY-MM-DD", help="Inclusive end date, capped at completed US trading days")
    btc = parser.add_mutually_exclusive_group()
    btc.add_argument("--include-btc", dest="include_btc", action="store_true", help="Include BTC (default); requires the existing Binance cache")
    btc.add_argument("--no-btc", dest="include_btc", action="store_false", help="Run the four ETFs only")
    parser.set_defaults(include_btc=True)
    parser.add_argument("--cost-sensitivity", action="store_true", help="Also report ETF 0.05%% per-side slippage, without choosing a rule")
    parser.add_argument("--csv", type=Path, help="Write all metrics, calendar-year returns, fixed periods, data quality and gate results in a single CSV")
    return parser


def rates_for(symbols, slip):
    btc = config.get_fee_profile("binance_spot_bnb")
    return {s: (btc["fee_rate"], btc["slippage_rate"]) if s == "BTC" else (0., slip) for s in symbols}


def ready_dates(bars, start=None):
    earliest = max(frame.index[WARMUP + 1] for frame in bars.values())
    if start is not None:
        earliest = max(earliest, pd.Timestamp(start))
    common = next(iter(bars.values())).index
    for frame in bars.values():
        common = common.intersection(frame.index)
    return common[common >= earliest]


def signal_targets(bars, rule, dates):
    # Shift BEFORE selecting dates: a fill never uses its own session close.
    return {s: pd.Series(trend_actions(frame, rule) == 1, index=frame.index).shift(1, fill_value=False).reindex(dates)
            for s, frame in bars.items()}


def asset_curve(symbol, bars, rule, rates):
    actions = trend_actions(bars, rule)
    result = simulate(bars, actions, np.full(len(bars), 4, dtype=np.int8),
                      timeframe="1Day", fee_rate=rates[symbol][0], slippage_rate=rates[symbol][1],
                      test_start=bars.index[WARMUP], liquidate_at_end=False)
    dates = bars.index[WARMUP + 1:]
    sleeves = portfolio({symbol: bars.loc[dates]}, signal_targets({symbol: bars}, rule, dates), rates)
    # Independently guard production per-asset reinvestment accounting, beyond
    # the reference gate's per-trade-return equivalence.
    fast_equity = result.equity.iloc[2:].to_numpy()
    if not np.allclose(sleeves.equity, fast_equity, rtol=0, atol=1e-9):
        raise RuntimeError("Daily sleeve accounting differs from fast engine; research stopped")
    return Curve(pd.Series(fast_equity, index=dates), sleeves.exposure, sleeves.entries, sleeves.turnover, sleeves.costs)


def run_research(data, *, sensitivity=False, progress=print):
    if any(len(frame) < WARMUP + 3 for frame in data.values()):
        raise ValueError("Each included asset needs at least 223 completed trading-day observations")
    records, summary, years, periods = [], [], [], []

    def record(name, rule, cost, curve, span):
        values = dict(name=name, rule=rule, cost=cost, span=span, **metrics(curve))
        summary.append(values)
        records.append(dict(section="metrics", **values))
        annual, fixed = period_tables(curve.equity)
        for year, value in annual.items():
            row = dict(name=name, rule=rule, cost=cost, span=span, year=int(year), return_pct=value)
            years.append(row)
            records.append(dict(section="calendar_year", **row))
        for row in fixed:
            row = dict(name=name, rule=rule, cost=cost, span=span, **row)
            periods.append(row)
            records.append(dict(section="fixed_period", **row))

    etfs = {s: data[s] for s in ETF_SYMBOLS}
    mixed_dates = ready_dates(data) if "BTC" in data else None
    if mixed_dates is not None and len(mixed_dates) < 2:
        raise ValueError("ETF_BTC_EQUAL: insufficient common dates after warm-up")
    mixed_start = mixed_dates[0] if mixed_dates is not None else None
    groups = [("ETF_EQUAL", etfs, None, "full")]
    if mixed_start is not None:
        groups += [("ETF_EQUAL", etfs, mixed_start, "BTC_matched"),
                   ("ETF_BTC_EQUAL", data, mixed_start, "full")]
    for slip in ((.0002, .0005) if sensitivity else (.0002,)):
        cost = f"ETF_side_{slip * 100:.2f}pct"
        rates = rates_for(data, slip)
        progress(f"Costs: {cost}; BTC binance_spot_bnb fee={rates.get('BTC', (0, 0))[0]:g}, slippage={rates.get('BTC', (0, 0))[1]:g}")
        for s, frame in data.items():
            dates = frame.index[WARMUP + 1:]
            for rule in RULES:
                record(s, rule, cost, asset_curve(s, frame, rule, rates), "asset_full")
            target = {s: pd.Series(True, index=dates)}
            record(s, "buy_hold", cost, portfolio({s: frame.loc[dates]}, target, rates), "asset_full")
        for name, source, start, span in groups:
            dates = mixed_dates if span == "BTC_matched" else ready_dates(source, start)
            if len(dates) < 2:
                raise ValueError(f"{name}: insufficient common dates after warm-up")
            trimmed = {s: frame.loc[dates] for s, frame in source.items()}
            union = pd.DatetimeIndex([])
            for frame in source.values():
                union = union.union(frame.loc[dates[0]:dates[-1]].index)
            progress(f"{name}/{span}: {dates[0].date()} -> {dates[-1].date()}, {len(dates)} common sessions; {len(union.difference(dates))} sessions excluded for missing sleeves")
            causal = "BTC" in source
            for rule in RULES:
                record(name, rule, cost, portfolio(trimmed, signal_targets(source, rule, dates), rates,
                                                  causal_transfers=causal), span)
            target = {s: pd.Series(True, index=dates) for s in source}
            record(name, "buy_hold", cost, portfolio(trimmed, target, rates, causal_transfers=causal), span)
            spy = {"SPY": data["SPY"].loc[dates]}
            record(name + "_SPY", "buy_hold", cost, portfolio(spy, {"SPY": target["SPY"]}, rates), span)
            balanced = {s: data[s].loc[dates] for s in ("SPY", "TLT")}
            record(name + "_60_40", "buy_hold", cost, portfolio(balanced, {s: target[s] for s in balanced}, rates,
                                                                weights=(.6, .4)), span)
    return records, pd.DataFrame(summary), pd.DataFrame(years), pd.DataFrame(periods)


def print_tables(summary, years, periods):
    identifiers = ["name", "rule", "cost", "span"]
    columns = identifiers + ["start", "end", "total_return_pct", "cagr_pct", "volatility_pct", "sharpe",
                             "max_drawdown_pct", "calmar", "time_invested_pct", "trades_per_year",
                             "worst_calendar_year", "worst_year_return_pct"]
    print("\nMetrics (percent columns in percentage points; trades/year counts position entries, excluding rebalance adjustments):")
    print(summary[columns].to_string(index=False, float_format=lambda value: f"{value:.4f}", na_rep="N/A"))
    print("\nCalendar-year returns (%; endpoint years may be partial):")
    print(years.pivot(index=identifiers, columns="year", values="return_pct").to_string(float_format=lambda value: f"{value:.4f}", na_rep="N/A"))
    print("\nFixed sub-period returns (%; actual covered dates shown, no extrapolation):")
    print(periods.to_string(index=False, float_format=lambda value: f"{value:.4f}", na_rep="N/A"))


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        output = csv_output_path(args.csv) if args.csv else None
        if output is not None and (output.suffix.lower() != ".csv" or "data_cache" in output.parts):
            raise ValueError("--csv must be a .csv report outside data_cache")
        end = completed_end(args.end_date)
        if end < START:
            raise ValueError("--end-date precedes the fixed 2016-01-01 data start")
        print("RESEARCH ONLY: fixed primary sma200_slope20; secondary sma200 is sensitivity only. Nothing goes live.")
        print("WARNING: the BTC rule was chosen after inspecting 2021-2026 BTC data; the ETF results are the first look at those assets.")
        print("Cash earns 0%; ignoring cash interest can understate returns of strategies that sit in cash.")
        print("Adjusted OHLC includes splits and dividends; dividends are not credited a second time.")
        print("Signals use 220 preceding observations; fills follow the decision at the next available open.")
        print("BTC signals use US-calendar sampled daily closes; fills use the first UTC daily open after the prior US close (including weekends).")
        print("ETF capital rebalances at the first common US session open of each month; mixed BTC capital transfers are fixed from the preceding US close to avoid future ETF opening prices.")
        print("Portfolios retain inactive sleeves as cash; costs apply to actual asset turnover. Holdings are marked at the final close without forced liquidation.")
        print("Daily volatility and Sharpe use sqrt(252), zero risk-free rate; CAGR uses actual elapsed calendar days. Exposure is the average target-weight fraction of invested sleeves.")
        print(f"Completed US date cutoff: {end.date()}; US reference calendar: SPY dates (holidays are not gaps).")
        data = load_etfs(end, download=args.download)
        gate = daily_equivalence_gate(data["SPY"])
        if args.include_btc:
            data["BTC"] = load_btc(data["SPY"].index, end, download=args.download)
        quality = [quality_report(s, frame, data["SPY"].index) for s, frame in data.items()]
        print("\nData quality (missing counts relative to SPY over the entire requested reference span, including unavailable early BTC history):")
        print(pd.DataFrame(quality).drop(columns="missing_dates").to_string(index=False))
        for row in quality:
            if row["missing_dates"]:
                dates = row["missing_dates"].split(",")
                display = ",".join(dates[:20])
                suffix = f" ... ({len(dates)} total; full list in --csv)" if len(dates) > 20 else ""
                print(f"{row['symbol']} missing US dates: {display}{suffix}")
        records, summary, years, periods = run_research(data, sensitivity=args.cost_sensitivity)
        print_tables(summary, years, periods)
        if output is not None:
            records += [dict(section="data_quality", **row) for row in quality]
            records += [dict(section="equivalence", dataset=label, trades_checked=count, result="pass", tolerance=1e-9) for label, count in gate.items()]
            output.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(records).to_csv(output, index=False)
            print(f"\nCSV saved: {output}")
        return 0
    except (ValueError, RuntimeError, OSError, KeyError) as error:
        print(f"RESEARCH STOPPED: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
