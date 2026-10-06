"""Pre-registered BTC walk-forward search, isolated from live trading.

Download explicitly first. Default search never loads final-holdout rows. The
holdout command evaluates only a previously frozen PASS candidate, without search.
"""

import argparse
import csv
import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

import binance_data as data
import config
from fast_search_engine import simulate
from report_output import csv_output_path
from search_equivalence import equivalence_gate
from search_space import TIMEFRAMES, apply_regime, enumerate_configs, family_signals, merge_regime
from strategy import get_strategy
from walk_forward import (FIRST_TRAIN, RANDOM_DRAWS, RANDOM_SEED, build_folds, compound,
                          holdout_bounds, pass_rule, select_top, selection_score, window_metrics,
                          random_baseline)

CANDIDATE_PATH = Path("strategy_search_candidate.json")
PROTOCOL = {
    "version": 1, "first_train": "2019-01-01", "train_months": 12, "test_months": 3,
    "step_months": 3, "holdout_months": 6, "random_draws": RANDOM_DRAWS, "seed": RANDOM_SEED,
    "eligibility": {"min_trades": 20, "min_per_week": 1, "max_per_day": 5, "min_invested_percent": 5},
    "score": "compounded return / compounded max drawdown; tie: more trades then config id",
    "pass": ["return>0", "return>=random_p95", "mean_trade>0", "positive_folds>=55%", "double_cost_return>0"],
    "rsi": "Wilder, simple-average seed; flat RSI=50", "regime_v1": [200, 20, 240],
}


def costs():
    primary, alpaca = config.get_fee_profile("binance_spot_bnb"), config.get_fee_profile("alpaca")
    return {"binance_spot_bnb": primary, "alpaca": alpaca,
            "2x_binance": {key: value * 2 for key, value in primary.items()}}


def protocol_hash():
    payload = {"protocol": PROTOCOL, "configs": [c.to_dict() for c in enumerate_configs()], "costs": costs()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def prepare_history(hourly):
    frames = {tf: data.resample_bars(hourly, tf) for tf in TIMEFRAMES}
    states = {tf: merge_regime(frame, frames["4Hour"], tf) for tf, frame in frames.items()}
    return frames, states


def evaluate_configs(hourly, folds, *, cutoff, configs=None, progress=True):
    """One continuous simulation per config per cost; retain bounded window metrics."""
    cutoff = data.utc(cutoff)
    # Defensive boundary: even a caller handing in future rows cannot expose them.
    hourly = hourly.loc[hourly.index + pd.Timedelta(hours=1) <= cutoff]
    configs = tuple(configs or enumerate_configs())
    frames, states = prepare_history(hourly)
    signal_cache, summaries = {}, {}
    for cost_name, rates in costs().items():
        summary = {"folds": [{"train": {}, "test": {}} for _ in folds]}
        if progress:
            print(f"Simulating {len(configs)} configs at {cost_name}: {rates}", flush=True)
        for number, c in enumerate(configs, 1):
            frame = frames[c.timeframe]
            key = (c.timeframe, c.family, c.first, c.second)
            if key not in signal_cache:
                signal_cache[key] = family_signals(frame, c)
            actions, reasons = apply_regime(signal_cache[key], states[c.timeframe], c.regime_gate)
            result = simulate(frame, actions, reasons, timeframe=c.timeframe,
                              test_start=FIRST_TRAIN, fee_rate=rates["fee_rate"],
                              slippage_rate=rates["slippage_rate"], stop_loss=c.stop_loss,
                              max_hold_hours=c.max_hold_hours)
            for fold, windows in zip(folds, summary["folds"]):
                windows["train"][c.config_id] = window_metrics(result, fold.train_start, fold.train_end)
                windows["test"][c.config_id] = window_metrics(result, fold.test_start, fold.test_end)
            if progress and (number % 50 == 0 or number == len(configs)):
                print(f"  {cost_name}: {number}/{len(configs)} configs", flush=True)
        summaries[cost_name] = summary
    return summaries, frames


def summarize_walk_forward(summaries, folds):
    primary = summaries["binance_spot_bnb"]
    selections, eligible_sets, rows = [], [], []
    config_lookup = {c.config_id: c.to_dict() for c in enumerate_configs()}
    for fold, windows in zip(folds, primary["folds"]):
        selected, eligible_names = select_top(windows["train"])
        selections.append(selected)
        eligible_sets.append(eligible_names)
        if len(selected) < 5:
            rows.append({"row_type": "INSUFFICIENT_ELIGIBLE", "fold": fold.number,
                         "eligible_count": len(eligible_names), "issue": "At least five eligible configs required"})
        for rank, name in enumerate(selected, 1):
            train, test = windows["train"][name], windows["test"][name]
            rows.append({"row_type": "SELECTION", "fold": fold.number, "rank": rank,
                         "config_id": name, "parameters_json": json.dumps(config_lookup.get(name, {}), sort_keys=True),
                         "eligible_count": len(eligible_names),
                         "train_start": str(fold.train_start), "train_end": str(fold.train_end),
                         "test_start": str(fold.test_start), "test_end": str(fold.test_end),
                         "fee_profile": "binance_spot_bnb", **costs()["binance_spot_bnb"],
                         "train_score": selection_score(train), "train_return_percent": train["return_percent"],
                         "train_trades": train["trades"], "oos_return_percent": test["return_percent"],
                         "oos_max_drawdown_percent": test["max_drawdown_percent"]})
    complete = bool(folds) and all(len(names) == 5 for names in selections)
    portfolio = {}
    for cost_name, summary in summaries.items():
        returns, top1, trade_sum, trade_count = [], [], 0.0, 0
        for fold, names, windows in zip(folds, selections, summary["folds"]):
            # Freeze primary fee selection; costs never trigger a new selection.
            value = float(np.mean([windows["test"][name]["return_percent"] for name in names])) if len(names) == 5 else np.nan
            first = windows["test"][names[0]]["return_percent"] if len(names) == 5 else np.nan
            returns.append(value)
            top1.append(first)
            for name in names:
                trade_sum += windows["test"][name]["trade_return_sum"]
                trade_count += windows["test"][name]["trades"]
            rows.append({"row_type": "PORTFOLIO_FOLD", "fold": fold.number, "fee_profile": cost_name,
                         **costs()[cost_name], "oos_return_percent": value, "top1_return_percent": first,
                         "config_ids": ",".join(names)})
        portfolio[cost_name] = {"return_percent": compound(returns) if complete else np.nan,
                                "top1_return_percent": compound(top1) if complete else np.nan,
                                "positive_fold_percent": np.mean(np.array(returns) > 0) * 100 if complete else np.nan,
                                "average_trade_return_percent": trade_sum / trade_count if trade_count else None,
                                "trades": trade_count, "fold_returns": returns}
        rows.append({"row_type": "PORTFOLIO_TOTAL", "fee_profile": cost_name, **costs()[cost_name],
                     **{key: value for key, value in portfolio[cost_name].items() if key != "fold_returns"}})
    if complete:
        test_returns = [{name: metrics["return_percent"] for name, metrics in windows["test"].items()}
                        for windows in primary["folds"]]
        distribution, _ = random_baseline(eligible_sets, test_returns)
    else:
        distribution = np.full(RANDOM_DRAWS, np.nan)
    percentiles = {percent: float(np.percentile(distribution, percent)) for percent in (5, 25, 50, 75, 95)}
    for percent, value in percentiles.items():
        rows.append({"row_type": "RANDOM_PERCENTILE", "fee_profile": "binance_spot_bnb",
                     "percentile": percent, "return_percent": value, "draws": RANDOM_DRAWS, "seed": RANDOM_SEED})
    main = portfolio["binance_spot_bnb"]
    passed, criteria = pass_rule(compounded_return=main["return_percent"], random_p95=percentiles[95],
                                 average_trade_return=main["average_trade_return_percent"],
                                 positive_fold_percent=main["positive_fold_percent"],
                                 double_cost_return=portfolio["2x_binance"]["return_percent"])
    for criterion in criteria:
        rows.append({"row_type": "PASS_RULE", "fee_profile": "binance_spot_bnb", **criterion})
    rows.append({"row_type": "VERDICT", "passed": passed, "complete_folds": complete})
    train_scores, test_scores = [], []
    for names, windows in zip(selections, primary["folds"]):
        if len(names) == 5:
            train_scores.extend(selection_score(windows["train"][name]) for name in names)
            test_scores.extend(selection_score(windows["test"][name]) for name in names)
    degradation = {"row_type": "IS_OOS_DEGRADATION", "average_train_score": float(np.mean(train_scores)) if train_scores else None,
                   "average_oos_score": float(np.mean(test_scores)) if test_scores else None}
    rows.append(degradation)
    return {"rows": rows, "passed": passed, "criteria": criteria, "portfolio": portfolio,
            "random_percentiles": percentiles, "selections": selections, "degradation": degradation}


def benchmark_rows(frames, folds):
    from fast_search_engine import Simulation
    rows = []
    bars = frames["4Hour"]
    strategy = get_strategy("regime_only_4h")
    prepared = strategy.prepare_indicators(bars)
    decisions = [strategy.decide_at(prepared, i) for i in range(len(bars))]
    actions = np.array([1 if d.action == "BUY" else -1 if d.action == "SELL" else 0 for d in decisions])
    reasons = np.full(len(bars), 4)
    duration = pd.Timedelta(hours=4)
    benchmark_start, benchmark_end = folds[0].test_start, folds[-1].test_end
    market = bars.loc[(bars.index >= benchmark_start) & (bars.index + duration <= benchmark_end)]
    if market.empty or market.index[0] != benchmark_start or market.index[-1] + duration != benchmark_end:
        raise ValueError("BTC benchmark does not cover OOS boundaries")
    for name, rates in costs().items():
        regime = simulate(bars, actions, reasons, timeframe="4Hour", test_start=FIRST_TRAIN,
                          fee_rate=rates["fee_rate"], slippage_rate=rates["slippage_rate"])
        # One continuous buy-and-hold benchmark, bought once, not repurchased each fold.
        quantity = (1 - rates["fee_rate"]) / (market.iloc[0]["open"] * (1 + rates["slippage_rate"]))
        curve = pd.Series(quantity * market["close"].to_numpy(), index=market.index + duration)
        curve.iloc[-1] *= (1 - rates["fee_rate"]) * (1 - rates["slippage_rate"])
        curve.loc[benchmark_start] = 1.0
        btc = Simulation(curve.sort_index(), pd.Series(True, index=market.index), pd.DataFrame())
        for label, result in (("BTC_buy_hold", btc), ("regime_only_4h", regime)):
            returns = []
            for fold in folds:
                metrics = window_metrics(result, fold.test_start, fold.test_end)
                returns.append(metrics["return_percent"])
                rows.append({"row_type": "BENCHMARK_FOLD", "benchmark": label, "fold": fold.number,
                             "fee_profile": name, **rates, "oos_return_percent": metrics["return_percent"]})
            rows.append({"row_type": "BENCHMARK_TOTAL", "benchmark": label, "fee_profile": name,
                         **rates, "return_percent": compound(returns)})
    return rows


def quality_rows(frames, folds, cutoff):
    rows = []
    for timeframe, bars in frames.items():
        full = data.data_quality(bars, timeframe, data.HISTORY_START, cutoff)
        print(f"DATA QUALITY {timeframe}: {full}", flush=True)
        rows.append({"row_type": "DATA_QUALITY", "timeframe": timeframe, "phase": "ALL_ALLOWED", **full})
        for fold in folds:
            combined = []
            for phase, start, end in (("TRAIN", fold.train_start, fold.train_end), ("TEST", fold.test_start, fold.test_end)):
                quality = data.data_quality(bars, timeframe, start, end)
                rows.append({"row_type": "DATA_QUALITY", "timeframe": timeframe, "fold": fold.number, "phase": phase,
                             "start_time": str(start), "end_time": str(end), **quality})
                combined.append(f"{phase} expected={quality['expected']} missing={quality['missing']} zero={quality['zero_volume']}")
            print(f"  fold {fold.number} {timeframe}: {'; '.join(combined)}", flush=True)
    return rows


def write_csv(path, rows):
    target = csv_output_path(path)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV saved to {target}", flush=True)


def validate_paths(cache_path, output, candidate_path):
    cache, candidate = Path(cache_path).resolve(), csv_output_path(candidate_path)
    targets = [candidate]
    if output is not None:
        targets.append(csv_output_path(output))
    for target in targets:
        if target == cache or (target.exists() and cache.exists() and target.samefile(cache)):
            raise ValueError("Report/candidate output must not overwrite the research cache")
    if len(targets) == 2 and (targets[0] == targets[1] or
                             (targets[0].exists() and targets[1].exists() and targets[0].samefile(targets[1]))):
        raise ValueError("CSV and frozen candidate paths must differ")


def run_search(*, as_of=None, cache_path=data.CACHE_PATH, output="strategy_search.csv", candidate_path=CANDIDATE_PATH):
    validate_paths(cache_path, output, candidate_path)
    configs = enumerate_configs()
    holdout_start, holdout_end = holdout_bounds(as_of or pd.Timestamp.now(tz="UTC"))
    folds = build_folds(holdout_start)
    if not folds:
        raise ValueError("No complete walk-forward folds before holdout")
    print(f"Pre-registered configs: {len(configs)} | folds: {len(folds)} | random draws: {RANDOM_DRAWS}, seed: {RANDOM_SEED}", flush=True)
    print(f"Holdout EXCLUDED: [{holdout_start}, {holdout_end}); reading cache only before {holdout_start}", flush=True)
    hourly = data.load_history(end=holdout_start, cache_path=cache_path)
    if hourly.empty or hourly.index.min() > data.HISTORY_START or hourly.index.max() + pd.Timedelta(hours=1) < holdout_start:
        raise ValueError("Cache does not cover the allowed history; run python strategy_search.py --download")
    print(f"Loaded {len(hourly):,} completed hourly bars. Running equivalence gate before search.", flush=True)
    gate = equivalence_gate(hourly, progress=lambda message: print(message, flush=True))
    frames, _ = prepare_history(hourly)
    rows = quality_rows(frames, folds, holdout_start)
    for family, status in gate.items():
        rows.append({"row_type": "EQUIVALENCE", "family": family, "passed": True, **status})
    print("Informational 4H Binance/Alpaca parity check on the permitted overlap...", flush=True)
    try:
        parity = data.parity_check(frames["4Hour"], end=holdout_start)
        print(f"PARITY: {parity}", flush=True)
        rows.append({"row_type": "PARITY", **parity})
    except Exception as error:
        print(f"PARITY unavailable (informational): {error}", flush=True)
        rows.append({"row_type": "PARITY", "issue": str(error)})
    summaries, frames = evaluate_configs(hourly, folds, cutoff=holdout_start)
    report = summarize_walk_forward(summaries, folds)
    rows.extend(report["rows"])
    benchmarks = benchmark_rows(frames, folds)
    rows.extend(benchmarks)
    for benchmark in benchmarks:
        if benchmark["row_type"] == "BENCHMARK_TOTAL":
            print(f"{benchmark['benchmark']} {benchmark['fee_profile']}: OOS={benchmark['return_percent']:.4f}%", flush=True)
    for name, metrics in report["portfolio"].items():
        print(f"{name}: TOP-5 OOS={metrics['return_percent']:.4f}% | TOP-1 OOS={metrics['top1_return_percent']:.4f}%", flush=True)
    print(f"Random-selection OOS percentiles: {report['random_percentiles']}", flush=True)
    print(f"IS-vs-OOS degradation: {report['degradation']}", flush=True)
    for rule in report["criteria"]:
        print(f"{'PASS' if rule['passed'] else 'FAIL'} {rule['criterion']}: value={rule['value']} threshold={rule['threshold']}", flush=True)
    print(f"OVERALL: {'PASS' if report['passed'] else 'FAIL'}", flush=True)
    if report["passed"]:
        selected = report["selections"][-1]
        last_train = folds[-1]
        candidate = {"protocol_hash": protocol_hash(), "passed": True, "config_ids": selected,
                     "train_start": str(last_train.train_start), "train_end": str(last_train.train_end),
                     "holdout_start": str(holdout_start), "holdout_end": str(holdout_end),
                     "configs": [c.to_dict() for c in configs if c.config_id in selected]}
        target = csv_output_path(candidate_path)
        target.write_text(json.dumps(candidate, indent=2), encoding="utf-8")
        print(f"Frozen candidate (last registered train window {last_train.train_start} -> {last_train.train_end}): {selected}; saved to {target}", flush=True)
        print(json.dumps(candidate["configs"], sort_keys=True), flush=True)
        rows.append({"row_type": "FROZEN_CANDIDATE", "config_ids": ",".join(selected),
                     "train_start": candidate["train_start"], "train_end": candidate["train_end"]})
    if output is not None:
        write_csv(output, rows)
    return report


def run_final_holdout(*, cache_path=data.CACHE_PATH, candidate_path=CANDIDATE_PATH, output="strategy_search_holdout.csv"):
    # No re-selection, no walk-forward search, no PASS declaration from holdout.
    validate_paths(cache_path, output, candidate_path)
    with Path(candidate_path).open(encoding="utf-8") as handle:
        candidate = json.load(handle)
    if not candidate.get("passed") or candidate.get("protocol_hash") != protocol_hash():
        raise ValueError("Holdout requires a frozen PASS candidate with this exact protocol")
    lookup = {c.config_id: c for c in enumerate_configs()}
    ids = candidate["config_ids"]
    if len(ids) != 5 or len(set(ids)) != 5 or any(name not in lookup for name in ids):
        raise ValueError("Frozen candidate must contain five unique registered configs")
    start, end = data.utc(candidate["holdout_start"]), data.utc(candidate["holdout_end"])
    if not start < end or end - pd.DateOffset(months=6) != start:
        raise ValueError("Frozen holdout must span exactly six calendar months")
    expected_folds = build_folds(start)
    if not expected_folds or data.utc(candidate["train_start"]) != expected_folds[-1].train_start or data.utc(candidate["train_end"]) != expected_folds[-1].train_end:
        raise ValueError("Frozen selection must come from the last registered training window")
    if data.utc(candidate["train_end"]) > start:
        raise ValueError("Frozen training must end before the holdout")
    # Gate reads only pre-holdout sample, even on explicit holdout evaluation.
    allowed = data.load_history(end=start, cache_path=cache_path)
    equivalence_gate(allowed, progress=lambda message: print(message, flush=True))
    hourly = data.load_history(end=end, cache_path=cache_path)
    if hourly.empty or hourly.index.max() + pd.Timedelta(hours=1) < end:
        raise ValueError("Cache does not cover frozen holdout end; use --download")
    frames, states = prepare_history(hourly)
    rows = []
    for name, rates in costs().items():
        returns = []
        for identity in ids:
            c = lookup[identity]
            actions, reasons = apply_regime(family_signals(frames[c.timeframe], c), states[c.timeframe], c.regime_gate)
            result = simulate(frames[c.timeframe], actions, reasons, timeframe=c.timeframe, test_start=FIRST_TRAIN,
                              fee_rate=rates["fee_rate"], slippage_rate=rates["slippage_rate"],
                              stop_loss=c.stop_loss, max_hold_hours=c.max_hold_hours)
            metrics = window_metrics(result, start, end)
            returns.append(metrics["return_percent"])
            rows.append({"row_type": "HOLDOUT_CONFIG", "config_id": identity, "fee_profile": name, **rates, **metrics})
        value = float(np.mean(returns))
        rows.append({"row_type": "HOLDOUT_PORTFOLIO", "fee_profile": name, **rates,
                     "return_percent": value, "start_time": str(start), "end_time": str(end)})
        print(f"FROZEN HOLDOUT {name}: TOP-5={value:.4f}%", flush=True)
    if output is not None:
        write_csv(output, rows)
    return rows


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--download", action="store_true", help="Explicitly ingest BTCUSDT hourly bars, 2018 -> now (includes holdout)")
    mode.add_argument("--final-holdout", action="store_true", help="Evaluate only a previously frozen PASS candidate")
    parser.add_argument("--cache", default=str(data.CACHE_PATH))
    parser.add_argument("--candidate", default=str(CANDIDATE_PATH))
    parser.add_argument("--output", help="CSV path; defaults to strategy_search.csv or strategy_search_holdout.csv")
    parser.add_argument("--as-of", help="Freeze UTC completed-candle cutoff for download/search (ISO-8601)")
    return parser


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.as_of and args.final_holdout:
            raise ValueError("Final holdout uses its frozen cutoff; --as-of is not allowed")
        if args.download:
            data.download_history(cache_path=args.cache, end_time=args.as_of)
        elif args.final_holdout:
            run_final_holdout(cache_path=args.cache, candidate_path=args.candidate,
                              output=args.output or "strategy_search_holdout.csv")
        else:
            run_search(as_of=args.as_of, cache_path=args.cache, candidate_path=args.candidate,
                       output=args.output or "strategy_search.csv")
    except (ValueError, RuntimeError, OSError, sqlite3.Error) as error:
        parser.exit(1, f"STRATEGY SEARCH STOPPED: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
