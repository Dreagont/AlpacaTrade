python -m venv .venv
.venv\Scripts\activate

pip install alpaca-py python-dotenv pandas numpy

## Research fee profiles and paper reporting

Backtests now default to `binance_spot_bnb`. All five research CLIs accept
`--fee-profile alpaca|binance_spot_bnb|mexc_spot_taker|zero_cost` and include
`fee_profile`, `fee_rate`, and `slippage_rate` in report headers and CSV rows.
Rates are fixed research assumptions as of 2026, based on public fee pages;
slippage is a modeling assumption. Market orders use taker rates.

| Profile | Fee per side | Slippage per side |
| --- | ---: | ---: |
| `alpaca` | 0.25% | 0.05% |
| `binance_spot_bnb` (default) | 0.075% | 0.02% |
| `mexc_spot_taker` | 0.05% | 0.02% |
| `zero_cost` | 0% | 0% |

Sources: [Alpaca crypto fees](https://docs.alpaca.markets/us/docs/crypto-fees),
[Binance BNB fee discount](https://www.binance.com/en/support/faq/detail/115000583311),
[MEXC spot fee calculation](https://www.mexc.co/en-GB/learn/article/mexc-spot-trading-fees-maker-taker-rates-calculator/1).
Promotions, tiers, pairs, and regional schedules can differ from these assumptions.
The existing engine withholds BUY fees in BTC and deducts SELL fees in USD;
for `binance_spot_bnb` this approximates paying fees separately in BNB.
The compatibility constants `BACKTEST_FEE_PERCENT` and
`BACKTEST_SLIPPAGE_PERCENT` derive from the configured default profile.

```bash
python fee_sensitivity.py
python fee_sensitivity.py --blocks 2 --block-days 90 --end-time 2026-10-06T00:00:00Z --csv
python paper_report.py --compare-backtest
python paper_report.py --mark-price 65000 --csv
python paper_report.py --no-mark --csv paper_report.csv
```

`fee_sensitivity.py` uses the same adjacent-block period semantics as
`regime_filter_research.py`: defaults are sixteen 90-day blocks ending at a common
completed-candle boundary, one continuous simulation per profile. It fetches bars
once, then reruns the strategy for every profile, so SL/TP paths can differ.
Its compounded trade return includes any final open position marked at final
close after hypothetical exit fees/slippage; reported costs also include that
hypothetical exit. Trade count is completed round trips, with an open-position
flag shown separately. BTC buy & hold uses the same span and shows both raw and
profile-cost-adjusted returns. CSV output is optional (`--csv [path]`).

`paper_report.py` opens `--db` (default `trading_bot.db`) with SQLite URI `mode=ro`
and `PRAGMA query_only=ON`, without importing `database.py`, `main.py`, or
`broker.py`. It reads BTC BUY/SELL records, sorts recorded timestamps, and pairs
one BUY position at a time in FIFO order; partial SELLs are combined until the
credited BTC is exhausted. It requires recorded asset deltas and strategy/order
role provenance. Leading SELLs are unpaired. Overlapping BUYs, missing deltas,
duplicate identities, tied timestamps, and inconsistent quantities are reported
as ambiguous; ownership is not inferred beyond a ledger gap. Non-filled orders
with execution evidence are also flagged. Unresolved rows are excluded from PnL
totals and listed separately.

The BUY debit is `fill_price * gross filled quantity` (the requested budget is
preserved separately); credited BTC comes from `asset_quantity_delta`.
Recorded PnL uses actual SELL fill prices and quantities minus this debit.
**The current ledger has no recorded SELL quote-fee field, so exact actual net
Alpaca PnL cannot be recovered.** The report explicitly distinguishes recorded
PnL before that unrecorded SELL fee, unavailable actual net PnL, and an Alpaca net
estimate using the fixed `alpaca` taker rate. It does not treat this estimate as
a recorded fee.

For each hypothetical profile, the same actual BUY debit buys
`debit / actual_buy_fill_price * (1 - fee_rate)` BTC. SELL proceeds use actual
SELL fill prices and the same sold fractions of the original credited position,
then deduct only the selected profile's `fee_rate`. **No simulated slippage is
added to real fills.** CSV rows preserve profile slippage as metadata and record
`applied_slippage_rate=0`. An open position uses `--mark-price`, or a read-only
latest-trade market-data fetch, with a hypothetical exit fee for re-pricing.
`--no-mark` skips that fetch and leaves open total PnL unavailable while reporting
realized PnL from any partial exits. Marked values and totals are labeled.
CSV output rejects database paths, filesystem aliases, and SQLite sidecars.

`--compare-backtest` runs the configured live strategy on 4H bars with the selected
fee profile, historical warm-up, and no forced final liquidation. It seeds flat
one candle before the first paper fill's candle so an entry in that first candle
is eligible. The end is the last fill's candle close for closed ledgers or report
time for an open position; history uses completed candles only. It lists paper
and backtest BUY/SELL events side by side, requiring unique same-side matches
within four hours, and flags unmatched events, ambiguous candidates, and strategy
mismatches. Recorded order timestamps can reflect reconciliation rather than
execution time; this and a flat initial state can explain mismatches. Paper
re-pricing itself remains based solely on recorded execution prices.

The golden regression test explicitly pins `alpaca`; its fixture is unchanged.
These commands only perform research/reporting and never submit orders.

## Pre-registered BTC strategy search

```bash
python strategy_search.py --download
python strategy_search.py
# Only after a PASS produced a frozen candidate, and only when requested:
python strategy_search.py --final-holdout
```

This research path uses public Binance `BTCUSDT` 1h klines, with no API key or
trading client. The [Binance public API documentation](https://github.com/binance/binance-spot-api-docs/blob/master/rest-api.md)
describes the kline endpoint. Download starts at 2018-01-01, paginates in batches
of up to 1000, retries temporary errors with bounded backoff, and commits each
page to `data_cache/binance_btcusdt_1h.sqlite`. It prints progress immediately and
resumes at the last cached hour. Missing hours and historical truncated maintenance
candles remain missing; download does not invent OHLC or heal old gaps. UTC 2h/4h
resampling drops buckets without all constituent 1h candles. The cache is ignored
by Git; all read connections use URI `mode=ro`, query-only SQL, and explicit close.

The default search fixes the end to the current completed 4h UTC boundary and
holds back the preceding **six calendar months**. Its SQL query reads only rows
completed by the holdout start, including the cached sample used for equivalence.
`--download` is explicit ingestion of the entire history through now, including
holdout storage; it does not run a search. Optional `--as-of <UTC timestamp>` fixes
the download/search cutoff for reproduction. `--cache`, `--output`, and
`--candidate` select research artifact paths. Output paths cannot overwrite the
trading database, the research cache, or one another.

The frozen enumeration is **936 configurations**: 26 signal parameter pairs,
12 overlay combinations (regime off/on, SL none/3%/6%, max hold none/120h), and
three timeframes (1Hour/2Hour/4Hour). Families and parameters are exactly the
registered trend SMA, MA cross, Donchian, and RSI pullback lists. RSI uses Wilder
smoothing seeded by the first simple average; flat RSI is 50. Donchian channels
exclude the current candle. Frozen 4h regime v1 uses SMA200, slope20, completed
close timestamps, and the same 240-minute stale rule as short-term research.
Missing/unavailable regime forces SELL for gated configs.

The NumPy engine decides at completed candle t and executes at open t+1, using
reference open-exit priority and conservative intrabar stop-before-target behavior.
BUY fees reduce base quantity and SELL fees reduce quote proceeds. Every closed
position fully reinvests into its next entry; the equity curve marks at each candle
close. Numba accelerates the same kernel if installed; it is optional.
Before every search, a fail-closed equivalence gate checks all four families,
all 12 overlay combinations and all three timeframes (144 cases per dataset)
against the unchanged reference backtester. It checks seeded synthetic and cached
real data independently, including identical trade timestamps/reasons and absolute
per-trade return tolerance `1e-9`. The gate uses the long-lookback representative
parameter pair of each family. Missing cached history or a failed gate stops the
search. An Alpaca/Binance 4h-close parity check on a bounded pre-holdout overlap is
informational; venue/currency differences and fetch failures do not affect selection.

Calendar folds start with train `[2019-01-01, 2020-01-01)` and test
`[2020-01-01, 2020-04-01)`, advancing three months until the last full test ends
before holdout. As of 2026-10-06 this gives **25 folds**, ending at 2026-04-01;
unused days before the holdout are not OOS segments. Training windows overlap;
OOS windows do not. Each config runs once continuously per cost scenario, with
2018 indicator warm-up and a flat initial simulation state at 2019-01-01. Indicators
and simulation are causal; selections and window metrics access only their own
prefixes. Positions carry across train/test boundaries, with no forced fold exits.
Completed trades belong to the half-open window containing their exit timestamp;
their average uses the complete round-trip return on notional, including an entry
before the window when applicable. Exposure is sampled on available candle opens.

Train eligibility is exactly >=20 trades, >=1 trade/week, <=5/day, and >=5% invested.
The score is compounded marked return divided by compounded marked max drawdown;
ties prefer more trades, then stable config ID. Zero drawdown has score +infinity
for positive return, -infinity for negative return, or zero for zero return. TOP-5
is equal-weight at each fold start; its fold return is the mean of the five normalized
constituent marked returns. Fold returns compound in chronological order. TOP-1 is
reported alongside it. Continuous constituent positions are valued at the boundary;
this research portfolio does not simulate extra live reallocation orders. No fold
may silently shrink TOP-5 when fewer than five configs qualify; that run cannot PASS.

The random control samples five distinct configs uniformly from each fold's same
eligible set for 1000 paths, seed `20261006`, then compounds each path's OOS returns.
Primary selection uses `binance_spot_bnb`; Alpaca and 2x Binance (both fee and
slippage doubled) rerun full paths while keeping the primary selections fixed.
Benchmarks use the same OOS segments: continuous frozen `regime_only_4h`, and BTC
buy & hold bought once at the first OOS open with a hypothetical costed exit at
the last OOS close. The latter pays costs once on each side, without repurchasing
at every fold. Data quality is reported for each timeframe and each train/test fold.

PASS requires all five registered criteria: TOP-5 OOS return >0, at least random
p95, average completed trade return >0, positive folds >=55%, and return >0 at 2x
Binance costs. Console output shows each criterion, the overall verdict, random
percentiles, benchmarks, and mean selected IS/OOS scores. CSV contains quality,
gate/parity, selections with parameters, portfolio/benchmark rows, random percentiles,
PASS rules, and IS/OOS degradation. No historical result changes these rules.

Only PASS writes `strategy_search_candidate.json`, freezing TOP-5 from the **last
registered fold's training window** plus its original holdout boundaries and protocol
fingerprint. The explicit holdout command requires that candidate, rejects protocol
changes, reads only through its frozen holdout end, evaluates those five configs,
and performs no new selection. Holdout is never run automatically. No live files,
strategy registry entries, orders, or trading database writes are part of this search.

Verification (all automated tests offline):

```bash
python -m compileall -q *.py tests
python -m unittest discover -s tests -t . -v
python strategy_search.py --help
```

`.github/workflows/research-tests.yml` runs the offline suite on Ubuntu and Windows
with Python 3.12. It performs no data download, strategy search, holdout evaluation,
or live trading.
