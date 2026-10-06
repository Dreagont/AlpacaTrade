# Fixed daily multi-asset research

Run from the repository root:

```sh
python multi_asset_research.py --download
python multi_asset_research.py --cost-sensitivity
python multi_asset_research.py --no-btc --end-date 2025-12-31 --csv multi_asset_results.csv
```

This research does not import a broker, change live configuration, or place orders.
Only the fixed `sma200_slope20` primary and `sma200` secondary rules are evaluated.
The secondary rule and ETF cost sensitivity do not select or promote a rule.
The BTC rule was chosen after inspecting 2021–2026 BTC data; these ETF results
are the first look at those assets. Cash earns 0%, which omits interest available
to a strategy holding cash.

## Data and execution

SPY, TLT, GLD and UUP use `StockHistoricalDataClient`, the existing `.env` keys,
`StockBarsRequest(timeframe=TimeFrame.Day, adjustment=Adjustment.ALL, feed=DataFeed.SIP)`
and a fixed 2016-01-01 start. Alpaca adjusts all OHLC fields; prices are used once
and dividends are not credited separately. SIP entitlement rejection alone triggers
an explicitly reported IEX fallback. Authentication and other failures stop the run.
Source: [Alpaca request documentation](https://alpaca.markets/sdks/python/api_reference/data/stock/requests.html).

CSV price caches and JSON provenance live under the already ignored
`data_cache/multi_asset/`. Loading refuses caches without `adjustment=all` and a
known source feed. `--download` refreshes the full adjusted ETF history, rather than
appending differently adjusted vintages. A cache request ending before the requested
cutoff produces a refresh warning. Actual covered spans always appear in the output.

Today is excluded before 16:00 America/New_York, with daylight saving time respected;
`--end-date` is inclusive and cannot authorize future/uncompleted sessions. The US
reference calendar is SPY's completed dates, so weekends and holidays are not missing
observations. Diagnostics include missing dates within the full SPY span, including
an asset's unavailable early history. SPY cannot identify missing sessions in its own
source history; use an exchange calendar independently if that diagnostic is needed.

BTCUSDT daily candles aggregate the existing `binance_data` hourly cache/download.
All 24 unique UTC hours are required; no incomplete day is synthesized. For US date D,
use the latest complete BTC candle with close time <= D at 16:00 America/New_York.
If the most recent UTC candle is unavailable, the latest older completed daily bar
remains eligible, as registered. Execution uses an observed UTC midnight open from
the hourly cache, including days with an incomplete set of later hourly observations;
an unobserved opening price stays missing. BTC signals apply the fixed SMA rule to this US-calendar
close series, using 220 preceding US observations.

The user-approved BTC execution convention fills at the first UTC daily open strictly
after the signal's US close. A Friday signal can fill at Saturday 00:00 UTC. Using the
open of the completed BTC candle mapped to Monday would fill before the Friday signal;
this implementation avoids that temporal inversion. BTC is marked at the latest
completed daily close available on each US session. ETF decisions at close t fill at
the next observed session's open; missing rows are never filled with invented prices.
The first decision follows 220 observations, and metrics begin at its next opening fill.

The fast daily engine must first match the unchanged `backtest.run_backtest` on synthetic
daily data with gaps and a real adjusted SPY sample. Each dataset has 18 cases:
both fixed rules plus a deterministic execution schedule, each at three cost schedules
and with/without final liquidation. Entry/exit times and reasons must match exactly;
per-trade returns must match within absolute 1e-9. A mismatch stops research. An
additional production check matches each asset's reinvested curve to sleeve accounting.

## Portfolio accounting

ETF_EQUAL has four 25% sleeves; ETF_BTC_EQUAL has five 20% sleeves. Each sleeve applies
its rule independently. Uninvested sleeve capital remains cash without redistribution.
Portfolio reporting starts only once every sleeve has 220 prior observations and a
subsequent execution date. Common available session dates are used; excluded sessions
are reported, and the signals are shifted on each asset's original calendar before
selecting the common dates. Do not interpret periods with excluded sessions as a
complete daily path.

ETF portfolios rebalance sleeve capital at the first available common US opening
session of each month. Capital transfers sum to zero. An invested donor sleeve sells
enough to fund its transfer including sale costs; an invested recipient buys with its
incoming capital including purchase costs. Transfers between cash sleeves incur no
asset turnover. Capital is equalized before transaction costs; costs and opening gaps
can leave small differences afterward.

BTC's approved fill can precede the ETF opening session. For mixed portfolios, monthly
USD transfers are therefore fixed from the preceding US closing sleeve capital and
executed at the sleeves' respective next opens. This prevents the earlier BTC fill's
size from using later ETF opening prices. Opening gaps mean mixed sleeve weights can
differ from equal weights after execution. This timing convention applies identically
to the mixed strategy and its equal-weight buy-and-hold benchmark. It is an explicit
causal approximation to simultaneous equal-weight rebalancing across these calendars.

ETF entry/exit commission is zero, with 0.02% spread/slippage per side. With
`--cost-sensitivity`, repeat the same reports at 0.05%. BTC always uses the existing
`binance_spot_bnb` profile (fee 0.075%, slippage 0.02% per side). As in the reference
engine, buy fees reduce received asset quantity and sell fees reduce cash proceeds;
this approximates BNB-paid fees. Both signal changes and actual rebalance trades pay
costs. All curves finish marked to close, without an artificial terminal sale.

Each asset includes buy-and-hold on the same execution dates. Each portfolio includes
buy-and-hold of its own assets, SPY buy-and-hold and monthly-rebalanced 60/40 SPY/TLT
on exactly that portfolio's dates. ETF_EQUAL is reported over its full span and from
ETF_BTC_EQUAL's start date and on its exact common dates when BTC is included.

## Reports and verification

Reports contain total return, CAGR, volatility, Sharpe, compounded maximum drawdown,
Calmar, time invested, entry trades per year, worst calendar year, calendar-year returns,
and the fixed 2016–2019, 2020–2021, 2022 and 2023–latest periods. Empty periods are N/A;
actual covered dates identify partial periods. Endpoint calendar years can be partial.

The initial capital baseline is included in returns and drawdowns, including the first
entry cost. CAGR and trades per year use inclusive elapsed calendar days divided by
365.25. Daily volatility and Sharpe use sample standard deviation (`ddof=1`), sqrt(252),
and a 0% risk-free rate. Calmar is CAGR / positive maximum drawdown. Sharpe and Calmar
are N/A for zero variance or zero drawdown respectively. Portfolio time invested is
the average target-weight fraction of invested sleeves. Trades/year counts entries,
including an open final position, and excludes monthly resizing; CSV additionally
reports actual market-notional turnover and costs relative to initial capital.

`--csv` writes a single file with a `section` column for metrics, calendar years,
fixed periods, data quality and equivalence results. It cannot overwrite the trading
database, its aliases or sidecars, or write into the source cache directory.

```sh
python -m compileall -q *.py tests
python -m unittest discover -s tests -t . -v
python multi_asset_research.py --help
```

CI executes offline tests on Ubuntu and Windows with Python 3.12. Its directory-form
compile command also works with PowerShell's wildcard handling. Offline data clients
are mocked; tests never require credentials, download data, or submit orders.
