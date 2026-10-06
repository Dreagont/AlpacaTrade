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
