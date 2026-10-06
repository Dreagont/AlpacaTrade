SYMBOL = "BTC/USD"

DATABASE_PATH = "trading_bot.db"

BACKTEST_STARTING_CAPITAL = 1000.0
BACKTEST_LOOKBACK_DAYS = 30
# As of 2026, public fee pages: fixed research assumptions for market-order takers.
# https://docs.alpaca.markets/us/docs/crypto-fees
# https://www.binance.com/en/support/faq/detail/115000583311
# https://www.mexc.co/en-GB/learn/article/mexc-spot-trading-fees-maker-taker-rates-calculator/1
# Promotions, tiers and regional rates can differ. Slippage is a modeling assumption.
FEE_PROFILES = {
    "alpaca":           {"fee_rate": 0.0025,  "slippage_rate": 0.0005},
    "binance_spot_bnb": {"fee_rate": 0.00075, "slippage_rate": 0.0002},
    "mexc_spot_taker":  {"fee_rate": 0.0005,  "slippage_rate": 0.0002},
    "zero_cost":        {"fee_rate": 0.0,     "slippage_rate": 0.0},
}
BACKTEST_FEE_PROFILE = "binance_spot_bnb"


def get_fee_profile(name):
    """Return a copy so callers cannot accidentally change the shared schedule."""
    if name not in FEE_PROFILES:
        raise ValueError(
            f"Unknown fee profile {name!r}. Valid names: {', '.join(FEE_PROFILES)}"
        )
    return dict(FEE_PROFILES[name])


# Compatibility aliases, derived from the selected research profile.
BACKTEST_FEE_PERCENT = get_fee_profile(BACKTEST_FEE_PROFILE)["fee_rate"]
BACKTEST_SLIPPAGE_PERCENT = get_fee_profile(BACKTEST_FEE_PROFILE)["slippage_rate"]


def add_fee_profile_argument(parser):
    parser.add_argument(
        "--fee-profile", choices=tuple(FEE_PROFILES), default=BACKTEST_FEE_PROFILE,
        help="Market-order taker fee/slippage assumptions (default: %(default)s)",
    )


def print_fee_profile(name, fee_rate=None, slippage_rate=None):
    rates = get_fee_profile(name) if name != "custom" else {}
    fee_rate = rates.get("fee_rate") if fee_rate is None else fee_rate
    slippage_rate = rates.get("slippage_rate") if slippage_rate is None else slippage_rate
    print(f"fee_profile={name} | fee_rate={fee_rate:g} | slippage_rate={slippage_rate:g}")
    print("BUY fee withheld in base asset; SELL fee deducted from quote proceeds. "
          "For binance_spot_bnb this approximates fees paid separately in BNB.")
BACKTEST_SAVE_TRADES_CSV = True
BACKTEST_ZERO_COST_DIAGNOSTIC = False
