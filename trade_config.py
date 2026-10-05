"""Settings that define the live trading strategy and its risk limits."""

TRADE_AMOUNT_USD = 20
MAX_POSITION_USD = 100

FAST_MA = 10
SLOW_MA = 30
RSI_PERIOD = 14
RSI_BUY_THRESHOLD = 70

CHECK_INTERVAL_SECONDS = 60
LIVE_TIMEFRAME = "5Min"
LIVE_STRATEGY = "ma_rsi_crossover"

STOP_LOSS_PERCENT = 0.02
TAKE_PROFIT_PERCENT = 0.04

# Broker-side stop-limit is an optional safety net and is disabled until enabled
# deliberately after paper validation. It is not guaranteed execution.
ENABLE_BROKER_STOP_LIMIT = False
BROKER_STOP_LIMIT_OFFSET_PERCENT = 0.005
BROKER_STOP_LIMIT_PRICE_INCREMENT = 1.0

PAPER_SMOKE_TEST_AMOUNT_USD = 20.0
