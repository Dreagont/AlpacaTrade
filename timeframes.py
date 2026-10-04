import re

from alpaca.data.timeframe import TimeFrame, TimeFrameUnit


def parse_timeframe(value: str) -> tuple[TimeFrame, float]:
    """Parse Alpaca-style timeframes and return the frame plus minutes per bar."""
    match = re.fullmatch(r"(\d+)(Min|Hour|Day|Week|Month)", value)
    if not match:
        raise ValueError("Timeframe must look like 5Min, 1Hour, or 1Day")

    amount = int(match.group(1))
    if amount <= 0:
        raise ValueError("Timeframe amount must be positive")

    units = {
        "Min": (TimeFrameUnit.Minute, 1),
        "Hour": (TimeFrameUnit.Hour, 60),
        "Day": (TimeFrameUnit.Day, 24 * 60),
        "Week": (TimeFrameUnit.Week, 7 * 24 * 60),
        "Month": (TimeFrameUnit.Month, 30.436875 * 24 * 60),
    }
    unit, minutes_per_unit = units[match.group(2)]
    return TimeFrame(amount, unit), amount * minutes_per_unit
