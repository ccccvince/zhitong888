from __future__ import annotations

from datetime import datetime, time
from typing import Optional

from app.security import beijing_now


class MarketClock:
    """A-share trading session state machine locked to Beijing time (Asia/Shanghai)."""

    def __init__(self) -> None:
        self._trade_dates: set[str] = set()

    def set_trade_dates(self, dates: set[str]) -> None:
        self._trade_dates = dates

    def is_trade_day(self, dt: Optional[datetime] = None) -> bool:
        dt = dt or beijing_now()
        d_str = dt.strftime("%Y-%m-%d")
        if self._trade_dates:
            return d_str in self._trade_dates
        # Fallback to Mon-Fri if calendar has not been fetched yet
        return dt.weekday() < 5

    def session_status(self, dt: Optional[datetime] = None) -> dict[str, str | bool]:
        dt = dt or beijing_now()
        if not self.is_trade_day(dt):
            return {
                "phase": "CLOSED_HOLIDAY",
                "label": "非交易日/休市",
                "can_match": False,
                "is_trading_hours": False,
            }

        t = dt.time()
        if t < time(9, 15):
            return {"phase": "PRE_OPEN", "label": "盘前准备", "can_match": False, "is_trading_hours": False}
        if time(9, 15) <= t < time(9, 25):
            return {"phase": "CALL_AUCTION", "label": "早盘集合竞价", "can_match": False, "is_trading_hours": True}
        if time(9, 25) <= t < time(9, 30):
            return {"phase": "AUCTION_GAP", "label": "待开盘", "can_match": False, "is_trading_hours": True}
        if time(9, 30) <= t < time(11, 30):
            return {"phase": "MORNING_CONTINUOUS", "label": "早盘交易中", "can_match": True, "is_trading_hours": True}
        if time(11, 30) <= t < time(13, 0):
            return {"phase": "LUNCH_BREAK", "label": "午间休市", "can_match": False, "is_trading_hours": False}
        if time(13, 0) <= t < time(14, 57):
            return {"phase": "AFTERNOON_CONTINUOUS", "label": "午盘交易中", "can_match": True, "is_trading_hours": True}
        if time(14, 57) <= t < time(15, 0):
            return {"phase": "CLOSE_AUCTION", "label": "尾盘集合竞价", "can_match": False, "is_trading_hours": True}
        return {"phase": "AFTER_CLOSE", "label": "已收盘", "can_match": False, "is_trading_hours": False}
