from __future__ import annotations

import codecs
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
from typing import Optional

import adata
import httpx

from app.engine.paper_broker import infer_limit_prices
from app.models import (
    LimitUpScreenResponse,
    LimitUpStockItem,
    MarketIndexSnapshot,
    MarketMode,
    MinuteBar,
    OrderBookLevel,
    QuoteSnapshot,
)
from app.security import beijing_now
from app.storage import now_str

# Built-in reference quotes so simulation & offline unit tests always have realistic baselines.
POPULAR_STOCKS: dict[str, dict[str, str | float]] = {
    "600519": {"name": "贵州茅台", "price": 1445.00, "pre_close": 1453.90, "turnover": 0.13, "pe": 21.91},
    "300750": {"name": "宁德时代", "price": 265.40, "pre_close": 263.80, "turnover": 0.78, "pe": 24.60},
    "000001": {"name": "平安银行", "price": 12.17, "pre_close": 12.30, "turnover": 0.53, "pe": 5.43},
    "002594": {"name": "比亚迪", "price": 288.50, "pre_close": 286.00, "turnover": 0.92, "pe": 25.10},
    "600036": {"name": "招商银行", "price": 38.45, "pre_close": 38.20, "turnover": 0.36, "pe": 6.45},
    "000858": {"name": "五粮液", "price": 146.80, "pre_close": 147.50, "turnover": 0.44, "pe": 17.80},
    "601318": {"name": "中国平安", "price": 56.20, "pre_close": 55.90, "turnover": 0.58, "pe": 8.95},
    "688981": {"name": "中芯国际", "price": 95.30, "pre_close": 94.10, "turnover": 2.45, "pe": 118.40},
    "300059": {"name": "东方财富", "price": 26.85, "pre_close": 26.40, "turnover": 3.15, "pe": 42.30},
    "601138": {"name": "工业富联", "price": 24.60, "pre_close": 24.35, "turnover": 1.12, "pe": 21.40},
}

INDEX_DEFINITIONS: list[dict[str, str | float]] = [
    {"symbol": "sh000001", "code": "000001", "name": "上证指数", "seed_price": 3602.07, "seed_pre_close": 3602.57, "seed_amount": 424_293_166_077.0},
    {"symbol": "sz399001", "code": "399001", "name": "深证成指", "seed_price": 10764.88, "seed_pre_close": 10690.78, "seed_amount": 470_329_539_263.0},
    {"symbol": "sz399006", "code": "399006", "name": "创业板指", "seed_price": 2336.80, "seed_pre_close": 2320.02, "seed_amount": 222_087_640_341.0},
    {"symbol": "sh000688", "code": "000688", "name": "科创50", "seed_price": 1253.91, "seed_pre_close": 1213.68, "seed_amount": 63_363_620_610.0},
    {"symbol": "sh000300", "code": "000300", "name": "沪深300", "seed_price": 3981.91, "seed_pre_close": 3979.80, "seed_amount": 239_539_869_538.0},
]
INDEX_BY_SYMBOL: dict[str, dict[str, str | float]] = {str(d["symbol"]): d for d in INDEX_DEFINITIONS}


def to_tencent_symbol(code: str) -> str:
    if code.startswith(("6", "5", "900")):
        return f"sh{code}"
    if code.startswith(("8", "4", "92")):
        return f"bj{code}"
    return f"sz{code}"


def classify_board(stock_code: str) -> str:
    """Classify an A-share code into 主板 / 创业板 / 科创板 / 北交所."""
    if stock_code.startswith(("300", "301")):
        return "创业板"
    if stock_code.startswith(("688", "689")):
        return "科创板"
    if stock_code.startswith(("8", "4", "92")):
        return "北交所"
    return "主板"


def is_limit_up_bar(stock_code: str, close_price: float, pre_close: float, short_name: str = "") -> bool:
    """Return True when daily close reaches A-share daily limit-up threshold (supports 5% ST, 10%, 20%, 30%)."""
    if close_price <= 0 or pre_close <= 0:
        return False
    board = classify_board(stock_code)
    upper_name = short_name.upper()
    if board == "主板" and ("ST" in upper_name or "退" in upper_name):
        limit_up = int(pre_close * 1.05 * 100 + 0.5 + 1e-9) / 100.0
    else:
        limit_up, _ = infer_limit_prices(stock_code, pre_close)
    return close_price >= (limit_up - 0.005)


def parse_eastmoney_zt_pool_item(row: dict, trade_date_iso: str) -> Optional[dict]:
    """Parse one item from Eastmoney getTopicZTPool response."""
    code = str(row.get("c") or "").strip()
    if len(code) != 6 or not code.isdigit():
        return None
    raw_price = _safe_float(str(row.get("p") or "0"))
    # Eastmoney getTopicZTPool encodes p in milli-yuan (e.g. 7130 -> 7.13)
    price = round(raw_price / 1000.0, 2) if raw_price >= 100 else round(raw_price, 2)
    return {
        "stock_code": code,
        "short_name": str(row.get("n") or code).strip() or code,
        "trade_date": trade_date_iso,
        "price": price,
        "change_pct": round(_safe_float(str(row.get("zdp") or "0")), 2),
        "amount": round(_safe_float(str(row.get("amount") or "0")), 2),
        "turnover_ratio": round(_safe_float(str(row.get("hs") or "0")), 2),
        "lbc": max(1, _safe_int(str(row.get("lbc") or "1"), 1)),
        "industry": str(row.get("hybk") or "").strip(),
    }


def _safe_float(val: str, default: float = 0.0) -> float:
    try:
        return float(val) if val not in ("", None, "-", "nan") else default
    except (ValueError, TypeError):
        return default


def _safe_int(val: str, default: int = 0) -> int:
    try:
        return int(float(val)) if val not in ("", None, "-", "nan") else default
    except (ValueError, TypeError):
        return default


def _split_tencent_lines(text: str) -> list[tuple[str, list[str]]]:
    rows: list[tuple[str, list[str]]] = []
    for line in text.strip().splitlines():
        if "~" not in line or '="' not in line:
            continue
        lhs, body = line.split('="', 1)
        symbol = lhs.strip().removeprefix("v_")
        body = body.rstrip('"; \r\n')
        rows.append((symbol, body.split("~")))
    return rows


def _parse_tencent_amount(parts: list[str]) -> float:
    amount = 0.0
    if len(parts) > 35 and parts[35] and "/" in parts[35]:
        seg = parts[35].split("/")
        if len(seg) >= 3:
            amount = _safe_float(seg[2])
    if amount <= 0 and len(parts) > 37:
        amount = _safe_float(parts[37]) * 10_000.0
    return round(amount, 2)


def compute_sma_series(closes: list[float], period: int, decimals: int = 2) -> list[Optional[float]]:
    """Compute rolling simple moving average aligned with closes."""
    if period <= 0:
        return [None] * len(closes)
    out: list[Optional[float]] = []
    running = 0.0
    for idx, close in enumerate(closes):
        val = float(close)
        running += val
        if idx >= period:
            running -= float(closes[idx - period])
        if idx >= period - 1:
            out.append(round(running / period, decimals))
        else:
            out.append(None)
    return out


def compute_macd_series(
    closes: list[float], fast_period: int = 12, slow_period: int = 26, signal_period: int = 9
) -> list[tuple[float, float, float]]:
    """Compute (DIF, DEA, MACD histogram) aligned with closes."""
    if not closes:
        return []
    k_fast = 2.0 / (fast_period + 1)
    k_slow = 2.0 / (slow_period + 1)
    k_sig = 2.0 / (signal_period + 1)
    ema_fast = float(closes[0])
    ema_slow = float(closes[0])
    dea = 0.0
    out: list[tuple[float, float, float]] = []

    for idx, raw_c in enumerate(closes):
        close = float(raw_c)
        if idx == 0:
            dif = 0.0
            dea = 0.0
        else:
            ema_fast = ema_fast * (1.0 - k_fast) + close * k_fast
            ema_slow = ema_slow * (1.0 - k_slow) + close * k_slow
            dif = ema_fast - ema_slow
            dea = dea * (1.0 - k_sig) + dif * k_sig
        macd_val = 2.0 * (dif - dea)
        out.append((round(dif, 3), round(dea, 3), round(macd_val, 3)))
    return out


def compute_kline_indicators(bars: list[dict]) -> list[dict]:
    """Compute MA5/MA10/MA20 and MACD(12, 26, 9) in chronological order."""
    if not bars:
        return []
    ordered = sorted(bars, key=lambda b: str(b["trade_date"]))
    closes = [float(b["close"]) for b in ordered]
    ma5_series = compute_sma_series(closes, 5, decimals=2)
    ma10_series = compute_sma_series(closes, 10, decimals=2)
    ma20_series = compute_sma_series(closes, 20, decimals=2)
    macd_series = compute_macd_series(closes, 12, 26, 9)

    result: list[dict] = []
    for idx, bar in enumerate(ordered):
        dif, dea, macd_val = macd_series[idx]
        enriched = dict(bar)
        enriched["ma5"] = ma5_series[idx]
        enriched["ma10"] = ma10_series[idx]
        enriched["ma20"] = ma20_series[idx]
        enriched["dif"] = dif
        enriched["dea"] = dea
        enriched["macd"] = macd_val
        result.append(enriched)

    return result


def parse_tencent_indices(text: str) -> dict[str, MarketIndexSnapshot]:
    """Parse major A-share index lines (keyed by full market symbol like sh000001) without 6-digit code collision."""
    result: dict[str, MarketIndexSnapshot] = {}
    ts = now_str()

    for symbol, parts in _split_tencent_lines(text):
        meta = INDEX_BY_SYMBOL.get(symbol)
        if not meta or len(parts) < 38:
            continue

        price = _safe_float(parts[3])
        pre_close = _safe_float(parts[4])
        if price <= 0 and pre_close > 0:
            price = pre_close
        change = _safe_float(parts[31], round(price - pre_close, 2) if pre_close > 0 else 0.0)
        change_pct = _safe_float(
            parts[32],
            round((change / pre_close) * 100, 2) if pre_close > 0 else 0.0,
        )
        result[symbol] = MarketIndexSnapshot(
            symbol=symbol,
            code=str(meta["code"]),
            name=parts[1].strip() or str(meta["name"]),
            price=round(price, 2),
            pre_close=round(pre_close, 2),
            change=round(change, 2),
            change_pct=round(change_pct, 2),
            amount=_parse_tencent_amount(parts),
            updated_at=ts,
        )
    return result


def parse_tencent_payload(text: str) -> dict[str, QuoteSnapshot]:
    """Parse batch response from https://qt.gtimg.cn/q=... into QuoteSnapshot objects."""
    result: dict[str, QuoteSnapshot] = {}
    ts = now_str()

    for symbol, parts in _split_tencent_lines(text):
        # Prevent sh000001 (上证指数) from overwriting sz000001 (平安银行) when payloads mix
        if symbol in INDEX_BY_SYMBOL or len(parts) < 40:
            continue

        short_name = parts[1].strip()
        stock_code = parts[2].strip()
        if not stock_code or len(stock_code) != 6:
            continue

        price = _safe_float(parts[3])
        pre_close = _safe_float(parts[4])
        open_p = _safe_float(parts[5])
        if price <= 0 and pre_close > 0:
            price = pre_close
        if open_p <= 0 and price > 0:
            open_p = pre_close if pre_close > 0 else price

        vol_lots = _safe_float(parts[6])
        volume_shares = vol_lots * 100.0

        high_raw = _safe_float(parts[33], price)
        low_raw = _safe_float(parts[34], price)
        high_p = high_raw if high_raw > 0 else price
        low_p = low_raw if low_raw > 0 else price
        change = _safe_float(parts[31], round(price - pre_close, 2) if pre_close > 0 else 0.0)
        change_pct = _safe_float(
            parts[32],
            round((change / pre_close) * 100, 2) if pre_close > 0 else 0.0,
        )
        amount = _parse_tencent_amount(parts)

        turnover_ratio = round(_safe_float(parts[38]), 2) if len(parts) > 38 else 0.0
        pe_ratio = round(_safe_float(parts[39]), 2) if len(parts) > 39 else 0.0
        amplitude = round(_safe_float(parts[43]), 2) if len(parts) > 43 else 0.0
        if amplitude <= 0 and pre_close > 0 and high_p >= low_p > 0:
            amplitude = round(((high_p - low_p) / pre_close) * 100, 2)

        limit_up = _safe_float(parts[47]) if len(parts) > 47 else 0.0
        limit_down = _safe_float(parts[48]) if len(parts) > 48 else 0.0
        if (limit_up <= 0 or limit_down <= 0) and pre_close > 0:
            limit_up, limit_down = infer_limit_prices(stock_code, pre_close)

        bids: list[OrderBookLevel] = []
        for idx in range(5):
            bp = _safe_float(parts[9 + idx * 2])
            bv_lots = _safe_int(parts[10 + idx * 2])
            if bp > 0 and bv_lots > 0:
                bids.append(OrderBookLevel(price=round(bp, 2), volume=bv_lots * 100))

        asks: list[OrderBookLevel] = []
        for idx in range(5):
            ap = _safe_float(parts[19 + idx * 2])
            av_lots = _safe_int(parts[20 + idx * 2])
            if ap > 0 and av_lots > 0:
                asks.append(OrderBookLevel(price=round(ap, 2), volume=av_lots * 100))

        # Preserve sealed limit-up/down indicator if book side is empty at limit price
        if not asks and limit_up > 0 and price >= limit_up - 0.005:
            asks = [OrderBookLevel(price=limit_up, volume=0)]
        if not bids and limit_down > 0 and price <= limit_down + 0.005:
            bids = [OrderBookLevel(price=limit_down, volume=0)]

        result[stock_code] = QuoteSnapshot(
            stock_code=stock_code,
            short_name=short_name or stock_code,
            price=round(price, 2),
            pre_close=round(pre_close, 2),
            open=round(open_p, 2),
            high=round(high_p, 2),
            low=round(low_p, 2),
            change=round(change, 2),
            change_pct=round(change_pct, 2),
            volume=volume_shares,
            amount=amount,
            turnover_ratio=turnover_ratio,
            pe_ratio=pe_ratio,
            amplitude=amplitude,
            limit_up=round(limit_up, 2),
            limit_down=round(limit_down, 2),
            asks=asks,
            bids=bids,
            source="tencent_qt+adata",
            updated_at=ts,
        )
    return result


class MarketDataManager:
    """Fetches real A-share quotes (Tencent batch + adata) and generates simulated ticks for 24/7 testing."""

    def __init__(self, network_enabled: bool = True) -> None:
        self.network_enabled = network_enabled
        self._lock = threading.RLock()
        self._quotes: dict[str, QuoteSnapshot] = {}
        self._sim_quotes: dict[str, QuoteSnapshot] = {}
        self._indices: dict[str, MarketIndexSnapshot] = {}
        self._sim_indices: dict[str, MarketIndexSnapshot] = {}
        self._sim_mins: dict[str, list[MinuteBar]] = {}
        self._daily_cache: dict[str, tuple[float, int, list[dict]]] = {}
        self._min_cache: dict[str, tuple[float, list[MinuteBar]]] = {}
        self._zt_pool_cache: dict[str, tuple[float, list[dict]]] = {}
        self._trade_cal_cache: tuple[float, list[str]] | None = None
        self._concept_cache: dict[str, tuple[float, list[str]]] = {}

    def get_cached_quotes(self, mode: MarketMode = MarketMode.REAL) -> dict[str, QuoteSnapshot]:
        with self._lock:
            if mode == MarketMode.SIMULATED:
                merged = dict(self._quotes)
                merged.update(self._sim_quotes)
                return merged
            return dict(self._quotes)

    def get_cached_indices(self, mode: MarketMode = MarketMode.REAL) -> list[MarketIndexSnapshot]:
        with self._lock:
            src = self._sim_indices if (mode == MarketMode.SIMULATED and self._sim_indices) else self._indices
            return [
                src[str(d["symbol"])]
                for d in INDEX_DEFINITIONS
                if str(d["symbol"]) in src
            ]

    def get_quote(self, stock_code: str, mode: MarketMode = MarketMode.REAL) -> Optional[QuoteSnapshot]:
        with self._lock:
            if mode == MarketMode.SIMULATED and stock_code in self._sim_quotes:
                return self._sim_quotes.get(stock_code)
            return self._quotes.get(stock_code)

    @staticmethod
    def _build_seed_index(meta: dict[str, str | float]) -> MarketIndexSnapshot:
        price = float(meta["seed_price"])
        pre_close = float(meta["seed_pre_close"])
        change = round(price - pre_close, 2)
        change_pct = round((change / pre_close) * 100, 2) if pre_close > 0 else 0.0
        return MarketIndexSnapshot(
            symbol=str(meta["symbol"]),
            code=str(meta["code"]),
            name=str(meta["name"]),
            price=round(price, 2),
            pre_close=round(pre_close, 2),
            change=change,
            change_pct=change_pct,
            amount=float(meta["seed_amount"]),
            updated_at=now_str(),
        )

    def fetch_indices(self, mode: MarketMode = MarketMode.REAL) -> list[MarketIndexSnapshot]:
        """Fetch major A-share indices (上证指数/深证成指/创业板指/科创50/沪深300) or step gently in SIMULATED mode."""
        with self._lock:
            need_real = (mode == MarketMode.REAL) or (len(self._indices) < len(INDEX_DEFINITIONS))

        fetched: dict[str, MarketIndexSnapshot] = {}
        if self.network_enabled and need_real:
            symbols = ",".join(str(d["symbol"]) for d in INDEX_DEFINITIONS)
            try:
                resp = httpx.get(
                    f"https://qt.gtimg.cn/q={symbols}",
                    headers={"Referer": "https://gu.qq.com/"},
                    timeout=4.0,
                )
                if resp.status_code == 200:
                    text = resp.content.decode("gbk", errors="ignore")
                    fetched = parse_tencent_indices(text)
            except Exception:
                fetched = {}

        now_full = now_str()
        with self._lock:
            for meta in INDEX_DEFINITIONS:
                sym = str(meta["symbol"])
                if sym in fetched:
                    self._indices[sym] = fetched[sym]
                elif sym not in self._indices:
                    self._indices[sym] = self._build_seed_index(meta)

                if mode == MarketMode.SIMULATED:
                    prev = self._sim_indices.get(sym) or self._indices[sym]
                    step_pct = random.gauss(0.0, 0.0012)
                    next_p = round(max(100.0, prev.price * (1.0 + step_pct)), 2)
                    chg = round(next_p - prev.pre_close, 2)
                    chg_pct = round((chg / prev.pre_close) * 100, 2) if prev.pre_close > 0 else 0.0
                    self._sim_indices[sym] = MarketIndexSnapshot(
                        symbol=prev.symbol,
                        code=prev.code,
                        name=prev.name,
                        price=next_p,
                        pre_close=prev.pre_close,
                        change=chg,
                        change_pct=chg_pct,
                        amount=round(prev.amount + random.uniform(2e7, 1.5e8), 2),
                        updated_at=now_full,
                    )

            return self.get_cached_indices(mode)

    def _build_seed_quote(self, code: str, short_name: str = "") -> QuoteSnapshot:
        seed = POPULAR_STOCKS.get(code)
        base_price = float(seed["price"]) if seed else 20.00
        pre_close = float(seed["pre_close"]) if seed else 19.80
        name = short_name or (str(seed["name"]) if seed else f"A股{code}")
        lu, ld = infer_limit_prices(code, pre_close)
        change = round(base_price - pre_close, 2)
        change_pct = round((change / pre_close) * 100, 2) if pre_close > 0 else 0.0
        high_p = round(max(base_price, pre_close * 1.005), 2)
        low_p = round(min(base_price, pre_close * 0.995), 2)
        amplitude = round(((high_p - low_p) / pre_close) * 100, 2) if pre_close > 0 else 0.0
        turnover_ratio = float(seed.get("turnover", 0.75)) if seed else 0.85
        pe_ratio = float(seed.get("pe", 18.50)) if seed else 18.50
        asks, bids = self._build_order_book(base_price, lu, ld)
        return QuoteSnapshot(
            stock_code=code,
            short_name=name,
            price=round(base_price, 2),
            pre_close=round(pre_close, 2),
            open=round(pre_close * 1.002, 2),
            high=high_p,
            low=low_p,
            change=change,
            change_pct=change_pct,
            volume=1_250_000.0,
            amount=round(base_price * 1_250_000.0, 2),
            turnover_ratio=round(turnover_ratio, 2),
            pe_ratio=round(pe_ratio, 2),
            amplitude=amplitude,
            limit_up=lu,
            limit_down=ld,
            asks=asks,
            bids=bids,
            source="seed_baseline",
            updated_at=now_str(),
        )

    @staticmethod
    def _build_order_book(
        price: float, limit_up: float, limit_down: float
    ) -> tuple[list[OrderBookLevel], list[OrderBookLevel]]:
        tick = 0.01
        asks: list[OrderBookLevel] = []
        bids: list[OrderBookLevel] = []

        if limit_up > 0 and price >= limit_up - 0.005:
            asks = [OrderBookLevel(price=limit_up, volume=0)]
        else:
            for i in range(5):
                ap = round(price + i * tick, 2)
                if limit_up > 0 and ap > limit_up:
                    break
                asks.append(OrderBookLevel(price=ap, volume=random.randint(3, 80) * 100))

        if limit_down > 0 and price <= limit_down + 0.005:
            bids = [OrderBookLevel(price=limit_down, volume=0)]
        else:
            for i in range(5):
                bp = round(price - (i + 1) * tick, 2)
                if bp <= 0 or (limit_down > 0 and bp < limit_down):
                    break
                bids.append(OrderBookLevel(price=bp, volume=random.randint(3, 80) * 100))

        return asks, bids

    def fetch_real_quotes(self, stock_items: list[tuple[str, str]]) -> dict[str, QuoteSnapshot]:
        """Fetch live quotes for [(code, short_name), ...]; falls back gracefully if offline."""
        if not stock_items:
            return {}
        codes = [c for c, _ in stock_items]
        names_map = {c: n for c, n in stock_items}
        fetched: dict[str, QuoteSnapshot] = {}

        if self.network_enabled:
            symbols = ",".join(to_tencent_symbol(c) for c in codes)
            try:
                resp = httpx.get(
                    f"https://qt.gtimg.cn/q={symbols}",
                    headers={"Referer": "https://gu.qq.com/"},
                    timeout=4.0,
                )
                if resp.status_code == 200:
                    text = resp.content.decode("gbk", errors="ignore")
                    fetched = parse_tencent_payload(text)
            except Exception:
                fetched = {}

            # Try adata 5-level fallback only for missing codes (capped cleanly)
            missing = [c for c in codes if c not in fetched]
            for mc in missing[:3]:
                try:
                    df5 = adata.stock.market.get_market_five(mc)
                    if df5 is not None and not df5.empty:
                        row = df5.iloc[0]
                        asks = [
                            OrderBookLevel(price=round(float(row[f"s{i}"]), 2), volume=int(float(row[f"sv{i}"])))
                            for i in range(1, 6)
                            if _safe_float(str(row.get(f"s{i}", 0))) > 0 and _safe_int(str(row.get(f"sv{i}", 0))) > 0
                        ]
                        bids = [
                            OrderBookLevel(price=round(float(row[f"b{i}"]), 2), volume=int(float(row[f"bv{i}"])))
                            for i in range(1, 6)
                            if _safe_float(str(row.get(f"b{i}", 0))) > 0 and _safe_int(str(row.get(f"bv{i}", 0))) > 0
                        ]
                        ref_p = bids[0].price if bids else (asks[0].price if asks else 10.0)
                        lu, ld = infer_limit_prices(mc, ref_p)
                        fetched[mc] = QuoteSnapshot(
                            stock_code=mc,
                            short_name=str(row.get("short_name") or names_map.get(mc) or mc),
                            price=ref_p,
                            pre_close=ref_p,
                            open=ref_p,
                            high=ref_p,
                            low=ref_p,
                            limit_up=lu,
                            limit_down=ld,
                            asks=asks,
                            bids=bids,
                            source="adata_five",
                            updated_at=now_str(),
                        )
                except Exception:
                    pass

        with self._lock:
            for code, sname in stock_items:
                if code in fetched:
                    self._quotes[code] = fetched[code]
                elif code not in self._quotes:
                    self._quotes[code] = self._build_seed_quote(code, sname)
            return {c: self._quotes[c] for c, _ in stock_items if c in self._quotes}

    def step_simulated_quotes(self, stock_items: list[tuple[str, str]]) -> dict[str, QuoteSnapshot]:
        """Advance simulated quotes independently from real market cache for 24/7 paper-trading."""
        with self._lock:
            need_init = [item for item in stock_items if item[0] not in self._quotes and item[0] not in self._sim_quotes]
        if need_init:
            self.fetch_real_quotes(need_init)

        now_bj = beijing_now()
        now_label = now_bj.strftime("%H:%M:%S")
        now_full = now_bj.strftime("%Y-%m-%d %H:%M:%S")
        updated: dict[str, QuoteSnapshot] = {}

        with self._lock:
            for code, sname in stock_items:
                prev = self._sim_quotes.get(code) or self._quotes.get(code) or self._build_seed_quote(code, sname)
                lu = prev.limit_up or infer_limit_prices(code, prev.pre_close)[0]
                ld = prev.limit_down or infer_limit_prices(code, prev.pre_close)[1]

                # Random walk within +/- 0.55% per step, bounded inside [limit_down, limit_up]
                pct_step = random.gauss(0.0, 0.0028)
                raw_next = prev.price * (1.0 + pct_step)
                next_price = round(max(ld if ld > 0 else 0.01, min(lu if lu > 0 else 100_000.0, raw_next)), 2)
                if next_price <= 0:
                    next_price = prev.price

                pre_close = prev.pre_close if prev.pre_close > 0 else prev.price
                change = round(next_price - pre_close, 2)
                change_pct = round((change / pre_close) * 100, 2) if pre_close > 0 else 0.0
                step_vol = random.randint(20, 450) * 100.0
                asks, bids = self._build_order_book(next_price, lu, ld)
                new_high = round(max(prev.high or next_price, next_price), 2)
                new_low = round(min(prev.low if prev.low > 0 else next_price, next_price), 2)
                amplitude = round(((new_high - new_low) / pre_close) * 100, 2) if pre_close > 0 else prev.amplitude
                turnover_ratio = round(prev.turnover_ratio + random.uniform(0.0, 0.02), 2)

                new_q = QuoteSnapshot(
                    stock_code=code,
                    short_name=prev.short_name or sname or code,
                    price=next_price,
                    pre_close=round(pre_close, 2),
                    open=prev.open if prev.open > 0 else next_price,
                    high=new_high,
                    low=new_low,
                    change=change,
                    change_pct=change_pct,
                    volume=prev.volume + step_vol,
                    amount=round(prev.amount + step_vol * next_price, 2),
                    turnover_ratio=turnover_ratio,
                    pe_ratio=prev.pe_ratio,
                    amplitude=amplitude,
                    limit_up=round(lu, 2),
                    limit_down=round(ld, 2),
                    asks=asks,
                    bids=bids,
                    source="simulated_tick",
                    updated_at=now_full,
                )
                self._sim_quotes[code] = new_q
                updated[code] = new_q

                bars = self._sim_mins.setdefault(code, [])
                if not bars and next_price > 0:
                    # Seed a few trailing points so intraday chart looks continuous right away
                    base_dt = now_bj - timedelta(minutes=8)
                    seed_p = pre_close if pre_close > 0 else next_price
                    for idx in range(4):
                        t_pt = (base_dt + timedelta(minutes=idx * 2)).strftime("%H:%M:%S")
                        interp = round(seed_p + (next_price - seed_p) * ((idx + 1) / 5.0), 2)
                        bars.append(MinuteBar(time=t_pt, price=interp, avg_price=interp, volume=step_vol))
                avg_p = round(sum(b.price for b in bars[-19:] + [MinuteBar(time=now_label, price=next_price)]) / min(20, len(bars) + 1), 2)
                bars.append(MinuteBar(time=now_label, price=next_price, avg_price=avg_p, volume=step_vol))
                if len(bars) > 240:
                    del bars[: len(bars) - 240]

        return updated

    def refresh_quotes(
        self, stock_items: list[tuple[str, str]], mode: MarketMode
    ) -> dict[str, QuoteSnapshot]:
        if mode == MarketMode.SIMULATED:
            return self.step_simulated_quotes(stock_items)
        return self.fetch_real_quotes(stock_items)

    def fetch_intraday_mins(self, stock_code: str, mode: MarketMode) -> list[MinuteBar]:
        if mode == MarketMode.SIMULATED:
            with self._lock:
                sim_bars = list(self._sim_mins.get(stock_code, []))
            if sim_bars:
                return sim_bars

        if not self.network_enabled:
            q = self.get_quote(stock_code)
            if q:
                return [MinuteBar(time=q.updated_at[-8:] or "09:30:00", price=q.price, avg_price=q.price, volume=q.volume)]
            return []

        now_ts = time.time()
        with self._lock:
            cached = self._min_cache.get(stock_code)
            if cached and (now_ts - cached[0]) < 15.0:
                return cached[1]

        bars: list[MinuteBar] = []
        if self.network_enabled:
            sym = to_tencent_symbol(stock_code)
            try:
                mresp = httpx.get(
                    "https://web.ifzq.gtimg.cn/appstock/app/minute/query",
                    params={"code": sym},
                    timeout=3.5,
                )
                if mresp.status_code == 200:
                    mdata = ((mresp.json().get("data") or {}).get(sym) or {}).get("data") or {}
                    lines = mdata.get("data") or []
                    prev_cum_lots = 0.0
                    cum_amt = 0.0
                    for ln in lines:
                        parts_ln = str(ln).strip().split()
                        if len(parts_ln) < 3:
                            continue
                        raw_t = parts_ln[0]
                        hhmm = f"{raw_t[:2]}:{raw_t[2:4]}" if len(raw_t) >= 4 else raw_t
                        px = _safe_float(parts_ln[1])
                        cum_lots = _safe_float(parts_ln[2])
                        if px <= 0:
                            continue
                        delta_shares = max(0.0, (cum_lots - prev_cum_lots) * 100.0)
                        prev_cum_lots = cum_lots
                        if len(parts_ln) >= 4 and _safe_float(parts_ln[3]) > 0:
                            cum_amt = _safe_float(parts_ln[3])
                        else:
                            cum_amt += delta_shares * px
                        tot_shares = cum_lots * 100.0
                        avg_px = round(cum_amt / tot_shares, 2) if tot_shares > 0 else round(px, 2)
                        bars.append(
                            MinuteBar(
                                time=hhmm,
                                price=round(px, 2),
                                avg_price=avg_px,
                                volume=delta_shares,
                            )
                        )
            except Exception:
                bars = []

            if not bars:
                try:
                    df = adata.stock.market.get_market_min(stock_code)
                    if df is not None and not df.empty:
                        for _, row in df.iterrows():
                            t_str = str(row.get("trade_time", ""))
                            hhmm = t_str.split(" ")[-1][:5] if " " in t_str else t_str[:5]
                            bars.append(
                                MinuteBar(
                                    time=hhmm,
                                    price=round(float(row["price"]), 2),
                                    avg_price=round(float(row.get("avg_price", row["price"])), 2),
                                    volume=float(row.get("volume", 0.0)),
                                )
                            )
                except Exception:
                    bars = []

        if not bars:
            q = self.get_quote(stock_code)
            if q:
                bars = [MinuteBar(time="最新", price=q.price, avg_price=q.price, volume=q.volume)]

        with self._lock:
            self._min_cache[stock_code] = (now_ts, bars)
        return bars

    def fetch_daily_kline(self, stock_code: str, days: int = 60) -> list[dict]:
        now_ts = time.time()
        with self._lock:
            cached = self._daily_cache.get(stock_code)
            if cached and (now_ts - cached[0]) < 300.0 and cached[1] >= days:
                full_bars = cached[2]
                return full_bars[-days:] if len(full_bars) > days else list(full_bars)

        raw_bars: list[dict] = []
        if self.network_enabled:
            sym = to_tencent_symbol(stock_code)
            try:
                kresp = httpx.get(
                    "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get",
                    params={"param": f"{sym},day,,,{max(120, days + 60)},qfq"},
                    timeout=4.0,
                )
                if kresp.status_code == 200:
                    kdata = (kresp.json().get("data") or {}).get(sym) or {}
                    rows = kdata.get("qfqday") or kdata.get("day") or []
                    prev_c = 0.0
                    for row in rows:
                        if len(row) < 6:
                            continue
                        o = _safe_float(str(row[1]))
                        c = _safe_float(str(row[2]))
                        h = _safe_float(str(row[3]))
                        l = _safe_float(str(row[4]))
                        v = _safe_float(str(row[5])) * 100.0
                        pct = round(((c - prev_c) / prev_c) * 100, 2) if prev_c > 0 else 0.0
                        raw_bars.append(
                            {
                                "trade_date": str(row[0]),
                                "open": round(o, 2),
                                "close": round(c, 2),
                                "high": round(h, 2),
                                "low": round(l, 2),
                                "volume": v,
                                "amount": round(v * c, 2),
                                "change_pct": pct,
                                "turnover_ratio": 0.0,
                            }
                        )
                        prev_c = c
            except Exception:
                raw_bars = []

            if not raw_bars:
                # Fallback to adata with extended lookback for MA20 & MACD warmup
                lookback_cal_days = max(220, days * 2 + 120)
                start_date = (beijing_now() - timedelta(days=lookback_cal_days)).strftime("%Y-%m-%d")
                try:
                    df = adata.stock.market.get_market(stock_code, start_date=start_date)
                    if df is not None and not df.empty:
                        for _, row in df.iterrows():
                            raw_bars.append(
                                {
                                    "trade_date": str(row["trade_date"]),
                                    "open": round(float(row["open"]), 2),
                                    "high": round(float(row["high"]), 2),
                                    "low": round(float(row["low"]), 2),
                                    "close": round(float(row["close"]), 2),
                                    "volume": float(row.get("volume", 0.0)),
                                    "amount": float(row.get("amount", 0.0)),
                                    "change_pct": round(float(row.get("change_pct", 0.0)), 2),
                                    "turnover_ratio": round(float(row.get("turnover_ratio", 0.0)), 2),
                                }
                            )
                except Exception:
                    raw_bars = []

        if not raw_bars:
            # Generate deterministic fallback bars with warmup history if offline so MA20 & MACD populate accurately
            q = self.get_quote(stock_code) or self._build_seed_quote(stock_code)
            base = q.pre_close if q.pre_close > 0 else q.price
            rng = random.Random(stock_code)
            total_gen = max(days + 60, 120)
            weekdays: list[str] = []
            cursor = beijing_now()
            while len(weekdays) < total_gen:
                if cursor.weekday() < 5:
                    weekdays.append(cursor.strftime("%Y-%m-%d"))
                cursor -= timedelta(days=1)
            weekdays.reverse()

            cur = base * 0.94
            for d in weekdays:
                o = round(cur, 2)
                mean_pull = ((base - cur) / base) * 0.10 if base > 0 else 0.0
                ret = mean_pull + rng.uniform(-0.024, 0.026)
                c = round(max(0.5, o * (1.0 + ret)), 2)
                h = round(max(o, c) * (1.0 + rng.uniform(0.002, 0.015)), 2)
                l = round(min(o, c) * (1.0 - rng.uniform(0.002, 0.015)), 2)
                pct = round(((c - o) / o) * 100, 2) if o > 0 else 0.0
                vol = float(rng.randint(80_000, 320_000))
                raw_bars.append(
                    {
                        "trade_date": d,
                        "open": o,
                        "high": h,
                        "low": l,
                        "close": c,
                        "volume": vol,
                        "amount": round(vol * c, 2),
                        "change_pct": pct,
                        "turnover_ratio": round(rng.uniform(0.4, 2.2), 2),
                    }
                )
                cur = c

        enriched = compute_kline_indicators(raw_bars)
        with self._lock:
            self._daily_cache[stock_code] = (now_ts, days, enriched)
        return enriched[-days:] if len(enriched) > days else list(enriched)

    def fetch_trade_calendar(self) -> set[str]:
        if not self.network_enabled:
            return set()
        try:
            df = adata.stock.info.trade_calendar()
            if df is not None and not df.empty:
                open_days = df[df["trade_status"] == 1]["trade_date"].astype(str).tolist()
                return set(open_days)
        except Exception:
            pass
        return set()

    def search_stocks(self, query: str) -> list[dict[str, str]]:
        q = query.strip()
        if not q:
            return [
                {"stock_code": c, "short_name": str(info["name"]), "market": to_tencent_symbol(c)[:2].upper()}
                for c, info in list(POPULAR_STOCKS.items())[:8]
            ]

        results: list[dict[str, str]] = []
        seen: set[str] = set()

        if self.network_enabled:
            try:
                resp = httpx.get(
                    "https://smartbox.gtimg.cn/s3/",
                    params={"v": "2", "q": q, "t": "gp"},
                    timeout=3.0,
                )
                if resp.status_code == 200 and '="' in resp.text:
                    raw_val = resp.text.split('="', 1)[1].rstrip('"; \r\n')
                    decoded = codecs.decode(raw_val, "unicode_escape")
                    for entry in decoded.split("^"):
                        segs = entry.split("~")
                        if len(segs) >= 5:
                            mkt, code, name, _, kind = segs[:5]
                            if mkt in ("sh", "sz", "bj") and len(code) == 6 and code.isdigit() and "GP" in kind:
                                if code not in seen:
                                    seen.add(code)
                                    results.append(
                                        {"stock_code": code, "short_name": name, "market": mkt.upper()}
                                    )
            except Exception:
                pass

        # Fallback / supplement from built-in popular list
        q_lower = q.lower()
        for code, info in POPULAR_STOCKS.items():
            name = str(info["name"])
            if (q_lower in code or q_lower in name.lower()) and code not in seen:
                seen.add(code)
                results.append(
                    {"stock_code": code, "short_name": name, "market": to_tencent_symbol(code)[:2].upper()}
                )

        # Direct 6-digit code pass-through if user typed exact code not returned
        if q.isdigit() and len(q) == 6 and q not in seen:
            results.append({"stock_code": q, "short_name": q, "market": to_tencent_symbol(q)[:2].upper()})

        return results[:15]

    def _recent_open_dates(self, days: int) -> list[str]:
        """Return up to `days` most recent trading dates (excluding today before 09:30 open)."""
        now_dt = beijing_now()
        today_str = now_dt.strftime("%Y-%m-%d")
        before_open = now_dt.strftime("%H:%M") < "09:30"

        sorted_open: list[str] = []
        now_ts = time.time()
        with self._lock:
            cal_cached = self._trade_cal_cache
        if cal_cached and (now_ts - cal_cached[0]) < 3600.0:
            sorted_open = list(cal_cached[1])
        elif self.network_enabled:
            open_set = self.fetch_trade_calendar()
            if open_set:
                sorted_open = sorted(open_set)
                with self._lock:
                    self._trade_cal_cache = (now_ts, sorted_open)

        eligible = [d for d in sorted_open if d <= today_str]
        if before_open and eligible and eligible[-1] == today_str:
            eligible = eligible[:-1]
        if len(eligible) >= days:
            return eligible[-days:]

        # Fallback weekday series when calendar unavailable or offline
        weekdays: list[str] = []
        cursor = now_dt - timedelta(days=1 if before_open else 0)
        while len(weekdays) < days:
            if cursor.weekday() < 5:
                weekdays.append(cursor.strftime("%Y-%m-%d"))
            cursor -= timedelta(days=1)
        weekdays.reverse()
        return weekdays

    def _fetch_day_zt_pool(self, trade_date_iso: str, client: Optional[httpx.Client] = None) -> tuple[list[dict], bool]:
        """Fetch one trading day's limit-up pool from Eastmoney with TTL cache."""
        today_iso = beijing_now().strftime("%Y-%m-%d")
        ttl = 60.0 if trade_date_iso == today_iso else 3600.0
        now_ts = time.time()
        with self._lock:
            cached = self._zt_pool_cache.get(trade_date_iso)
            if cached and (now_ts - cached[0]) < ttl:
                return list(cached[1]), True

        if not self.network_enabled:
            return [], False

        date_compact = trade_date_iso.replace("-", "")
        own_client = client is None
        http = client or httpx.Client(timeout=5.0, headers={"Referer": "https://quote.eastmoney.com/"})
        collected: list[dict] = []
        ok = False
        try:
            for page in range(5):
                resp = http.get(
                    "https://push2ex.eastmoney.com/getTopicZTPool",
                    params={
                        "ut": "7eea3edcaed734bea9cbfc24409ed989",
                        "dpt": "wz.ztzt",
                        "Pageindex": page,
                        "pagesize": 300,
                        "sort": "fbt:asc",
                        "date": date_compact,
                    },
                )
                if resp.status_code != 200:
                    break
                payload = resp.json()
                data = payload.get("data")
                ok = True
                if not isinstance(data, dict):
                    break
                pool = data.get("pool") or []
                for row in pool:
                    parsed = parse_eastmoney_zt_pool_item(row, trade_date_iso)
                    if parsed:
                        collected.append(parsed)
                tc = int(data.get("tc") or len(collected))
                if not pool or len(collected) >= tc:
                    break
        except Exception:
            if not ok:
                return [], False
        finally:
            if own_client:
                http.close()

        if ok:
            with self._lock:
                self._zt_pool_cache[trade_date_iso] = (time.time(), collected)
        return collected, ok

    def _batch_fetch_tencent_quotes(self, codes: list[str]) -> dict[str, QuoteSnapshot]:
        """Fetch current quotes for matched screener stocks in batches of 60 without touching simulated cache."""
        if not codes:
            return {}
        out: dict[str, QuoteSnapshot] = {}
        if not self.network_enabled:
            with self._lock:
                for c in codes:
                    if c in self._quotes:
                        out[c] = self._quotes[c]
            return out

        unique_codes = list(dict.fromkeys(codes))
        with httpx.Client(timeout=4.0, headers={"Referer": "https://gu.qq.com/"}) as http:
            for start in range(0, len(unique_codes), 60):
                chunk = unique_codes[start : start + 60]
                symbols = ",".join(to_tencent_symbol(c) for c in chunk)
                try:
                    resp = http.get(f"https://qt.gtimg.cn/q={symbols}")
                    if resp.status_code == 200:
                        text = resp.content.decode("gbk", errors="ignore")
                        out.update(parse_tencent_payload(text))
                except Exception:
                    continue
        return out

    def screen_limit_up_stocks(
        self,
        days: int = 5,
        board: str = "ALL",
        min_limit_count: int = 1,
        only_watchlist: bool = False,
        keyword: str = "",
        watchlist_items: list[tuple[str, str]] | None = None,
    ) -> LimitUpScreenResponse:
        """Scan A-share stocks that hit daily limit-up within the last N trading days."""
        days = max(1, min(int(days), 20))
        min_limit_count = max(1, int(min_limit_count))
        board_filter = (board or "ALL").strip()
        kw = (keyword or "").strip().lower()
        watch_list = watchlist_items or []
        watch_codes = {c for c, _ in watch_list}
        trade_dates = self._recent_open_dates(days)

        agg: dict[str, dict] = {}
        source = "kline_scan"
        any_zt_ok = False

        if self.network_enabled and trade_dates:
            day_results: list[tuple[list[dict], bool]] = []
            with httpx.Client(timeout=5.0, headers={"Referer": "https://quote.eastmoney.com/"}) as http:
                workers = min(6, max(1, len(trade_dates)))
                with ThreadPoolExecutor(max_workers=workers) as pool_exec:
                    futures = [pool_exec.submit(self._fetch_day_zt_pool, d, http) for d in trade_dates]
                    for fut in as_completed(futures):
                        try:
                            day_results.append(fut.result())
                        except Exception:
                            day_results.append(([], False))

            any_zt_ok = any(ok for _, ok in day_results)
            if any_zt_ok:
                source = "eastmoney_zt_pool"
                for rows, ok in day_results:
                    if not ok:
                        continue
                    for r in rows:
                        code = r["stock_code"]
                        entry = agg.get(code)
                        if not entry:
                            agg[code] = {
                                "stock_code": code,
                                "short_name": r["short_name"],
                                "dates": {r["trade_date"]},
                                "max_lbc": int(r["lbc"]),
                                "last_date": r["trade_date"],
                                "price": float(r["price"]),
                                "change_pct": float(r["change_pct"]),
                                "turnover_ratio": float(r["turnover_ratio"]),
                                "amount": float(r["amount"]),
                                "industry": r.get("industry", ""),
                            }
                        else:
                            entry["dates"].add(r["trade_date"])
                            entry["max_lbc"] = max(entry["max_lbc"], int(r["lbc"]))
                            if r["industry"] and not entry["industry"]:
                                entry["industry"] = r["industry"]
                            if r["trade_date"] >= entry["last_date"]:
                                entry["last_date"] = r["trade_date"]
                                entry["short_name"] = r["short_name"]
                                entry["price"] = float(r["price"])
                                entry["change_pct"] = float(r["change_pct"])
                                entry["turnover_ratio"] = float(r["turnover_ratio"])
                                entry["amount"] = float(r["amount"])
                                if r["industry"]:
                                    entry["industry"] = r["industry"]

        # Offline / fallback K-line scan across watchlist + baseline universe
        if not any_zt_ok:
            source = "kline_scan"
            candidates = list(watch_list)
            if not only_watchlist:
                for c, info in POPULAR_STOCKS.items():
                    if c not in watch_codes:
                        candidates.append((c, str(info["name"])))

            observed_dates: set[str] = set()
            for code, name in candidates:
                bars = self.fetch_daily_kline(code, days=max(12, days + 6))
                if len(bars) < 2:
                    continue
                start_idx = max(1, len(bars) - days)
                hit_dates: set[str] = set()
                streak = 0
                max_streak = 0
                for idx in range(start_idx, len(bars)):
                    prev_close = float(bars[idx - 1]["close"])
                    close_p = float(bars[idx]["close"])
                    t_date = str(bars[idx]["trade_date"])
                    observed_dates.add(t_date)
                    if is_limit_up_bar(code, close_p, prev_close, name):
                        hit_dates.add(t_date)
                        streak += 1
                        max_streak = max(max_streak, streak)
                    else:
                        streak = 0
                if not hit_dates:
                    continue
                q_live = self.get_quote(code)
                last_bar = bars[-1]
                agg[code] = {
                    "stock_code": code,
                    "short_name": (q_live.short_name if q_live and q_live.short_name else name) or code,
                    "dates": hit_dates,
                    "max_lbc": max(1, max_streak),
                    "last_date": max(hit_dates),
                    "price": q_live.price if q_live else float(last_bar["close"]),
                    "change_pct": q_live.change_pct if q_live else float(last_bar.get("change_pct", 0.0)),
                    "turnover_ratio": q_live.turnover_ratio if q_live else float(last_bar.get("turnover_ratio", 0.0)),
                    "amount": q_live.amount if q_live else float(last_bar.get("amount", 0.0)),
                    "industry": "",
                }
            if observed_dates:
                trade_dates = sorted(observed_dates)[-days:]

        # Apply filters prior to live quote enrichment
        filtered: list[dict] = []
        for entry in agg.values():
            code = entry["stock_code"]
            board_name = classify_board(code)
            if board_filter != "ALL" and board_name != board_filter:
                continue
            if only_watchlist and code not in watch_codes:
                continue
            if len(entry["dates"]) < min_limit_count:
                continue
            if kw:
                haystack = f"{code} {entry['short_name']} {entry['industry']}".lower()
                if kw not in haystack:
                    continue
            entry["board"] = board_name
            filtered.append(entry)

        live_quotes = (
            self._batch_fetch_tencent_quotes([e["stock_code"] for e in filtered])
            if source == "eastmoney_zt_pool" and filtered
            else {}
        )

        items: list[LimitUpStockItem] = []
        for entry in filtered:
            code = entry["stock_code"]
            q = live_quotes.get(code)
            sorted_dates = sorted(entry["dates"], reverse=True)
            items.append(
                LimitUpStockItem(
                    stock_code=code,
                    short_name=(q.short_name if q and q.short_name and q.short_name != code else entry["short_name"]),
                    board=entry["board"],
                    last_price=round(q.price if q else entry["price"], 2),
                    latest_change_pct=round(q.change_pct if q else entry["change_pct"], 2),
                    turnover_ratio=round(q.turnover_ratio if q else entry["turnover_ratio"], 2),
                    amount=round(q.amount if q else entry["amount"], 2),
                    limit_up_count=len(sorted_dates),
                    max_consecutive_boards=entry["max_lbc"],
                    last_limit_up_date=sorted_dates[0],
                    limit_up_dates=sorted_dates,
                    industry=entry["industry"],
                    in_watchlist=(code in watch_codes),
                )
            )

        items.sort(
            key=lambda x: (
                -x.limit_up_count,
                -x.max_consecutive_boards,
                -int(x.last_limit_up_date.replace("-", "") or 0),
                -x.latest_change_pct,
                x.stock_code,
            )
        )
        return LimitUpScreenResponse(
            days=days,
            trade_dates=trade_dates,
            total_hits=len(items),
            source=source,
            scanned_at=now_str(),
            items=items,
        )

    @staticmethod
    def _summarize_daily_technicals(daily: list[dict], current_price: float) -> dict[str, float | str | None]:
        """Compute MA5/MA10/MA20, 5d/20d returns, 20d range, and MACD signal from daily bars."""
        if not daily:
            return {
                "ma5": None,
                "ma10": None,
                "ma20": None,
                "change_5d_pct": None,
                "change_20d_pct": None,
                "high_20d": None,
                "low_20d": None,
                "macd_signal": "数据不足",
            }
        last = daily[-1]
        tail_20 = daily[-20:]
        ref_p = current_price if current_price > 0 else float(last.get("close") or 0)
        c5 = float(daily[-6]["close"]) if len(daily) >= 6 else float(daily[0]["close"])
        c20 = float(tail_20[0]["close"]) if tail_20 else float(daily[0]["close"])
        chg5 = round(((ref_p - c5) / c5) * 100, 2) if c5 > 0 else None
        chg20 = round(((ref_p - c20) / c20) * 100, 2) if c20 > 0 else None
        dif = float(last.get("dif") or 0.0)
        dea = float(last.get("dea") or 0.0)
        macd = float(last.get("macd") or 0.0)
        if dif > dea and macd > 0:
            macd_signal = "MACD 多头 (DIF>DEA)"
        elif dif < dea and macd < 0:
            macd_signal = "MACD 空头 (DIF<DEA)"
        else:
            macd_signal = "MACD 震荡走平"
        return {
            "ma5": last.get("ma5"),
            "ma10": last.get("ma10"),
            "ma20": last.get("ma20"),
            "change_5d_pct": chg5,
            "change_20d_pct": chg20,
            "high_20d": round(max(float(b["high"]) for b in tail_20), 2),
            "low_20d": round(min(float(b["low"]) for b in tail_20), 2),
            "macd_signal": macd_signal,
        }

    def fetch_index_detail(self, symbol: str, mode: MarketMode = MarketMode.REAL) -> dict:
        """Fetch live OHLCV snapshot + daily K-line bars & technicals for a major A-share index."""
        sym = symbol.strip().lower()
        meta = INDEX_BY_SYMBOL.get(sym)
        if not meta:
            raise ValueError(f"未知指数代码: {symbol}")

        # Start from cached or seed snapshot
        cached_list = self.get_cached_indices(mode)
        snap = next((it for it in cached_list if it.symbol == sym), None) or self._build_seed_index(meta)
        open_p = snap.pre_close
        high_p = max(snap.price, snap.pre_close)
        low_p = min(snap.price, snap.pre_close)
        volume = 0.0
        amplitude = round(((high_p - low_p) / snap.pre_close) * 100, 2) if snap.pre_close > 0 else 0.0

        if self.network_enabled:
            try:
                resp = httpx.get(
                    f"https://qt.gtimg.cn/q={sym}",
                    headers={"Referer": "https://gu.qq.com/"},
                    timeout=3.5,
                )
                if resp.status_code == 200:
                    text = resp.content.decode("gbk", errors="ignore")
                    parsed_map = parse_tencent_indices(text)
                    if sym in parsed_map:
                        snap = parsed_map[sym]
                    for line_sym, parts in _split_tencent_lines(text):
                        if line_sym == sym and len(parts) > 35:
                            open_p = _safe_float(parts[5], snap.pre_close)
                            high_p = _safe_float(parts[33], snap.price)
                            low_p = _safe_float(parts[34], snap.price)
                            volume = _safe_float(parts[6]) * 100.0
                            amplitude = (
                                round(_safe_float(parts[43]), 2)
                                if len(parts) > 43 and _safe_float(parts[43]) > 0
                                else (
                                    round(((high_p - low_p) / snap.pre_close) * 100, 2)
                                    if snap.pre_close > 0
                                    else 0.0
                                )
                            )
            except Exception:
                pass

        # Fetch daily bars via Tencent index kline API (cached in _daily_cache under f"IDX:{sym}")
        cache_key = f"IDX:{sym}"
        now_ts = time.time()
        daily: list[dict] = []
        with self._lock:
            cached = self._daily_cache.get(cache_key)
            if cached and (now_ts - cached[0]) < 300.0:
                daily = list(cached[2])

        if not daily:
            raw_bars: list[dict] = []
            if self.network_enabled:
                try:
                    kresp = httpx.get(
                        "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get",
                        params={"param": f"{sym},day,,,60"},
                        timeout=4.0,
                    )
                    if kresp.status_code == 200:
                        kdata = (kresp.json().get("data") or {}).get(sym) or {}
                        rows = kdata.get("day") or kdata.get("qfqday") or []
                        prev_c = 0.0
                        for row in rows:
                            if len(row) < 6:
                                continue
                            o = _safe_float(str(row[1]))
                            c = _safe_float(str(row[2]))
                            h = _safe_float(str(row[3]))
                            l = _safe_float(str(row[4]))
                            v = _safe_float(str(row[5])) * 100.0
                            pct = round(((c - prev_c) / prev_c) * 100, 2) if prev_c > 0 else 0.0
                            raw_bars.append(
                                {
                                    "trade_date": str(row[0]),
                                    "open": round(o, 2),
                                    "close": round(c, 2),
                                    "high": round(h, 2),
                                    "low": round(l, 2),
                                    "volume": v,
                                    "amount": round(v * c, 2),
                                    "change_pct": pct,
                                    "turnover_ratio": 0.0,
                                }
                            )
                            prev_c = c
                except Exception:
                    raw_bars = []

            if not raw_bars:
                base = snap.pre_close or snap.price
                rng = random.Random(sym)
                weekdays: list[str] = []
                cursor = beijing_now()
                while len(weekdays) < 80:
                    if cursor.weekday() < 5:
                        weekdays.append(cursor.strftime("%Y-%m-%d"))
                    cursor -= timedelta(days=1)
                weekdays.reverse()
                cur = base * 0.96
                for d in weekdays:
                    o = round(cur, 2)
                    ret = ((base - cur) / base) * 0.1 + rng.uniform(-0.014, 0.015)
                    c = round(max(100.0, o * (1.0 + ret)), 2)
                    raw_bars.append(
                        {
                            "trade_date": d,
                            "open": o,
                            "high": round(max(o, c) * (1.0 + rng.uniform(0.001, 0.009)), 2),
                            "low": round(min(o, c) * (1.0 - rng.uniform(0.001, 0.009)), 2),
                            "close": c,
                            "volume": float(rng.randint(500_000, 2_000_000)),
                            "amount": snap.amount,
                            "change_pct": round(((c - o) / o) * 100, 2) if o > 0 else 0.0,
                            "turnover_ratio": 0.0,
                        }
                    )
                    cur = c

            daily = compute_kline_indicators(raw_bars)
            with self._lock:
                self._daily_cache[cache_key] = (now_ts, len(daily), daily)

        return {
            "kind": "INDEX",
            "symbol": snap.symbol,
            "code": snap.code,
            "name": snap.name,
            "snapshot": {
                **snap.model_dump(),
                "open": round(open_p, 2),
                "high": round(high_p, 2),
                "low": round(low_p, 2),
                "volume": volume,
                "amplitude": amplitude,
            },
            "daily": daily[-45:],
            "summary": self._summarize_daily_technicals(daily, snap.price),
        }

    def _fetch_stock_concepts(self, code: str, timeout_sec: float = 2.0) -> list[str]:
        now_ts = time.time()
        with self._lock:
            cc = self._concept_cache.get(code)
            if cc and (now_ts - cc[0]) < 3600.0:
                return list(cc[1])
        if not self.network_enabled:
            return []
        concepts: list[str] = []

        def _query() -> list[str]:
            cdf = adata.stock.info.get_concept_ths(code)
            if cdf is not None and not cdf.empty and "name" in cdf.columns:
                return [str(x).strip() for x in cdf["name"].dropna().tolist() if str(x).strip()][:8]
            return []

        pool_exec = ThreadPoolExecutor(max_workers=1)
        try:
            fut = pool_exec.submit(_query)
            concepts = fut.result(timeout=timeout_sec)
        except Exception:
            concepts = []
        finally:
            pool_exec.shutdown(wait=False, cancel_futures=True)

        with self._lock:
            self._concept_cache[code] = (now_ts, concepts)
        return concepts

    def fetch_stock_detail(self, stock_code: str, mode: MarketMode = MarketMode.REAL) -> dict:
        """Fetch comprehensive quote, technical summary, concepts, intraday mins, and daily bars (works 24/7 including market closed hours)."""
        code = stock_code.strip()
        q = self.get_quote(code, mode=mode)
        if not q:
            q = self.refresh_quotes([(code, code)], mode=mode).get(code) or self._build_seed_quote(code)

        with ThreadPoolExecutor(max_workers=3) as pool_exec:
            f_daily = pool_exec.submit(self.fetch_daily_kline, code, 60)
            f_mins = pool_exec.submit(self.fetch_intraday_mins, code, mode)
            f_concepts = pool_exec.submit(self._fetch_stock_concepts, code, 2.0)
            daily = f_daily.result()
            mins = f_mins.result()
            concepts = f_concepts.result()

        q_dict = q.model_dump()
        last_trade_date = ""
        if daily:
            last_bar = daily[-1]
            last_trade_date = str(last_bar.get("trade_date") or "")
            prev_bar_close = float(daily[-2]["close"]) if len(daily) >= 2 else float(last_bar.get("open") or q.pre_close)
            # Backfill intraday fields from the latest completed trading day when off-hours/pre-open returns 0
            if float(q_dict.get("open") or 0) <= 0:
                q_dict["open"] = float(last_bar.get("open") or q.price)
            if float(q_dict.get("high") or 0) <= 0:
                q_dict["high"] = float(last_bar.get("high") or q.price)
            if float(q_dict.get("low") or 0) <= 0:
                q_dict["low"] = float(last_bar.get("low") or q.price)
            if float(q_dict.get("volume") or 0) <= 0 and float(last_bar.get("volume") or 0) > 0:
                q_dict["volume"] = float(last_bar["volume"])
            if float(q_dict.get("amount") or 0) <= 0 and float(last_bar.get("amount") or 0) > 0:
                q_dict["amount"] = float(last_bar["amount"])
            if float(q_dict.get("turnover_ratio") or 0) <= 0 and float(last_bar.get("turnover_ratio") or 0) > 0:
                q_dict["turnover_ratio"] = float(last_bar["turnover_ratio"])
            if float(q_dict.get("amplitude") or 0) <= 0 and prev_bar_close > 0:
                h_v = float(q_dict.get("high") or 0)
                l_v = float(q_dict.get("low") or 0)
                if h_v >= l_v > 0:
                    q_dict["amplitude"] = round(((h_v - l_v) / prev_bar_close) * 100, 2)

        return {
            "kind": "STOCK",
            "stock_code": q.stock_code,
            "short_name": q.short_name,
            "board": classify_board(q.stock_code),
            "last_trade_date": last_trade_date,
            "concepts": concepts,
            "quote": q_dict,
            "mins": [m.model_dump() for m in mins],
            "daily": daily[-45:],
            "summary": self._summarize_daily_technicals(daily, float(q_dict.get("price") or q.price)),
        }
