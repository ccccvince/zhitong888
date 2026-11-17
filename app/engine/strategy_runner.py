from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from app.engine.market_feed import compute_macd_series, compute_sma_series
from app.engine.paper_broker import PaperBroker
from app.models import (
    AutoStrategyRuleRecord,
    AutoStrategyRuleRequest,
    OrderRecord,
    OrderRequest,
    OrderSide,
    OrderStatus,
    QuoteSnapshot,
)
from app.security import beijing_now
from app.storage import Database, now_str


class StrategyRunner:
    """Evaluates lightweight rule-based auto-trading strategies on incoming quotes per user."""

    COOLDOWN_SEC = 30

    def __init__(self, db: Database, broker: PaperBroker) -> None:
        self.db = db
        self.broker = broker

    def add_strategy(
        self, req: AutoStrategyRuleRequest, user_id: str = "default"
    ) -> AutoStrategyRuleRecord:
        if req.order_qty % 100 != 0:
            raise ValueError("自动策略单次委托数量必须为 100 股的整数倍")
        if req.strategy_type == "MA_CROSS" and req.fast_period >= req.slow_period:
            raise ValueError("双均线策略要求快线周期严格小于慢线周期")
        strategy_id = uuid.uuid4().hex[:10].upper()
        now = now_str()
        with self.db.connect() as conn:
            conn.execute(
                """
                INSERT INTO auto_strategies (
                    strategy_id, user_id, stock_code, strategy_type, lower_price, upper_price,
                    fast_period, slow_period, order_qty, enabled, last_signal_tag, last_action_at, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '', NULL, ?)
                """,
                (
                    strategy_id,
                    user_id,
                    req.stock_code,
                    req.strategy_type,
                    req.lower_price,
                    req.upper_price,
                    req.fast_period,
                    req.slow_period,
                    req.order_qty,
                    1 if req.enabled else 0,
                    now,
                ),
            )
            row = conn.execute(
                "SELECT * FROM auto_strategies WHERE strategy_id = ? AND user_id = ?",
                (strategy_id, user_id),
            ).fetchone()
        return self._to_record(row)

    def list_strategies(self, user_id: str = "default") -> list[AutoStrategyRuleRecord]:
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM auto_strategies WHERE user_id = ? ORDER BY created_at DESC",
                (user_id,),
            ).fetchall()
        return [self._to_record(r) for r in rows]

    def toggle_strategy(
        self, strategy_id: str, enabled: bool, user_id: str = "default"
    ) -> AutoStrategyRuleRecord:
        with self.db.connect() as conn:
            conn.execute(
                "UPDATE auto_strategies SET enabled = ? WHERE strategy_id = ? AND user_id = ?",
                (1 if enabled else 0, strategy_id, user_id),
            )
            row = conn.execute(
                "SELECT * FROM auto_strategies WHERE strategy_id = ? AND user_id = ?",
                (strategy_id, user_id),
            ).fetchone()
            if not row:
                raise ValueError(f"自动策略 {strategy_id} 不存在")
        return self._to_record(row)

    def delete_strategy(self, strategy_id: str, user_id: str = "default") -> None:
        with self.db.connect() as conn:
            conn.execute(
                "DELETE FROM auto_strategies WHERE strategy_id = ? AND user_id = ?",
                (strategy_id, user_id),
            )

    def evaluate(
        self,
        quotes: dict[str, QuoteSnapshot],
        can_trade: bool,
        daily_bars_map: Optional[dict[str, list[dict]]] = None,
        user_id: str = "default",
    ) -> list[OrderRecord]:
        if not can_trade or not quotes:
            return []

        now_dt = beijing_now()
        now = now_dt.strftime("%Y-%m-%d %H:%M:%S")
        submitted_orders: list[OrderRecord] = []

        with self.db.connect() as conn:
            rules = conn.execute(
                "SELECT * FROM auto_strategies WHERE enabled = 1 AND user_id = ?", (user_id,)
            ).fetchall()
            if not rules:
                return []
            acc_row = conn.execute(
                "SELECT current_trade_date FROM account WHERE user_id = ?", (user_id,)
            ).fetchone()
            ref_trade_date = (
                str(acc_row["current_trade_date"]) if acc_row else now_dt.strftime("%Y-%m-%d")
            )
            open_rows = conn.execute(
                "SELECT stock_code, side, strategy_name FROM orders WHERE user_id = ? AND status IN (?, ?)",
                (user_id, OrderStatus.PENDING.value, OrderStatus.PARTIAL_FILLED.value),
            ).fetchall()
            open_pairs = {(r["stock_code"], r["side"]) for r in open_rows}

        positions = {p.stock_code: p for p in self.broker.get_positions(quotes, user_id=user_id)}

        for r in rules:
            code = r["stock_code"]
            q = quotes.get(code)
            if not q or q.price <= 0:
                continue

            if r["last_action_at"]:
                try:
                    last_dt = datetime.strptime(r["last_action_at"], "%Y-%m-%d %H:%M:%S")
                    if (now_dt - last_dt).total_seconds() < self.COOLDOWN_SEC:
                        continue
                except ValueError:
                    pass

            stype = r["strategy_type"]
            lower_p = float(r["lower_price"])
            upper_p = float(r["upper_price"])
            fast_p = int(r["fast_period"]) if "fast_period" in r.keys() else 5
            slow_p = int(r["slow_period"]) if "slow_period" in r.keys() else 20
            last_signal_tag = str(r["last_signal_tag"]) if "last_signal_tag" in r.keys() else ""
            qty = int(r["order_qty"])
            pos = positions.get(code)
            avail_qty = pos.available_qty if pos else 0

            req: OrderRequest | None = None
            one_shot = False
            next_signal_tag = last_signal_tag

            if stype == "GRID_MEAN_REVERT":
                if lower_p > 0 and q.price <= lower_p and (code, OrderSide.BUY.value) not in open_pairs:
                    buy_px = q.asks[0].price if (q.asks and q.asks[0].price > 0) else q.price
                    req = OrderRequest(
                        stock_code=code,
                        side=OrderSide.BUY,
                        price=buy_px,
                        quantity=qty,
                        strategy_name=f"网格低吸({r['strategy_id'][:4]})",
                    )
                elif (
                    upper_p > 0
                    and q.price >= upper_p
                    and avail_qty >= qty
                    and (code, OrderSide.SELL.value) not in open_pairs
                ):
                    sell_px = q.bids[0].price if (q.bids and q.bids[0].price > 0) else q.price
                    req = OrderRequest(
                        stock_code=code,
                        side=OrderSide.SELL,
                        price=sell_px,
                        quantity=qty,
                        strategy_name=f"网格高抛({r['strategy_id'][:4]})",
                    )
            elif stype == "BREAKOUT_BUY":
                if upper_p > 0 and q.price >= upper_p and (code, OrderSide.BUY.value) not in open_pairs:
                    buy_px = q.asks[0].price if (q.asks and q.asks[0].price > 0) else q.price
                    req = OrderRequest(
                        stock_code=code,
                        side=OrderSide.BUY,
                        price=buy_px,
                        quantity=qty,
                        strategy_name=f"突破追涨({r['strategy_id'][:4]})",
                    )
                    one_shot = True
            elif stype == "TAKE_PROFIT_STOP_LOSS":
                if avail_qty > 0 and (code, OrderSide.SELL.value) not in open_pairs:
                    sell_qty = min(qty, avail_qty)
                    if sell_qty % 100 != 0 and sell_qty != avail_qty:
                        sell_qty = (sell_qty // 100) * 100
                    if sell_qty > 0:
                        sell_px = q.bids[0].price if (q.bids and q.bids[0].price > 0) else q.price
                        if upper_p > 0 and q.price >= upper_p:
                            req = OrderRequest(
                                stock_code=code,
                                side=OrderSide.SELL,
                                price=sell_px,
                                quantity=sell_qty,
                                strategy_name=f"触发止盈({r['strategy_id'][:4]})",
                            )
                            one_shot = True
                        elif lower_p > 0 and q.price <= lower_p:
                            req = OrderRequest(
                                stock_code=code,
                                side=OrderSide.SELL,
                                price=sell_px,
                                quantity=sell_qty,
                                strategy_name=f"触发止损({r['strategy_id'][:4]})",
                            )
                            one_shot = True
            elif stype in ("MA_CROSS", "MACD_CROSS"):
                bars = (daily_bars_map or {}).get(code)
                if bars and len(bars) >= 2:
                    ordered = sorted(bars, key=lambda b: str(b["trade_date"]))
                    closes = [float(b["close"]) for b in ordered]
                    last_bar_date = str(ordered[-1]["trade_date"])
                    if last_bar_date == ref_trade_date:
                        closes[-1] = float(q.price)
                        bar_date = last_bar_date
                    else:
                        closes.append(float(q.price))
                        bar_date = ref_trade_date

                    signal_side: Optional[OrderSide] = None
                    if stype == "MA_CROSS" and len(closes) >= slow_p + 1:
                        fast_s = compute_sma_series(closes, fast_p, decimals=4)
                        slow_s = compute_sma_series(closes, slow_p, decimals=4)
                        pf, ps = fast_s[-2], slow_s[-2]
                        cf, cs = fast_s[-1], slow_s[-1]
                        if pf is not None and ps is not None and cf is not None and cs is not None:
                            if pf <= ps and cf > cs:
                                signal_side = OrderSide.BUY
                            elif pf >= ps and cf < cs:
                                signal_side = OrderSide.SELL
                    elif stype == "MACD_CROSS" and len(closes) >= 20:
                        macd_s = compute_macd_series(closes, 12, 26, 9)
                        pdif, pdea, _ = macd_s[-2]
                        cdif, cdea, _ = macd_s[-1]
                        if pdif <= pdea and cdif > cdea:
                            signal_side = OrderSide.BUY
                        elif pdif >= pdea and cdif < cdea:
                            signal_side = OrderSide.SELL

                    if signal_side is not None:
                        tag = f"{bar_date}:{signal_side.value}"
                        if tag != last_signal_tag:
                            if signal_side == OrderSide.BUY and (code, OrderSide.BUY.value) not in open_pairs:
                                buy_px = q.asks[0].price if (q.asks and q.asks[0].price > 0) else q.price
                                label = f"MA{fast_p}/{slow_p}金叉" if stype == "MA_CROSS" else "MACD金叉"
                                req = OrderRequest(
                                    stock_code=code,
                                    side=OrderSide.BUY,
                                    price=buy_px,
                                    quantity=qty,
                                    strategy_name=f"{label}({r['strategy_id'][:4]})",
                                )
                                next_signal_tag = tag
                            elif (
                                signal_side == OrderSide.SELL
                                and avail_qty > 0
                                and (code, OrderSide.SELL.value) not in open_pairs
                            ):
                                sell_qty = min(qty, avail_qty)
                                if sell_qty % 100 != 0 and sell_qty != avail_qty:
                                    sell_qty = (sell_qty // 100) * 100
                                if sell_qty > 0:
                                    sell_px = q.bids[0].price if (q.bids and q.bids[0].price > 0) else q.price
                                    label = f"MA{fast_p}/{slow_p}死叉" if stype == "MA_CROSS" else "MACD死叉"
                                    req = OrderRequest(
                                        stock_code=code,
                                        side=OrderSide.SELL,
                                        price=sell_px,
                                        quantity=sell_qty,
                                        strategy_name=f"{label}({r['strategy_id'][:4]})",
                                    )
                                    next_signal_tag = tag

            if req is None:
                continue

            order_rec = self.broker.place_order(req, q, quotes_map=quotes, user_id=user_id)
            submitted_orders.append(order_rec)
            if order_rec.status in (OrderStatus.PENDING, OrderStatus.PARTIAL_FILLED):
                open_pairs.add((code, req.side.value))
            disable_flag = 0 if (one_shot and order_rec.status != OrderStatus.REJECTED) else 1
            persisted_tag = next_signal_tag if order_rec.status != OrderStatus.REJECTED else last_signal_tag
            with self.db.connect() as conn:
                conn.execute(
                    """
                    UPDATE auto_strategies
                    SET last_action_at = ?, enabled = ?, last_signal_tag = ?
                    WHERE strategy_id = ? AND user_id = ?
                    """,
                    (now, disable_flag, persisted_tag, r["strategy_id"], user_id),
                )

        return submitted_orders

    @staticmethod
    def _to_record(row) -> AutoStrategyRuleRecord:
        d = dict(row)
        d["enabled"] = bool(d["enabled"])
        return AutoStrategyRuleRecord(**d)
