from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from typing import Optional

from app.config import settings
from app.models import (
    AccountSummary,
    MarketMode,
    OrderBookLevel,
    OrderRecord,
    OrderRequest,
    OrderSide,
    OrderStatus,
    PositionRecord,
    QuoteSnapshot,
    RiskConfig,
    TradeRecord,
)
from app.security import today_str
from app.storage import Database, now_str


def infer_limit_prices(stock_code: str, pre_close: float) -> tuple[float, float]:
    """Infer A-share daily price limits from board prefix."""
    if pre_close <= 0:
        return 0.0, 0.0
    if stock_code.startswith(("300", "301", "688", "689")):
        ratio = 0.20
    elif stock_code.startswith(("8", "4", "92")):
        ratio = 0.30
    else:
        ratio = 0.10
    return round(pre_close * (1 + ratio), 2), round(pre_close * (1 - ratio), 2)


def calc_fees(side: OrderSide, amount: float) -> tuple[float, float, float, float]:
    """Return (commission, transfer_fee, stamp_tax, total_fee)."""
    if amount <= 0:
        return 0.0, 0.0, 0.0, 0.0
    commission = max(round(amount * settings.commission_rate, 2), settings.min_commission)
    transfer_fee = round(amount * settings.transfer_fee_rate, 2)
    stamp_tax = round(amount * settings.stamp_tax_rate, 2) if side == OrderSide.SELL else 0.0
    total_fee = round(commission + transfer_fee + stamp_tax, 2)
    return commission, transfer_fee, stamp_tax, total_fee


class PaperBroker:
    """Multi-user isolated A-share paper trading broker with T+1 and 5-level book matching."""

    def __init__(self, db: Database) -> None:
        self.db = db

    def get_account_row(self, user_id: str = "default") -> dict:
        with self.db.connect() as conn:
            row = conn.execute("SELECT * FROM account WHERE user_id = ?", (user_id,)).fetchone()
            if row:
                return dict(row)
        self.db.ensure_user_workspace(user_id)
        with self.db.connect() as conn:
            row = conn.execute("SELECT * FROM account WHERE user_id = ?", (user_id,)).fetchone()
            return dict(row)

    def set_market_mode(self, mode: MarketMode, user_id: str = "default") -> None:
        with self.db.connect() as conn:
            conn.execute(
                "UPDATE account SET market_mode = ?, updated_at = ? WHERE user_id = ?",
                (mode.value, now_str(), user_id),
            )

    def reset_account(self, initial_cash: float, user_id: str = "default") -> AccountSummary:
        now = now_str()
        tdate = today_str()
        with self.db.connect() as conn:
            mode_row = conn.execute(
                "SELECT market_mode FROM account WHERE user_id = ?", (user_id,)
            ).fetchone()
            mode = mode_row["market_mode"] if mode_row else MarketMode.REAL.value
            conn.execute("DELETE FROM positions WHERE user_id = ?", (user_id,))
            conn.execute("DELETE FROM orders WHERE user_id = ?", (user_id,))
            conn.execute("DELETE FROM trades WHERE user_id = ?", (user_id,))
            conn.execute("DELETE FROM nav_history WHERE user_id = ?", (user_id,))
            conn.execute(
                """
                UPDATE account
                SET initial_cash = ?, available_cash = ?, frozen_cash = 0.0,
                    current_trade_date = ?, market_mode = ?, updated_at = ?
                WHERE user_id = ?
                """,
                (initial_cash, initial_cash, tdate, mode, now, user_id),
            )
            conn.execute(
                """
                INSERT INTO nav_history (user_id, trade_date, total_assets, available_cash, market_value, daily_return_pct, recorded_at)
                VALUES (?, ?, ?, ?, 0.0, 0.0, ?)
                """,
                (user_id, tdate, initial_cash, initial_cash, now),
            )
        return self.get_account_summary({}, user_id=user_id)

    def get_positions(
        self, quotes: dict[str, QuoteSnapshot] | None = None, user_id: str = "default"
    ) -> list[PositionRecord]:
        quotes = quotes or {}
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM positions WHERE user_id = ? AND total_qty > 0 ORDER BY stock_code",
                (user_id,),
            ).fetchall()
        res: list[PositionRecord] = []
        for r in rows:
            code = r["stock_code"]
            q = quotes.get(code)
            last_price = q.price if q and q.price > 0 else r["avg_cost"]
            mv = round(last_price * r["total_qty"], 2)
            cost_val = r["avg_cost"] * r["total_qty"]
            pnl = round(mv - cost_val, 2)
            pnl_pct = round((pnl / cost_val) * 100, 2) if cost_val > 0 else 0.0
            res.append(
                PositionRecord(
                    stock_code=code,
                    short_name=r["short_name"],
                    total_qty=r["total_qty"],
                    available_qty=r["available_qty"],
                    today_bought_qty=r["today_bought_qty"],
                    frozen_qty=r["frozen_qty"],
                    avg_cost=round(r["avg_cost"], 4),
                    last_price=round(last_price, 3),
                    market_value=mv,
                    unrealized_pnl=pnl,
                    unrealized_pnl_pct=pnl_pct,
                )
            )
        return res

    def get_account_summary(
        self, quotes: dict[str, QuoteSnapshot] | None = None, user_id: str = "default"
    ) -> AccountSummary:
        acc = self.get_account_row(user_id=user_id)
        positions = self.get_positions(quotes, user_id=user_id)
        mv = round(sum(p.market_value for p in positions), 2)
        total_assets = round(acc["available_cash"] + acc["frozen_cash"] + mv, 2)
        total_pnl = round(total_assets - acc["initial_cash"], 2)
        total_pnl_pct = round((total_pnl / acc["initial_cash"]) * 100, 2) if acc["initial_cash"] > 0 else 0.0
        return AccountSummary(
            initial_cash=round(acc["initial_cash"], 2),
            available_cash=round(acc["available_cash"], 2),
            frozen_cash=round(acc["frozen_cash"], 2),
            market_value=mv,
            total_assets=total_assets,
            total_pnl=total_pnl,
            total_pnl_pct=total_pnl_pct,
            current_trade_date=acc["current_trade_date"],
            market_mode=MarketMode(acc["market_mode"]),
        )

    def list_orders(self, limit: int = 100, user_id: str = "default") -> list[OrderRecord]:
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM orders WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        return [OrderRecord(**dict(r)) for r in rows]

    def list_trades(self, limit: int = 100, user_id: str = "default") -> list[TradeRecord]:
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM trades WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        return [TradeRecord(**dict(r)) for r in rows]

    def list_nav_history(self, user_id: str = "default") -> list[dict]:
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM nav_history WHERE user_id = ? ORDER BY trade_date ASC",
                (user_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_risk_config(self, user_id: str = "default") -> RiskConfig:
        with self.db.connect() as conn:
            row = conn.execute("SELECT * FROM risk_config WHERE user_id = ?", (user_id,)).fetchone()
            if not row:
                return RiskConfig()
            return RiskConfig(
                max_single_position_pct=float(row["max_single_position_pct"]),
                max_order_amount=float(row["max_order_amount"]),
                max_total_loss_pct=float(row["max_total_loss_pct"]),
                block_buy_on_loss=bool(row["block_buy_on_loss"]),
            )

    def update_risk_config(self, cfg: RiskConfig, user_id: str = "default") -> RiskConfig:
        now = now_str()
        with self.db.connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO risk_config (
                    user_id, max_single_position_pct, max_order_amount, max_total_loss_pct, block_buy_on_loss, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    round(cfg.max_single_position_pct, 2),
                    round(cfg.max_order_amount, 2),
                    round(cfg.max_total_loss_pct, 2),
                    1 if cfg.block_buy_on_loss else 0,
                    now,
                ),
            )
        return self.get_risk_config(user_id=user_id)

    def _check_buy_risk(
        self,
        conn,
        user_id: str,
        acc: dict,
        req: OrderRequest,
        price: float,
        qty: int,
        quote: Optional[QuoteSnapshot],
        quotes_map: Optional[dict[str, QuoteSnapshot]],
    ) -> Optional[str]:
        row = conn.execute("SELECT * FROM risk_config WHERE user_id = ?", (user_id,)).fetchone()
        if not row:
            return None
        max_pos_pct = float(row["max_single_position_pct"])
        max_amt = float(row["max_order_amount"])
        max_loss_pct = float(row["max_total_loss_pct"])
        block_on_loss = bool(row["block_buy_on_loss"])

        est_amt = round(price * qty, 2)
        if max_amt > 0 and est_amt > max_amt + 1e-6:
            return f"风控拦截: 单笔委托金额 ¥{est_amt:,.2f} 超过上限 ¥{max_amt:,.2f}"

        pos_rows = conn.execute(
            "SELECT * FROM positions WHERE user_id = ? AND total_qty > 0", (user_id,)
        ).fetchall()
        mv_total = 0.0
        target_pos_mv = 0.0
        for p in pos_rows:
            pcode = p["stock_code"]
            pq = (quotes_map or {}).get(pcode)
            if pq is None and pcode == req.stock_code:
                pq = quote
            px = pq.price if (pq and pq.price > 0) else float(p["avg_cost"])
            val = px * int(p["total_qty"])
            mv_total += val
            if pcode == req.stock_code:
                target_pos_mv = val

        total_assets = float(acc["available_cash"]) + float(acc["frozen_cash"]) + mv_total
        if max_loss_pct > 0 and block_on_loss and acc["initial_cash"] > 0:
            cur_loss_pct = ((float(acc["initial_cash"]) - total_assets) / float(acc["initial_cash"])) * 100.0
            if cur_loss_pct >= max_loss_pct - 1e-6:
                return f"风控熔断: 账户总亏损 {cur_loss_pct:.2f}% 已达阈值 {max_loss_pct:.2f}%，已禁止新买入"

        if max_pos_pct < 100.0 and total_assets > 0:
            open_buy = conn.execute(
                """
                SELECT COALESCE(SUM(frozen_cash), 0.0) AS s FROM orders
                WHERE user_id = ? AND stock_code = ? AND side = ? AND status IN (?, ?)
                """,
                (
                    user_id,
                    req.stock_code,
                    OrderSide.BUY.value,
                    OrderStatus.PENDING.value,
                    OrderStatus.PARTIAL_FILLED.value,
                ),
            ).fetchone()["s"]
            projected_val = target_pos_mv + float(open_buy or 0.0) + est_amt
            projected_pct = (projected_val / total_assets) * 100.0
            if projected_pct > max_pos_pct + 0.01:
                return f"风控拦截: 买入后 {req.stock_code} 仓位将达 {projected_pct:.2f}%，超过单票上限 {max_pos_pct:.1f}%"

        return None

    def place_order(
        self,
        req: OrderRequest,
        quote: Optional[QuoteSnapshot] = None,
        match_now: bool = True,
        quotes_map: Optional[dict[str, QuoteSnapshot]] = None,
        user_id: str = "default",
    ) -> OrderRecord:
        now = now_str()
        order_id = uuid.uuid4().hex[:12].upper()
        price = round(req.price, 2)
        qty = req.quantity

        with self.db.connect() as conn:
            acc_row = conn.execute("SELECT * FROM account WHERE user_id = ?", (user_id,)).fetchone()
            if not acc_row:
                self.db._ensure_user_workspace_tx(conn, user_id)
                acc_row = conn.execute("SELECT * FROM account WHERE user_id = ?", (user_id,)).fetchone()
            acc = dict(acc_row)
            tdate = acc["current_trade_date"]

            short_name = quote.short_name if quote else req.stock_code
            if short_name == req.stock_code:
                w = conn.execute(
                    "SELECT short_name FROM watchlist WHERE user_id = ? AND stock_code = ?",
                    (user_id, req.stock_code),
                ).fetchone()
                if w:
                    short_name = w["short_name"]

            # Price limit check if quote is available
            if quote and quote.pre_close > 0:
                lu, ld = (
                    (quote.limit_up, quote.limit_down)
                    if quote.limit_up > 0 and quote.limit_down > 0
                    else infer_limit_prices(req.stock_code, quote.pre_close)
                )
                if price > lu + 0.001 or price < ld - 0.001:
                    return self._insert_rejected(
                        conn, user_id, order_id, req, short_name, price, tdate, now,
                        f"委托价格 {price:.2f} 超出今日涨跌停区间 [{ld:.2f}, {lu:.2f}]",
                    )

            if req.side == OrderSide.BUY:
                if qty < 100 or qty % 100 != 0:
                    return self._insert_rejected(
                        conn, user_id, order_id, req, short_name, price, tdate, now,
                        "A股买入数量必须为 100 股的整数倍",
                    )
                risk_err = self._check_buy_risk(conn, user_id, acc, req, price, qty, quote, quotes_map)
                if risk_err:
                    return self._insert_rejected(
                        conn, user_id, order_id, req, short_name, price, tdate, now, risk_err,
                    )
                est_amt = price * qty
                _, _, _, est_fee = calc_fees(OrderSide.BUY, est_amt)
                freeze_amt = round(est_amt + est_fee, 2)
                if acc["available_cash"] < freeze_amt:
                    return self._insert_rejected(
                        conn, user_id, order_id, req, short_name, price, tdate, now,
                        f"可用资金不足: 需要 {freeze_amt:.2f}, 可用 {acc['available_cash']:.2f}",
                    )
                conn.execute(
                    """
                    UPDATE account
                    SET available_cash = ROUND(available_cash - ?, 2),
                        frozen_cash = ROUND(frozen_cash + ?, 2),
                        updated_at = ?
                    WHERE user_id = ?
                    """,
                    (freeze_amt, freeze_amt, now, user_id),
                )
                frozen_cash = freeze_amt
            else:
                pos = conn.execute(
                    "SELECT * FROM positions WHERE user_id = ? AND stock_code = ?",
                    (user_id, req.stock_code),
                ).fetchone()
                if not pos or pos["total_qty"] <= 0:
                    return self._insert_rejected(
                        conn, user_id, order_id, req, short_name, price, tdate, now,
                        "未持有该股票，无法卖出",
                    )
                avail = pos["available_qty"]
                if avail < qty:
                    return self._insert_rejected(
                        conn, user_id, order_id, req, short_name, price, tdate, now,
                        f"可用持仓不足(T+1限制): 总持仓 {pos['total_qty']} 股, 今日可用 {avail} 股, 申请卖出 {qty} 股",
                    )
                if qty % 100 != 0 and qty != avail:
                    return self._insert_rejected(
                        conn, user_id, order_id, req, short_name, price, tdate, now,
                        "卖出数量需为 100 股整数倍，或一次性卖出全部可用余股",
                    )
                short_name = pos["short_name"]
                conn.execute(
                    """
                    UPDATE positions
                    SET available_qty = available_qty - ?,
                        frozen_qty = frozen_qty + ?,
                        updated_at = ?
                    WHERE user_id = ? AND stock_code = ?
                    """,
                    (qty, qty, now, user_id, req.stock_code),
                )
                frozen_cash = 0.0

            conn.execute(
                """
                INSERT INTO orders (
                    order_id, user_id, stock_code, short_name, side, price, quantity,
                    filled_quantity, avg_filled_price, frozen_cash, status,
                    reject_reason, strategy_name, trade_date, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 0, 0.0, ?, ?, '', ?, ?, ?, ?)
                """,
                (
                    order_id, user_id, req.stock_code, short_name, req.side.value, price, qty,
                    frozen_cash, OrderStatus.PENDING.value, req.strategy_name, tdate, now, now,
                ),
            )
            order_row = dict(
                conn.execute("SELECT * FROM orders WHERE order_id = ?", (order_id,)).fetchone()
            )

        record = OrderRecord(**order_row)
        if quote is not None and match_now:
            self.match_quote(quote, user_id=user_id)
            with self.db.connect() as conn:
                refreshed = conn.execute(
                    "SELECT * FROM orders WHERE order_id = ?", (order_id,)
                ).fetchone()
                if refreshed:
                    record = OrderRecord(**dict(refreshed))
        return record

    def _insert_rejected(
        self, conn, user_id: str, order_id: str, req: OrderRequest, short_name: str,
        price: float, tdate: str, now: str, reason: str,
    ) -> OrderRecord:
        conn.execute(
            """
            INSERT INTO orders (
                order_id, user_id, stock_code, short_name, side, price, quantity,
                filled_quantity, avg_filled_price, frozen_cash, status,
                reject_reason, strategy_name, trade_date, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 0, 0.0, 0.0, ?, ?, ?, ?, ?, ?)
            """,
            (
                order_id, user_id, req.stock_code, short_name, req.side.value, price, req.quantity,
                OrderStatus.REJECTED.value, reason, req.strategy_name, tdate, now, now,
            ),
        )
        row = conn.execute("SELECT * FROM orders WHERE order_id = ?", (order_id,)).fetchone()
        return OrderRecord(**dict(row))

    def cancel_order(self, order_id: str, user_id: str = "default") -> OrderRecord:
        now = now_str()
        with self.db.connect() as conn:
            row = conn.execute(
                "SELECT * FROM orders WHERE order_id = ? AND user_id = ?",
                (order_id, user_id),
            ).fetchone()
            if not row:
                raise ValueError(f"订单 {order_id} 不存在")
            if row["status"] not in (OrderStatus.PENDING.value, OrderStatus.PARTIAL_FILLED.value):
                raise ValueError(f"订单状态为 {row['status']}，不可撤单")

            order_uid = str(row["user_id"] or user_id)
            if row["side"] == OrderSide.BUY.value:
                refund = round(row["frozen_cash"], 2)
                if refund > 0:
                    conn.execute(
                        """
                        UPDATE account
                        SET available_cash = ROUND(available_cash + ?, 2),
                            frozen_cash = MAX(0.0, ROUND(frozen_cash - ?, 2)),
                            updated_at = ?
                        WHERE user_id = ?
                        """,
                        (refund, refund, now, order_uid),
                    )
            else:
                remaining = row["quantity"] - row["filled_quantity"]
                if remaining > 0:
                    conn.execute(
                        """
                        UPDATE positions
                        SET available_qty = available_qty + ?,
                            frozen_qty = MAX(0, frozen_qty - ?),
                            updated_at = ?
                        WHERE user_id = ? AND stock_code = ?
                        """,
                        (remaining, remaining, now, order_uid, row["stock_code"]),
                    )

            conn.execute(
                """
                UPDATE orders
                SET status = ?, frozen_cash = 0.0, updated_at = ?
                WHERE order_id = ? AND user_id = ?
                """,
                (OrderStatus.CANCELLED.value, now, order_id, order_uid),
            )
            updated = conn.execute("SELECT * FROM orders WHERE order_id = ?", (order_id,)).fetchone()
            return OrderRecord(**dict(updated))

    def match_quote(
        self, quote: QuoteSnapshot, user_id: Optional[str] = None
    ) -> list[TradeRecord]:
        """Match active orders for a stock against its 5-level book (for user_id or all users)."""
        if quote.price <= 0:
            return []
        executed_trades: list[TradeRecord] = []
        now = now_str()

        with self.db.connect() as conn:
            if user_id is not None:
                active_orders = conn.execute(
                    """
                    SELECT * FROM orders
                    WHERE stock_code = ? AND user_id = ? AND status IN (?, ?)
                    ORDER BY created_at ASC
                    """,
                    (
                        quote.stock_code,
                        user_id,
                        OrderStatus.PENDING.value,
                        OrderStatus.PARTIAL_FILLED.value,
                    ),
                ).fetchall()
            else:
                active_orders = conn.execute(
                    """
                    SELECT * FROM orders
                    WHERE stock_code = ? AND status IN (?, ?)
                    ORDER BY created_at ASC
                    """,
                    (quote.stock_code, OrderStatus.PENDING.value, OrderStatus.PARTIAL_FILLED.value),
                ).fetchall()
            if not active_orders:
                return []

            lu, ld = (
                (quote.limit_up, quote.limit_down)
                if quote.limit_up > 0 and quote.limit_down > 0
                else infer_limit_prices(quote.stock_code, quote.pre_close)
            )
            asks = quote.asks or [OrderBookLevel(price=quote.price, volume=100_000)]
            bids = quote.bids or [OrderBookLevel(price=quote.price, volume=100_000)]

            for order in active_orders:
                order_uid = str(order["user_id"] or "default")
                side = OrderSide(order["side"])
                remaining = order["quantity"] - order["filled_quantity"]
                if remaining <= 0:
                    continue

                if side == OrderSide.BUY and quote.asks and quote.asks[0].volume == 0 and lu > 0 and quote.price >= lu - 0.005:
                    continue
                if side == OrderSide.SELL and quote.bids and quote.bids[0].volume == 0 and ld > 0 and quote.price <= ld + 0.005:
                    continue

                levels = asks if side == OrderSide.BUY else bids
                fills: list[tuple[float, int]] = []
                rem = remaining
                for lvl in levels:
                    if rem <= 0 or lvl.price <= 0 or lvl.volume <= 0:
                        continue
                    price_ok = (lvl.price <= order["price"]) if side == OrderSide.BUY else (lvl.price >= order["price"])
                    if not price_ok:
                        continue
                    avail_vol = (lvl.volume // 100) * 100 if rem >= 100 else lvl.volume
                    take_qty = min(rem, avail_vol)
                    if take_qty <= 0:
                        continue
                    fills.append((round(lvl.price, 2), take_qty))
                    rem -= take_qty

                if not fills:
                    continue

                total_fill_qty = sum(q for _, q in fills)
                total_fill_amt = round(sum(p * q for p, q in fills), 2)
                batch_avg_price = round(total_fill_amt / total_fill_qty, 4)
                commission, transfer_fee, stamp_tax, total_fee = calc_fees(side, total_fill_amt)

                trade_id = uuid.uuid4().hex[:12].upper()
                tdate = order["trade_date"]
                conn.execute(
                    """
                    INSERT INTO trades (
                        trade_id, user_id, order_id, stock_code, short_name, side, price, quantity,
                        amount, commission, transfer_fee, stamp_tax, total_fee, trade_date, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        trade_id, order_uid, order["order_id"], order["stock_code"], order["short_name"],
                        side.value, round(batch_avg_price, 3), total_fill_qty, total_fill_amt,
                        commission, transfer_fee, stamp_tax, total_fee, tdate, now,
                    ),
                )
                t_row = dict(conn.execute("SELECT * FROM trades WHERE trade_id = ?", (trade_id,)).fetchone())
                executed_trades.append(TradeRecord(**t_row))

                new_filled_qty = order["filled_quantity"] + total_fill_qty
                prev_amt = order["filled_quantity"] * order["avg_filled_price"]
                new_avg_price = round((prev_amt + total_fill_amt) / new_filled_qty, 4)
                is_done = new_filled_qty >= order["quantity"]
                new_status = OrderStatus.FILLED.value if is_done else OrderStatus.PARTIAL_FILLED.value

                if side == OrderSide.BUY:
                    actual_cost = round(total_fill_amt + total_fee, 2)
                    cur_frozen = order["frozen_cash"]
                    used_from_frozen = min(cur_frozen, actual_cost)
                    extra_from_cash = round(actual_cost - used_from_frozen, 2)
                    remaining_frozen = round(cur_frozen - used_from_frozen, 2)

                    refund_to_cash = remaining_frozen if is_done else 0.0
                    final_order_frozen = 0.0 if is_done else remaining_frozen

                    conn.execute(
                        """
                        UPDATE account
                        SET frozen_cash = MAX(0.0, ROUND(frozen_cash - ?, 2)),
                            available_cash = ROUND(available_cash + ? - ?, 2),
                            updated_at = ?
                        WHERE user_id = ?
                        """,
                        (used_from_frozen + refund_to_cash, refund_to_cash, extra_from_cash, now, order_uid),
                    )
                    conn.execute(
                        """
                        UPDATE orders
                        SET filled_quantity = ?, avg_filled_price = ?, frozen_cash = ?,
                            status = ?, updated_at = ?
                        WHERE order_id = ?
                        """,
                        (new_filled_qty, new_avg_price, final_order_frozen, new_status, now, order["order_id"]),
                    )
                    self._upsert_buy_position(
                        conn, order_uid, order["stock_code"], order["short_name"], total_fill_qty, actual_cost, now
                    )
                else:
                    net_proceeds = round(total_fill_amt - total_fee, 2)
                    conn.execute(
                        """
                        UPDATE account
                        SET available_cash = ROUND(available_cash + ?, 2),
                            updated_at = ?
                        WHERE user_id = ?
                        """,
                        (net_proceeds, now, order_uid),
                    )
                    conn.execute(
                        """
                        UPDATE orders
                        SET filled_quantity = ?, avg_filled_price = ?, status = ?, updated_at = ?
                        WHERE order_id = ?
                        """,
                        (new_filled_qty, new_avg_price, new_status, now, order["order_id"]),
                    )
                    self._deduct_sell_position(conn, order_uid, order["stock_code"], total_fill_qty, now)

        return executed_trades

    def _upsert_buy_position(
        self, conn, user_id: str, stock_code: str, short_name: str, qty: int, total_cost_with_fee: float, now: str
    ) -> None:
        pos = conn.execute(
            "SELECT * FROM positions WHERE user_id = ? AND stock_code = ?",
            (user_id, stock_code),
        ).fetchone()
        if not pos:
            avg_cost = total_cost_with_fee / qty
            conn.execute(
                """
                INSERT INTO positions (
                    user_id, stock_code, short_name, total_qty, available_qty, today_bought_qty, frozen_qty, avg_cost, updated_at
                ) VALUES (?, ?, ?, ?, 0, ?, 0, ?, ?)
                """,
                (user_id, stock_code, short_name, qty, qty, avg_cost, now),
            )
        else:
            new_total = pos["total_qty"] + qty
            old_cost = pos["total_qty"] * pos["avg_cost"]
            new_avg_cost = (old_cost + total_cost_with_fee) / new_total
            conn.execute(
                """
                UPDATE positions
                SET short_name = ?, total_qty = ?, today_bought_qty = today_bought_qty + ?,
                    avg_cost = ?, updated_at = ?
                WHERE user_id = ? AND stock_code = ?
                """,
                (short_name, new_total, qty, new_avg_cost, now, user_id, stock_code),
            )

    def _deduct_sell_position(
        self, conn, user_id: str, stock_code: str, qty: int, now: str
    ) -> None:
        pos = conn.execute(
            "SELECT * FROM positions WHERE user_id = ? AND stock_code = ?",
            (user_id, stock_code),
        ).fetchone()
        if not pos:
            return
        new_total = max(0, pos["total_qty"] - qty)
        new_frozen = max(0, pos["frozen_qty"] - qty)
        if new_total == 0:
            conn.execute(
                "DELETE FROM positions WHERE user_id = ? AND stock_code = ?",
                (user_id, stock_code),
            )
        else:
            conn.execute(
                """
                UPDATE positions
                SET total_qty = ?, frozen_qty = ?, updated_at = ?
                WHERE user_id = ? AND stock_code = ?
                """,
                (new_total, new_frozen, now, user_id, stock_code),
            )

    def settle_day(
        self,
        quotes: dict[str, QuoteSnapshot] | None = None,
        next_trade_date: str | None = None,
        user_id: str = "default",
    ) -> AccountSummary:
        """End-of-day settlement for one user: cancel open orders, unlock T+1 shares, save NAV."""
        now = now_str()
        with self.db.connect() as conn:
            open_orders = conn.execute(
                "SELECT order_id FROM orders WHERE user_id = ? AND status IN (?, ?)",
                (user_id, OrderStatus.PENDING.value, OrderStatus.PARTIAL_FILLED.value),
            ).fetchall()
        for row in open_orders:
            self.cancel_order(row["order_id"], user_id=user_id)

        summary_before = self.get_account_summary(quotes, user_id=user_id)
        with self.db.connect() as conn:
            conn.execute(
                """
                UPDATE positions
                SET available_qty = total_qty, today_bought_qty = 0, frozen_qty = 0, updated_at = ?
                WHERE user_id = ?
                """,
                (now, user_id),
            )
            prev_nav = conn.execute(
                "SELECT total_assets FROM nav_history WHERE user_id = ? AND trade_date < ? ORDER BY trade_date DESC LIMIT 1",
                (user_id, summary_before.current_trade_date),
            ).fetchone()
            base_assets = prev_nav["total_assets"] if prev_nav else summary_before.initial_cash
            daily_ret = (
                round(((summary_before.total_assets - base_assets) / base_assets) * 100, 2)
                if base_assets > 0
                else 0.0
            )
            conn.execute(
                """
                INSERT OR REPLACE INTO nav_history (user_id, trade_date, total_assets, available_cash, market_value, daily_return_pct, recorded_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    summary_before.current_trade_date,
                    summary_before.total_assets,
                    summary_before.available_cash,
                    summary_before.market_value,
                    daily_ret,
                    now,
                ),
            )

            if not next_trade_date:
                cur_dt = datetime.strptime(summary_before.current_trade_date, "%Y-%m-%d")
                nxt = cur_dt + timedelta(days=1)
                while nxt.weekday() >= 5:
                    nxt += timedelta(days=1)
                next_trade_date = nxt.strftime("%Y-%m-%d")

            conn.execute(
                "UPDATE account SET current_trade_date = ?, updated_at = ? WHERE user_id = ?",
                (next_trade_date, now, user_id),
            )

        return self.get_account_summary(quotes, user_id=user_id)
