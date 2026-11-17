from __future__ import annotations

from pathlib import Path

from app.engine.paper_broker import PaperBroker, calc_fees, infer_limit_prices
from app.models import OrderBookLevel, OrderRequest, OrderSide, OrderStatus, QuoteSnapshot, RiskConfig
from app.storage import Database


def make_quote(
    code: str = "000001",
    name: str = "平安银行",
    price: float = 10.00,
    pre_close: float = 10.00,
    asks: list[OrderBookLevel] | None = None,
    bids: list[OrderBookLevel] | None = None,
) -> QuoteSnapshot:
    lu, ld = infer_limit_prices(code, pre_close)
    return QuoteSnapshot(
        stock_code=code,
        short_name=name,
        price=price,
        pre_close=pre_close,
        open=pre_close,
        high=price,
        low=price,
        limit_up=lu,
        limit_down=ld,
        asks=asks if asks is not None else [OrderBookLevel(price=price, volume=50_000)],
        bids=bids if bids is not None else [OrderBookLevel(price=round(price - 0.01, 2), volume=50_000)],
    )


def test_fee_calculation() -> None:
    comm, trans, stamp, total = calc_fees(OrderSide.BUY, 10_000.0)
    assert comm == 5.0  # 10000 * 0.00025 = 2.5 -> min 5.0
    assert trans == 0.1
    assert stamp == 0.0
    assert total == 5.1

    comm_s, trans_s, stamp_s, total_s = calc_fees(OrderSide.SELL, 100_000.0)
    assert comm_s == 25.0
    assert trans_s == 1.0
    assert stamp_s == 50.0  # sell stamp tax 0.05%
    assert total_s == 76.0


def test_buy_lot_and_limit_price_validation(tmp_path: Path) -> None:
    db = Database(tmp_path / "t1.db")
    broker = PaperBroker(db)
    q = make_quote(price=10.00, pre_close=10.00)

    # Non-multiple of 100 rejected
    o_bad_qty = broker.place_order(
        OrderRequest(stock_code="000001", side=OrderSide.BUY, price=10.00, quantity=150), q
    )
    assert o_bad_qty.status == OrderStatus.REJECTED

    # Exceeds limit_up (11.00) rejected
    o_bad_px = broker.place_order(
        OrderRequest(stock_code="000001", side=OrderSide.BUY, price=11.50, quantity=100), q
    )
    assert o_bad_px.status == OrderStatus.REJECTED


def test_buy_price_improvement_refund_and_t_plus_1(tmp_path: Path) -> None:
    db = Database(tmp_path / "t2.db")
    broker = PaperBroker(db)
    q = make_quote(
        price=10.00,
        pre_close=10.00,
        asks=[OrderBookLevel(price=10.00, volume=10_000)],
    )

    # Place buy limit @ 10.50 -> fills @ ask1 10.00, unused frozen cash refunded
    order = broker.place_order(
        OrderRequest(stock_code="000001", side=OrderSide.BUY, price=10.50, quantity=1000), q
    )
    assert order.status == OrderStatus.FILLED
    assert order.filled_quantity == 1000
    assert order.avg_filled_price == 10.00

    summary = broker.get_account_summary({"000001": q})
    assert summary.frozen_cash == 0.0
    # 1,000,000 - (1000*10.0 + 5.0 + 0.1) = 989,994.90
    assert summary.available_cash == 989_994.90

    # Check T+1: total 1000, available 0 today
    positions = broker.get_positions({"000001": q})
    assert len(positions) == 1
    assert positions[0].total_qty == 1000
    assert positions[0].available_qty == 0
    assert positions[0].today_bought_qty == 1000

    # Same-day sell must be rejected by T+1 guard
    sell_today = broker.place_order(
        OrderRequest(stock_code="000001", side=OrderSide.SELL, price=10.20, quantity=500), q
    )
    assert sell_today.status == OrderStatus.REJECTED
    assert "T+1" in sell_today.reject_reason

    # After EOD settlement, shares unlock for selling on T+1
    q_next = make_quote(
        price=10.50,
        pre_close=10.00,
        bids=[OrderBookLevel(price=10.50, volume=10_000)],
    )
    broker.settle_day({"000001": q_next})
    positions_t1 = broker.get_positions({"000001": q_next})
    assert positions_t1[0].available_qty == 1000
    assert positions_t1[0].today_bought_qty == 0

    sell_t1 = broker.place_order(
        OrderRequest(stock_code="000001", side=OrderSide.SELL, price=10.50, quantity=1000), q_next
    )
    assert sell_t1.status == OrderStatus.FILLED
    assert broker.get_positions({"000001": q_next}) == []


def test_partial_fill_and_cancel_refund(tmp_path: Path) -> None:
    db = Database(tmp_path / "t3.db")
    broker = PaperBroker(db)
    q = make_quote(
        price=10.10,
        pre_close=10.00,
        asks=[
            OrderBookLevel(price=10.10, volume=200),
            OrderBookLevel(price=10.20, volume=200),
            OrderBookLevel(price=10.80, volume=500),  # above limit price
        ],
    )

    # Order 600 @ 10.30 -> fills 400 across ask1+ask2, 200 remains pending
    order = broker.place_order(
        OrderRequest(stock_code="000001", side=OrderSide.BUY, price=10.30, quantity=600), q
    )
    assert order.status == OrderStatus.PARTIAL_FILLED
    assert order.filled_quantity == 400
    assert round(order.avg_filled_price, 2) == 10.15

    # Cancel remaining 200 -> frozen cash becomes 0 and unspent returned
    cancelled = broker.cancel_order(order.order_id)
    assert cancelled.status == OrderStatus.CANCELLED
    assert cancelled.frozen_cash == 0.0
    summary = broker.get_account_summary({"000001": q})
    assert summary.frozen_cash == 0.0


def test_sealed_limit_up_blocks_buy(tmp_path: Path) -> None:
    db = Database(tmp_path / "t4.db")
    broker = PaperBroker(db)
    q_sealed = make_quote(
        price=11.00,
        pre_close=10.00,
        asks=[OrderBookLevel(price=11.00, volume=0)],
        bids=[OrderBookLevel(price=11.00, volume=500_000)],
    )
    order = broker.place_order(
        OrderRequest(stock_code="000001", side=OrderSide.BUY, price=11.00, quantity=100),
        q_sealed,
    )
    assert order.status == OrderStatus.PENDING
    assert order.filled_quantity == 0


def test_risk_guardrails_max_amount_and_single_position_including_pending(tmp_path: Path) -> None:
    db = Database(tmp_path / "risk1.db")
    broker = PaperBroker(db)
    broker.update_risk_config(
        RiskConfig(
            max_single_position_pct=20.0,
            max_order_amount=120_000.0,
            max_total_loss_pct=0.0,
            block_buy_on_loss=True,
        )
    )
    q = make_quote(price=10.00, pre_close=10.00)

    # 1. Single order exceeds max_order_amount (10 * 13000 = 130,000 > 120,000)
    o_amt = broker.place_order(
        OrderRequest(stock_code="000001", side=OrderSide.BUY, price=10.00, quantity=13_000),
        q,
    )
    assert o_amt.status == OrderStatus.REJECTED
    assert "单笔委托金额" in o_amt.reject_reason

    # 2. First pending order freezes 100,000 (~10% of 1,000,000)
    o_pending = broker.place_order(
        OrderRequest(stock_code="000001", side=OrderSide.BUY, price=10.00, quantity=10_000),
        q,
        match_now=False,
    )
    assert o_pending.status == OrderStatus.PENDING

    # 3. Second order (110,000) pushes total 000001 exposure over 20% -> rejected
    o_pos_limit = broker.place_order(
        OrderRequest(stock_code="000001", side=OrderSide.BUY, price=10.00, quantity=11_000),
        q,
        match_now=False,
    )
    assert o_pos_limit.status == OrderStatus.REJECTED
    assert "仓位将达" in o_pos_limit.reject_reason

    # 4. Smaller order within 20% cap succeeds
    o_ok = broker.place_order(
        OrderRequest(stock_code="000001", side=OrderSide.BUY, price=10.00, quantity=9_000),
        q,
        match_now=True,
    )
    assert o_ok.status == OrderStatus.FILLED


def test_risk_circuit_breaker_blocks_buy_allows_sell(tmp_path: Path) -> None:
    db = Database(tmp_path / "risk2.db")
    broker = PaperBroker(db)
    q_buy = make_quote(
        price=10.00,
        pre_close=10.00,
        asks=[OrderBookLevel(price=10.00, volume=100_000)],
    )
    o_buy = broker.place_order(
        OrderRequest(stock_code="000001", side=OrderSide.BUY, price=10.00, quantity=20_000),
        q_buy,
    )
    assert o_buy.status == OrderStatus.FILLED

    # Drop price to 8.00 (~4% total asset drawdown on 1M initial_cash) and unlock T+1
    q_drop = make_quote(
        price=8.00,
        pre_close=8.00,
        asks=[OrderBookLevel(price=8.00, volume=100_000)],
        bids=[OrderBookLevel(price=8.00, volume=100_000)],
    )
    broker.settle_day({"000001": q_drop})
    broker.update_risk_config(
        RiskConfig(
            max_single_position_pct=100.0,
            max_order_amount=0.0,
            max_total_loss_pct=3.5,
            block_buy_on_loss=True,
        )
    )

    # Buy blocked by total drawdown circuit breaker
    o_blocked = broker.place_order(
        OrderRequest(stock_code="000001", side=OrderSide.BUY, price=8.00, quantity=100),
        q_drop,
        quotes_map={"000001": q_drop},
    )
    assert o_blocked.status == OrderStatus.REJECTED
    assert "风控熔断" in o_blocked.reject_reason

    # Sell remains allowed to cut losses
    o_sell = broker.place_order(
        OrderRequest(stock_code="000001", side=OrderSide.SELL, price=8.00, quantity=5_000),
        q_drop,
        quotes_map={"000001": q_drop},
    )
    assert o_sell.status == OrderStatus.FILLED
