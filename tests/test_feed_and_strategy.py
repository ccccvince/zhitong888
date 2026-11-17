from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from app.engine.backtester import run_backtest
from app.engine.market_feed import (
    MarketDataManager,
    classify_board,
    compute_kline_indicators,
    compute_macd_series,
    is_limit_up_bar,
    parse_eastmoney_zt_pool_item,
    parse_tencent_indices,
    parse_tencent_payload,
)
from app.engine.paper_broker import PaperBroker
from app.engine.strategy_runner import StrategyRunner
from app.models import (
    AutoStrategyRuleRequest,
    BacktestRequest,
    MarketMode,
    OrderBookLevel,
    OrderSide,
    OrderStatus,
    QuoteSnapshot,
)
from app.storage import Database


def test_parse_tencent_payload_and_orderbook_lots_to_shares() -> None:
    fields = [""] * 55
    fields[1] = "平安银行"
    fields[2] = "000001"
    fields[3] = "12.17"
    fields[4] = "12.30"
    fields[5] = "12.25"
    fields[6] = "1000"  # 1000 lots -> 100,000 shares
    fields[9], fields[10] = "12.17", "110"
    fields[11], fields[12] = "12.16", "200"
    fields[19], fields[20] = "12.18", "144"
    fields[21], fields[22] = "12.19", "191"
    fields[31] = "-0.13"
    fields[32] = "-1.06"
    fields[33] = "12.32"
    fields[34] = "12.11"
    fields[35] = "12.17/1000/1217000"
    fields[38] = "0.53"
    fields[39] = "5.43"
    fields[43] = "1.71"
    fields[47] = "13.53"
    fields[48] = "11.07"
    payload = 'v_sz000001="' + "~".join(fields) + '";'

    parsed = parse_tencent_payload(payload)
    assert "000001" in parsed
    q = parsed["000001"]
    assert q.short_name == "平安银行"
    assert q.price == 12.17
    assert q.volume == 100_000.0
    assert q.turnover_ratio == 0.53
    assert q.pe_ratio == 5.43
    assert q.amplitude == 1.71
    assert q.bids[0].price == 12.17 and q.bids[0].volume == 11_000
    assert q.asks[0].price == 12.18 and q.asks[0].volume == 14_400
    assert q.limit_up == 13.53 and q.limit_down == 11.07


def test_parse_tencent_indices_no_symbol_collision_with_000001() -> None:
    idx_fields = [""] * 50
    idx_fields[1] = "上证指数"
    idx_fields[2] = "000001"
    idx_fields[3] = "3602.07"
    idx_fields[4] = "3602.57"
    idx_fields[31] = "-0.50"
    idx_fields[32] = "-0.01"
    idx_fields[35] = "3602.07/274584385/424293166077"

    stk_fields = [""] * 50
    stk_fields[1] = "平安银行"
    stk_fields[2] = "000001"
    stk_fields[3] = "12.17"
    stk_fields[4] = "12.30"
    stk_fields[31] = "-0.13"
    stk_fields[32] = "-1.06"
    stk_fields[35] = "12.17/1000/1217000"

    mixed = (
        'v_sh000001="' + "~".join(idx_fields) + '";\n'
        'v_sz000001="' + "~".join(stk_fields) + '";\n'
    )
    indices = parse_tencent_indices(mixed)
    stocks = parse_tencent_payload(mixed)

    assert list(indices.keys()) == ["sh000001"]
    assert indices["sh000001"].name == "上证指数"
    assert indices["sh000001"].price == 3602.07
    assert indices["sh000001"].amount == 424_293_166_077.0

    assert stocks["000001"].short_name == "平安银行"
    assert stocks["000001"].price == 12.17


def test_compute_kline_indicators_and_offline_lookback() -> None:
    bars = [
        {"trade_date": f"2026-01-{i+1:02d}", "open": float(10 + i), "high": float(11 + i), "low": float(9 + i), "close": float(10 + i), "volume": 1000.0}
        for i in range(25)
    ]
    enriched = compute_kline_indicators(bars)
    assert enriched[3]["ma5"] is None
    assert enriched[4]["ma5"] == 12.0  # mean(10,11,12,13,14)
    assert enriched[9]["ma10"] == 14.5
    assert enriched[18]["ma20"] is None
    assert enriched[19]["ma20"] == 19.5
    assert enriched[-1]["dif"] > 0
    assert enriched[-1]["macd"] == round(2.0 * (enriched[-1]["dif"] - enriched[-1]["dea"]), 3)

    mgr = MarketDataManager(network_enabled=False)
    daily = mgr.fetch_daily_kline("600519", days=60)
    assert len(daily) == 60
    # Warmed-up lookback ensures MA20 & MACD are populated from bar 0
    assert daily[0]["ma20"] is not None
    assert "dif" in daily[0] and "dea" in daily[0] and "macd" in daily[0]

    indices = mgr.fetch_indices(MarketMode.SIMULATED)
    assert len(indices) == 5
    assert {idx.symbol for idx in indices} == {"sh000001", "sz399001", "sz399006", "sh000688", "sh000300"}


def test_strategy_runner_grid_and_breakout(tmp_path: Path) -> None:
    db = Database(tmp_path / "strat.db")
    broker = PaperBroker(db)
    runner = StrategyRunner(db, broker)

    runner.add_strategy(
        AutoStrategyRuleRequest(
            stock_code="000001",
            strategy_type="BREAKOUT_BUY",
            upper_price=12.50,
            order_qty=200,
        )
    )

    q_below = {
        "000001": QuoteSnapshot(
            stock_code="000001",
            short_name="平安银行",
            price=12.20,
            pre_close=12.00,
            limit_up=13.20,
            limit_down=10.80,
            asks=[OrderBookLevel(price=12.20, volume=10_000)],
            bids=[OrderBookLevel(price=12.19, volume=10_000)],
        )
    }
    assert runner.evaluate(q_below, can_trade=True) == []

    q_breakout = {
        "000001": QuoteSnapshot(
            stock_code="000001",
            short_name="平安银行",
            price=12.60,
            pre_close=12.00,
            limit_up=13.20,
            limit_down=10.80,
            asks=[OrderBookLevel(price=12.60, volume=10_000)],
            bids=[OrderBookLevel(price=12.59, volume=10_000)],
        )
    }
    orders = runner.evaluate(q_breakout, can_trade=True)
    assert len(orders) == 1
    assert orders[0].status == OrderStatus.FILLED
    assert orders[0].quantity == 200
    # One-shot BREAKOUT_BUY automatically disables after trigger
    assert runner.list_strategies()[0].enabled is False


def test_strategy_runner_ma_cross_and_same_day_dedup(tmp_path: Path) -> None:
    db = Database(tmp_path / "strat_ma.db")
    broker = PaperBroker(db)
    runner = StrategyRunner(db, broker)

    with pytest.raises(ValueError):
        runner.add_strategy(
            AutoStrategyRuleRequest(
                stock_code="000001",
                strategy_type="MA_CROSS",
                fast_period=10,
                slow_period=5,
                order_qty=100,
            )
        )

    rule = runner.add_strategy(
        AutoStrategyRuleRequest(
            stock_code="000001",
            strategy_type="MA_CROSS",
            fast_period=3,
            slow_period=5,
            order_qty=200,
        )
    )

    # Prior 5 bars strictly before today: MA3=10.3333 <= MA5=10.6000
    prior_closes = [11.0, 11.0, 11.0, 10.0, 10.0]
    bars = [
        {"trade_date": f"2026-01-{i+1:02d}", "close": c}
        for i, c in enumerate(prior_closes)
    ]
    # Live quote at 13.50 lifts MA3=11.1667 > MA5=11.1000 -> Golden Cross BUY
    q_up = {
        "000001": QuoteSnapshot(
            stock_code="000001",
            short_name="平安银行",
            price=13.50,
            pre_close=12.50,
            limit_up=13.75,
            limit_down=11.25,
            asks=[OrderBookLevel(price=13.50, volume=10_000)],
            bids=[OrderBookLevel(price=13.49, volume=10_000)],
        )
    }
    buy_orders = runner.evaluate(q_up, can_trade=True, daily_bars_map={"000001": bars})
    assert len(buy_orders) == 1
    assert buy_orders[0].side == OrderSide.BUY
    assert buy_orders[0].status == OrderStatus.FILLED

    saved = runner.list_strategies()[0]
    assert ":BUY" in saved.last_signal_tag

    # Even if cooldown timestamp is cleared, same-day tag prevents duplicate BUY order
    with db.connect() as conn:
        conn.execute(
            "UPDATE auto_strategies SET last_action_at = '2020-01-01 09:30:00' WHERE strategy_id = ?",
            (rule.strategy_id,),
        )
    assert runner.evaluate(q_up, can_trade=True, daily_bars_map={"000001": bars}) == []


def test_strategy_runner_macd_cross(tmp_path: Path) -> None:
    db = Database(tmp_path / "strat_macd.db")
    broker = PaperBroker(db)
    runner = StrategyRunner(db, broker)
    runner.add_strategy(
        AutoStrategyRuleRequest(
            stock_code="000001",
            strategy_type="MACD_CROSS",
            order_qty=100,
        )
    )

    # Steady decline then strong rebound flips DIF from below DEA to above DEA
    hist_closes = [round(13.0 - 0.1 * i, 2) for i in range(25)]  # 13.0 -> 10.6
    # Verify our chosen series guarantees DIF <= DEA prior bar and DIF > DEA on 11.65
    macd_check = compute_macd_series(hist_closes + [11.65], 12, 26, 9)
    assert macd_check[-2][0] <= macd_check[-2][1] and macd_check[-1][0] > macd_check[-1][1]

    bars = [{"trade_date": f"2026-02-{i+1:02d}", "close": c} for i, c in enumerate(hist_closes)]
    q_rebound = {
        "000001": QuoteSnapshot(
            stock_code="000001",
            short_name="平安银行",
            price=11.65,
            pre_close=10.60,
            limit_up=11.66,
            limit_down=9.54,
            asks=[OrderBookLevel(price=11.65, volume=10_000)],
            bids=[OrderBookLevel(price=11.64, volume=10_000)],
        )
    }
    orders = runner.evaluate(q_rebound, can_trade=True, daily_bars_map={"000001": bars})
    assert len(orders) == 1
    assert orders[0].side == OrderSide.BUY
    assert "MACD" in orders[0].strategy_name


def test_backtester_strategies_t_plus_1_and_validation() -> None:
    # Build 120 daily bars with two complete smooth sine-like waves so MA/MACD/GRID trade cleanly
    import math

    start = datetime(2026, 1, 1)
    bars: list[dict] = []
    for i in range(120):
        close = round(10.0 + 1.8 * math.sin(i / 6.5), 2)
        bars.append(
            {
                "trade_date": (start + timedelta(days=i)).strftime("%Y-%m-%d"),
                "open": close,
                "high": round(close * 1.01, 2),
                "low": round(close * 0.99, 2),
                "close": close,
                "volume": 100_000.0,
            }
        )

    for stype in ("MA_CROSS", "MACD_CROSS", "GRID"):
        req = BacktestRequest(
            stock_code="000001",
            strategy_type=stype,
            days=90,
            initial_cash=500_000.0,
            position_pct=90.0,
            fast_period=5,
            slow_period=15,
            grid_lower=8.8,
            grid_upper=11.4,
            stop_loss_pct=0.0,
            take_profit_pct=0.0,
        )
        res = run_backtest(req, bars, short_name="平安银行")
        assert res.bar_count == 90
        assert len(res.equity_curve) == 90
        assert res.trade_count >= 2
        assert res.closed_trades >= 1
        assert 0.0 <= res.win_rate_pct <= 100.0
        assert res.max_drawdown_pct >= 0.0
        for sig in res.signals:
            assert sig.quantity % 100 == 0
            assert sig.fee >= 5.0
        # T+1 check: no BUY and SELL on the same trade_date
        buy_dates = {s.trade_date for s in res.signals if s.side == "BUY"}
        sell_dates = {s.trade_date for s in res.signals if s.side == "SELL"}
        assert buy_dates.isdisjoint(sell_dates)

    with pytest.raises(ValueError):
        run_backtest(
            BacktestRequest(stock_code="000001", strategy_type="MA_CROSS", fast_period=20, slow_period=10),
            bars,
        )
    with pytest.raises(ValueError):
        run_backtest(
            BacktestRequest(stock_code="000001", strategy_type="GRID", grid_lower=12.0, grid_upper=10.0),
            bars,
        )


def test_limit_up_screener_helpers_and_offline_filtering(monkeypatch) -> None:
    assert classify_board("600519") == "主板"
    assert classify_board("300750") == "创业板"
    assert classify_board("688981") == "科创板"
    assert classify_board("830799") == "北交所"
    assert classify_board("920001") == "北交所"

    assert is_limit_up_bar("000001", 11.00, 10.00) is True
    assert is_limit_up_bar("000001", 10.99, 10.00) is False
    assert is_limit_up_bar("300750", 12.00, 10.00) is True
    assert is_limit_up_bar("688981", 11.98, 10.00) is False
    assert is_limit_up_bar("830799", 13.00, 10.00) is True
    assert is_limit_up_bar("600001", 10.50, 10.00, short_name="*ST示例") is True
    assert is_limit_up_bar("600001", 10.48, 10.00, short_name="ST示例") is False

    em = parse_eastmoney_zt_pool_item(
        {"c": "300750", "n": "宁德时代", "p": 265400, "zdp": 20.0, "amount": 1.2e9, "hs": 5.5, "lbc": 2, "hybk": "电池"},
        "2026-11-16",
    )
    assert em is not None
    assert em["price"] == 265.40
    assert em["lbc"] == 2
    assert em["industry"] == "电池"
    assert parse_eastmoney_zt_pool_item({"c": "BAD"}, "2026-11-16") is None

    mgr = MarketDataManager(network_enabled=False)

    def fake_daily(code: str, days: int = 60) -> list[dict]:
        dates = ["2026-11-10", "2026-11-11", "2026-11-12", "2026-11-13", "2026-11-14", "2026-11-15"]
        if code == "000001":
            # Two consecutive 10% limit-up days: 11.00 -> 12.10
            closes = [10.00, 10.00, 11.00, 12.10, 12.05, 12.00]
        elif code == "300750":
            # One 20% limit-up day: 100.00 -> 120.00
            closes = [100.00, 100.00, 100.00, 100.00, 120.00, 119.50]
        else:
            closes = [50.0, 50.2, 50.1, 50.3, 50.4, 50.2]
        return [
            {
                "trade_date": d,
                "open": c,
                "high": c,
                "low": c,
                "close": c,
                "volume": 100_000.0,
                "amount": c * 100_000.0,
                "change_pct": 0.0,
                "turnover_ratio": 1.2,
            }
            for d, c in zip(dates, closes)
        ]

    monkeypatch.setattr(mgr, "fetch_daily_kline", fake_daily)
    watch = [("000001", "平安银行"), ("600519", "贵州茅台")]

    res_all = mgr.screen_limit_up_stocks(days=5, board="ALL", min_limit_count=1, watchlist_items=watch)
    assert res_all.source == "kline_scan"
    assert [it.stock_code for it in res_all.items] == ["000001", "300750"]
    assert res_all.items[0].limit_up_count == 2
    assert res_all.items[0].max_consecutive_boards == 2
    assert res_all.items[0].in_watchlist is True
    assert res_all.items[1].board == "创业板"
    assert res_all.items[1].in_watchlist is False

    res_cyb = mgr.screen_limit_up_stocks(days=5, board="创业板", watchlist_items=watch)
    assert [it.stock_code for it in res_cyb.items] == ["300750"]

    res_2plus = mgr.screen_limit_up_stocks(days=5, min_limit_count=2, watchlist_items=watch)
    assert [it.stock_code for it in res_2plus.items] == ["000001"]

    res_watch = mgr.screen_limit_up_stocks(days=5, only_watchlist=True, watchlist_items=watch)
    assert [it.stock_code for it in res_watch.items] == ["000001"]

    res_kw = mgr.screen_limit_up_stocks(days=5, keyword="宁德", watchlist_items=watch)
    assert [it.stock_code for it in res_kw.items] == ["300750"]
