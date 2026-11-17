from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.main import create_app


def test_full_api_workflow(tmp_path: Path) -> None:
    app = create_app(
        db_path=tmp_path / "api_test.db",
        network_enabled=False,
        enable_background_loop=False,
        auth_required=False,
    )

    with TestClient(app) as client:
        # 1. Home HTML & initial state
        html = client.get("/")
        assert html.status_code == 200
        assert "A股智能盯盘与模拟交易系统" in html.text

        st = client.get("/api/state").json()
        assert st["account"]["initial_cash"] == 1_000_000.0
        assert len(st["watchlist"]) >= 6
        assert len(st["indices"]) == 5
        assert {idx["symbol"] for idx in st["indices"]} == {"sh000001", "sz399001", "sz399006", "sh000688", "sh000300"}
        assert "turnover_ratio" in st["watchlist"][0]["quote"]
        assert "pe_ratio" in st["watchlist"][0]["quote"]
        assert "amplitude" in st["watchlist"][0]["quote"]

        # 2. Switch to SIMULATED mode & step tick
        mode_res = client.post("/api/mode", json={"mode": "SIMULATED"}).json()
        assert mode_res["account"]["market_mode"] == "SIMULATED"
        assert mode_res["session"]["effective_can_match"] is True

        tick_res = client.post("/api/sim/tick").json()
        assert tick_res["watchlist"][0]["quote"]["source"] == "simulated_tick"

        # 3. Stock search & watchlist add/delete
        search = client.get("/api/stocks/search?q=601318").json()
        assert any(s["stock_code"] == "601318" for s in search)

        wl_add = client.post("/api/watchlist", json={"stock_code": "601318", "short_name": "中国平安"})
        assert wl_add.status_code == 200
        assert any(w["stock_code"] == "601318" for w in wl_add.json()["watchlist"])

        # 4. Chart data (intraday + daily bars with MA & MACD)
        chart = client.get("/api/quotes/000001/chart").json()
        assert len(chart["mins"]) >= 1
        assert len(chart["daily"]) == 60
        assert chart["daily"][0]["ma5"] is not None
        assert chart["daily"][0]["ma20"] is not None
        assert "macd" in chart["daily"][-1] and "dif" in chart["daily"][-1] and "dea" in chart["daily"][-1]

        # 5. Place buy order in SIMULATED mode using limit_up/current price -> immediate fill
        q_000001 = next(w["quote"] for w in tick_res["watchlist"] if w["stock_code"] == "000001")
        buy_px = q_000001["asks"][0]["price"] if q_000001["asks"] else q_000001["price"]
        buy_res = client.post(
            "/api/orders",
            json={
                "stock_code": "000001",
                "side": "BUY",
                "price": buy_px,
                "quantity": 500,
            },
        ).json()
        assert buy_res["order"]["status"] == "FILLED"
        assert buy_res["state"]["positions"][0]["total_qty"] == 500
        assert buy_res["state"]["positions"][0]["available_qty"] == 0

        # Same-day sell rejected by T+1
        sell_same_day = client.post(
            "/api/orders",
            json={
                "stock_code": "000001",
                "side": "SELL",
                "price": buy_px,
                "quantity": 500,
            },
        ).json()
        assert sell_same_day["order"]["status"] == "REJECTED"

        # Settle day unlocks shares
        settle_res = client.post("/api/account/settle").json()
        assert settle_res["positions"][0]["available_qty"] == 500

        # 6. Alert rule triggers immediately on PRICE_ABOVE 1.0
        alert_res = client.post(
            "/api/alerts",
            json={
                "stock_code": "000001",
                "condition": "PRICE_ABOVE",
                "threshold": 1.0,
                "cooldown_sec": 60,
                "note": "API测试",
            },
        ).json()
        assert alert_res["rule"]["stock_code"] == "000001"
        assert len(alert_res["state"]["alert_events"]) >= 1

        # 7. Auto strategy creation
        strat_res = client.post(
            "/api/strategies",
            json={
                "stock_code": "600519",
                "strategy_type": "GRID_MEAN_REVERT",
                "lower_price": 1200.0,
                "upper_price": 1800.0,
                "order_qty": 100,
            },
        ).json()
        assert strat_res["strategy"]["strategy_type"] == "GRID_MEAN_REVERT"

        # 8. Watchlist metadata & pinning
        patch_wl = client.patch(
            "/api/watchlist/601318",
            json={"is_pinned": True, "group_name": "金融", "note": "核心蓝筹"},
        )
        assert patch_wl.status_code == 200
        wl_after = patch_wl.json()["watchlist"]
        assert wl_after[0]["stock_code"] == "601318"
        assert wl_after[0]["is_pinned"] is True
        assert wl_after[0]["group_name"] == "金融"
        assert wl_after[0]["note"] == "核心蓝筹"
        assert client.patch("/api/watchlist/999999", json={"is_pinned": True}).status_code == 404

        # 9. Risk config persistence & order rejection
        risk_save = client.post(
            "/api/risk",
            json={
                "max_single_position_pct": 15.0,
                "max_order_amount": 30_000.0,
                "max_total_loss_pct": 10.0,
                "block_buy_on_loss": True,
            },
        )
        assert risk_save.status_code == 200
        assert client.get("/api/risk").json()["max_order_amount"] == 30_000.0

        risk_rej = client.post(
            "/api/orders",
            json={"stock_code": "000001", "side": "BUY", "price": 12.0, "quantity": 3000},
        ).json()
        assert risk_rej["order"]["status"] == "REJECTED"
        assert "风控拦截" in risk_rej["order"]["reject_reason"]

        # 10. MA_CROSS strategy & validation
        ma_bad = client.post(
            "/api/strategies",
            json={"stock_code": "000001", "strategy_type": "MA_CROSS", "fast_period": 20, "slow_period": 5, "order_qty": 100},
        )
        assert ma_bad.status_code == 400
        ma_ok = client.post(
            "/api/strategies",
            json={"stock_code": "000001", "strategy_type": "MA_CROSS", "fast_period": 5, "slow_period": 20, "order_qty": 100},
        )
        assert ma_ok.status_code == 200
        assert ma_ok.json()["strategy"]["strategy_type"] == "MA_CROSS"

        # 11. Historical daily backtest
        bt_res = client.post(
            "/api/backtest",
            json={"stock_code": "000001", "strategy_type": "MA_CROSS", "days": 60, "fast_period": 5, "slow_period": 20},
        )
        assert bt_res.status_code == 200
        bt_json = bt_res.json()
        assert bt_json["bar_count"] == 60
        assert len(bt_json["equity_curve"]) == 60
        assert "max_drawdown_pct" in bt_json

        # 12. CSV exports with UTF-8 BOM
        orders_csv = client.get("/api/export/orders.csv")
        assert orders_csv.status_code == 200
        assert orders_csv.content.startswith(b"\xef\xbb\xbf")
        assert "委托编号" in orders_csv.content.decode("utf-8-sig")

        trades_csv = client.get("/api/export/trades.csv")
        assert trades_csv.status_code == 200
        assert trades_csv.content.startswith(b"\xef\xbb\xbf")
        assert "成交编号" in trades_csv.content.decode("utf-8-sig")

        # 13. Limit-up screener API (offline kline_scan fallback)
        def fake_kline(code: str, days: int = 60) -> list[dict]:
            closes = [10.0, 10.0, 12.0, 12.0, 12.1] if code == "300750" else [10.0] * 5
            return [
                {"trade_date": f"2026-11-1{i}", "open": c, "high": c, "low": c, "close": c, "volume": 1000.0, "amount": 10000.0, "change_pct": 0.0, "turnover_ratio": 1.0}
                for i, c in enumerate(closes)
            ]

        app.state.ctx.feed.fetch_daily_kline = fake_kline  # type: ignore[method-assign]
        scr_res = client.get("/api/screener/limit-up?days=5&board=创业板&min_count=1&only_watchlist=false&q=宁德")
        assert scr_res.status_code == 200
        scr_json = scr_res.json()
        assert scr_json["source"] == "kline_scan"
        assert scr_json["total_hits"] == 1
        assert scr_json["items"][0]["stock_code"] == "300750"

        # 14. Reset account cleanly
        reset_res = client.post("/api/account/reset", json={"initial_cash": 500_000.0}).json()
        assert reset_res["account"]["initial_cash"] == 500_000.0
        assert reset_res["positions"] == []


def test_ai_endpoints_api(tmp_path: Path) -> None:
    import httpx
    from app.engine.ai_assistant import AIAssistantService

    app = create_app(
        db_path=tmp_path / "ai_api_test.db",
        network_enabled=False,
        enable_background_loop=False,
        auth_required=False,
    )

    def mock_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "qwen2.5:14b"}]})
        if request.url.path.endswith("/chat/completions"):
            return httpx.Response(200, json={"model": "qwen2.5:14b", "choices": [{"message": {"content": "API AI 回答正常"}}]})
        return httpx.Response(404)

    transport = httpx.MockTransport(mock_handler)
    app.state.ctx.ai_assistant = AIAssistantService(
        app.state.ctx.db,
        tcp_probe=lambda h, p, t: (True, 1.2, f"TCP 端口 {h}:{p} 可达 (1.2ms)"),
        http_client_factory=lambda t: httpx.Client(transport=transport, timeout=t),
    )

    with TestClient(app) as client:
        # 1. Default AI config + port metadata in /api/state
        st = client.get("/api/state").json()
        assert "ai_config" in st
        assert st["ai_config"]["host"] == "127.0.0.1"
        assert st["ai_config"]["port"] == 11434
        assert "api_key" not in st["ai_config"]

        # 2. Local fallback ask works before enabling external AI
        fb = client.post("/api/ai/ask", json={"task_type": "DIAGNOSE_STOCK", "stock_code": "600519"})
        assert fb.status_code == 200
        assert fb.json()["source"] == "local_rule_fallback"

        # 3. Reject invalid base_url
        bad = client.post(
            "/api/ai/config",
            json={"provider_name": "Bad", "base_url": "not-a-url", "model": "x", "enabled": True},
        )
        assert bad.status_code == 400

        # 4. Save valid config & test endpoint health + LLM ask
        saved = client.post(
            "/api/ai/config",
            json={
                "provider_name": "Ollama Test",
                "base_url": "http://127.0.0.1:11434",
                "model": "qwen2.5:14b",
                "api_key": "sk-test9999",
                "enabled": True,
            },
        )
        assert saved.status_code == 200
        cfg = saved.json()["ai_config"]
        assert cfg["chat_endpoint"] == "http://127.0.0.1:11434/v1/chat/completions"
        assert cfg["api_key_masked"] == "sk-t...9999"

        chk = client.post("/api/ai/test", json={"probe_chat": True})
        assert chk.status_code == 200
        assert chk.json()["status"] == "OK"
        assert chk.json()["discovered_models"] == ["qwen2.5:14b"]

        ans = client.post("/api/ai/ask", json={"task_type": "CHAT", "prompt": "测试问股", "stock_code": "600519"})
        assert ans.status_code == 200
        assert ans.json()["source"] == "llm"
        assert "API AI 回答正常" in ans.json()["reply"]
