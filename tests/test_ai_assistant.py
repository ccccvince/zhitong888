from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from app.engine.ai_assistant import (
    AIAssistantService,
    mask_api_key,
    parse_ai_endpoint,
    parse_custom_headers,
)
from app.models import (
    AIAskRequest,
    AIConfigTestRequest,
    AIConfigUpdateRequest,
    LimitUpScreenResponse,
    LimitUpStockItem,
    QuoteSnapshot,
)
from app.storage import Database


def test_parse_ai_endpoint_and_key_masking() -> None:
    ep1 = parse_ai_endpoint("http://127.0.0.1:11434")
    assert ep1["scheme"] == "http"
    assert ep1["host"] == "127.0.0.1"
    assert ep1["port"] == 11434
    assert ep1["base_url"] == "http://127.0.0.1:11434/v1"
    assert ep1["chat_endpoint"] == "http://127.0.0.1:11434/v1/chat/completions"
    assert ep1["models_endpoint"] == "http://127.0.0.1:11434/v1/models"

    ep2 = parse_ai_endpoint("https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions/")
    assert ep2["port"] == 443
    assert ep2["base_url"] == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert ep2["models_endpoint"] == "https://dashscope.aliyuncs.com/compatible-mode/v1/models"

    for bad in ("", "127.0.0.1:11434", "ftp://127.0.0.1:11434/v1", "http://127.0.0.1:99999/v1"):
        with pytest.raises(ValueError):
            parse_ai_endpoint(bad)

    assert mask_api_key("") == "未设置"
    assert mask_api_key("123456") == "12****"
    assert mask_api_key("sk-abcd1234wxyz") == "sk-a...wxyz"


def test_ai_config_persistence_key_preservation_and_clear(tmp_path: Path) -> None:
    db = Database(tmp_path / "ai_cfg.db")
    svc = AIAssistantService(db)

    init_view = svc.get_config_view()
    assert init_view.enabled is False
    assert init_view.host == "127.0.0.1"
    assert init_view.port == 11434
    assert init_view.has_api_key is False

    saved = svc.save_config(
        AIConfigUpdateRequest(
            provider_name="DeepSeek",
            base_url="https://api.deepseek.com",
            model="deepseek-chat",
            api_key="sk-secretkey6789",
            enabled=True,
        )
    )
    assert saved.host == "api.deepseek.com"
    assert saved.port == 443
    assert saved.has_api_key is True
    assert saved.api_key_masked == "sk-s...6789"
    assert "sk-secretkey6789" not in saved.model_dump_json()

    # Updating other fields with empty api_key preserves saved key
    updated = svc.save_config(
        AIConfigUpdateRequest(
            provider_name="DeepSeek",
            base_url="https://api.deepseek.com/v1",
            model="deepseek-reasoner",
            api_key="",
            enabled=True,
        )
    )
    assert updated.has_api_key is True
    assert updated.api_key_masked == "sk-s...6789"
    assert updated.model == "deepseek-reasoner"

    # Explicitly clearing key
    cleared = svc.save_config(
        AIConfigUpdateRequest(
            provider_name="DeepSeek",
            base_url="https://api.deepseek.com/v1",
            model="deepseek-chat",
            clear_api_key=True,
            enabled=True,
        )
    )
    assert cleared.has_api_key is False
    assert cleared.api_key_masked == "未设置"


def test_ai_health_check_tcp_and_mocked_openai_models_chat(tmp_path: Path) -> None:
    db = Database(tmp_path / "ai_health.db")

    # 1. TCP unreachable case
    svc_down = AIAssistantService(
        db,
        tcp_probe=lambda h, p, t: (False, None, f"TCP 端口 {h}:{p} 连接失败: ConnectionRefused"),
    )
    down_res = svc_down.test_connection(AIConfigTestRequest(probe_chat=True))
    assert down_res.tcp_reachable is False
    assert down_res.status == "ERROR"

    # 2. TCP reachable + mocked /models & /chat/completions HTTP 200
    seen_auth: list[str] = []

    def transport_handler(request: httpx.Request) -> httpx.Response:
        seen_auth.append(request.headers.get("authorization", ""))
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "qwen2.5:14b"}, {"id": "deepseek-r1:7b"}]})
        if request.url.path.endswith("/chat/completions"):
            return httpx.Response(200, json={"choices": [{"message": {"content": "OK"}}]})
        return httpx.Response(404)

    mock_transport = httpx.MockTransport(transport_handler)
    svc_ok = AIAssistantService(
        db,
        tcp_probe=lambda h, p, t: (True, 1.8, f"TCP 端口 {h}:{p} 可达 (1.8ms)"),
        http_client_factory=lambda t: httpx.Client(transport=mock_transport, timeout=t),
    )
    svc_ok.save_config(
        AIConfigUpdateRequest(
            provider_name="Local vLLM",
            base_url="http://127.0.0.1:8000/v1",
            model="qwen2.5:14b",
            api_key="sk-local-token",
            enabled=True,
        )
    )
    ok_res = svc_ok.test_connection(AIConfigTestRequest(probe_chat=True))
    assert ok_res.tcp_reachable is True
    assert ok_res.tcp_latency_ms == 1.8
    assert ok_res.http_status == 200
    assert ok_res.chat_probe_ok is True
    assert ok_res.status == "OK"
    assert ok_res.discovered_models == ["qwen2.5:14b", "deepseek-r1:7b"]
    assert all(a == "Bearer sk-local-token" for a in seen_auth)
    assert svc_ok.get_config_view().last_check_status == "OK"

    # 3. Auth failure 401 -> WARN
    auth_fail_transport = httpx.MockTransport(
        lambda req: httpx.Response(401, json={"error": {"message": "Invalid API Key"}})
    )
    svc_warn = AIAssistantService(
        db,
        tcp_probe=lambda h, p, t: (True, 2.0, "ok"),
        http_client_factory=lambda t: httpx.Client(transport=auth_fail_transport, timeout=t),
    )
    warn_res = svc_warn.test_connection(AIConfigTestRequest(probe_chat=True))
    assert warn_res.tcp_reachable is True
    assert warn_res.http_status == 401
    assert warn_res.status == "WARN"
    assert "Invalid API Key" in warn_res.message


def test_ai_ask_llm_with_drafts_and_local_fallback(tmp_path: Path) -> None:
    db = Database(tmp_path / "ai_ask.db")
    quote = QuoteSnapshot(
        stock_code="600519",
        short_name="贵州茅台",
        price=1450.0,
        pre_close=1440.0,
        open=1442.0,
        high=1460.0,
        low=1438.0,
        change=10.0,
        change_pct=0.69,
        volume=200000,
        amount=290000000.0,
        turnover_ratio=0.16,
        amplitude=1.53,
        pe_ratio=22.1,
        limit_up=1584.0,
        limit_down=1296.0,
        updated_at="2026-11-17 13:45:00",
    )
    bars = [
        {
            "trade_date": f"2026-10-{i:02d}",
            "open": 1400.0 + i,
            "high": 1410.0 + i,
            "low": 1395.0 + i,
            "close": 1405.0 + i,
            "change_pct": 0.4,
        }
        for i in range(1, 26)
    ]
    state_payload = {
        "account": {
            "current_trade_date": "2026-11-17",
            "market_mode": "REAL",
            "total_assets": 1000000.0,
            "available_cash": 850000.0,
            "market_value": 150000.0,
            "total_pnl": 0.0,
            "total_pnl_pct": 0.0,
        },
        "risk_config": {"max_single_position_pct": 100.0, "max_order_amount": 0.0, "max_total_loss_pct": 0.0},
        "watchlist": [{"stock_code": "600519", "short_name": "贵州茅台", "group_name": "核心", "is_pinned": True, "note": "低吸观察"}],
        "positions": [
            {
                "stock_code": "600519",
                "short_name": "贵州茅台",
                "total_qty": 100,
                "available_qty": 0,
                "today_bought_qty": 100,
                "frozen_qty": 0,
                "avg_cost": 1450.0,
                "last_price": 1450.0,
                "market_value": 145000.0,
                "unrealized_pnl": 0.0,
                "unrealized_pnl_pct": 0.0,
            }
        ],
        "orders": [],
        "trades": [],
        "strategies": [],
    }

    # 1. Local fallback when AI disabled
    svc_off = AIAssistantService(db)
    fb = svc_off.ask(AIAskRequest(task_type="DIAGNOSE_STOCK", stock_code="600519"), state_payload, quote, bars)
    assert fb.ok is True
    assert fb.source == "local_rule_fallback"
    assert "贵州茅台" in fb.reply
    assert len(fb.drafts) == 1
    assert fb.drafts[0].kind == "BACKTEST"

    rev = svc_off.ask(AIAskRequest(task_type="DAILY_REVIEW"), state_payload, quote, bars)
    assert "总仓位 **15.0%**" in rev.reply
    assert "贵州茅台(100股)" in rev.reply

    # 2. Active OpenAI-compatible mock with nested JSON drafts (valid + invalid draft filtered out)
    captured_payload: dict = {}

    def llm_handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_payload
        captured_payload = json.loads(request.content.decode("utf-8"))
        content = (
            "### 贵州茅台盘面诊断\n"
            "- 当前处于短期多头格局，可关注 MA5 支撑。\n\n"
            "```json\n"
            "{\n"
            '  "drafts": [\n'
            '    {"kind": "BACKTEST", "title": "回测 MA5/20", "summary": "90日+8%止损", "payload": {"stock_code": "600519", "strategy_type": "MA_CROSS", "days": 90, "fast_period": 5, "slow_period": 20, "stop_loss_pct": 8}},\n'
            '    {"kind": "STRATEGY", "title": "非法均线策略应被过滤", "summary": "fast>=slow", "payload": {"stock_code": "600519", "strategy_type": "MA_CROSS", "fast_period": 20, "slow_period": 5, "order_qty": 100}},\n'
            '    {"kind": "ALERT", "title": "突破预警", "summary": "涨超1480提醒", "payload": {"stock_code": "600519", "condition": "PRICE_ABOVE", "threshold": 1480}}\n'
            "  ]\n"
            "}\n"
            "```"
        )
        return httpx.Response(200, json={"model": "deepseek-chat", "choices": [{"message": {"content": content}}]})

    mock_llm = httpx.MockTransport(llm_handler)
    svc_on = AIAssistantService(
        db,
        http_client_factory=lambda t: httpx.Client(transport=mock_llm, timeout=t),
    )
    svc_on.save_config(
        AIConfigUpdateRequest(
            provider_name="DeepSeek",
            base_url="https://api.deepseek.com/v1",
            model="deepseek-chat",
            api_key="sk-1234567890",
            enabled=True,
        )
    )
    scr = LimitUpScreenResponse(
        days=5,
        trade_dates=["2026-11-17"],
        total_hits=1,
        source="eastmoney_zt_pool",
        scanned_at="2026-11-17 13:45:00",
        items=[
            LimitUpStockItem(
                stock_code="002187",
                short_name="广百股份",
                board="主板",
                last_price=8.44,
                latest_change_pct=10.04,
                turnover_ratio=26.4,
                amount=1.5e9,
                limit_up_count=4,
                max_consecutive_boards=4,
                last_limit_up_date="2026-11-17",
                limit_up_dates=["2026-11-17"],
                industry="一般零售",
            )
        ],
    )
    ans = svc_on.ask(
        AIAskRequest(task_type="SCREENER_SUMMARY", prompt="归纳主线并给茅台策略", stock_code="600519"),
        state_payload,
        quote,
        bars,
        screener_resp=scr,
    )
    assert ans.ok is True
    assert ans.source == "llm"
    assert ans.model_used == "deepseek-chat"
    assert "```json" not in ans.reply
    assert [d.kind for d in ans.drafts] == ["BACKTEST", "ALERT"]
    assert "贵州茅台" in captured_payload["messages"][1]["content"]
    assert "广百股份" in captured_payload["messages"][1]["content"]

    # 3. Active endpoint HTTP 500 -> transparent error
    err_transport = httpx.MockTransport(lambda req: httpx.Response(500, json={"error": {"message": "Upstream overloaded"}}))
    svc_err = AIAssistantService(
        db,
        http_client_factory=lambda t: httpx.Client(transport=err_transport, timeout=t),
    )
    err_ans = svc_err.ask(AIAskRequest(task_type="CHAT", prompt="hi"), state_payload, quote, bars)
    assert err_ans.ok is False
    assert err_ans.source == "llm_error"
    assert "Upstream overloaded" in err_ans.reply


def test_custom_ai_protocols_exact_url_and_headers(tmp_path: Path) -> None:
    # 1. Custom non-mainstream paths & exact '#' URL override
    ep_custom = parse_ai_endpoint("https://custom-relay.top:8443/gateway/v2")
    assert ep_custom["chat_endpoint"] == "https://custom-relay.top:8443/gateway/v2/chat/completions"

    ep_exact = parse_ai_endpoint("https://worker.example.com/my-fn#")
    assert ep_exact["exact_endpoint"] is True
    assert ep_exact["base_url"] == "https://worker.example.com/my-fn#"
    assert ep_exact["chat_endpoint"] == "https://worker.example.com/my-fn"

    ep_claude = parse_ai_endpoint("https://claude-proxy.top/v1", api_format="anthropic_messages")
    assert ep_claude["chat_endpoint"] == "https://claude-proxy.top/v1/messages"

    ep_resp = parse_ai_endpoint("https://resp-proxy.top/openai", api_format="openai_responses")
    assert ep_resp["chat_endpoint"] == "https://resp-proxy.top/openai/responses"

    # 2. Custom headers parsing (JSON and key: value lines)
    assert parse_custom_headers('{"X-Relay-Auth": "tok-1", "HTTP-Referer": "https://a.com"}') == {
        "X-Relay-Auth": "tok-1",
        "HTTP-Referer": "https://a.com",
    }
    assert parse_custom_headers("X-Token: abc\nX-App: zhitong") == {
        "X-Token": "abc",
        "X-App": "zhitong",
    }
    with pytest.raises(ValueError, match="JSON"):
        parse_custom_headers("{bad json")

    # 3. Platform public AI with Anthropic Messages protocol, keyless relay & custom headers
    db = Database(tmp_path / "custom_proto.db")
    db.set_public_ai_config(
        provider_name="自定义 Claude 公益站",
        api_format="anthropic_messages",
        base_url="https://free-claude-relay.example.com/custom",
        model="claude-sonnet-4-5",
        daily_limit=0,
        api_key="",
        custom_headers="X-Relay-Pass: secret-99",
        enabled=True,
    )

    seen_reqs: list[tuple[str, dict[str, str], dict]] = []

    def proto_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models") and request.method == "GET":
            return httpx.Response(200, json={"data": [{"id": "custom-alpha"}, {"id": "claude-sonnet-4-5"}]})
        body = json.loads(request.content.decode("utf-8"))
        seen_reqs.append((str(request.url), dict(request.headers), body))
        if request.url.path.endswith("/messages"):
            return httpx.Response(
                200,
                json={"model": "claude-sonnet-4-5", "content": [{"type": "text", "text": "**自定义通道诊断成功**"}]},
            )
        if request.url.path.endswith("/responses"):
            return httpx.Response(200, json={"model": "custom-o3", "output_text": "**Responses OK**"})
        return httpx.Response(200, json={"choices": [{"message": {"content": "**Exact OK**"}}]})

    svc = AIAssistantService(
        db,
        tcp_probe=lambda h, p, t: (True, 6.2, "ok"),
        http_client_factory=lambda t: httpx.Client(transport=httpx.MockTransport(proto_handler), timeout=t),
    )

    # Discover models even when model is left blank
    disc = svc.test_platform_ai(
        base_url="https://free-claude-relay.example.com/custom",
        model="",
    )
    assert disc["ok"] is True
    assert disc["chat_probe_ok"] is None
    assert disc["discovered_models"] == ["custom-alpha", "claude-sonnet-4-5"]

    # Live ask via Anthropic Messages + custom header without requiring API key
    state_payload = {
        "account": {
            "current_trade_date": "2026-11-18",
            "market_mode": "REAL",
            "total_assets": 1000000.0,
            "available_cash": 1000000.0,
            "market_value": 0.0,
            "total_pnl": 0.0,
            "total_pnl_pct": 0.0,
        },
        "risk_config": {"max_single_position_pct": 100.0, "max_order_amount": 0.0, "max_total_loss_pct": 0.0},
        "watchlist": [],
        "positions": [],
        "orders": [],
        "trades": [],
        "strategies": [],
    }
    res_anth = svc.ask(AIAskRequest(task_type="CHAT", prompt="分析"), state_payload)
    assert res_anth.ok is True
    assert res_anth.source == "platform_trial_llm"
    assert "自定义通道诊断成功" in res_anth.reply
    url_hit, hdrs_hit, payload_hit = seen_reqs[-1]
    assert url_hit == "https://free-claude-relay.example.com/custom/messages"
    assert hdrs_hit.get("x-relay-pass") == "secret-99"
    assert hdrs_hit.get("anthropic-version") == "2023-06-01"
    assert "system" in payload_hit

    # Switch to OpenAI Responses + exact URL '#' mode
    db.set_public_ai_config(
        provider_name="Responses 网关",
        api_format="openai_responses",
        base_url="https://resp.example.com/v1",
        model="custom-o3",
        daily_limit=0,
        api_key="sk-custom-test",
        enabled=True,
    )
    res_resp = svc.ask(AIAskRequest(task_type="CHAT", prompt="hi"), state_payload)
    assert res_resp.ok is True
    assert "Responses OK" in res_resp.reply
    assert seen_reqs[-1][0] == "https://resp.example.com/v1/responses"
    assert "input" in seen_reqs[-1][2]
