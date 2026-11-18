from __future__ import annotations

import sqlite3
from pathlib import Path

from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.main import create_app
from app.storage import Database


def test_public_auth_invite_isolation_and_ssrf(tmp_path: Path) -> None:
    db_file = tmp_path / "multi_user.db"
    app = create_app(
        db_path=db_file,
        network_enabled=False,
        enable_background_loop=False,
        auth_required=True,
        open_register=False,
    )
    app.state.ctx.db.ensure_bootstrap_admin("boss", "AdminPass123!")

    # 1. Anonymous visitor cannot read state, trade, or connect WebSocket
    with TestClient(app) as anon:
        assert anon.get("/").status_code == 200
        assert anon.get("/healthz").status_code == 200
        me = anon.get("/api/auth/me").json()
        assert me["authenticated"] is False
        assert me["auth_required"] is True
        assert anon.get("/api/state").status_code == 401
        assert anon.post("/api/ai/ask", json={"task_type": "CHAT", "prompt": "hi"}).status_code == 401
        try:
            with anon.websocket_connect("/ws") as ws:
                ws.receive_text()
            raise AssertionError("Anonymous WebSocket should not remain open")
        except (WebSocketDisconnect, Exception):
            pass

    # 2. Admin logs in and creates single-use & multi-use invite codes
    with TestClient(app) as admin_client:
        login_bad = admin_client.post(
            "/api/auth/login", json={"username": "boss", "password": "wrong"}
        )
        assert login_bad.status_code == 401

        login_ok = admin_client.post(
            "/api/auth/login", json={"username": "boss", "password": "AdminPass123!"}
        )
        assert login_ok.status_code == 200
        assert login_ok.json()["user"]["role"] == "ADMIN"

        inv1 = admin_client.post(
            "/api/admin/invites", json={"max_uses": 1, "count": 1, "note": "one-shot"}
        )
        assert inv1.status_code == 200
        code_once = inv1.json()["created"][0]["code"]

        inv2 = admin_client.post(
            "/api/admin/invites", json={"max_uses": 2, "count": 1, "custom_code": "TEAM888"}
        )
        assert inv2.status_code == 200
        assert inv2.json()["created"][0]["code"] == "TEAM888"

    # 3. Registration enforcement & single-use invite exhaustion
    with TestClient(app) as client_a, TestClient(app) as client_b:
        no_inv = client_a.post(
            "/api/auth/register", json={"username": "alice", "password": "UserPass123"}
        )
        assert no_inv.status_code == 400

        reg_a = client_a.post(
            "/api/auth/register",
            json={
                "username": "alice",
                "display_name": "Alice",
                "password": "UserPass123",
                "invite_code": code_once,
            },
        )
        assert reg_a.status_code == 200
        assert reg_a.json()["user"]["role"] == "USER"

        # Single-use code must now reject Bob
        reuse_fail = client_b.post(
            "/api/auth/register",
            json={
                "username": "bob",
                "password": "UserPass123",
                "invite_code": code_once,
            },
        )
        assert reuse_fail.status_code == 400

        reg_b = client_b.post(
            "/api/auth/register",
            json={
                "username": "bob",
                "password": "UserPass123",
                "invite_code": "team888",
            },
        )
        assert reg_b.status_code == 200

        # Non-admin cannot access admin overview
        assert client_a.get("/api/admin/overview").status_code == 403

        # 4. Complete workspace isolation between Alice and Bob
        client_a.post("/api/mode", json={"mode": "SIMULATED"})
        tick_a = client_a.post("/api/sim/tick").json()
        q_a = next(w["quote"] for w in tick_a["watchlist"] if w["stock_code"] == "000001")
        buy_px = q_a["asks"][0]["price"] if q_a["asks"] else q_a["price"]
        ord_a = client_a.post(
            "/api/orders",
            json={"stock_code": "000001", "side": "BUY", "price": buy_px, "quantity": 200},
        ).json()
        assert ord_a["order"]["status"] == "FILLED"
        oid_a = ord_a["order"]["order_id"]

        st_a = client_a.get("/api/state").json()
        st_b = client_b.get("/api/state").json()
        assert st_a["account"]["market_mode"] == "SIMULATED"
        assert st_b["account"]["market_mode"] == "REAL"
        assert len(st_a["positions"]) == 1
        assert len(st_b["positions"]) == 0
        assert st_b["account"]["available_cash"] == 1_000_000.0
        # Real quote cache remains uncontaminated for Bob
        assert all(w["quote"]["source"] != "simulated_tick" for w in st_b["watchlist"] if w["quote"])

        # Bob cannot cancel Alice's order (IDOR guard)
        assert client_b.post(f"/api/orders/{oid_a}/cancel").status_code == 400

        # Authenticated WebSocket works cleanly with isolated user state
        with client_b.websocket_connect("/ws") as ws_b:
            ws_state = ws_b.receive_json()
            assert ws_state["positions"] == []
            assert ws_state["account"]["market_mode"] == "REAL"

        # 5. SSRF guards on AI base_url and webhook_url in public auth mode
        for bad_url in (
            "http://127.0.0.1:11434/v1",
            "http://localhost:8000/v1",
            "http://169.254.169.254/latest",
            "http://10.0.0.5:8080/v1",
        ):
            r = client_a.post(
                "/api/ai/config",
                json={"provider_name": "Bad", "base_url": bad_url, "model": "qwen", "enabled": True},
            )
            assert r.status_code == 400

        bad_hook = client_a.post(
            "/api/alerts",
            json={
                "stock_code": "600519",
                "condition": "PRICE_ABOVE",
                "threshold": 1500.0,
                "webhook_url": "http://169.254.169.254/latest/meta-data",
            },
        )
        assert bad_hook.status_code == 400


def test_legacy_single_tenant_sqlite_migration(tmp_path: Path) -> None:
    db_file = tmp_path / "legacy.db"
    conn = sqlite3.connect(str(db_file))
    conn.executescript(
        """
        CREATE TABLE account (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            initial_cash REAL NOT NULL,
            available_cash REAL NOT NULL,
            frozen_cash REAL NOT NULL,
            current_trade_date TEXT NOT NULL,
            market_mode TEXT NOT NULL DEFAULT 'REAL',
            updated_at TEXT NOT NULL
        );
        INSERT INTO account VALUES (1, 500000.0, 400000.0, 0.0, '2026-11-17', 'REAL', '2026-11-17 10:00:00');
        CREATE TABLE watchlist (
            stock_code TEXT PRIMARY KEY,
            short_name TEXT NOT NULL,
            added_at TEXT NOT NULL
        );
        INSERT INTO watchlist VALUES ('600519', '贵州茅台', '2026-11-17 10:00:00');
        CREATE TABLE positions (
            stock_code TEXT PRIMARY KEY,
            short_name TEXT NOT NULL,
            total_qty INTEGER NOT NULL,
            available_qty INTEGER NOT NULL,
            today_bought_qty INTEGER NOT NULL DEFAULT 0,
            frozen_qty INTEGER NOT NULL DEFAULT 0,
            avg_cost REAL NOT NULL,
            updated_at TEXT NOT NULL
        );
        INSERT INTO positions VALUES ('600519', '贵州茅台', 100, 100, 0, 0, 1450.0, '2026-11-17 10:00:00');
        """
    )
    conn.commit()
    conn.close()

    db = Database(db_file)
    with db.connect() as migrated:
        acc = migrated.execute("SELECT * FROM account WHERE user_id = 'default'").fetchone()
        assert acc is not None
        assert float(acc["initial_cash"]) == 500000.0
        wl = migrated.execute("SELECT * FROM watchlist WHERE user_id = 'default'").fetchall()
        assert len(wl) == 1 and wl[0]["stock_code"] == "600519"
        pos = migrated.execute("SELECT * FROM positions WHERE user_id = 'default'").fetchone()
        assert pos is not None and pos["total_qty"] == 100


def test_platform_trial_ai_quota_and_fallback(tmp_path: Path, monkeypatch) -> None:
    import httpx
    from app.config import settings
    from app.engine.ai_assistant import AIAssistantService
    from app.models import AIAskRequest

    monkeypatch.setattr(settings, "public_ai_base_url", "https://api.deepseek.com/v1")
    monkeypatch.setattr(settings, "public_ai_model", "deepseek-chat")
    monkeypatch.setattr(settings, "public_ai_api_key", "sk-platform-secret-hidden")
    monkeypatch.setattr(settings, "public_ai_daily_limit", 1)

    db = Database(tmp_path / "trial.db", seed_default_workspace=False)
    user = db.create_user("trialuser", "Secret123", require_invite=False)
    uid = str(user["user_id"])

    transport = httpx.MockTransport(
        lambda req: httpx.Response(200, json={"model": "deepseek-chat", "choices": [{"message": {"content": "平台体验回答"}}]})
    )
    svc = AIAssistantService(
        db,
        http_client_factory=lambda t: httpx.Client(transport=transport, timeout=t),
        allow_private_endpoints=False,
    )
    cfg = svc.get_config_view(user_id=uid)
    assert cfg.platform_trial_available is True
    assert cfg.trial_remaining_today == 1
    assert "sk-platform-secret-hidden" not in cfg.model_dump_json()

    st = {"account": {"current_trade_date": "2026-11-17", "market_mode": "REAL"}, "watchlist": []}
    r1 = svc.ask(AIAskRequest(task_type="CHAT", prompt="分析行情"), st, user_id=uid)
    assert r1.source == "platform_trial_llm"
    assert "平台体验回答" in r1.reply
    assert svc.get_config_view(user_id=uid).trial_remaining_today == 0

    # 2nd call hits daily cap and safely degrades to local fallback
    r2 = svc.ask(AIAskRequest(task_type="DAILY_REVIEW"), st, user_id=uid)
    assert r2.source == "local_rule_fallback"
    assert "用完" in r2.warnings[0]


def test_email_registration_reset_password_and_alert_notifications(tmp_path: Path) -> None:
    import json
    import re
    import httpx
    from app.engine.email_service import ResendEmailService
    from app.models import QuoteSnapshot

    sent_emails: list[dict] = []

    def resend_handler(request: httpx.Request) -> httpx.Response:
        assert request.headers.get("authorization") == "Bearer re_mock_test_key"
        body = json.loads(request.content.decode("utf-8"))
        sent_emails.append(body)
        return httpx.Response(200, json={"id": f"email_{len(sent_emails)}"})

    transport = httpx.MockTransport(resend_handler)
    email_svc = ResendEmailService(
        api_key="re_mock_test_key",
        default_from="智瞳量化 <noreply@zhitong888.ccwu.cc>",
        http_client_factory=lambda t: httpx.Client(transport=transport, timeout=t),
    )

    app = create_app(
        db_path=tmp_path / "email_flow.db",
        network_enabled=False,
        enable_background_loop=False,
        auth_required=True,
        invite_required=False,
        email_verification_required=True,
        email_service=email_svc,
    )
    app.state.ctx.db.ensure_bootstrap_admin("admin", "AdminPass123!")

    with TestClient(app) as client:
        me = client.get("/api/auth/me").json()
        assert me["invite_required"] is False
        assert me["email_verification_required"] is True
        assert me["email_service_configured"] is True

        # Register fails cleanly without email OTP
        no_code = client.post(
            "/api/auth/register",
            json={"username": "trader01", "password": "Pass1234", "email": "trader01@example.com"},
        )
        assert no_code.status_code == 400

        # Send code -> 1st succeeds, immediate 2nd hits 60s cooldown (429)
        s1 = client.post("/api/auth/send-code", json={"email": "trader01@example.com", "purpose": "REGISTER"})
        assert s1.status_code == 200
        assert len(sent_emails) == 1
        otp_match = re.search(r"(\d{6})", str(sent_emails[-1]["subject"]))
        assert otp_match is not None
        reg_otp = otp_match.group(1)

        s2 = client.post("/api/auth/send-code", json={"email": "trader01@example.com", "purpose": "REGISTER"})
        assert s2.status_code == 429

        # Register succeeds WITHOUT invite_code using verified email + OTP
        reg_ok = client.post(
            "/api/auth/register",
            json={
                "username": "trader01",
                "password": "Pass1234",
                "email": "trader01@example.com",
                "email_code": reg_otp,
            },
        )
        assert reg_ok.status_code == 200
        assert reg_ok.json()["user"]["email"] == "trader01@example.com"
        assert reg_ok.json()["user"]["email_verified"] is True

        # Alert rule with notify_email=True dispatches Resend alert email on trigger
        app.state.ctx.feed._quotes["600519"] = QuoteSnapshot(
            stock_code="600519",
            short_name="贵州茅台",
            price=1550.0,
            pre_close=1500.0,
            change=50.0,
            change_pct=3.33,
        )
        alert_res = client.post(
            "/api/alerts",
            json={
                "stock_code": "600519",
                "condition": "PRICE_ABOVE",
                "threshold": 1520.0,
                "notify_email": True,
            },
        )
        assert alert_res.status_code == 200
        evs = alert_res.json()["state"]["alert_events"]
        assert len(evs) == 1
        assert evs[0]["email_status"] == "SENT"
        assert sent_emails[-1]["to"] == ["trader01@example.com"]

        # Logout, reset password via email OTP, and login using email directly
        client.post("/api/auth/logout")
        rs_send = client.post(
            "/api/auth/send-code",
            json={"email": "trader01@example.com", "purpose": "RESET_PASSWORD"},
        )
        assert rs_send.status_code == 200
        reset_otp = re.search(r"(\d{6})", str(sent_emails[-1]["subject"])).group(1)

        rs_done = client.post(
            "/api/auth/reset-password",
            json={"email": "trader01@example.com", "code": reset_otp, "new_password": "NewPass999"},
        )
        assert rs_done.status_code == 200

        login_by_email = client.post(
            "/api/auth/login", json={"username": "trader01@example.com", "password": "NewPass999"}
        )
        assert login_by_email.status_code == 200
        assert login_by_email.json()["user"]["username"] == "trader01"

    # Admin updates email settings & sends test email
    with TestClient(app) as adm:
        assert adm.post("/api/auth/login", json={"username": "admin", "password": "AdminPass123!"}).status_code == 200
        upd = adm.post(
            "/api/admin/email",
            json={
                "from_email": "智瞳量化 <notice@chanderaskhar.xyz>",
                "email_verification_required": True,
                "invite_required": False,
            },
        )
        assert upd.status_code == 200
        tst = adm.post("/api/admin/email/test", json={"to_email": "owner@example.com"})
        assert tst.status_code == 200
        assert sent_emails[-1]["from"] == "智瞳量化 <notice@chanderaskhar.xyz>"
        assert sent_emails[-1]["to"] == ["owner@example.com"]


def test_brevo_api_email_sending(monkeypatch) -> None:
    import json
    import httpx
    from app.config import settings
    from app.engine.email_service import UniversalEmailService

    monkeypatch.setattr(settings, "brevo_api_key", "xkeysib-mock-brevo-key")
    monkeypatch.setattr(settings, "resend_api_key", "")
    monkeypatch.setattr(settings, "email_from", "智瞳量化 <noreply@mydomain.com>")

    captured_payload = {}

    def brevo_handler(req: httpx.Request) -> httpx.Response:
        nonlocal captured_payload
        assert req.url == "https://api.brevo.com/v3/smtp/email"
        assert req.headers.get("api-key") == "xkeysib-mock-brevo-key"
        captured_payload = json.loads(req.content.decode("utf-8"))
        return httpx.Response(201, json={"messageId": "<brevo-test-123@smtp-relay.mailin.fr>"})

    svc = UniversalEmailService(
        http_client_factory=lambda t: httpx.Client(transport=httpx.MockTransport(brevo_handler), timeout=t)
    )
    assert svc.get_provider_name() == "Brevo"
    ok, msg_id = svc.send_email(
        to_email="user@qq.com",
        subject="测试",
        html_body="<p>内容</p>",
    )
    assert ok is True
    assert msg_id == "<brevo-test-123@smtp-relay.mailin.fr>"
    assert captured_payload["sender"]["email"] == "noreply@mydomain.com"
    assert captured_payload["to"] == [{"email": "user@qq.com"}]


def test_admin_page_telemetry_public_ai_and_announcement(tmp_path: Path) -> None:
    app = create_app(
        db_path=tmp_path / "admin_page.db",
        network_enabled=False,
        enable_background_loop=False,
        auth_required=True,
        open_register=True,
        email_verification_required=False,
    )
    app.state.ctx.db.ensure_bootstrap_admin("boss", "AdminPass123!")
    with TestClient(app) as admin_cli, TestClient(app) as user_cli:
        page = admin_cli.get("/admin")
        assert page.status_code == 200
        assert "智瞳量化 · 管理后台" in page.text

        # Non-admin user cannot call new admin routes
        reg = user_cli.post(
            "/api/auth/register",
            json={"username": "collab1", "password": "UserPass123!", "display_name": "协作成员"},
        )
        assert reg.status_code == 200
        assert user_cli.post("/api/admin/registration", json={"invite_required": True}).status_code == 403
        assert user_cli.post("/api/admin/announcement", json={"title": "t", "content": "c", "enabled": True}).status_code == 403

        # Admin logs in and inspects telemetry overview
        assert admin_cli.post("/api/auth/login", json={"username": "boss", "password": "AdminPass123!"}).status_code == 200
        ov = admin_cli.get("/api/admin/overview")
        assert ov.status_code == 200
        data = ov.json()
        assert data["telemetry"]["counts"]["users_total"] == 2
        assert len(data["telemetry"]["trends_14d"]) == 14
        assert any(u["username"] == "collab1" for u in data["telemetry"]["user_summaries"])

        # Toggle invite mode via /api/admin/registration
        t1 = admin_cli.post("/api/admin/registration", json={"invite_required": True})
        assert t1.status_code == 200 and t1.json()["invite_required"] is True
        t2 = admin_cli.post("/api/admin/registration", json={"invite_required": False})
        assert t2.status_code == 200 and t2.json()["open_register"] is True

        # SSRF blocked on public AI base URL when private endpoints disallowed
        bad_ai = admin_cli.post(
            "/api/admin/public-ai",
            json={
                "provider_name": "Inner",
                "base_url": "http://127.0.0.1:11434/v1",
                "model": "qwen2.5:7b",
                "api_key": "sk-secret",
                "daily_limit": 5,
            },
        )
        assert bad_ai.status_code == 400

        # Valid public AI saves masked key and updates user auth quota dynamically
        ok_ai = admin_cli.post(
            "/api/admin/public-ai",
            json={
                "provider_name": "智瞳体验",
                "base_url": "https://api.deepseek.com/v1",
                "model": "deepseek-chat",
                "api_key": "sk-platform-demo-987654",
                "daily_limit": 12,
                "temperature": 0.4,
                "max_tokens": 2048,
                "timeout_sec": 45,
                "system_prompt": "严格按A股T+1分析",
            },
        )
        assert ok_ai.status_code == 200
        pub_ai = ok_ai.json()["public_ai"]
        assert pub_ai["configured"] is True
        assert pub_ai["temperature"] == 0.4
        assert pub_ai["max_tokens"] == 2048
        assert pub_ai["timeout_sec"] == 45
        assert "T+1" in pub_ai["system_prompt"]
        assert "987654" not in pub_ai["api_key_masked"]
        me_u = user_cli.get("/api/auth/me").json()
        assert me_u["user"]["ai_daily_limit"] == 12

        # daily_limit=0 with enabled=True works as unlimited; enabled=False disables it
        unlim = admin_cli.post(
            "/api/admin/public-ai",
            json={
                "enabled": True,
                "provider_name": "智瞳体验",
                "base_url": "https://api.deepseek.com/v1",
                "model": "deepseek-chat",
                "daily_limit": 0,
            },
        )
        assert unlim.status_code == 200 and unlim.json()["public_ai"]["configured"] is True
        dis = admin_cli.post(
            "/api/admin/public-ai",
            json={
                "enabled": False,
                "provider_name": "智瞳体验",
                "base_url": "https://api.deepseek.com/v1",
                "model": "deepseek-chat",
                "daily_limit": 0,
            },
        )
        assert dis.status_code == 200 and dis.json()["public_ai"]["configured"] is False

        # Arbitrary custom AI endpoint with custom path/format/headers & keyless proxy works
        bad_hdr = admin_cli.post(
            "/api/admin/public-ai",
            json={
                "enabled": True,
                "provider_name": "BadHdr",
                "base_url": "https://my-custom-ai.example.com/custom-v2",
                "model": "my-model-1",
                "custom_headers": "{invalid json",
            },
        )
        assert bad_hdr.status_code == 400

        custom_ai = admin_cli.post(
            "/api/admin/public-ai",
            json={
                "enabled": True,
                "provider_name": "自有网关",
                "api_format": "anthropic_messages",
                "base_url": "https://my-custom-ai.example.com/custom-v2",
                "model": "custom-claude-relay",
                "clear_api_key": True,
                "custom_headers": "X-Proxy-Token: secret123",
                "daily_limit": 0,
            },
        )
        assert custom_ai.status_code == 200
        cp = custom_ai.json()["public_ai"]
        assert cp["configured"] is True
        assert cp["api_format"] == "anthropic_messages"
        assert cp["chat_endpoint"] == "https://my-custom-ai.example.com/custom-v2/messages"
        assert cp["has_custom_headers"] is True
        assert cp["custom_header_keys"] == ["X-Proxy-Token"]

        # Index & stock instant detail endpoints
        idx_res = user_cli.get("/api/indices/sh000001/detail")
        assert idx_res.status_code == 200
        idx_data = idx_res.json()
        assert idx_data["kind"] == "INDEX"
        assert idx_data["name"] == "上证指数"
        assert len(idx_data["daily"]) > 10
        assert idx_data["summary"]["ma5"] is not None
        assert user_cli.get("/api/indices/unknown999/detail").status_code == 404

        stk_res = user_cli.get("/api/stocks/600519/detail")
        assert stk_res.status_code == 200
        stk_data = stk_res.json()
        assert stk_data["kind"] == "STOCK"
        assert stk_data["board"] == "主板"
        assert stk_data["quote"]["stock_code"] == "600519"
        assert stk_data["quote"]["open"] > 0
        assert len(stk_data["daily"]) > 10
        assert len(stk_data["mins"]) >= 1
        assert stk_data["last_trade_date"] != ""

        # Foreground announcement toggles cleanly
        ann_res = admin_cli.post(
            "/api/admin/announcement",
            json={"title": "系统公告", "content": "欢迎多人协作体验", "enabled": True},
        )
        assert ann_res.status_code == 200
        pub_ann = user_cli.get("/api/announcement").json()
        assert pub_ann["enabled"] is True
        assert pub_ann["title"] == "系统公告"
