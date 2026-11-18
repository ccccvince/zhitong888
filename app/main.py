from __future__ import annotations

import asyncio
import csv
import io
import os
import time
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any, Optional

from fastapi import Body, Depends, FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.engine.ai_assistant import (
    AIAssistantService,
    mask_api_key,
    normalize_ai_format,
    parse_ai_endpoint,
    parse_custom_headers,
)
from app.engine.alert_engine import AlertEngine
from app.engine.backtester import run_backtest
from app.engine.clock import MarketClock
from app.engine.email_service import ResendEmailService
from app.engine.market_feed import MarketDataManager, POPULAR_STOCKS
from app.engine.paper_broker import PaperBroker
from app.engine.strategy_runner import StrategyRunner
from app.models import (
    AIAskRequest,
    AIAskResponse,
    AIConfigTestRequest,
    AIConfigUpdateRequest,
    AIConfigView,
    AIHealthCheckResponse,
    AdminAnnouncementRequest,
    AdminEmailSettingsRequest,
    AdminEmailTestRequest,
    AdminPublicAIConfigRequest,
    AdminPublicAITestRequest,
    AdminRegistrationModeRequest,
    AlertRuleRequest,
    AuthLoginRequest,
    AuthRegisterRequest,
    AuthStatusResponse,
    AuthUserView,
    AutoStrategyRuleRequest,
    BacktestRequest,
    BindEmailRequest,
    ChangePasswordRequest,
    InviteCreateRequest,
    LimitUpScreenResponse,
    MarketMode,
    MarketModeRequest,
    OrderRequest,
    ResetAccountRequest,
    ResetPasswordEmailRequest,
    RiskConfig,
    SendEmailCodeRequest,
    TogglePayload,
    WatchlistAddRequest,
    WatchlistUpdateRequest,
)
from app.security import (
    LoginRateLimiter,
    create_session_token,
    resolve_secret_key,
    today_str,
    verify_session_token,
)
from app.storage import Database, is_valid_email, normalize_email, now_str

STATIC_DIR = Path(__file__).resolve().parent / "static"
SESSION_COOKIE_NAME = "zt_session"


class TradingAppContext:
    def __init__(
        self,
        db_path: Path | str | None = None,
        network_enabled: bool = True,
        auth_required: Optional[bool] = None,
        open_register: Optional[bool] = None,
        invite_required: Optional[bool] = None,
        email_verification_required: Optional[bool] = None,
        allow_private_ai_endpoints: Optional[bool] = None,
        allow_private_webhooks: Optional[bool] = None,
        email_service: Optional[ResendEmailService] = None,
    ) -> None:
        resolved_db = db_path or os.environ.get("PAPER_TRADE_DB") or settings.db_path
        self.auth_required = settings.auth_required if auth_required is None else auth_required
        if invite_required is not None:
            self._explicit_invite_required: Optional[bool] = invite_required
        elif open_register is not None:
            self._explicit_invite_required = not open_register
        else:
            self._explicit_invite_required = None
        self._explicit_email_verify = email_verification_required

        if allow_private_ai_endpoints is not None:
            self.allow_private_ai_endpoints = allow_private_ai_endpoints
        elif "ALLOW_PRIVATE_AI_ENDPOINTS" in os.environ:
            self.allow_private_ai_endpoints = settings.allow_private_ai_endpoints
        else:
            self.allow_private_ai_endpoints = not self.auth_required

        if allow_private_webhooks is not None:
            self.allow_private_webhooks = allow_private_webhooks
        elif "ALLOW_PRIVATE_WEBHOOKS" in os.environ:
            self.allow_private_webhooks = settings.allow_private_webhooks
        else:
            self.allow_private_webhooks = not self.auth_required

        self.db = Database(resolved_db, seed_default_workspace=not self.auth_required)
        self.secret_key = resolve_secret_key(self.db.db_path.parent, settings.session_secret)
        self.login_limiter = LoginRateLimiter(max_failures=10, window_sec=300)
        self.email_rate_limiter = LoginRateLimiter(max_failures=15, window_sec=3600)
        self.email_service = email_service or ResendEmailService()

        self.broker = PaperBroker(self.db)
        self.alert_engine = AlertEngine(
            self.db,
            allow_private_webhooks=self.allow_private_webhooks,
            email_notifier=self._dispatch_alert_email,
        )
        self.strategy_runner = StrategyRunner(self.db, self.broker)
        self.clock = MarketClock()
        self.feed = MarketDataManager(network_enabled=network_enabled)
        self.ai_assistant = AIAssistantService(
            self.db, allow_private_endpoints=self.allow_private_ai_endpoints
        )
        self.user_clients: dict[str, set[WebSocket]] = {}
        self._tick_lock = asyncio.Lock()
        self.started_monotonic = time.monotonic()
        self.started_at = now_str()

        if settings.admin_username and settings.admin_password:
            self.db.ensure_bootstrap_admin(settings.admin_username, settings.admin_password)

    def is_invite_required(self) -> bool:
        if self._explicit_invite_required is not None:
            return self._explicit_invite_required
        stored = self.db.get_system_setting("invite_required", "")
        if stored in ("true", "false"):
            return stored == "true"
        return settings.invite_required

    def is_email_verification_required(self) -> bool:
        if self._explicit_email_verify is not None:
            return self._explicit_email_verify
        stored = self.db.get_system_setting("email_verification_required", "")
        if stored in ("true", "false"):
            return stored == "true"
        return settings.email_verification_required and self.email_service.is_configured()

    def get_resend_from(self) -> str:
        stored = self.db.get_system_setting("resend_from", "").strip()
        return stored or self.email_service.default_from

    def _dispatch_alert_email(self, user_id: str, stock_name: str, stock_code: str, msg: str) -> str:
        if not self.email_service.is_configured():
            return "NO_PROVIDER"
        u = self.db.get_user_by_id(user_id)
        if not u:
            return "SKIPPED"
        email = str(u.get("email") or "").strip()
        if not email or not bool(u.get("email_verified", 0)):
            return "UNVERIFIED_EMAIL"
        if not bool(u.get("alert_email_enabled", 1)):
            return "DISABLED_BY_USER"
        ok, detail = self.email_service.send_alert_email(
            to_email=email,
            stock_name=stock_name,
            stock_code=stock_code,
            alert_message=msg,
            from_email=self.get_resend_from(),
        )
        return "SENT" if ok else f"ERR:{detail[:60]}"

    def get_watchlist_rows(self, user_id: str = "default") -> list[dict[str, Any]]:
        with self.db.connect() as conn:
            rows = conn.execute(
                """
                SELECT stock_code, short_name, is_pinned, group_name, note, added_at
                FROM watchlist WHERE user_id = ?
                ORDER BY is_pinned DESC, added_at ASC
                """,
                (user_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_watchlist_items(self, user_id: str = "default") -> list[tuple[str, str]]:
        return [(r["stock_code"], r["short_name"]) for r in self.get_watchlist_rows(user_id=user_id)]

    def fetch_strategy_daily_bars(self, user_id: Optional[str] = None) -> dict[str, list[dict]]:
        """Fetch daily K-line bars needed by enabled technical-indicator auto strategies."""
        with self.db.connect() as conn:
            if user_id is not None:
                rules = conn.execute(
                    "SELECT stock_code, strategy_type, slow_period FROM auto_strategies WHERE enabled = 1 AND user_id = ?",
                    (user_id,),
                ).fetchall()
            else:
                rules = conn.execute(
                    "SELECT stock_code, strategy_type, slow_period FROM auto_strategies WHERE enabled = 1"
                ).fetchall()
        needed: dict[str, int] = {}
        for r in rules:
            if r["strategy_type"] in ("MA_CROSS", "MACD_CROSS"):
                span = max(60, int(r["slow_period"] or 20) + 35)
                code = str(r["stock_code"])
                if code not in needed or span > needed[code]:
                    needed[code] = span
        result: dict[str, list[dict]] = {}
        for code, days in needed.items():
            result[code] = self.feed.fetch_daily_kline(code, days=days)
        return result

    def _collect_order_trades(self, order_ids: list[str], user_id: str = "default") -> list[dict]:
        """Collect persisted trade fills associated with a list of order IDs."""
        if not order_ids:
            return []
        collected: list[dict] = []
        with self.db.connect() as conn:
            for oid in order_ids:
                rows = conn.execute(
                    "SELECT * FROM trades WHERE order_id = ? AND user_id = ? ORDER BY created_at DESC",
                    (oid, user_id),
                ).fetchall()
                collected.extend(dict(r) for r in rows)
        return collected

    def build_state_payload(
        self,
        user_id: str = "default",
        new_alerts: list[dict] | None = None,
        new_trades: list[dict] | None = None,
    ) -> dict[str, Any]:
        acc_row = self.broker.get_account_row(user_id=user_id)
        mode = MarketMode(acc_row["market_mode"])
        quotes = self.feed.get_cached_quotes(mode)
        watchlist_rows = self.get_watchlist_rows(user_id=user_id)

        positions_raw = self.broker.get_positions(quotes, user_id=user_id)
        account = self.broker.get_account_summary(quotes, user_id=user_id)
        session = dict(self.clock.session_status())
        effective_can_match = (account.market_mode == MarketMode.SIMULATED) or bool(
            session.get("can_match", False)
        )
        session["effective_can_match"] = effective_can_match

        watchlist_payload = []
        for row in watchlist_rows:
            code = row["stock_code"]
            name = row["short_name"]
            q = quotes.get(code)
            display_name = q.short_name if (q and q.short_name and q.short_name != code) else name
            watchlist_payload.append(
                {
                    "stock_code": code,
                    "short_name": display_name,
                    "is_pinned": bool(row.get("is_pinned", 0)),
                    "group_name": row.get("group_name") or "默认",
                    "note": row.get("note") or "",
                    "quote": q.model_dump() if q else None,
                }
            )

        return {
            "account": account.model_dump(),
            "risk_config": self.broker.get_risk_config(user_id=user_id).model_dump(),
            "ai_config": self.ai_assistant.get_config_view(user_id=user_id).model_dump(),
            "session": session,
            "indices": [idx.model_dump() for idx in self.feed.get_cached_indices(mode)],
            "watchlist": watchlist_payload,
            "positions": [p.model_dump() for p in positions_raw],
            "orders": [o.model_dump() for o in self.broker.list_orders(120, user_id=user_id)],
            "trades": [t.model_dump() for t in self.broker.list_trades(120, user_id=user_id)],
            "alert_rules": [r.model_dump() for r in self.alert_engine.list_rules(user_id=user_id)],
            "alert_events": [e.model_dump() for e in self.alert_engine.list_events(80, user_id=user_id)],
            "strategies": [s.model_dump() for s in self.strategy_runner.list_strategies(user_id=user_id)],
            "nav_history": self.broker.list_nav_history(user_id=user_id),
            "new_alerts": new_alerts or [],
            "new_trades": new_trades or [],
            "server_time": now_str(),
        }

    async def broadcast_state(
        self,
        user_id: str = "default",
        new_alerts: list[dict] | None = None,
        new_trades: list[dict] | None = None,
    ) -> dict[str, Any]:
        payload = self.build_state_payload(
            user_id=user_id, new_alerts=new_alerts, new_trades=new_trades
        )
        clients = self.user_clients.get(user_id)
        if not clients:
            return payload
        dead: list[WebSocket] = []
        for ws in list(clients):
            try:
                await ws.send_json(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            clients.discard(ws)
        if not clients:
            self.user_clients.pop(user_id, None)
        return payload

    def _list_active_tick_users(self) -> list[str]:
        active: set[str] = set(self.user_clients.keys())
        if not self.auth_required:
            active.add("default")
        with self.db.connect() as conn:
            for q in (
                "SELECT DISTINCT user_id FROM orders WHERE status IN ('PENDING', 'PARTIAL_FILLED')",
                "SELECT DISTINCT user_id FROM auto_strategies WHERE enabled = 1",
                "SELECT DISTINCT user_id FROM alert_rules WHERE enabled = 1",
            ):
                for row in conn.execute(q).fetchall():
                    active.add(str(row["user_id"]))
        return sorted(active)

    def _sync_resolved_short_names(self, quotes: dict[str, Any]) -> None:
        if not quotes:
            return
        with self.db.connect() as conn:
            unnamed = conn.execute(
                "SELECT user_id, stock_code, short_name FROM watchlist WHERE short_name = stock_code"
            ).fetchall()
            for r in unnamed:
                code = str(r["stock_code"])
                q = quotes.get(code)
                if q and q.short_name and q.short_name != code:
                    conn.execute(
                        "UPDATE watchlist SET short_name = ? WHERE user_id = ? AND stock_code = ?",
                        (q.short_name, r["user_id"], code),
                    )

    async def run_tick(
        self, force_sim_step: bool = False, trigger_user_id: Optional[str] = None
    ) -> dict[str, Any]:
        async with self._tick_lock:
            if trigger_user_id is not None:
                uid = trigger_user_id
                acc = self.broker.get_account_row(user_id=uid)
                mode = MarketMode.SIMULATED if force_sim_step else MarketMode(acc["market_mode"])
                watch_items = self.get_watchlist_items(user_id=uid)
                pos_items = [
                    (p.stock_code, p.short_name)
                    for p in self.broker.get_positions(user_id=uid)
                ]
                seen_codes = {c for c, _ in watch_items}
                all_items = list(watch_items) + [it for it in pos_items if it[0] not in seen_codes]
                if mode == MarketMode.SIMULATED:
                    quotes, _ = await asyncio.gather(
                        asyncio.to_thread(self.feed.step_simulated_quotes, all_items),
                        asyncio.to_thread(self.feed.fetch_indices, MarketMode.SIMULATED),
                    )
                else:
                    quotes, _ = await asyncio.gather(
                        asyncio.to_thread(self.feed.refresh_quotes, all_items, MarketMode.REAL),
                        asyncio.to_thread(self.feed.fetch_indices, MarketMode.REAL),
                    )
                self._sync_resolved_short_names(quotes)
                session = self.clock.session_status()
                effective_can_match = (mode == MarketMode.SIMULATED) or bool(session.get("can_match", False))
                new_trades: list[dict] = []
                if effective_can_match:
                    for q in quotes.values():
                        filled = self.broker.match_quote(q, user_id=uid)
                        new_trades.extend([t.model_dump() for t in filled])
                    daily_bars_map = await asyncio.to_thread(self.fetch_strategy_daily_bars, uid)
                    strat_orders = await asyncio.to_thread(
                        self.strategy_runner.evaluate, quotes, True, daily_bars_map, uid
                    )
                    filled_ids = [so.order_id for so in strat_orders if so.filled_quantity > 0]
                    new_trades.extend(self._collect_order_trades(filled_ids, user_id=uid))
                triggered = await asyncio.to_thread(self.alert_engine.evaluate, quotes, uid)
                new_alerts = [e.model_dump() for e in triggered]
                return await self.broadcast_state(
                    user_id=uid, new_alerts=new_alerts, new_trades=new_trades
                )

            # Shared background market tick across all active users
            active_users = self._list_active_tick_users()
            all_items = self.db.list_all_watched_and_held_items()
            if not all_items and not self.auth_required:
                all_items = self.get_watchlist_items("default")

            real_quotes, _ = await asyncio.gather(
                asyncio.to_thread(self.feed.fetch_real_quotes, all_items),
                asyncio.to_thread(self.feed.fetch_indices, MarketMode.REAL),
            )
            self._sync_resolved_short_names(real_quotes)

            any_sim = False
            user_modes: dict[str, MarketMode] = {}
            for uid in active_users:
                acc_row = self.broker.get_account_row(user_id=uid)
                m = MarketMode(acc_row["market_mode"])
                user_modes[uid] = m
                if m == MarketMode.SIMULATED:
                    any_sim = True

            if any_sim:
                await asyncio.gather(
                    asyncio.to_thread(self.feed.step_simulated_quotes, all_items),
                    asyncio.to_thread(self.feed.fetch_indices, MarketMode.SIMULATED),
                )

            session = self.clock.session_status()
            session_can_match = bool(session.get("can_match", False))
            daily_bars_map = await asyncio.to_thread(self.fetch_strategy_daily_bars, None)

            last_payload: dict[str, Any] = {}
            for uid in active_users:
                mode = user_modes[uid]
                user_quotes = self.feed.get_cached_quotes(mode)
                effective_can_match = (mode == MarketMode.SIMULATED) or session_can_match
                new_trades = []
                if effective_can_match:
                    for q in user_quotes.values():
                        filled = self.broker.match_quote(q, user_id=uid)
                        new_trades.extend([t.model_dump() for t in filled])
                    strat_orders = await asyncio.to_thread(
                        self.strategy_runner.evaluate, user_quotes, True, daily_bars_map, uid
                    )
                    filled_ids = [so.order_id for so in strat_orders if so.filled_quantity > 0]
                    new_trades.extend(self._collect_order_trades(filled_ids, user_id=uid))

                triggered = await asyncio.to_thread(self.alert_engine.evaluate, user_quotes, uid)
                new_alerts = [e.model_dump() for e in triggered]

                if self.user_clients.get(uid):
                    last_payload = await self.broadcast_state(
                        user_id=uid, new_alerts=new_alerts, new_trades=new_trades
                    )
            return last_payload


def _to_auth_user_view(row: dict[str, Any], db: Optional[Database] = None) -> AuthUserView:
    today = today_str()
    used = int(row.get("ai_calls_today") or 0) if row.get("ai_calls_date") == today else 0
    if db is not None:
        pub_cfg = db.get_public_ai_config()
        daily_limit = int(pub_cfg.get("daily_limit") or 0)
        has_trial = bool(
            pub_cfg.get("enabled", True)
            and pub_cfg.get("api_key")
            and pub_cfg.get("base_url")
            and pub_cfg.get("model")
        )
    else:
        daily_limit = settings.public_ai_daily_limit
        has_trial = bool(
            settings.public_ai_api_key
            and settings.public_ai_base_url
            and settings.public_ai_model
        )
    return AuthUserView(
        user_id=str(row["user_id"]),
        username=str(row["username"]),
        display_name=str(row.get("display_name") or row["username"]),
        email=str(row.get("email") or ""),
        email_verified=bool(row.get("email_verified", 0)),
        alert_email_enabled=bool(row.get("alert_email_enabled", 1)),
        role="ADMIN" if row.get("role") == "ADMIN" else "USER",
        is_active=bool(row.get("is_active", 1)),
        ai_calls_today=used,
        ai_daily_limit=daily_limit if has_trial else 0,
        created_at=str(row.get("created_at") or ""),
        last_login_at=row.get("last_login_at"),
    )


def create_app(
    db_path: Path | str | None = None,
    network_enabled: bool = True,
    enable_background_loop: bool = True,
    auth_required: Optional[bool] = None,
    open_register: Optional[bool] = None,
    invite_required: Optional[bool] = None,
    email_verification_required: Optional[bool] = None,
    allow_private_ai_endpoints: Optional[bool] = None,
    allow_private_webhooks: Optional[bool] = None,
    email_service: Optional[ResendEmailService] = None,
) -> FastAPI:
    ctx = TradingAppContext(
        db_path=db_path,
        network_enabled=network_enabled,
        auth_required=auth_required,
        open_register=open_register,
        invite_required=invite_required,
        email_verification_required=email_verification_required,
        allow_private_ai_endpoints=allow_private_ai_endpoints,
        allow_private_webhooks=allow_private_webhooks,
        email_service=email_service,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        bg_task: asyncio.Task | None = None
        cal_task: asyncio.Task | None = None

        async def load_calendar() -> None:
            dates = await asyncio.to_thread(ctx.feed.fetch_trade_calendar)
            if dates:
                ctx.clock.set_trade_dates(dates)

        async def poll_loop() -> None:
            while True:
                try:
                    await ctx.run_tick()
                except Exception:
                    pass
                await asyncio.sleep(settings.poll_interval_sec)

        if enable_background_loop:
            cal_task = asyncio.create_task(load_calendar())
            await ctx.run_tick()
            bg_task = asyncio.create_task(poll_loop())
        else:
            base_items = (
                ctx.db.list_all_watched_and_held_items()
                if ctx.auth_required
                else ctx.get_watchlist_items("default")
            )
            await asyncio.gather(
                asyncio.to_thread(ctx.feed.refresh_quotes, base_items, MarketMode.REAL),
                asyncio.to_thread(ctx.feed.fetch_indices, MarketMode.REAL),
            )

        try:
            yield
        finally:
            if bg_task:
                bg_task.cancel()
                with suppress(asyncio.CancelledError):
                    await bg_task
            if cal_task:
                cal_task.cancel()
                with suppress(asyncio.CancelledError):
                    await cal_task

    app = FastAPI(title="智瞳量化 · A股智能盯盘与模拟交易系统", version="2.0.0", lifespan=lifespan)
    app.state.ctx = ctx

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    def _extract_token(request: Request) -> str:
        cookie_tok = request.cookies.get(SESSION_COOKIE_NAME) or ""
        if cookie_tok:
            return cookie_tok
        auth_hdr = request.headers.get("authorization") or ""
        if auth_hdr.lower().startswith("bearer "):
            return auth_hdr[7:].strip()
        return ""

    def _resolve_user(request: Request) -> Optional[dict[str, Any]]:
        if not ctx.auth_required:
            return {
                "user_id": "default",
                "username": "local",
                "display_name": "本地账户",
                "role": "ADMIN",
                "is_active": 1,
                "ai_calls_today": 0,
                "ai_calls_date": today_str(),
                "created_at": "",
                "last_login_at": None,
            }
        token = _extract_token(request)
        uid = verify_session_token(token, ctx.secret_key)
        if not uid:
            return None
        return ctx.db.get_user_by_id(uid)

    async def require_user(request: Request) -> dict[str, Any]:
        user = _resolve_user(request)
        if not user:
            raise HTTPException(status_code=401, detail="请先登录账号")
        if not bool(user.get("is_active", 1)):
            raise HTTPException(status_code=403, detail="当前账号已被管理员停用")
        return user

    async def require_admin(user: dict[str, Any] = Depends(require_user)) -> dict[str, Any]:
        if user.get("role") != "ADMIN":
            raise HTTPException(status_code=403, detail="仅管理员可执行该操作")
        return user

    def _set_cookie(resp: Response, token: str, request: Request) -> None:
        forwarded_proto = (request.headers.get("x-forwarded-proto") or "").split(",")[0].strip().lower()
        is_https = (request.url.scheme == "https") or (forwarded_proto == "https")
        resp.set_cookie(
            key=SESSION_COOKIE_NAME,
            value=token,
            httponly=True,
            samesite="lax",
            secure=is_https,
            max_age=86400 * 14,
            path="/",
        )

    @app.api_route("/067511d8e4e6d461c19d8499ec80ad16.txt", methods=["GET", "HEAD"])
    async def mailjet_validation_file() -> Response:
        return Response(content="", media_type="text/plain")

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/admin")
    async def admin_page() -> FileResponse:
        return FileResponse(STATIC_DIR / "admin.html")

    @app.get("/api/announcement")
    async def public_announcement() -> dict[str, Any]:
        ann = ctx.db.get_announcement()
        if not ann["enabled"]:
            return {"enabled": False, "title": "", "content": "", "updated_at": ann.get("updated_at", "")}
        return ann

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {
            "status": "ok",
            "time": now_str(),
            "auth_required": ctx.auth_required,
            "quotes_cached": len(ctx.feed.get_cached_quotes(MarketMode.REAL)),
        }

    # --- Authentication & Account Routes ---

    @app.get("/api/auth/me", response_model=AuthStatusResponse)
    async def auth_me(request: Request) -> AuthStatusResponse:
        user = _resolve_user(request)
        inv_req = ctx.is_invite_required()
        email_req = ctx.is_email_verification_required()
        email_cfg = ctx.email_service.is_configured()
        if not user or not bool(user.get("is_active", 1)):
            return AuthStatusResponse(
                authenticated=False,
                auth_required=ctx.auth_required,
                open_register=not inv_req,
                invite_required=inv_req,
                email_verification_required=email_req,
                email_service_configured=email_cfg,
                user=None,
            )
        return AuthStatusResponse(
            authenticated=True,
            auth_required=ctx.auth_required,
            open_register=not inv_req,
            invite_required=inv_req,
            email_verification_required=email_req,
            email_service_configured=email_cfg,
            user=_to_auth_user_view(user, ctx.db),
        )

    def _client_ip(request: Request) -> str:
        return (
            (request.headers.get("cf-connecting-ip") or "").strip()
            or (request.headers.get("x-forwarded-for") or "").split(",")[0].strip()
            or (request.client.host if request.client else "unknown")
        )

    @app.post("/api/auth/send-code")
    async def auth_send_code(req: SendEmailCodeRequest, request: Request) -> dict[str, Any]:
        if not ctx.email_service.is_configured():
            raise HTTPException(status_code=400, detail="邮件服务尚未配置 RESEND_API_KEY")
        clean_email = normalize_email(req.email)
        if not is_valid_email(clean_email):
            raise HTTPException(status_code=400, detail="请输入有效的邮箱地址")

        ip_key = f"send_code:{_client_ip(request)}"
        if ctx.email_rate_limiter.is_blocked(ip_key):
            raise HTTPException(status_code=429, detail="请求验证码过于频繁，请稍后再试")

        if req.purpose in ("REGISTER", "BIND_EMAIL"):
            existing = ctx.db.get_user_by_email(clean_email)
            if existing:
                raise HTTPException(status_code=400, detail="该邮箱已被注册或绑定，请直接登录或找回密码")
        elif req.purpose == "RESET_PASSWORD":
            existing = ctx.db.get_user_by_email(clean_email)
            if not existing:
                raise HTTPException(status_code=400, detail="该邮箱尚未注册或绑定账号")

        try:
            code = ctx.db.create_email_code(clean_email, req.purpose, ttl_sec=600, cooldown_sec=60)
        except ValueError as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc

        purpose_labels = {
            "REGISTER": "账号注册",
            "RESET_PASSWORD": "重置密码",
            "BIND_EMAIL": "绑定邮箱",
        }
        ok, detail = await asyncio.to_thread(
            ctx.email_service.send_verification_code,
            to_email=clean_email,
            code=code,
            purpose_label=purpose_labels.get(req.purpose, "安全验证"),
            from_email=ctx.get_resend_from(),
        )
        if not ok:
            with ctx.db.connect() as conn:
                conn.execute(
                    "DELETE FROM email_codes WHERE email = ? AND purpose = ? AND consumed = 0",
                    (clean_email, req.purpose.upper()),
                )
            raise HTTPException(status_code=400, detail=detail)

        ctx.email_rate_limiter.record_failure(ip_key)
        return {"ok": True, "message": f"验证码已发送至 {clean_email}（10分钟内有效）"}

    @app.post("/api/auth/register")
    async def auth_register(req: AuthRegisterRequest, request: Request) -> JSONResponse:
        clean_email = normalize_email(req.email)
        require_email_code = ctx.is_email_verification_required() and ctx.email_service.is_configured()
        email_verified = False

        if require_email_code:
            if not is_valid_email(clean_email):
                raise HTTPException(status_code=400, detail="请填写有效的注册邮箱地址")
            if not req.email_code.strip():
                raise HTTPException(status_code=400, detail="请先获取并填写 6 位邮箱验证码")
            if not ctx.db.verify_and_consume_email_code(clean_email, "REGISTER", req.email_code):
                raise HTTPException(status_code=400, detail="邮箱验证码无效或已过期，请重新获取")
            email_verified = True
        elif clean_email and req.email_code.strip():
            if not ctx.db.verify_and_consume_email_code(clean_email, "REGISTER", req.email_code):
                raise HTTPException(status_code=400, detail="邮箱验证码无效或已过期")
            email_verified = True

        try:
            user = ctx.db.create_user(
                username=req.username,
                password=req.password,
                display_name=req.display_name,
                email=clean_email,
                email_verified=email_verified,
                invite_code=req.invite_code,
                require_invite=ctx.is_invite_required(),
                role="USER",
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        token = create_session_token(str(user["user_id"]), ctx.secret_key)
        payload = {"ok": True, "user": _to_auth_user_view(user, ctx.db).model_dump()}
        resp = JSONResponse(content=payload)
        _set_cookie(resp, token, request)
        return resp

    @app.post("/api/auth/reset-password")
    async def auth_reset_password(req: ResetPasswordEmailRequest) -> dict[str, Any]:
        clean_email = normalize_email(req.email)
        if not is_valid_email(clean_email):
            raise HTTPException(status_code=400, detail="邮箱格式不正确")
        if not ctx.db.verify_and_consume_email_code(clean_email, "RESET_PASSWORD", req.code):
            raise HTTPException(status_code=400, detail="验证码无效或已过期")
        try:
            ctx.db.reset_password_by_email(clean_email, req.new_password)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, "message": "密码重置成功，请使用新密码登录"}

    @app.post("/api/auth/email")
    async def auth_bind_or_update_email(
        req: BindEmailRequest, user: dict[str, Any] = Depends(require_user)
    ) -> dict[str, Any]:
        if not ctx.auth_required:
            raise HTTPException(status_code=400, detail="当前未启用多用户登录模式")
        uid = str(user["user_id"])
        clean_email = normalize_email(req.email)
        if clean_email:
            if ctx.email_service.is_configured():
                if not req.code.strip():
                    raise HTTPException(status_code=400, detail="请先获取并填写邮箱验证码")
                if not ctx.db.verify_and_consume_email_code(clean_email, "BIND_EMAIL", req.code):
                    raise HTTPException(status_code=400, detail="邮箱验证码无效或已过期")
                verified = True
            else:
                verified = False
            try:
                updated = ctx.db.update_user_email(
                    uid,
                    email=clean_email,
                    verified=verified,
                    alert_email_enabled=req.alert_email_enabled,
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        elif req.alert_email_enabled is not None:
            updated = ctx.db.update_user_email(uid, alert_email_enabled=req.alert_email_enabled)
        else:
            raise HTTPException(status_code=400, detail="请提供要绑定的邮箱地址")
        return {"ok": True, "user": _to_auth_user_view(updated, ctx.db).model_dump()}

    @app.post("/api/auth/login")
    async def auth_login(req: AuthLoginRequest, request: Request) -> JSONResponse:
        client_ip = (
            (request.headers.get("cf-connecting-ip") or "").strip()
            or (request.headers.get("x-forwarded-for") or "").split(",")[0].strip()
            or (request.client.host if request.client else "unknown")
        )
        rate_key = f"{client_ip}:{req.username.strip().lower()}"
        if ctx.login_limiter.is_blocked(rate_key):
            raise HTTPException(status_code=429, detail="登录失败次数过多，请 5 分钟后再试")

        user = ctx.db.authenticate_user(req.username, req.password)
        if not user:
            ctx.login_limiter.record_failure(rate_key)
            raise HTTPException(status_code=401, detail="用户名或密码错误")
        if not bool(user.get("is_active", 1)):
            raise HTTPException(status_code=403, detail="当前账号已被管理员停用")

        ctx.login_limiter.clear(rate_key)
        token = create_session_token(str(user["user_id"]), ctx.secret_key)
        payload = {"ok": True, "user": _to_auth_user_view(user, ctx.db).model_dump()}
        resp = JSONResponse(content=payload)
        _set_cookie(resp, token, request)
        return resp

    @app.post("/api/auth/logout")
    async def auth_logout() -> JSONResponse:
        resp = JSONResponse(content={"ok": True})
        resp.delete_cookie(key=SESSION_COOKIE_NAME, path="/")
        return resp

    @app.post("/api/auth/password")
    async def auth_change_password(
        req: ChangePasswordRequest, user: dict[str, Any] = Depends(require_user)
    ) -> dict[str, Any]:
        if not ctx.auth_required:
            raise HTTPException(status_code=400, detail="当前未启用多用户登录模式")
        try:
            ctx.db.change_password(str(user["user_id"]), req.old_password, req.new_password)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, "message": "密码修改成功"}

    # --- Admin Routes ---

    def _serialize_public_ai(pub_ai: dict[str, Any]) -> dict[str, Any]:
        pub_key = str(pub_ai.get("api_key") or "")
        raw_hdrs = str(pub_ai.get("custom_headers") or "")
        hdr_keys: list[str] = []
        if raw_hdrs:
            try:
                hdr_keys = list(parse_custom_headers(raw_hdrs).keys())
            except ValueError:
                hdr_keys = []
        fmt = normalize_ai_format(pub_ai.get("api_format"))
        base = str(pub_ai.get("base_url") or "")
        chat_ep = ""
        if base:
            try:
                chat_ep = parse_ai_endpoint(base, allow_private=True, api_format=fmt)["chat_endpoint"]
            except ValueError:
                chat_ep = ""
        return {
            "enabled": bool(pub_ai.get("enabled", True)),
            "provider_name": str(pub_ai.get("provider_name") or "自定义 AI 接口"),
            "api_format": fmt,
            "base_url": base,
            "chat_endpoint": chat_ep,
            "model": str(pub_ai.get("model") or ""),
            "daily_limit": int(pub_ai.get("daily_limit") or 0),
            "temperature": float(pub_ai.get("temperature") if pub_ai.get("temperature") is not None else 0.3),
            "max_tokens": int(pub_ai.get("max_tokens") or 1500),
            "timeout_sec": int(pub_ai.get("timeout_sec") or 30),
            "system_prompt": str(pub_ai.get("system_prompt") or ""),
            "has_api_key": bool(pub_key),
            "api_key_masked": mask_api_key(pub_key) if pub_key else "免密钥/未设置",
            "has_custom_headers": bool(hdr_keys),
            "custom_header_keys": hdr_keys,
            "configured": bool(pub_ai.get("enabled", True) and base and pub_ai.get("model")),
        }

    @app.get("/api/admin/overview")
    async def admin_overview(admin: dict[str, Any] = Depends(require_admin)) -> dict[str, Any]:
        inv_req = ctx.is_invite_required()
        quotes = ctx.feed.get_cached_quotes(MarketMode.REAL)
        telemetry = ctx.db.get_admin_telemetry(
            quotes_cached=len(quotes),
            market_mode=MarketMode.REAL.value,
            uptime_seconds=time.monotonic() - ctx.started_monotonic,
        )
        pub_ai = ctx.db.get_public_ai_config()
        return {
            "open_register": not inv_req,
            "invite_required": inv_req,
            "site_url": settings.site_url,
            "started_at": ctx.started_at,
            "email_config": {
                "configured": ctx.email_service.is_configured(),
                "api_key_masked": mask_api_key(ctx.email_service.api_key),
                "from_email": ctx.get_resend_from(),
                "email_verification_required": ctx.is_email_verification_required(),
                "site_url": settings.site_url,
            },
            "public_ai": _serialize_public_ai(pub_ai),
            "announcement": ctx.db.get_announcement(),
            "telemetry": telemetry,
            "users": ctx.db.list_users(),
            "invites": ctx.db.list_invite_codes(),
        }

    @app.post("/api/admin/registration")
    async def admin_update_registration(
        req: AdminRegistrationModeRequest, admin: dict[str, Any] = Depends(require_admin)
    ) -> dict[str, Any]:
        ctx.db.set_system_setting("invite_required", "true" if req.invite_required else "false")
        ctx._explicit_invite_required = None
        inv_req = ctx.is_invite_required()
        return {"ok": True, "open_register": not inv_req, "invite_required": inv_req}

    @app.post("/api/admin/public-ai")
    async def admin_update_public_ai(
        req: AdminPublicAIConfigRequest, admin: dict[str, Any] = Depends(require_admin)
    ) -> dict[str, Any]:
        clean_base = req.base_url.strip()
        fmt = normalize_ai_format(req.api_format)
        if clean_base:
            try:
                ep = parse_ai_endpoint(
                    clean_base,
                    allow_private=ctx.allow_private_ai_endpoints,
                    resolve_dns=not ctx.allow_private_ai_endpoints,
                    api_format=fmt,
                )
                clean_base = ep["base_url"]
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        clean_headers = req.custom_headers.strip()
        if clean_headers and not req.clear_custom_headers:
            try:
                parse_custom_headers(clean_headers)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        saved = ctx.db.set_public_ai_config(
            provider_name=req.provider_name,
            api_format=fmt,
            base_url=clean_base,
            model=req.model.strip(),
            daily_limit=req.daily_limit,
            api_key=req.api_key,
            clear_api_key=req.clear_api_key,
            custom_headers=clean_headers,
            clear_custom_headers=req.clear_custom_headers,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            timeout_sec=req.timeout_sec,
            system_prompt=req.system_prompt,
            enabled=req.enabled,
        )
        for active_uid in list(ctx.user_clients.keys()):
            await ctx.broadcast_state(user_id=active_uid)
        return {"ok": True, "public_ai": _serialize_public_ai(saved)}

    @app.post("/api/admin/public-ai/test")
    async def admin_test_public_ai(
        req: AdminPublicAITestRequest, admin: dict[str, Any] = Depends(require_admin)
    ) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(
                ctx.ai_assistant.test_platform_ai,
                api_format=req.api_format,
                base_url=req.base_url.strip(),
                model=req.model.strip(),
                api_key=req.api_key.strip(),
                custom_headers=req.custom_headers,
                clear_api_key=req.clear_api_key,
                clear_custom_headers=req.clear_custom_headers,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/admin/announcement")
    async def admin_update_announcement(
        req: AdminAnnouncementRequest, admin: dict[str, Any] = Depends(require_admin)
    ) -> dict[str, Any]:
        ann = ctx.db.set_announcement(title=req.title, content=req.content, enabled=req.enabled)
        return {"ok": True, "announcement": ann}

    @app.post("/api/admin/email")
    async def admin_update_email_settings(
        req: AdminEmailSettingsRequest, admin: dict[str, Any] = Depends(require_admin)
    ) -> dict[str, Any]:
        from_clean = req.from_email.strip()
        if "@" not in from_clean:
            raise HTTPException(status_code=400, detail="发件人邮箱格式无效（示例: 智瞳量化 <noreply@你的域名>）")
        ctx.db.set_system_setting("resend_from", from_clean)
        ctx.db.set_system_setting(
            "email_verification_required", "true" if req.email_verification_required else "false"
        )
        ctx.db.set_system_setting("invite_required", "true" if req.invite_required else "false")
        ctx._explicit_invite_required = None
        ctx._explicit_email_verify = None
        return {
            "ok": True,
            "email_config": {
                "configured": ctx.email_service.is_configured(),
                "api_key_masked": mask_api_key(ctx.email_service.api_key),
                "from_email": ctx.get_resend_from(),
                "email_verification_required": ctx.is_email_verification_required(),
                "invite_required": ctx.is_invite_required(),
                "site_url": settings.site_url,
            },
        }

    @app.post("/api/admin/email/test")
    async def admin_test_email(
        req: AdminEmailTestRequest, admin: dict[str, Any] = Depends(require_admin)
    ) -> dict[str, Any]:
        if not ctx.email_service.is_configured():
            raise HTTPException(status_code=400, detail="服务器尚未配置 RESEND_API_KEY")
        to_clean = normalize_email(req.to_email)
        if not is_valid_email(to_clean):
            raise HTTPException(status_code=400, detail="收件人邮箱格式不正确")
        ok, detail = await asyncio.to_thread(
            ctx.email_service.send_email,
            to_email=to_clean,
            subject="【智瞳量化】Resend 发信连通性测试成功",
            html_body=f"<p>你好，来自 <strong>{settings.site_url}</strong> 的 Resend 测试邮件已送达。</p><p>发送时间（北京时间）：{now_str()}</p>",
            text_body=f"智瞳量化 Resend 测试邮件送达 ({settings.site_url}) {now_str()}",
            from_email=ctx.get_resend_from(),
        )
        if not ok:
            raise HTTPException(status_code=400, detail=detail)
        return {"ok": True, "message": f"测试邮件已通过 Resend 提交发送至 {to_clean} (ID: {detail})"}

    @app.post("/api/admin/invites")
    async def admin_create_invites(
        req: InviteCreateRequest, admin: dict[str, Any] = Depends(require_admin)
    ) -> dict[str, Any]:
        try:
            created = ctx.db.create_invite_codes(
                created_by=str(admin["username"]),
                max_uses=req.max_uses,
                count=req.count,
                note=req.note,
                custom_code=req.custom_code,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"created": created, "invites": ctx.db.list_invite_codes()}

    @app.post("/api/admin/invites/{code}/toggle")
    async def admin_toggle_invite(
        code: str, payload: TogglePayload, admin: dict[str, Any] = Depends(require_admin)
    ) -> dict[str, Any]:
        try:
            inv = ctx.db.toggle_invite_code(code, payload.enabled)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"invite": inv, "invites": ctx.db.list_invite_codes()}

    @app.delete("/api/admin/invites/{code}")
    async def admin_delete_invite(
        code: str, admin: dict[str, Any] = Depends(require_admin)
    ) -> dict[str, Any]:
        ctx.db.delete_invite_code(code)
        return {"ok": True, "invites": ctx.db.list_invite_codes()}

    @app.post("/api/admin/users/{target_user_id}/toggle")
    async def admin_toggle_user(
        target_user_id: str,
        payload: TogglePayload,
        admin: dict[str, Any] = Depends(require_admin),
    ) -> dict[str, Any]:
        try:
            updated = ctx.db.toggle_user_active(target_user_id, payload.enabled)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not payload.enabled:
            clients = ctx.user_clients.pop(target_user_id, set())
            for ws in list(clients):
                with suppress(Exception):
                    await ws.close(code=4403)
        return {"user": updated, "users": ctx.db.list_users()}

    # --- Per-user Trading Terminal Routes ---

    @app.get("/api/state")
    async def get_state(user: dict[str, Any] = Depends(require_user)) -> dict[str, Any]:
        return ctx.build_state_payload(user_id=str(user["user_id"]))

    @app.post("/api/mode")
    async def switch_mode(
        req: MarketModeRequest, user: dict[str, Any] = Depends(require_user)
    ) -> dict[str, Any]:
        uid = str(user["user_id"])
        ctx.broker.set_market_mode(req.mode, user_id=uid)
        return await ctx.run_tick(
            force_sim_step=(req.mode == MarketMode.SIMULATED), trigger_user_id=uid
        )

    @app.post("/api/sim/tick")
    async def step_simulation(user: dict[str, Any] = Depends(require_user)) -> dict[str, Any]:
        uid = str(user["user_id"])
        acc = ctx.broker.get_account_row(user_id=uid)
        if acc["market_mode"] != MarketMode.SIMULATED.value:
            raise HTTPException(
                status_code=400,
                detail="当前为真实行情模式，请先切换至【仿真推演模式】后再执行单步推演",
            )
        return await ctx.run_tick(force_sim_step=True, trigger_user_id=uid)

    @app.post("/api/account/reset")
    async def reset_account(
        req: ResetAccountRequest, user: dict[str, Any] = Depends(require_user)
    ) -> dict[str, Any]:
        uid = str(user["user_id"])
        ctx.broker.reset_account(req.initial_cash, user_id=uid)
        return await ctx.broadcast_state(user_id=uid)

    @app.post("/api/account/settle")
    async def settle_day(user: dict[str, Any] = Depends(require_user)) -> dict[str, Any]:
        uid = str(user["user_id"])
        acc = ctx.broker.get_account_row(user_id=uid)
        mode = MarketMode(acc["market_mode"])
        quotes = ctx.feed.get_cached_quotes(mode)
        cur_date = acc["current_trade_date"]
        next_date = None
        if ctx.clock._trade_dates:
            future_dates = sorted(d for d in ctx.clock._trade_dates if d > cur_date)
            if future_dates:
                next_date = future_dates[0]
        ctx.broker.settle_day(quotes=quotes, next_trade_date=next_date, user_id=uid)
        return await ctx.broadcast_state(user_id=uid)

    @app.get("/api/stocks/search")
    async def search_stocks(
        q: str = Query(default=""), user: dict[str, Any] = Depends(require_user)
    ) -> list[dict[str, str]]:
        return await asyncio.to_thread(ctx.feed.search_stocks, q)

    @app.get("/api/screener/limit-up", response_model=LimitUpScreenResponse)
    async def screen_limit_up(
        days: int = Query(default=5, ge=1, le=20),
        board: str = Query(default="ALL"),
        min_count: int = Query(default=1, ge=1, le=20),
        only_watchlist: bool = Query(default=False),
        q: str = Query(default=""),
        user: dict[str, Any] = Depends(require_user),
    ) -> LimitUpScreenResponse:
        uid = str(user["user_id"])
        watch_items = ctx.get_watchlist_items(user_id=uid)
        return await asyncio.to_thread(
            ctx.feed.screen_limit_up_stocks,
            days,
            board,
            min_count,
            only_watchlist,
            q,
            watch_items,
        )

    @app.post("/api/watchlist")
    async def add_watchlist(
        req: WatchlistAddRequest, user: dict[str, Any] = Depends(require_user)
    ) -> dict[str, Any]:
        uid = str(user["user_id"])
        code = req.stock_code.strip()
        if not code.isdigit() or len(code) != 6:
            raise HTTPException(status_code=400, detail="请输入 6 位数字 A股代码")
        quotes = await asyncio.to_thread(
            ctx.feed.refresh_quotes, [(code, req.short_name or code)], MarketMode.REAL
        )
        q = quotes.get(code)
        resolved_name = req.short_name or (q.short_name if q and q.short_name else code)
        group_clean = req.group_name.strip() or "默认"
        with ctx.db.connect() as conn:
            conn.execute(
                """
                INSERT INTO watchlist (user_id, stock_code, short_name, is_pinned, group_name, note, added_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, stock_code) DO UPDATE SET short_name = excluded.short_name
                """,
                (uid, code, resolved_name, 1 if req.is_pinned else 0, group_clean, req.note.strip(), now_str()),
            )
        return await ctx.broadcast_state(user_id=uid)

    @app.patch("/api/watchlist/{stock_code}")
    async def update_watchlist(
        stock_code: str,
        req: WatchlistUpdateRequest,
        user: dict[str, Any] = Depends(require_user),
    ) -> dict[str, Any]:
        uid = str(user["user_id"])
        with ctx.db.connect() as conn:
            row = conn.execute(
                "SELECT * FROM watchlist WHERE user_id = ? AND stock_code = ?",
                (uid, stock_code),
            ).fetchone()
            if not row:
                raise HTTPException(status_code=404, detail=f"自选股 {stock_code} 不存在")
            new_group = (
                (req.group_name.strip() or "默认")
                if req.group_name is not None
                else (row["group_name"] or "默认")
            )
            new_pinned = (1 if req.is_pinned else 0) if req.is_pinned is not None else int(row["is_pinned"])
            new_note = req.note.strip() if req.note is not None else (row["note"] or "")
            conn.execute(
                """
                UPDATE watchlist SET group_name = ?, is_pinned = ?, note = ?
                WHERE user_id = ? AND stock_code = ?
                """,
                (new_group, new_pinned, new_note, uid, stock_code),
            )
        return await ctx.broadcast_state(user_id=uid)

    @app.delete("/api/watchlist/{stock_code}")
    async def remove_watchlist(
        stock_code: str, user: dict[str, Any] = Depends(require_user)
    ) -> dict[str, Any]:
        uid = str(user["user_id"])
        with ctx.db.connect() as conn:
            cur = conn.execute(
                "DELETE FROM watchlist WHERE user_id = ? AND stock_code = ?",
                (uid, stock_code),
            )
            if cur.rowcount == 0:
                raise HTTPException(status_code=404, detail=f"自选股 {stock_code} 不存在")
        return await ctx.broadcast_state(user_id=uid)

    @app.get("/api/quotes/{stock_code}/chart")
    async def get_stock_chart(
        stock_code: str, user: dict[str, Any] = Depends(require_user)
    ) -> dict[str, Any]:
        uid = str(user["user_id"])
        acc = ctx.broker.get_account_row(user_id=uid)
        mode = MarketMode(acc["market_mode"])
        q = ctx.feed.get_quote(stock_code, mode=mode)
        if not q:
            fetched = await asyncio.to_thread(
                ctx.feed.refresh_quotes, [(stock_code, stock_code)], mode
            )
            q = fetched.get(stock_code)
        mins, daily = await asyncio.gather(
            asyncio.to_thread(ctx.feed.fetch_intraday_mins, stock_code, mode),
            asyncio.to_thread(ctx.feed.fetch_daily_kline, stock_code, 60),
        )
        return {
            "stock_code": stock_code,
            "quote": q.model_dump() if q else None,
            "mins": [m.model_dump() for m in mins],
            "daily": daily,
        }

    @app.get("/api/indices/{symbol}/detail")
    async def get_index_detail(
        symbol: str, user: dict[str, Any] = Depends(require_user)
    ) -> dict[str, Any]:
        uid = str(user["user_id"])
        acc = ctx.broker.get_account_row(user_id=uid)
        mode = MarketMode(acc["market_mode"])
        try:
            return await asyncio.to_thread(ctx.feed.fetch_index_detail, symbol, mode)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/stocks/{stock_code}/detail")
    async def get_stock_detail(
        stock_code: str, user: dict[str, Any] = Depends(require_user)
    ) -> dict[str, Any]:
        uid = str(user["user_id"])
        acc = ctx.broker.get_account_row(user_id=uid)
        mode = MarketMode(acc["market_mode"])
        return await asyncio.to_thread(ctx.feed.fetch_stock_detail, stock_code, mode)

    @app.post("/api/orders")
    async def place_order(
        req: OrderRequest, user: dict[str, Any] = Depends(require_user)
    ) -> dict[str, Any]:
        uid = str(user["user_id"])
        acc = ctx.broker.get_account_row(user_id=uid)
        mode = MarketMode(acc["market_mode"])
        q = ctx.feed.get_quote(req.stock_code, mode=mode)
        if not q:
            fetched = await asyncio.to_thread(
                ctx.feed.refresh_quotes, [(req.stock_code, req.stock_code)], mode
            )
            q = fetched.get(req.stock_code)

        session = ctx.clock.session_status()
        effective_can_match = (mode == MarketMode.SIMULATED) or bool(session.get("can_match", False))
        order = ctx.broker.place_order(
            req,
            quote=q,
            match_now=effective_can_match,
            quotes_map=ctx.feed.get_cached_quotes(mode),
            user_id=uid,
        )
        new_trades = ctx._collect_order_trades(
            [order.order_id] if order.filled_quantity > 0 else [], user_id=uid
        )
        state = await ctx.broadcast_state(user_id=uid, new_trades=new_trades)
        return {"order": order.model_dump(), "state": state}

    @app.post("/api/orders/{order_id}/cancel")
    async def cancel_order(
        order_id: str, user: dict[str, Any] = Depends(require_user)
    ) -> dict[str, Any]:
        uid = str(user["user_id"])
        try:
            order = ctx.broker.cancel_order(order_id, user_id=uid)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        state = await ctx.broadcast_state(user_id=uid)
        return {"order": order.model_dump(), "state": state}

    @app.post("/api/alerts")
    async def create_alert(
        req: AlertRuleRequest, user: dict[str, Any] = Depends(require_user)
    ) -> dict[str, Any]:
        uid = str(user["user_id"])
        mode = MarketMode(ctx.broker.get_account_row(user_id=uid)["market_mode"])
        q = ctx.feed.get_quote(req.stock_code, mode=mode)
        sname = q.short_name if q else ""
        try:
            rule = ctx.alert_engine.add_rule(req, short_name=sname, user_id=uid)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        triggered = []
        if q:
            triggered = [e.model_dump() for e in ctx.alert_engine.evaluate({req.stock_code: q}, user_id=uid)]
        state = await ctx.broadcast_state(user_id=uid, new_alerts=triggered)
        return {"rule": rule.model_dump(), "state": state}

    @app.post("/api/alerts/{rule_id}/toggle")
    async def toggle_alert(
        rule_id: str,
        payload: dict[str, bool] = Body(...),
        user: dict[str, Any] = Depends(require_user),
    ) -> dict[str, Any]:
        uid = str(user["user_id"])
        try:
            rule = ctx.alert_engine.toggle_rule(rule_id, bool(payload.get("enabled", True)), user_id=uid)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        state = await ctx.broadcast_state(user_id=uid)
        return {"rule": rule.model_dump(), "state": state}

    @app.delete("/api/alerts/{rule_id}")
    async def delete_alert(
        rule_id: str, user: dict[str, Any] = Depends(require_user)
    ) -> dict[str, Any]:
        uid = str(user["user_id"])
        ctx.alert_engine.delete_rule(rule_id, user_id=uid)
        return await ctx.broadcast_state(user_id=uid)

    @app.post("/api/strategies")
    async def create_strategy(
        req: AutoStrategyRuleRequest, user: dict[str, Any] = Depends(require_user)
    ) -> dict[str, Any]:
        uid = str(user["user_id"])
        try:
            rule = ctx.strategy_runner.add_strategy(req, user_id=uid)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        acc = ctx.broker.get_account_row(user_id=uid)
        mode = MarketMode(acc["market_mode"])
        session = ctx.clock.session_status()
        effective_can_match = (mode == MarketMode.SIMULATED) or bool(session.get("can_match", False))
        new_trades: list[dict] = []
        if effective_can_match:
            quotes = ctx.feed.get_cached_quotes(mode)
            daily_bars_map = await asyncio.to_thread(ctx.fetch_strategy_daily_bars, uid)
            strat_orders = await asyncio.to_thread(
                ctx.strategy_runner.evaluate, quotes, True, daily_bars_map, uid
            )
            filled_ids = [so.order_id for so in strat_orders if so.filled_quantity > 0]
            new_trades = ctx._collect_order_trades(filled_ids, user_id=uid)
        state = await ctx.broadcast_state(user_id=uid, new_trades=new_trades)
        return {"strategy": rule.model_dump(), "state": state}

    @app.post("/api/strategies/{strategy_id}/toggle")
    async def toggle_strategy(
        strategy_id: str,
        payload: dict[str, bool] = Body(...),
        user: dict[str, Any] = Depends(require_user),
    ) -> dict[str, Any]:
        uid = str(user["user_id"])
        try:
            rule = ctx.strategy_runner.toggle_strategy(
                strategy_id, bool(payload.get("enabled", True)), user_id=uid
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        state = await ctx.broadcast_state(user_id=uid)
        return {"strategy": rule.model_dump(), "state": state}

    @app.delete("/api/strategies/{strategy_id}")
    async def delete_strategy(
        strategy_id: str, user: dict[str, Any] = Depends(require_user)
    ) -> dict[str, Any]:
        uid = str(user["user_id"])
        ctx.strategy_runner.delete_strategy(strategy_id, user_id=uid)
        return await ctx.broadcast_state(user_id=uid)

    @app.get("/api/risk")
    async def get_risk_config(user: dict[str, Any] = Depends(require_user)) -> dict[str, Any]:
        return ctx.broker.get_risk_config(user_id=str(user["user_id"])).model_dump()

    @app.post("/api/risk")
    async def update_risk_config(
        cfg: RiskConfig, user: dict[str, Any] = Depends(require_user)
    ) -> dict[str, Any]:
        uid = str(user["user_id"])
        saved = ctx.broker.update_risk_config(cfg, user_id=uid)
        state = await ctx.broadcast_state(user_id=uid)
        return {"risk_config": saved.model_dump(), "state": state}

    @app.post("/api/backtest")
    async def execute_backtest(
        req: BacktestRequest, user: dict[str, Any] = Depends(require_user)
    ) -> dict[str, Any]:
        uid = str(user["user_id"])
        warmup = max(req.slow_period + 15, 40)
        bars = await asyncio.to_thread(ctx.feed.fetch_daily_kline, req.stock_code, req.days + warmup)
        q = ctx.feed.get_quote(req.stock_code, MarketMode.REAL)
        short_name = q.short_name if q and q.short_name else ""
        if not short_name:
            for code, sname in ctx.get_watchlist_items(user_id=uid):
                if code == req.stock_code:
                    short_name = sname
                    break
        if not short_name and req.stock_code in POPULAR_STOCKS:
            short_name = str(POPULAR_STOCKS[req.stock_code]["name"])
        try:
            res = await asyncio.to_thread(run_backtest, req, bars, short_name or req.stock_code)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return res.model_dump()

    def _csv_response(filename: str, headers: list[str], rows: list[list[Any]]) -> Response:
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(headers)
        writer.writerows(rows)
        body = buf.getvalue().encode("utf-8-sig")
        return Response(
            content=body,
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.get("/api/export/trades.csv")
    async def export_trades_csv(user: dict[str, Any] = Depends(require_user)) -> Response:
        uid = str(user["user_id"])
        trades = ctx.broker.list_trades(limit=5000, user_id=uid)
        headers = [
            "成交编号", "委托编号", "代码", "名称", "方向", "成交价", "数量",
            "成交额", "佣金", "过户费", "印花税", "合计费用", "交易日", "成交时间",
        ]
        rows = [
            [
                t.trade_id, t.order_id, t.stock_code, t.short_name,
                "买入" if t.side.value == "BUY" else "卖出",
                f"{t.price:.3f}", t.quantity, f"{t.amount:.2f}",
                f"{t.commission:.2f}", f"{t.transfer_fee:.2f}", f"{t.stamp_tax:.2f}",
                f"{t.total_fee:.2f}", t.trade_date, t.created_at,
            ]
            for t in trades
        ]
        return _csv_response("trades_export.csv", headers, rows)

    @app.get("/api/export/orders.csv")
    async def export_orders_csv(user: dict[str, Any] = Depends(require_user)) -> Response:
        uid = str(user["user_id"])
        orders = ctx.broker.list_orders(limit=5000, user_id=uid)
        headers = [
            "委托编号", "代码", "名称", "方向", "委托价", "委托数量", "成交数量",
            "成交均价", "冻结资金", "状态", "拒单原因", "策略来源", "交易日", "委托时间",
        ]
        rows = [
            [
                o.order_id, o.stock_code, o.short_name,
                "买入" if o.side.value == "BUY" else "卖出",
                f"{o.price:.2f}", o.quantity, o.filled_quantity, f"{o.avg_filled_price:.3f}",
                f"{o.frozen_cash:.2f}", o.status.value, o.reject_reason, o.strategy_name,
                o.trade_date, o.created_at,
            ]
            for o in orders
        ]
        return _csv_response("orders_export.csv", headers, rows)

    @app.get("/api/ai/config", response_model=AIConfigView)
    async def get_ai_config(user: dict[str, Any] = Depends(require_user)) -> AIConfigView:
        return ctx.ai_assistant.get_config_view(user_id=str(user["user_id"]))

    @app.post("/api/ai/config")
    async def update_ai_config(
        req: AIConfigUpdateRequest, user: dict[str, Any] = Depends(require_user)
    ) -> dict[str, Any]:
        uid = str(user["user_id"])
        try:
            saved = ctx.ai_assistant.save_config(req, user_id=uid)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        state = await ctx.broadcast_state(user_id=uid)
        return {"ai_config": saved.model_dump(), "state": state}

    @app.post("/api/ai/test", response_model=AIHealthCheckResponse)
    async def test_ai_connection(
        req: AIConfigTestRequest | None = Body(default=None),
        user: dict[str, Any] = Depends(require_user),
    ) -> AIHealthCheckResponse:
        uid = str(user["user_id"])
        try:
            res = await asyncio.to_thread(ctx.ai_assistant.test_connection, req, uid)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        await ctx.broadcast_state(user_id=uid)
        return res

    @app.post("/api/ai/ask", response_model=AIAskResponse)
    async def ask_ai_assistant(
        req: AIAskRequest, user: dict[str, Any] = Depends(require_user)
    ) -> AIAskResponse:
        uid = str(user["user_id"])
        state_payload = ctx.build_state_payload(user_id=uid)
        watchlist = state_payload.get("watchlist") or []
        target_code = (req.stock_code or "").strip()
        if not target_code and watchlist:
            target_code = str(watchlist[0].get("stock_code") or "")

        acc_mode = MarketMode((state_payload.get("account") or {}).get("market_mode", MarketMode.REAL.value))
        quote = ctx.feed.get_quote(target_code, mode=acc_mode) if target_code else None
        if target_code and quote is None:
            fetched = await asyncio.to_thread(
                ctx.feed.refresh_quotes, [(target_code, target_code)], acc_mode
            )
            quote = fetched.get(target_code)

        bars: list[dict] = []
        if target_code:
            bars = await asyncio.to_thread(ctx.feed.fetch_daily_kline, target_code, 45)

        screener_resp = None
        if req.task_type == "SCREENER_SUMMARY":
            screener_resp = await asyncio.to_thread(
                ctx.feed.screen_limit_up_stocks,
                req.screener_days,
                "ALL",
                1,
                False,
                "",
                ctx.get_watchlist_items(user_id=uid),
            )

        res = await asyncio.to_thread(
            ctx.ai_assistant.ask,
            req,
            state_payload,
            quote,
            bars,
            screener_resp,
            uid,
        )
        if res.source == "platform_trial_llm":
            await ctx.broadcast_state(user_id=uid)
        return res

    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket) -> None:
        if ctx.auth_required:
            tok = ws.cookies.get(SESSION_COOKIE_NAME) or ws.query_params.get("token") or ""
            uid = verify_session_token(tok, ctx.secret_key)
            u = ctx.db.get_user_by_id(uid) if uid else None
            if not u or not bool(u.get("is_active", 1)):
                await ws.close(code=4401)
                return
            user_id = str(u["user_id"])
        else:
            user_id = "default"

        await ws.accept()
        clients = ctx.user_clients.setdefault(user_id, set())
        clients.add(ws)
        try:
            await ws.send_json(ctx.build_state_payload(user_id=user_id))
            while True:
                await ws.receive_text()
        except WebSocketDisconnect:
            clients.discard(ws)
            if not clients:
                ctx.user_clients.pop(user_id, None)
        except Exception:
            clients.discard(ws)
            if not clients:
                ctx.user_clients.pop(user_id, None)

    return app


app = create_app()
