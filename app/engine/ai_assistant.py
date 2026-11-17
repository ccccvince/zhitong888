from __future__ import annotations

import json
import re
import socket
import time
from collections import Counter
from typing import Any, Callable, Optional
from urllib.parse import urlsplit

import httpx

from app.config import settings
from app.engine.market_feed import (
    classify_board,
    compute_macd_series,
    compute_sma_series,
    is_limit_up_bar,
)
from app.models import (
    AIActionDraft,
    AIAskRequest,
    AIAskResponse,
    AIConfigTestRequest,
    AIConfigUpdateRequest,
    AIConfigView,
    AIHealthCheckResponse,
    AlertRuleRequest,
    AutoStrategyRuleRequest,
    BacktestRequest,
    LimitUpScreenResponse,
    PositionRecord,
    QuoteSnapshot,
)
from app.security import is_private_or_reserved_host
from app.storage import Database, now_str

TcpProbeFn = Callable[[str, int, float], tuple[bool, Optional[float], str]]
HttpClientFactory = Callable[[float], httpx.Client]


DEFAULT_AI_SYSTEM_PROMPT = (
    "你是严谨、客观的 A 股智能盯盘与量化投研副驾驶。请严格遵守以下准则：\n"
    "1. 仅基于系统提供的客观行情、五档盘口、已算出的均线/MACD 指标、涨停统计与模拟账户数据进行分析，严禁编造未提供的价格或基本面新闻。\n"
    "2. 结合 A 股 T+1 交收、涨跌幅（主板10%/双创20%/北交所30%）、100 股整数倍委托及滑点成本给出建议与明确风险提示，不作收益保证。\n"
    "3. 回答简洁、结构化（使用简短分点与加粗结论）。\n"
    "4. 如用户需求明确涉及跑回测、建自动策略、加预警或预填下单参数，可在回答末尾附带一个合法的 ```json 代码块（键为 \"drafts\" 数组）：\n"
    "   - kind 可选: BACKTEST / STRATEGY / ALERT / ORDER_PREFILL。\n"
    "   - 仅给出参数草稿供用户点击确认，不可声称已自动成交。"
)


def mask_api_key(raw_key: str) -> str:
    key = (raw_key or "").strip()
    if not key:
        return "未设置"
    if len(key) <= 8:
        return key[:2] + "****"
    return f"{key[:4]}...{key[-4:]}"


def parse_ai_endpoint(
    raw_base_url: str,
    allow_private: bool = True,
    resolve_dns: bool = False,
) -> dict[str, Any]:
    """Validate and normalize a universal OpenAI-compatible Base URL, extracting host, port, and endpoints."""
    cleaned = (raw_base_url or "").strip().rstrip("/")
    if not cleaned:
        raise ValueError("AI 接口地址不能为空")
    if "://" not in cleaned:
        raise ValueError("AI 接口地址必须包含协议头（如 http:// 或 https://）")

    parts = urlsplit(cleaned)
    scheme = parts.scheme.lower()
    if scheme not in ("http", "https"):
        raise ValueError("AI 接口仅支持 http:// 或 https:// 协议")

    host = parts.hostname
    if not host:
        raise ValueError("未能解析出有效的主机名或 IP")

    if not allow_private and is_private_or_reserved_host(host, resolve_dns=resolve_dns):
        raise ValueError(f"安全限制：公开环境不允许访问本机或内网 AI 地址 ({host})，请填写公网 HTTPS 接口")

    try:
        explicit_port = parts.port
    except ValueError as exc:
        raise ValueError("AI 接口端口号无效（范围需为 1~65535）") from exc

    port = explicit_port if explicit_port is not None else (443 if scheme == "https" else 80)
    if port < 1 or port > 65535:
        raise ValueError("AI 接口端口号超出有效范围 (1~65535)")

    path = (parts.path or "").rstrip("/")
    for suffix in ("/chat/completions", "/models"):
        if path.endswith(suffix):
            path = path[: -len(suffix)].rstrip("/")
            break
    if not path:
        path = "/v1"

    norm_base = f"{scheme}://{parts.netloc}{path}"
    return {
        "scheme": scheme,
        "host": host,
        "port": port,
        "base_path": path,
        "base_url": norm_base,
        "chat_endpoint": f"{norm_base}/chat/completions",
        "models_endpoint": f"{norm_base}/models",
    }


def default_tcp_probe(host: str, port: int, timeout_sec: float = 3.0) -> tuple[bool, Optional[float], str]:
    t0 = time.perf_counter()
    try:
        with socket.create_connection((host, port), timeout=max(0.5, min(timeout_sec, 5.0))):
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            return True, latency_ms, f"TCP 端口 {host}:{port} 可达 ({latency_ms}ms)"
    except OSError as exc:
        return False, None, f"TCP 端口 {host}:{port} 连接失败: {exc}"


def _default_http_client(timeout_sec: float, local_direct: bool = False) -> httpx.Client:
    return httpx.Client(timeout=timeout_sec, trust_env=not local_direct)


def _is_local_host(host: str) -> bool:
    h = (host or "").strip().lower()
    return h in ("127.0.0.1", "localhost", "::1") or h.startswith(("192.168.", "10.", "172.16.", "172.17."))


def _extract_drafts_from_reply(raw_text: str) -> tuple[str, list[AIActionDraft]]:
    """Extract optional ```json {"drafts": [...]} ``` block from LLM output and validate each item."""
    text = (raw_text or "").strip()
    drafts: list[AIActionDraft] = []
    if not text:
        return "", drafts

    pattern = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)
    matches = list(pattern.finditer(text))
    cleaned_text = text

    for match in reversed(matches):
        block = match.group(1).strip()
        if not block.startswith("{"):
            continue
        try:
            parsed = json.loads(block)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict) or "drafts" not in parsed or not isinstance(parsed["drafts"], list):
            continue

        validated: list[AIActionDraft] = []
        for item in parsed["drafts"]:
            if not isinstance(item, dict):
                continue
            kind = str(item.get("kind") or "").upper()
            payload = item.get("payload") if isinstance(item.get("payload"), dict) else {}
            title = str(item.get("title") or kind).strip()[:80]
            summary = str(item.get("summary") or "").strip()[:160]
            norm_payload = _validate_draft_payload(kind, payload)
            if norm_payload is None:
                continue
            validated.append(
                AIActionDraft(
                    kind=kind,  # type: ignore[arg-type]
                    title=title or kind,
                    summary=summary,
                    payload=norm_payload,
                )
            )
        drafts = validated + drafts
        cleaned_text = (cleaned_text[: match.start()] + cleaned_text[match.end() :]).strip()

    return cleaned_text, drafts[:4]


def _validate_draft_payload(kind: str, payload: dict[str, Any]) -> Optional[dict[str, Any]]:
    try:
        if kind == "BACKTEST":
            req = BacktestRequest(**payload)
            if req.strategy_type == "MA_CROSS" and req.fast_period >= req.slow_period:
                return None
            return req.model_dump()
        if kind == "STRATEGY":
            req = AutoStrategyRuleRequest(**payload)
            if req.order_qty % 100 != 0:
                return None
            if req.strategy_type == "MA_CROSS" and req.fast_period >= req.slow_period:
                return None
            return req.model_dump()
        if kind == "ALERT":
            req = AlertRuleRequest(**payload)
            return req.model_dump()
        if kind == "ORDER_PREFILL":
            code = str(payload.get("stock_code") or "").strip()
            side = str(payload.get("side") or "BUY").upper()
            qty = int(payload.get("quantity") or 100)
            price = float(payload.get("price") or 0.0)
            if len(code) != 6 or not code.isdigit() or side not in ("BUY", "SELL"):
                return None
            qty = max(100, (qty // 100) * 100)
            return {
                "stock_code": code,
                "side": side,
                "price": round(max(0.0, price), 2),
                "quantity": qty,
            }
    except Exception:
        return None
    return None


class AIAssistantService:
    """Universal OpenAI-compatible AI assistant with port/endpoint inspection and A-share quant context."""

    def __init__(
        self,
        db: Database,
        tcp_probe: TcpProbeFn = default_tcp_probe,
        http_client_factory: HttpClientFactory = _default_http_client,
        allow_private_endpoints: bool = True,
    ) -> None:
        self.db = db
        self.tcp_probe = tcp_probe
        self.http_client_factory = http_client_factory
        self.allow_private_endpoints = allow_private_endpoints

    def _open_client(self, host: str, timeout_sec: float) -> httpx.Client:
        if self.http_client_factory is _default_http_client:
            return _default_http_client(timeout_sec, local_direct=_is_local_host(host))
        return self.http_client_factory(timeout_sec)

    @staticmethod
    def _has_platform_trial() -> bool:
        return bool(
            settings.public_ai_api_key
            and settings.public_ai_base_url
            and settings.public_ai_model
            and settings.public_ai_daily_limit > 0
        )

    def _get_raw_row(self, user_id: str = "default") -> dict[str, Any]:
        with self.db.connect() as conn:
            row = conn.execute("SELECT * FROM ai_config WHERE user_id = ?", (user_id,)).fetchone()
            if row:
                return dict(row)
        return {
            "user_id": user_id,
            "provider_name": "Ollama 本地 (默认)",
            "base_url": "http://127.0.0.1:11434/v1",
            "model": "qwen2.5:14b",
            "api_key": "",
            "temperature": 0.3,
            "max_tokens": 1500,
            "timeout_sec": 30,
            "system_prompt": "",
            "enabled": 0,
            "last_check_status": "UNCONFIGURED",
            "last_check_message": "尚未测试连通性",
            "last_checked_at": None,
            "updated_at": now_str(),
        }

    def _row_to_view(self, row: dict[str, Any], user_id: str = "default") -> AIConfigView:
        raw_url = str(row.get("base_url") or "http://127.0.0.1:11434/v1")
        try:
            ep = parse_ai_endpoint(raw_url, allow_private=True)
        except ValueError:
            ep = {
                "scheme": "http",
                "host": "127.0.0.1",
                "port": 11434,
                "base_path": "/v1",
                "base_url": raw_url,
                "chat_endpoint": raw_url.rstrip("/") + "/chat/completions",
                "models_endpoint": raw_url.rstrip("/") + "/models",
            }
        raw_key = str(row.get("api_key") or "")
        trial_ok = self._has_platform_trial()
        trial_rem = 0
        if trial_ok:
            _, trial_rem = self.db.get_platform_ai_usage(user_id, settings.public_ai_daily_limit)
        return AIConfigView(
            provider_name=str(row.get("provider_name") or "通用 OpenAI 兼容"),
            base_url=ep["base_url"],
            scheme=ep["scheme"],
            host=ep["host"],
            port=ep["port"],
            base_path=ep["base_path"],
            chat_endpoint=ep["chat_endpoint"],
            models_endpoint=ep["models_endpoint"],
            model=str(row.get("model") or "qwen2.5:14b"),
            has_api_key=bool(raw_key.strip()),
            api_key_masked=mask_api_key(raw_key),
            temperature=float(row.get("temperature") if row.get("temperature") is not None else 0.3),
            max_tokens=int(row.get("max_tokens") or 1500),
            timeout_sec=int(row.get("timeout_sec") or 30),
            system_prompt=str(row.get("system_prompt") or ""),
            enabled=bool(row.get("enabled")),
            last_check_status=str(row.get("last_check_status") or "UNCONFIGURED"),
            last_check_message=str(row.get("last_check_message") or ""),
            last_checked_at=row.get("last_checked_at"),
            updated_at=row.get("updated_at"),
            allow_private_endpoints=self.allow_private_endpoints,
            platform_trial_available=trial_ok,
            trial_daily_limit=settings.public_ai_daily_limit if trial_ok else 0,
            trial_remaining_today=trial_rem,
            platform_trial_model=settings.public_ai_model if trial_ok else "",
        )

    def get_config_view(self, user_id: str = "default") -> AIConfigView:
        return self._row_to_view(self._get_raw_row(user_id=user_id), user_id=user_id)

    def save_config(self, req: AIConfigUpdateRequest, user_id: str = "default") -> AIConfigView:
        ep = parse_ai_endpoint(
            req.base_url,
            allow_private=self.allow_private_endpoints,
            resolve_dns=not self.allow_private_endpoints,
        )
        existing = self._get_raw_row(user_id=user_id)
        if req.clear_api_key:
            next_key = ""
        elif req.api_key is not None and req.api_key.strip():
            next_key = req.api_key.strip()
        else:
            next_key = str(existing.get("api_key") or "")

        changed_endpoint = (
            ep["base_url"] != existing.get("base_url")
            or req.model.strip() != existing.get("model")
            or next_key != str(existing.get("api_key") or "")
        )
        check_status = "UNCONFIGURED" if changed_endpoint else existing.get("last_check_status", "UNCONFIGURED")
        check_msg = (
            "配置已更新，可点击「测试端口与模型连通」"
            if changed_endpoint
            else existing.get("last_check_message", "")
        )
        checked_at = None if changed_endpoint else existing.get("last_checked_at")

        now = now_str()
        self.db.ensure_user_workspace(user_id)
        with self.db.connect() as conn:
            conn.execute(
                """
                UPDATE ai_config SET
                    provider_name = ?,
                    base_url = ?,
                    model = ?,
                    api_key = ?,
                    temperature = ?,
                    max_tokens = ?,
                    timeout_sec = ?,
                    system_prompt = ?,
                    enabled = ?,
                    last_check_status = ?,
                    last_check_message = ?,
                    last_checked_at = ?,
                    updated_at = ?
                WHERE user_id = ?
                """,
                (
                    req.provider_name.strip() or "通用 OpenAI 兼容",
                    ep["base_url"],
                    req.model.strip(),
                    next_key,
                    float(req.temperature),
                    int(req.max_tokens),
                    int(req.timeout_sec),
                    req.system_prompt.strip(),
                    1 if req.enabled else 0,
                    check_status,
                    check_msg,
                    checked_at,
                    now,
                    user_id,
                ),
            )
        return self.get_config_view(user_id=user_id)

    def test_connection(
        self, req: Optional[AIConfigTestRequest] = None, user_id: str = "default"
    ) -> AIHealthCheckResponse:
        saved = self._get_raw_row(user_id=user_id)
        use_override = req is not None and any(
            val is not None
            for val in (req.provider_name, req.base_url, req.model, req.api_key)
        )
        raw_base = (req.base_url if (req and req.base_url is not None) else saved["base_url"]) or ""
        provider_name = (
            req.provider_name
            if (req and req.provider_name is not None)
            else saved["provider_name"]
        )
        model_name = (req.model if (req and req.model is not None) else saved["model"]).strip()
        timeout_sec = int(req.timeout_sec if (req and req.timeout_sec is not None) else saved["timeout_sec"])
        probe_chat = bool(req.probe_chat) if req is not None else True

        if req and req.clear_api_key:
            api_key = ""
        elif req and req.api_key is not None and req.api_key.strip():
            api_key = req.api_key.strip()
        else:
            api_key = str(saved.get("api_key") or "")

        ep = parse_ai_endpoint(
            raw_base,
            allow_private=self.allow_private_endpoints,
            resolve_dns=not self.allow_private_endpoints,
        )
        preview_row = {
            **saved,
            "provider_name": provider_name,
            "base_url": ep["base_url"],
            "model": model_name,
            "api_key": api_key,
            "timeout_sec": timeout_sec,
        }
        view = self._row_to_view(preview_row, user_id=user_id)
        checked_at = now_str()

        tcp_ok, tcp_ms, tcp_msg = self.tcp_probe(ep["host"], ep["port"], float(timeout_sec))
        http_status: Optional[int] = None
        chat_probe_ok: Optional[bool] = None
        discovered_models: list[str] = []
        status = "ERROR"
        message = tcp_msg

        if tcp_ok:
            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            try:
                with self._open_client(ep["host"], float(min(timeout_sec, 15))) as client:
                    # 1. Discover models via /models
                    try:
                        m_resp = client.get(ep["models_endpoint"], headers=headers)
                        http_status = m_resp.status_code
                        if m_resp.status_code == 200:
                            m_data = m_resp.json()
                            if isinstance(m_data, dict) and isinstance(m_data.get("data"), list):
                                discovered_models = [
                                    str(it["id"])
                                    for it in m_data["data"]
                                    if isinstance(it, dict) and it.get("id")
                                ][:30]
                    except Exception as exc:
                        message = f"端口可达 ({tcp_ms}ms)，/models 检测异常: {type(exc).__name__}"

                    # 2. Optionally run a minimal chat probe to validate model & API key
                    if probe_chat:
                        c_resp = client.post(
                            ep["chat_endpoint"],
                            headers=headers,
                            json={
                                "model": model_name,
                                "messages": [{"role": "user", "content": "请回复 OK"}],
                                "max_tokens": 8,
                                "temperature": 0,
                            },
                        )
                        http_status = c_resp.status_code
                        if c_resp.status_code == 200:
                            c_body = c_resp.json()
                            if isinstance(c_body, dict) and c_body.get("choices"):
                                chat_probe_ok = True
                                status = "OK"
                                message = f"端口 {ep['host']}:{ep['port']} 连通正常，模型 {model_name} 响应成功 ({tcp_ms}ms)"
                            else:
                                chat_probe_ok = False
                                status = "WARN"
                                message = f"接口返回 HTTP 200，但响应格式非标准 OpenAI choices"
                        else:
                            chat_probe_ok = False
                            err_detail = self._extract_http_error(c_resp)
                            status = "WARN" if c_resp.status_code in (401, 403, 404, 422) else "ERROR"
                            message = f"TCP 端口 {ep['host']}:{ep['port']} 可达，但对话接口返回 HTTP {c_resp.status_code}: {err_detail}"
                    elif http_status == 200:
                        status = "OK"
                        message = f"端口 {ep['host']}:{ep['port']} 可达，发现 {len(discovered_models)} 个模型"
                    elif http_status is not None:
                        status = "WARN"
                        message = f"TCP 端口可达 ({tcp_ms}ms)，/models 返回 HTTP {http_status}"
            except Exception as exc:
                status = "ERROR"
                message = f"TCP 端口 {ep['host']}:{ep['port']} 可达，但 HTTP 请求失败: {type(exc).__name__}"

        # Always persist check result when testing saved config or same base_url
        if not use_override or ep["base_url"] == saved.get("base_url"):
            with self.db.connect() as conn:
                conn.execute(
                    """
                    UPDATE ai_config SET last_check_status = ?, last_check_message = ?, last_checked_at = ?
                    WHERE user_id = ?
                    """,
                    (status, message, checked_at, user_id),
                )
            view.last_check_status = status
            view.last_check_message = message
            view.last_checked_at = checked_at

        return AIHealthCheckResponse(
            endpoint_info=view,
            tcp_reachable=tcp_ok,
            tcp_latency_ms=tcp_ms,
            http_status=http_status,
            chat_probe_ok=chat_probe_ok,
            discovered_models=discovered_models,
            status=status,
            message=message,
            checked_at=checked_at,
        )

    @staticmethod
    def _extract_http_error(resp: httpx.Response) -> str:
        try:
            body = resp.json()
            if isinstance(body, dict):
                err = body.get("error")
                if isinstance(err, dict) and err.get("message"):
                    return str(err["message"])[:180]
                if body.get("detail"):
                    return str(body["detail"])[:180]
                if body.get("message"):
                    return str(body["message"])[:180]
        except Exception:
            pass
        return (resp.text or "")[:140] or "无详细错误信息"

    def ask(
        self,
        req: AIAskRequest,
        state_payload: dict[str, Any],
        stock_quote: Optional[QuoteSnapshot] = None,
        stock_bars: Optional[list[dict]] = None,
        screener_resp: Optional[LimitUpScreenResponse] = None,
        user_id: str = "default",
    ) -> AIAskResponse:
        t0 = time.perf_counter()
        raw_cfg = self._get_raw_row(user_id=user_id)
        view = self._row_to_view(raw_cfg, user_id=user_id)
        endpoint_label = f"{view.host}:{view.port} · {view.model}"

        context_text, target_code = self._assemble_context(
            req=req,
            state_payload=state_payload,
            stock_quote=stock_quote,
            stock_bars=stock_bars or [],
            screener_resp=screener_resp,
        )
        user_prompt = self._build_user_prompt(req, target_code, context_text)

        use_platform_trial = False
        trial_warnings: list[str] = []
        active_chat_endpoint = view.chat_endpoint
        active_host = view.host
        active_port = view.port
        active_model = view.model
        active_key = str(raw_cfg.get("api_key") or "").strip()
        active_label = endpoint_label

        if view.enabled:
            try:
                ep_check = parse_ai_endpoint(
                    view.base_url,
                    allow_private=self.allow_private_endpoints,
                    resolve_dns=not self.allow_private_endpoints,
                )
                active_chat_endpoint = ep_check["chat_endpoint"]
                active_host = ep_check["host"]
                active_port = ep_check["port"]
            except ValueError as exc:
                return AIAskResponse(
                    ok=False,
                    source="llm_error",
                    model_used=view.model,
                    endpoint_label=endpoint_label,
                    latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                    reply=f"**AI 配置受安全策略拦截**：{exc}",
                    drafts=[],
                    warnings=[str(exc)],
                    created_at=now_str(),
                )
        elif self._has_platform_trial():
            allowed, rem = self.db.consume_platform_ai_quota(user_id, settings.public_ai_daily_limit)
            if allowed:
                use_platform_trial = True
                pub_ep = parse_ai_endpoint(settings.public_ai_base_url, allow_private=True)
                active_chat_endpoint = pub_ep["chat_endpoint"]
                active_host = pub_ep["host"]
                active_port = pub_ep["port"]
                active_model = settings.public_ai_model
                active_key = settings.public_ai_api_key
                active_label = f"平台体验 AI · {active_model}"
                trial_warnings.append(
                    f"使用平台体验额度（今日剩余 {rem}/{settings.public_ai_daily_limit} 次）；可在「AI端口」配置个人 Key 不限次使用。"
                )
            else:
                fallback_reply, drafts = self._build_local_fallback(
                    req=req,
                    target_code=target_code,
                    state_payload=state_payload,
                    stock_quote=stock_quote,
                    stock_bars=stock_bars or [],
                    screener_resp=screener_resp,
                )
                return AIAskResponse(
                    ok=True,
                    source="local_rule_fallback",
                    model_used="本地量化指标摘要 (体验额度已用完)",
                    endpoint_label="今日体验额度已用完 · 本地引擎",
                    latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                    reply=fallback_reply,
                    drafts=drafts,
                    warnings=[
                        f"今日平台免费 AI 体验额度（{settings.public_ai_daily_limit} 次/天）已用完，已降级为本地规则摘要；点击「AI端口」填入你自己的 API Key 即可继续大模型对话。"
                    ],
                    created_at=now_str(),
                )
        else:
            fallback_reply, drafts = self._build_local_fallback(
                req=req,
                target_code=target_code,
                state_payload=state_payload,
                stock_quote=stock_quote,
                stock_bars=stock_bars or [],
                screener_resp=screener_resp,
            )
            return AIAskResponse(
                ok=True,
                source="local_rule_fallback",
                model_used="本地量化指标摘要 (未启用外部AI)",
                endpoint_label=f"未启用 · 默认配置 {endpoint_label}",
                latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                reply=fallback_reply,
                drafts=drafts,
                warnings=[
                    "当前未启用个人外部 AI 端口，以上为基于本地行情与指标规则的摘要；点击顶部「AI端口」保存并启用接口即可切换大模型推理。"
                ],
                created_at=now_str(),
            )

        sys_prompt = (view.system_prompt if not use_platform_trial else "") or DEFAULT_AI_SYSTEM_PROMPT
        headers = {"Content-Type": "application/json"}
        if active_key:
            headers["Authorization"] = f"Bearer {active_key}"

        try:
            with self._open_client(active_host, float(view.timeout_sec)) as client:
                resp = client.post(
                    active_chat_endpoint,
                    headers=headers,
                    json={
                        "model": active_model,
                        "messages": [
                            {"role": "system", "content": sys_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        "temperature": view.temperature,
                        "max_tokens": view.max_tokens,
                    },
                )
        except Exception as exc:
            err_msg = (
                f"连接 AI 服务失败 ({type(exc).__name__})"
                if use_platform_trial
                else f"连接 AI 端口 {active_host}:{active_port} 失败 ({type(exc).__name__})"
            )
            return AIAskResponse(
                ok=False,
                source="llm_error",
                model_used=active_model,
                endpoint_label=active_label,
                latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                reply=f"**AI 接口调用失败**：{err_msg}\n\n请检查顶部 **「AI端口」** 接口地址、密钥或网络连接。",
                drafts=[],
                warnings=[err_msg],
                created_at=now_str(),
            )

        if resp.status_code != 200:
            err_detail = self._extract_http_error(resp)
            err_msg = (
                f"平台 AI 服务返回 HTTP {resp.status_code}: {err_detail}"
                if use_platform_trial
                else f"AI 接口 {active_host}:{active_port} 返回 HTTP {resp.status_code}: {err_detail}"
            )
            return AIAskResponse(
                ok=False,
                source="llm_error",
                model_used=active_model,
                endpoint_label=active_label,
                latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                reply=f"**AI 接口返回异常 ({resp.status_code})**：{err_detail}",
                drafts=[],
                warnings=[err_msg],
                created_at=now_str(),
            )

        try:
            body = resp.json()
            choice0 = (body.get("choices") or [{}])[0]
            msg = choice0.get("message") or {}
            raw_content = str(msg.get("content") or "").strip()
            if not raw_content and msg.get("reasoning_content"):
                raw_content = str(msg.get("reasoning_content")).strip()
        except Exception as exc:
            err_msg = f"解析 AI 响应 JSON 失败 ({type(exc).__name__})"
            return AIAskResponse(
                ok=False,
                source="llm_error",
                model_used=active_model,
                endpoint_label=active_label,
                latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                reply=f"**AI 响应格式异常**：{err_msg}",
                drafts=[],
                warnings=[err_msg],
                created_at=now_str(),
            )

        clean_reply, drafts = _extract_drafts_from_reply(raw_content)
        return AIAskResponse(
            ok=True,
            source="platform_trial_llm" if use_platform_trial else "llm",
            model_used=str(body.get("model") or active_model),
            endpoint_label=active_label,
            latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
            reply=clean_reply or raw_content or "模型未返回有效文本内容。",
            drafts=drafts,
            warnings=trial_warnings,
            created_at=now_str(),
        )

    def _assemble_context(
        self,
        req: AIAskRequest,
        state_payload: dict[str, Any],
        stock_quote: Optional[QuoteSnapshot],
        stock_bars: list[dict],
        screener_resp: Optional[LimitUpScreenResponse],
    ) -> tuple[str, str]:
        sections: list[str] = []
        watchlist = state_payload.get("watchlist") or []
        target_code = (req.stock_code or "").strip()
        if not target_code and watchlist:
            target_code = str(watchlist[0].get("stock_code") or "")

        if target_code and (stock_quote is not None or stock_bars):
            watch_item = next((w for w in watchlist if w.get("stock_code") == target_code), None)
            pos_raw = next(
                (p for p in (state_payload.get("positions") or []) if p.get("stock_code") == target_code),
                None,
            )
            pos_obj = PositionRecord(**pos_raw) if pos_raw else None
            sections.append(self._format_stock_context(target_code, watch_item, stock_quote, stock_bars, pos_obj))

        if screener_resp is not None:
            watch_codes = {str(w.get("stock_code")) for w in watchlist}
            sections.append(self._format_screener_context(screener_resp, watch_codes))

        if req.include_account:
            sections.append(self._format_account_context(state_payload))

        return "\n\n".join(s for s in sections if s), target_code

    @staticmethod
    def _format_stock_context(
        code: str,
        watch_item: Optional[dict],
        q: Optional[QuoteSnapshot],
        bars: list[dict],
        pos: Optional[PositionRecord],
    ) -> str:
        lines = [f"### 当前标的上下文 ({code})"]
        if q:
            bid_vol = sum(b.volume for b in q.bids)
            ask_vol = sum(a.volume for a in q.asks)
            lines.append(
                f"- 名称/板块: {q.short_name} ({classify_board(code)}) | 现价: {q.price:.2f} ({q.change_pct:+.2f}%) | "
                f"今开: {q.open:.2f} 高: {q.high:.2f} 低: {q.low:.2f} 昨收: {q.pre_close:.2f}"
            )
            lines.append(
                f"- 量能/估值: 成交量 {q.volume / 100:.0f}手 | 成交额 {q.amount / 1e8:.2f}亿 | 换手率 {q.turnover_ratio:.2f}% | "
                f"振幅 {q.amplitude:.2f}% | PE {q.pe_ratio} | 涨停 {q.limit_up:.2f} / 跌停 {q.limit_down:.2f}"
            )
            lines.append(
                f"- 五档委托对比: 买五档总量 {bid_vol // 100}手 vs 卖五档总量 {ask_vol // 100}手"
            )
        if watch_item:
            lines.append(
                f"- 自选属性: 分组=[{watch_item.get('group_name', '默认')}] | 置顶={'是' if watch_item.get('is_pinned') else '否'} | "
                f"用户备注=\"{watch_item.get('note') or '无'}\""
            )
        if pos and pos.total_qty > 0:
            lines.append(
                f"- 当前持仓: 总持仓 {pos.total_qty}股 | T+1可用 {pos.available_qty}股 | 今买冻结 {pos.today_bought_qty}股 | "
                f"成本 {pos.avg_cost:.3f} | 浮盈 {pos.unrealized_pnl:+.2f} ({pos.unrealized_pnl_pct:+.2f}%)"
            )
        if bars:
            ordered = sorted(bars, key=lambda b: str(b["trade_date"]))
            closes = [float(b["close"]) for b in ordered]
            ma5 = compute_sma_series(closes, 5)
            ma10 = compute_sma_series(closes, 10)
            ma20 = compute_sma_series(closes, 20)
            macd = compute_macd_series(closes, 12, 26, 9)
            dif, dea, hist = macd[-1]
            lu_recent = []
            for idx in range(max(1, len(ordered) - 10), len(ordered)):
                if is_limit_up_bar(code, float(ordered[idx]["close"]), float(ordered[idx - 1]["close"]), q.short_name if q else ""):
                    lu_recent.append(str(ordered[idx]["trade_date"]))
            hi20 = max(float(b["high"]) for b in ordered[-20:])
            lo20 = min(float(b["low"]) for b in ordered[-20:])
            lines.append(
                f"- 已计算技术指标 ({ordered[-1]['trade_date']}): "
                f"MA5={ma5[-1] or '--'} | MA10={ma10[-1] or '--'} | MA20={ma20[-1] or '--'} | "
                f"MACD(DIF={dif:.3f}, DEA={dea:.3f}, HIST={hist:.3f}) | 近20日高/低: {hi20:.2f}/{lo20:.2f} | "
                f"近10日涨停日期: {', '.join(lu_recent) if lu_recent else '无'}"
            )
            tail_bars = ", ".join(
                f"{b['trade_date']}(收{float(b['close']):.2f}/{float(b.get('change_pct', 0.0)):+.2f}%)"
                for b in ordered[-5:]
            )
            lines.append(f"- 近5日日线摘要: {tail_bars}")
        return "\n".join(lines)

    @staticmethod
    def _format_screener_context(resp: LimitUpScreenResponse, watch_codes: set[str]) -> str:
        board_cnt = Counter(it.board for it in resp.items)
        ind_cnt = Counter(it.industry or "其他" for it in resp.items)
        top_industries = ", ".join(f"{k}({v}只)" for k, v in ind_cnt.most_common(8))
        leaders = [
            f"{it.short_name}({it.stock_code}, {it.board}, {it.industry or '--'}, {it.limit_up_count}板/{it.max_consecutive_boards}连板, 换手{it.turnover_ratio:.1f}%, 额{it.amount / 1e8:.2f}亿, 今日{it.latest_change_pct:+.2f}%)"
            for it in resp.items[:15]
        ]
        in_watch = [f"{it.short_name}({it.stock_code}, {it.limit_up_count}板)" for it in resp.items if it.stock_code in watch_codes]
        return "\n".join(
            [
                f"### 近 {resp.days} 日全市场涨停池 ({resp.trade_dates[0] if resp.trade_dates else '--'} ~ {resp.trade_dates[-1] if resp.trade_dates else '--'})",
                f"- 数据源: {resp.source} | 总命中: {resp.total_hits} 只 | 板块分布: {dict(board_cnt)}",
                f"- 高频行业题材 Top8: {top_industries or '暂无分类'}",
                f"- 与当前自选股重叠: {', '.join(in_watch) if in_watch else '无'}",
                "- 连板/活跃前 15 标的: " + ("；".join(leaders) if leaders else "无"),
            ]
        )

    @staticmethod
    def _calc_pos_ratio_pct(acc: dict[str, Any]) -> float:
        total_assets = float(acc.get("total_assets") or 0.0)
        market_value = float(acc.get("market_value") or 0.0)
        if total_assets <= 0:
            return 0.0
        return (market_value / total_assets) * 100.0

    @classmethod
    def _format_account_context(cls, state_payload: dict[str, Any]) -> str:
        acc = state_payload.get("account") or {}
        risk = state_payload.get("risk_config") or {}
        positions = state_payload.get("positions") or []
        orders = (state_payload.get("orders") or [])[:8]
        trades = (state_payload.get("trades") or [])[:8]
        strats = state_payload.get("strategies") or []
        rejected = [o for o in orders if o.get("status") == "REJECTED"]
        pos_ratio = cls._calc_pos_ratio_pct(acc)
        pos_details = "；".join(
            f"{p.get('short_name')}({p.get('stock_code')}, {p.get('total_qty')}股/可用{p.get('available_qty')}, 浮盈{float(p.get('unrealized_pnl_pct', 0)):+.2f}%)"
            for p in positions[:6]
        )
        return "\n".join(
            [
                "### 模拟账户与风控上下文",
                f"- 交易日: {acc.get('current_trade_date')} | 行情模式: {acc.get('market_mode')} | "
                f"总资产: {float(acc.get('total_assets', 0)):,.2f} | 可用资金: {float(acc.get('available_cash', 0)):,.2f} | "
                f"持仓市值: {float(acc.get('market_value', 0)):,.2f} ({pos_ratio:.1f}%) | 累计盈亏: {float(acc.get('total_pnl', 0)):+,.2f} ({float(acc.get('total_pnl_pct', 0)):+.2f}%)",
                f"- 风控规则: 单票上限 {risk.get('max_single_position_pct', 100)}% | 单笔限额 {risk.get('max_order_amount', 0)} | "
                f"总亏损熔断 {risk.get('max_total_loss_pct', 0)}% | 启用自动策略数: {sum(1 for s in strats if s.get('enabled'))}",
                f"- 持仓数: {len(positions)} 只（{pos_details or '空仓'}） | 近期成交: {len(trades)} 笔 | 近期风控拒单: {len(rejected)} 笔",
            ]
        )

    @staticmethod
    def _build_user_prompt(req: AIAskRequest, target_code: str, context_text: str) -> str:
        task_goals = {
            "DIAGNOSE_STOCK": f"请对标的 {target_code} 进行盘面、量价、均线/MACD 指标与操作风险诊断，给出关键支撑/压力位及建议参数草稿（如回测或预警思路）。",
            "SCREENER_SUMMARY": f"请梳理近 {req.screener_days} 日涨停股池的主线题材方向、连板梯队高度、大资金容量标的特征，并指出需规避的高位风险。",
            "DAILY_REVIEW": "请结合当前账户仓位、持仓盈亏、成交记录与风控配置，输出今日账户诊断与下一步仓位/风控改进建议。",
            "PARSE_BACKTEST_OR_RULE": "请将用户需求准确转译为客观分析与末尾 JSON drafts（BACKTEST / STRATEGY / ALERT / ORDER_PREFILL）。",
            "CHAT": "请基于以上实时行情与模拟账户上下文回答用户问题。",
        }
        goal = task_goals.get(req.task_type, task_goals["CHAT"])
        extra = f"\n\n用户补充指令：{req.prompt.strip()}" if req.prompt.strip() else ""
        return f"{context_text}\n\n### 任务目标\n{goal}{extra}"

    @classmethod
    def _build_local_fallback(
        cls,
        req: AIAskRequest,
        target_code: str,
        state_payload: dict[str, Any],
        stock_quote: Optional[QuoteSnapshot],
        stock_bars: list[dict],
        screener_resp: Optional[LimitUpScreenResponse],
    ) -> tuple[str, list[AIActionDraft]]:
        drafts: list[AIActionDraft] = []
        if req.task_type == "SCREENER_SUMMARY" and screener_resp is not None:
            ind_cnt = Counter(it.industry or "未分类" for it in screener_resp.items)
            board_cnt = Counter(it.board for it in screener_resp.items)
            top_ind = "、".join(f"**{k}**({v}只)" for k, v in ind_cnt.most_common(5))
            top3 = "；".join(
                f"{it.short_name}({it.stock_code}, {it.limit_up_count}板/{it.max_consecutive_boards}连板)"
                for it in screener_resp.items[:5]
            )
            reply = (
                f"### 近 {screener_resp.days} 日涨停池本地结构化归纳\n"
                f"- **总体规模**：共命中 **{screener_resp.total_hits}** 只涨停股（主板 {board_cnt.get('主板', 0)} / 创业板 {board_cnt.get('创业板', 0)} / 科创板 {board_cnt.get('科创板', 0)}）。\n"
                f"- **集中题材行业**：{top_ind or '暂无'}。\n"
                f"- **高标活跃前排**：{top3 or '暂无'}。\n"
                f"- **风险提示**：关注高换手多连板次日分歧承接情况，优先结合分时量价与止损风控。"
            )
            return reply, drafts

        if req.task_type == "DAILY_REVIEW":
            acc = state_payload.get("account") or {}
            pos = state_payload.get("positions") or []
            pos_ratio = cls._calc_pos_ratio_pct(acc)
            pos_desc = "、".join(f"{p.get('short_name')}({p.get('total_qty')}股)" for p in pos[:4])
            reply = (
                f"### 账户本地体检摘要 ({acc.get('current_trade_date', '--')})\n"
                f"- **总资产 / 收益率**：¥{float(acc.get('total_assets', 0)):,.2f}（{float(acc.get('total_pnl_pct', 0)):+.2f}%），总仓位 **{pos_ratio:.1f}%**。\n"
                f"- **当前持仓**：{len(pos)} 只标的中{f'（{pos_desc}）' if pos_desc else ''}，需注意当日新买入筹码受 T+1 限制次日方可卖出。\n"
                f"- **风控建议**：建议为高频波动标的配置单票仓位上限及自动化止损/预警规则。"
            )
            return reply, drafts

        if target_code and stock_quote:
            lines = [
                f"### {stock_quote.short_name} ({target_code}) 本地量价与指标摘要",
                f"- **最新盘面**：现价 **{stock_quote.price:.2f}** ({stock_quote.change_pct:+.2f}%)，换手率 **{stock_quote.turnover_ratio:.2f}%**，涨停 {stock_quote.limit_up:.2f} / 跌停 {stock_quote.limit_down:.2f}。",
            ]
            if stock_bars and len(stock_bars) >= 20:
                ordered = sorted(stock_bars, key=lambda b: str(b["trade_date"]))
                closes = [float(b["close"]) for b in ordered]
                ma5 = compute_sma_series(closes, 5)[-1]
                ma20 = compute_sma_series(closes, 20)[-1]
                dif, dea, _ = compute_macd_series(closes, 12, 26, 9)[-1]
                trend = "短期均线多头 (MA5 > MA20)" if (ma5 and ma20 and ma5 > ma20) else "短期均线偏弱 (MA5 ≤ MA20)"
                macd_state = "DIF 位于 DEA 上方" if dif >= dea else "DIF 位于 DEA 下方"
                lines.append(f"- **均线与趋势**：MA5={ma5 or '--'}，MA20={ma20 or '--'}，当前 **{trend}**；MACD {macd_state}。")
            drafts.append(
                AIActionDraft(
                    kind="BACKTEST",
                    title=f"为 {stock_quote.short_name} 运行 MA5/MA20 回测",
                    summary="90日日线回测 · 仓位95% · 止损8%",
                    payload=BacktestRequest(
                        stock_code=target_code,
                        strategy_type="MA_CROSS",
                        days=90,
                        fast_period=5,
                        slow_period=20,
                        position_pct=95,
                        stop_loss_pct=8.0,
                    ).model_dump(),
                )
            )
            return "\n".join(lines), drafts

        return "当前本地无足够标的数据可供分析，请先选择自选股或配置外部 AI 端口。", drafts
