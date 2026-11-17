from __future__ import annotations

import uuid
from datetime import datetime
from typing import Callable, Optional

import httpx

from app.config import settings
from app.models import (
    AlertCondition,
    AlertEventRecord,
    AlertRuleRecord,
    AlertRuleRequest,
    QuoteSnapshot,
)
from app.security import beijing_now, validate_outbound_http_url
from app.storage import Database, now_str

EmailNotifyFn = Callable[[str, str, str, str], str]


class AlertEngine:
    """Rule-based market-watching alert engine with multi-user isolation, SSRF protection, and email alerts."""

    def __init__(
        self,
        db: Database,
        allow_private_webhooks: Optional[bool] = None,
        email_notifier: Optional[EmailNotifyFn] = None,
    ) -> None:
        self.db = db
        self.allow_private_webhooks = (
            settings.allow_private_webhooks
            if allow_private_webhooks is None
            else allow_private_webhooks
        )
        self.email_notifier = email_notifier

    def _validate_webhook(self, raw_url: str, resolve_dns: bool = False) -> str:
        cleaned = (raw_url or "").strip()
        if not cleaned:
            return ""
        return validate_outbound_http_url(
            cleaned,
            allow_private=self.allow_private_webhooks,
            resolve_dns=resolve_dns,
            label="Webhook 地址",
        )

    def add_rule(
        self, req: AlertRuleRequest, short_name: str = "", user_id: str = "default"
    ) -> AlertRuleRecord:
        webhook_url = self._validate_webhook(req.webhook_url, resolve_dns=True)
        rule_id = uuid.uuid4().hex[:10].upper()
        now = now_str()
        with self.db.connect() as conn:
            name = short_name
            if not name:
                w = conn.execute(
                    "SELECT short_name FROM watchlist WHERE user_id = ? AND stock_code = ?",
                    (user_id, req.stock_code),
                ).fetchone()
                name = w["short_name"] if w else req.stock_code

            conn.execute(
                """
                INSERT INTO alert_rules (
                    rule_id, user_id, stock_code, short_name, condition, threshold,
                    cooldown_sec, enabled, note, webhook_url, notify_email, last_triggered_at, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, NULL, ?)
                """,
                (
                    rule_id, user_id, req.stock_code, name, req.condition.value,
                    req.threshold, req.cooldown_sec, req.note, webhook_url,
                    1 if req.notify_email else 0, now,
                ),
            )
            row = conn.execute(
                "SELECT * FROM alert_rules WHERE rule_id = ? AND user_id = ?", (rule_id, user_id)
            ).fetchone()
        return self._to_rule(row)

    def list_rules(self, user_id: str = "default") -> list[AlertRuleRecord]:
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM alert_rules WHERE user_id = ? ORDER BY created_at DESC", (user_id,)
            ).fetchall()
        return [self._to_rule(r) for r in rows]

    def toggle_rule(
        self, rule_id: str, enabled: bool, user_id: str = "default"
    ) -> AlertRuleRecord:
        with self.db.connect() as conn:
            conn.execute(
                "UPDATE alert_rules SET enabled = ? WHERE rule_id = ? AND user_id = ?",
                (1 if enabled else 0, rule_id, user_id),
            )
            row = conn.execute(
                "SELECT * FROM alert_rules WHERE rule_id = ? AND user_id = ?", (rule_id, user_id)
            ).fetchone()
            if not row:
                raise ValueError(f"预警规则 {rule_id} 不存在")
        return self._to_rule(row)

    def delete_rule(self, rule_id: str, user_id: str = "default") -> None:
        with self.db.connect() as conn:
            conn.execute(
                "DELETE FROM alert_rules WHERE rule_id = ? AND user_id = ?", (rule_id, user_id)
            )

    def list_events(self, limit: int = 80, user_id: str = "default") -> list[AlertEventRecord]:
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM alert_events WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        return [AlertEventRecord(**dict(r)) for r in rows]

    def evaluate(
        self, quotes: dict[str, QuoteSnapshot], user_id: Optional[str] = None
    ) -> list[AlertEventRecord]:
        if not quotes:
            return []
        triggered: list[AlertEventRecord] = []
        now_dt = beijing_now()
        now = now_dt.strftime("%Y-%m-%d %H:%M:%S")

        with self.db.connect() as conn:
            if user_id is not None:
                rules = conn.execute(
                    "SELECT * FROM alert_rules WHERE enabled = 1 AND user_id = ?", (user_id,)
                ).fetchall()
            else:
                rules = conn.execute("SELECT * FROM alert_rules WHERE enabled = 1").fetchall()

            for r in rules:
                q = quotes.get(r["stock_code"])
                if not q or q.price <= 0:
                    continue

                if r["last_triggered_at"]:
                    try:
                        last_dt = datetime.strptime(r["last_triggered_at"], "%Y-%m-%d %H:%M:%S")
                        if (now_dt - last_dt).total_seconds() < r["cooldown_sec"]:
                            continue
                    except ValueError:
                        pass

                cond = AlertCondition(r["condition"])
                th = float(r["threshold"])
                hit = False
                desc = ""

                if cond == AlertCondition.PRICE_ABOVE and q.price >= th:
                    hit = True
                    desc = f"现价 {q.price:.2f} ≥ 目标价 {th:.2f}"
                elif cond == AlertCondition.PRICE_BELOW and q.price <= th:
                    hit = True
                    desc = f"现价 {q.price:.2f} ≤ 预警价 {th:.2f}"
                elif cond == AlertCondition.PCT_ABOVE and q.change_pct >= th:
                    hit = True
                    desc = f"涨幅 {q.change_pct:+.2f}% ≥ {th:+.2f}%"
                elif cond == AlertCondition.PCT_BELOW and q.change_pct <= th:
                    hit = True
                    desc = f"跌幅 {q.change_pct:+.2f}% ≤ {th:+.2f}%"

                if not hit:
                    continue

                note_suffix = f" ({r['note']})" if r["note"] else ""
                msg = f"[{q.short_name} {q.stock_code}] {desc}{note_suffix}"
                webhook_status = "NONE"
                if r["webhook_url"]:
                    try:
                        safe_url = self._validate_webhook(r["webhook_url"], resolve_dns=True)
                        webhook_status = self._send_webhook(safe_url, msg)
                    except ValueError:
                        webhook_status = "ERR:BLOCKED_SSRF"

                event_id = uuid.uuid4().hex[:12].upper()
                rule_uid = str(r["user_id"] or "default")
                email_status = "NONE"
                want_email = bool(r["notify_email"]) if "notify_email" in r.keys() else False
                if want_email and self.email_notifier is not None:
                    try:
                        email_status = self.email_notifier(rule_uid, q.short_name, q.stock_code, msg)
                    except Exception as exc:
                        email_status = f"ERR:{type(exc).__name__}"

                conn.execute(
                    """
                    INSERT INTO alert_events (
                        event_id, user_id, rule_id, stock_code, short_name, message,
                        trigger_price, trigger_pct, webhook_status, email_status, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event_id, rule_uid, r["rule_id"], q.stock_code, q.short_name, msg,
                        q.price, q.change_pct, webhook_status, email_status, now,
                    ),
                )
                conn.execute(
                    "UPDATE alert_rules SET last_triggered_at = ? WHERE rule_id = ?",
                    (now, r["rule_id"]),
                )
                ev = conn.execute(
                    "SELECT * FROM alert_events WHERE event_id = ?", (event_id,)
                ).fetchone()
                triggered.append(AlertEventRecord(**dict(ev)))

        return triggered

    @staticmethod
    def _send_webhook(url: str, msg: str) -> str:
        payload = {
            "msg_type": "text",
            "msgtype": "text",
            "content": {"text": msg},
            "text": {"content": msg},
        }
        try:
            resp = httpx.post(url, json=payload, timeout=3.0)
            return f"HTTP_{resp.status_code}"
        except Exception as exc:
            return f"ERR:{type(exc).__name__}"

    @staticmethod
    def _to_rule(row) -> AlertRuleRecord:
        d = dict(row)
        d["enabled"] = bool(d["enabled"])
        d["notify_email"] = bool(d.get("notify_email", 0))
        return AlertRuleRecord(**d)
