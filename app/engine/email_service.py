from __future__ import annotations

import email.utils
import html
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Callable, Optional

import httpx

from app.config import settings

EmailHttpClientFactory = Callable[[float], httpx.Client]


def _default_email_client(timeout_sec: float) -> httpx.Client:
    return httpx.Client(timeout=timeout_sec)


def parse_sender_info(raw_sender: str, default_name: str = "智瞳量化") -> tuple[str, str]:
    cleaned = (raw_sender or "").strip()
    if not cleaned:
        return default_name, "noreply@example.com"
    name, addr = email.utils.parseaddr(cleaned)
    if not addr:
        addr = cleaned.strip("<> ")
    return name or default_name, addr


class UniversalEmailService:
    """Universal transactional email service supporting Brevo API, Resend API, and SMTP relay."""

    BREVO_API_URL = "https://api.brevo.com/v3/smtp/email"
    RESEND_API_URL = "https://api.resend.com/emails"

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_from: Optional[str] = None,
        http_client_factory: EmailHttpClientFactory = _default_email_client,
    ) -> None:
        self._explicit_api_key = api_key
        self._explicit_from = default_from
        self.http_client_factory = http_client_factory

    @property
    def api_key(self) -> str:
        if self._explicit_api_key is not None:
            return self._explicit_api_key.strip()
        return (
            settings.brevo_api_key.strip()
            or settings.resend_api_key.strip()
            or settings.smtp_pass.strip()
        )

    @property
    def default_from(self) -> str:
        if self._explicit_from is not None:
            return self._explicit_from.strip()
        return (
            settings.email_from.strip()
            or settings.resend_from.strip()
            or "智瞳量化 <onboarding@resend.dev>"
        )

    def is_configured(self) -> bool:
        return bool(
            self.api_key
            or (settings.smtp_host and settings.smtp_user and settings.smtp_pass)
        )

    def get_provider_name(self) -> str:
        k = self.api_key
        if settings.brevo_api_key or k.startswith("xkeysib-"):
            return "Brevo"
        if settings.resend_api_key or k.startswith("re_"):
            return "Resend"
        if settings.smtp_host:
            return f"SMTP ({settings.smtp_host})"
        return "Unconfigured"

    def _send_via_brevo(
        self, key: str, sender: str, recipient: str, subject: str, html_body: str, text_body: str
    ) -> tuple[bool, str]:
        sender_name, sender_email = parse_sender_info(sender)
        payload = {
            "sender": {"name": sender_name, "email": sender_email},
            "to": [{"email": recipient}],
            "subject": subject,
            "htmlContent": html_body,
        }
        if text_body:
            payload["textContent"] = text_body

        try:
            with self.http_client_factory(10.0) as client:
                resp = client.post(
                    self.BREVO_API_URL,
                    headers={
                        "api-key": key,
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                        "User-Agent": "zhitong-quant/2.0",
                    },
                    json=payload,
                )
        except Exception as exc:
            return False, f"请求 Brevo 网络异常 ({type(exc).__name__})"

        if 200 <= resp.status_code < 300:
            try:
                msg_id = str((resp.json() or {}).get("messageId") or "ok")
            except Exception:
                msg_id = "ok"
            return True, msg_id

        detail = ""
        try:
            body = resp.json()
            if isinstance(body, dict):
                detail = str(body.get("message") or body.get("code") or "")
        except Exception:
            detail = (resp.text or "")[:180]
        return False, f"Brevo 返回 HTTP {resp.status_code}: {detail or '发送失败'}"

    def _send_via_resend(
        self, key: str, sender: str, recipient: str, subject: str, html_body: str, text_body: str
    ) -> tuple[bool, str]:
        payload: dict[str, object] = {
            "from": sender,
            "to": [recipient],
            "subject": subject,
            "html": html_body,
        }
        if text_body:
            payload["text"] = text_body

        try:
            with self.http_client_factory(10.0) as client:
                resp = client.post(
                    self.RESEND_API_URL,
                    headers={
                        "Authorization": f"Bearer {key}",
                        "Content-Type": "application/json",
                        "User-Agent": "zhitong-quant/2.0",
                    },
                    json=payload,
                )
        except Exception as exc:
            return False, f"请求 Resend 网络异常 ({type(exc).__name__})"

        if 200 <= resp.status_code < 300:
            try:
                msg_id = str((resp.json() or {}).get("id") or "ok")
            except Exception:
                msg_id = "ok"
            return True, msg_id

        detail = ""
        try:
            err_json = resp.json()
            if isinstance(err_json, dict):
                detail = str(err_json.get("message") or err_json.get("name") or "")
        except Exception:
            detail = (resp.text or "")[:180]

        lower = detail.lower()
        if "only send testing emails to your own email" in lower:
            return (
                False,
                "Resend 当前为测试发信模式，仅限发往注册账号邮箱；若需发往任意邮箱，建议换用 Brevo 或完成域名验证",
            )
        return False, f"Resend 返回 HTTP {resp.status_code}: {detail or '发送失败'}"

    def _send_via_smtp(
        self, sender: str, recipient: str, subject: str, html_body: str, text_body: str
    ) -> tuple[bool, str]:
        host = settings.smtp_host
        port = settings.smtp_port
        user = settings.smtp_user
        passwd = settings.smtp_pass
        if not host:
            return False, "未配置 SMTP_HOST"

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = sender
        msg["To"] = recipient

        if text_body:
            msg.attach(MIMEText(text_body, "plain", "utf-8"))
        if html_body:
            msg.attach(MIMEText(html_body, "html", "utf-8"))

        try:
            if port == 465:
                server = smtplib.SMTP_SSL(host, port, timeout=10.0)
            else:
                server = smtplib.SMTP(host, port, timeout=10.0)
                server.starttls()
            if user and passwd:
                server.login(user, passwd)
            server.sendmail(sender, [recipient], msg.as_string())
            server.quit()
            return True, "smtp_ok"
        except Exception as exc:
            return False, f"SMTP 发信失败: {type(exc).__name__} {exc}"

    def send_email(
        self,
        *,
        to_email: str,
        subject: str,
        html_body: str,
        text_body: str = "",
        from_email: str = "",
    ) -> tuple[bool, str]:
        sender = (from_email or "").strip() or self.default_from
        recipient = (to_email or "").strip()
        if not recipient or "@" not in recipient:
            return False, "收件人邮箱格式无效"

        key = self.api_key
        # Check Brevo
        if settings.brevo_api_key or key.startswith("xkeysib-"):
            return self._send_via_brevo(key or settings.brevo_api_key, sender, recipient, subject, html_body, text_body)

        # Check Resend
        if settings.resend_api_key or key.startswith("re_"):
            return self._send_via_resend(key or settings.resend_api_key, sender, recipient, subject, html_body, text_body)

        # Check SMTP
        if settings.smtp_host:
            return self._send_via_smtp(sender, recipient, subject, html_body, text_body)

        if not key:
            return False, "未配置邮件服务密钥（支持 BREVO_API_KEY、RESEND_API_KEY 或 SMTP）"
        # Fallback to Resend
        return self._send_via_resend(key, sender, recipient, subject, html_body, text_body)

    def send_verification_code(
        self,
        *,
        to_email: str,
        code: str,
        purpose_label: str,
        from_email: str = "",
    ) -> tuple[bool, str]:
        safe_purpose = html.escape(purpose_label)
        safe_code = html.escape(code)
        site = html.escape(settings.site_url)
        subject = f"【智瞳量化】{purpose_label}验证码：{code}"
        html_body = f"""
        <div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','PingFang SC',sans-serif;max-width:520px;margin:0 auto;padding:24px;background:#0f172a;color:#e2e8f0;border-radius:12px;border:1px solid #1e293b;">
          <h2 style="margin:0 0 12px;font-size:18px;color:#f8fafc;">智瞳量化 · A股 AI 盯盘终端</h2>
          <p style="margin:0 0 14px;font-size:14px;color:#cbd5e1;">你正在进行 <strong>{safe_purpose}</strong> 操作，请在 <strong>10 分钟</strong>内输入以下验证码：</p>
          <div style="margin:18px 0;padding:14px 18px;background:#1e293b;border-radius:8px;text-align:center;font-size:28px;font-weight:700;letter-spacing:6px;color:#fbbf24;font-family:monospace;">{safe_code}</div>
          <p style="margin:0;font-size:12px;color:#94a3b8;">访问地址：<a href="{site}" style="color:#60a5fa;">{site}</a><br/>如非本人操作请忽略此邮件。</p>
        </div>
        """
        text_body = f"【智瞳量化】你的{purpose_label}验证码为：{code}（10分钟内有效）。地址：{settings.site_url}"
        return self.send_email(
            to_email=to_email,
            subject=subject,
            html_body=html_body,
            text_body=text_body,
            from_email=from_email,
        )

    def send_invite_email(
        self,
        *,
        to_email: str,
        invite_code: str,
        note: str = "",
        from_email: str = "",
    ) -> tuple[bool, str]:
        safe_code = html.escape(invite_code)
        safe_site = html.escape(settings.site_url)
        safe_note = html.escape(note or "智瞳量化 A 股模拟交易与 AI 盯盘系统内测邀请")
        subject = f"【智瞳量化】邀请你体验 A 股 AI 盯盘与模拟交易终端"
        html_body = f"""
        <div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','PingFang SC',sans-serif;max-width:520px;margin:0 auto;padding:24px;background:#0f172a;color:#e2e8f0;border-radius:12px;border:1px solid #1e293b;">
          <h2 style="margin:0 0 10px;font-size:18px;color:#f8fafc;">智瞳量化 · 邀请开通体验</h2>
          <p style="margin:0 0 14px;font-size:14px;color:#cbd5e1;">{safe_note}</p>
          <div style="margin:16px 0;padding:12px 16px;background:#1e293b;border-radius:8px;">
            <div style="font-size:12px;color:#94a3b8;margin-bottom:4px;">你的专属邀请码</div>
            <div style="font-size:22px;font-weight:700;letter-spacing:2px;color:#38bdf8;font-family:monospace;">{safe_code}</div>
          </div>
          <p style="margin:14px 0 0;font-size:13px;color:#cbd5e1;">点击访问并注册：<br/><a href="{safe_site}" style="color:#60a5fa;word-break:break-all;">{safe_site}</a></p>
        </div>
        """
        text_body = f"智瞳量化内测邀请：{safe_note}\n邀请码：{invite_code}\n注册地址：{settings.site_url}"
        return self.send_email(
            to_email=to_email,
            subject=subject,
            html_body=html_body,
            text_body=text_body,
            from_email=from_email,
        )

    def send_alert_email(
        self,
        *,
        to_email: str,
        stock_name: str,
        stock_code: str,
        alert_message: str,
        from_email: str = "",
    ) -> tuple[bool, str]:
        safe_title = html.escape(f"{stock_name} ({stock_code})")
        safe_msg = html.escape(alert_message)
        safe_site = html.escape(settings.site_url)
        subject = f"【盯盘预警】{stock_name} ({stock_code})"
        html_body = f"""
        <div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','PingFang SC',sans-serif;max-width:520px;margin:0 auto;padding:24px;background:#0f172a;color:#e2e8f0;border-radius:12px;border:1px solid #1e293b;">
          <h2 style="margin:0 0 10px;font-size:17px;color:#fbbf24;">盯盘价格预警触发 · {safe_title}</h2>
          <p style="margin:0 0 14px;font-size:14px;color:#f8fafc;padding:12px;background:#1e293b;border-radius:8px;border-left:4px solid #ef4444;">{safe_msg}</p>
          <p style="margin:0;font-size:12px;color:#94a3b8;">前往终端查看详情：<a href="{safe_site}" style="color:#60a5fa;">{safe_site}</a></p>
        </div>
        """
        return self.send_email(
            to_email=to_email,
            subject=subject,
            html_body=html_body,
            text_body=f"盯盘预警触发：{alert_message} ({settings.site_url})",
            from_email=from_email,
        )


# Backward-compatible alias
ResendEmailService = UniversalEmailService
