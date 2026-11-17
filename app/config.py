from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return default


@dataclass
class Settings:
    db_path: Path = field(
        default_factory=lambda: Path(
            os.environ.get("PAPER_TRADE_DB", str(DATA_DIR / "paper_trade.db"))
        )
    )
    initial_cash: float = 1_000_000.0
    commission_rate: float = 0.00025  # 万 2.5
    min_commission: float = 5.0       # 最低 5 元
    transfer_fee_rate: float = 0.00001  # 过户费 万 0.1
    stamp_tax_rate: float = 0.0005    # 印花税 万 5 (仅卖出)
    poll_interval_sec: float = 3.0    # 盯盘拉取间隔

    # Auth & public deployment controls
    auth_required: bool = field(default_factory=lambda: _env_bool("AUTH_REQUIRED", True))
    invite_required: bool = field(default_factory=lambda: _env_bool("INVITE_REQUIRED", False))
    open_register: bool = field(default_factory=lambda: _env_bool("ALLOW_OPEN_REGISTER", True))
    email_verification_required: bool = field(
        default_factory=lambda: _env_bool(
            "EMAIL_VERIFICATION_REQUIRED",
            bool(os.environ.get("RESEND_API_KEY", "").strip()),
        )
    )
    allow_private_ai_endpoints: bool = field(
        default_factory=lambda: _env_bool("ALLOW_PRIVATE_AI_ENDPOINTS", False)
    )
    allow_private_webhooks: bool = field(
        default_factory=lambda: _env_bool("ALLOW_PRIVATE_WEBHOOKS", False)
    )
    session_secret: str = field(default_factory=lambda: os.environ.get("APP_SECRET_KEY", ""))
    admin_username: str = field(default_factory=lambda: os.environ.get("ADMIN_USERNAME", "").strip())
    admin_password: str = field(default_factory=lambda: os.environ.get("ADMIN_PASSWORD", ""))

    # Email sending channels (Brevo, Resend, or standard SMTP)
    brevo_api_key: str = field(default_factory=lambda: os.environ.get("BREVO_API_KEY", "").strip())
    resend_api_key: str = field(default_factory=lambda: os.environ.get("RESEND_API_KEY", "").strip())
    smtp_host: str = field(default_factory=lambda: os.environ.get("SMTP_HOST", "").strip())
    smtp_port: int = field(default_factory=lambda: _env_int("SMTP_PORT", 587))
    smtp_user: str = field(default_factory=lambda: os.environ.get("SMTP_USER", "").strip())
    smtp_pass: str = field(default_factory=lambda: os.environ.get("SMTP_PASS", "").strip())
    email_from: str = field(
        default_factory=lambda: (
            os.environ.get("EMAIL_FROM")
            or os.environ.get("RESEND_FROM")
            or "智瞳量化 <onboarding@resend.dev>"
        ).strip()
    )
    resend_from: str = field(
        default_factory=lambda: os.environ.get("RESEND_FROM", "智瞳量化 <onboarding@resend.dev>").strip()
    )
    site_url: str = field(
        default_factory=lambda: os.environ.get("SITE_URL", "https://zhitong888.ccwu.cc").strip()
    )

    # Optional server-side platform AI trial key (never exposed to clients)
    public_ai_provider: str = field(
        default_factory=lambda: os.environ.get("PUBLIC_AI_PROVIDER", "智瞳公共体验 AI").strip()
    )
    public_ai_base_url: str = field(
        default_factory=lambda: os.environ.get("PUBLIC_AI_BASE_URL", "").strip()
    )
    public_ai_model: str = field(
        default_factory=lambda: os.environ.get("PUBLIC_AI_MODEL", "").strip()
    )
    public_ai_api_key: str = field(
        default_factory=lambda: os.environ.get("PUBLIC_AI_API_KEY", "").strip()
    )
    public_ai_daily_limit: int = field(
        default_factory=lambda: max(0, _env_int("PUBLIC_AI_DAILY_LIMIT", 10))
    )

    default_watchlist: list[dict[str, str]] = field(
        default_factory=lambda: [
            {"code": "600519", "name": "贵州茅台"},
            {"code": "300750", "name": "宁德时代"},
            {"code": "000001", "name": "平安银行"},
            {"code": "002594", "name": "比亚迪"},
            {"code": "600036", "name": "招商银行"},
            {"code": "000858", "name": "五粮液"},
        ]
    )


settings = Settings()
