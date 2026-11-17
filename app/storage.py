from __future__ import annotations

import hashlib
import os
import re
import secrets
import shutil
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterator, Optional

from app.config import settings
from app.models import MarketMode
from app.security import beijing_now, hash_password, now_str, today_str, verify_password

__all__ = ["Database", "beijing_now", "now_str", "today_str", "normalize_email", "is_valid_email"]

_USERNAME_RE = re.compile(r"^[\w.\-\u4e00-\u9fff]{2,32}$")
_EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")


def normalize_email(raw: str) -> str:
    return (raw or "").strip().lower()


def is_valid_email(raw: str) -> bool:
    return bool(_EMAIL_RE.match(normalize_email(raw)))


class Database:
    """Thread-safe multi-tenant SQLite store with WAL mode and Beijing timestamps."""

    def __init__(self, db_path: Path | str | None = None, seed_default_workspace: bool = True) -> None:
        self.db_path = Path(db_path) if db_path else settings.db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_pragmas()
        self._init_schema()
        if seed_default_workspace:
            self.ensure_user_workspace("default")

    def _init_pragmas(self) -> None:
        with self._lock:
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=10.0)
            try:
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("PRAGMA synchronous=NORMAL;")
                conn.execute("PRAGMA busy_timeout=5000;")
                conn.commit()
            finally:
                conn.close()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=10.0)
            conn.row_factory = sqlite3.Row
            try:
                conn.execute("PRAGMA busy_timeout=5000;")
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    @staticmethod
    def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        return {str(r["name"]) for r in rows}

    def _migrate_legacy_tables_if_needed(self, conn: sqlite3.Connection) -> None:
        """Upgrade legacy single-tenant tables (without user_id) cleanly to multi-tenant PKs."""
        acc_cols = self._table_columns(conn, "account")
        if acc_cols and "user_id" not in acc_cols:
            conn.executescript(
                """
                CREATE TABLE account_v2 (
                    user_id TEXT PRIMARY KEY,
                    initial_cash REAL NOT NULL,
                    available_cash REAL NOT NULL,
                    frozen_cash REAL NOT NULL,
                    current_trade_date TEXT NOT NULL,
                    market_mode TEXT NOT NULL DEFAULT 'REAL',
                    updated_at TEXT NOT NULL
                );
                INSERT INTO account_v2 (user_id, initial_cash, available_cash, frozen_cash, current_trade_date, market_mode, updated_at)
                SELECT 'default', initial_cash, available_cash, frozen_cash, current_trade_date, market_mode, updated_at FROM account;
                DROP TABLE account;
                ALTER TABLE account_v2 RENAME TO account;
                """
            )

        wl_cols = self._table_columns(conn, "watchlist")
        if wl_cols and "user_id" not in wl_cols:
            has_pin = "is_pinned" in wl_cols
            has_grp = "group_name" in wl_cols
            has_note = "note" in wl_cols
            pin_expr = "COALESCE(is_pinned, 0)" if has_pin else "0"
            grp_expr = "COALESCE(group_name, '默认')" if has_grp else "'默认'"
            note_expr = "COALESCE(note, '')" if has_note else "''"
            conn.executescript(
                f"""
                CREATE TABLE watchlist_v2 (
                    user_id TEXT NOT NULL DEFAULT 'default',
                    stock_code TEXT NOT NULL,
                    short_name TEXT NOT NULL,
                    is_pinned INTEGER NOT NULL DEFAULT 0,
                    group_name TEXT NOT NULL DEFAULT '默认',
                    note TEXT NOT NULL DEFAULT '',
                    added_at TEXT NOT NULL,
                    PRIMARY KEY (user_id, stock_code)
                );
                INSERT INTO watchlist_v2 (user_id, stock_code, short_name, is_pinned, group_name, note, added_at)
                SELECT 'default', stock_code, short_name, {pin_expr}, {grp_expr}, {note_expr}, added_at FROM watchlist;
                DROP TABLE watchlist;
                ALTER TABLE watchlist_v2 RENAME TO watchlist;
                """
            )

        pos_cols = self._table_columns(conn, "positions")
        if pos_cols and "user_id" not in pos_cols:
            conn.executescript(
                """
                CREATE TABLE positions_v2 (
                    user_id TEXT NOT NULL DEFAULT 'default',
                    stock_code TEXT NOT NULL,
                    short_name TEXT NOT NULL,
                    total_qty INTEGER NOT NULL,
                    available_qty INTEGER NOT NULL,
                    today_bought_qty INTEGER NOT NULL DEFAULT 0,
                    frozen_qty INTEGER NOT NULL DEFAULT 0,
                    avg_cost REAL NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (user_id, stock_code)
                );
                INSERT INTO positions_v2 (user_id, stock_code, short_name, total_qty, available_qty, today_bought_qty, frozen_qty, avg_cost, updated_at)
                SELECT 'default', stock_code, short_name, total_qty, available_qty, today_bought_qty, frozen_qty, avg_cost, updated_at FROM positions;
                DROP TABLE positions;
                ALTER TABLE positions_v2 RENAME TO positions;
                """
            )

        risk_cols = self._table_columns(conn, "risk_config")
        if risk_cols and "user_id" not in risk_cols:
            conn.executescript(
                """
                CREATE TABLE risk_config_v2 (
                    user_id TEXT PRIMARY KEY,
                    max_single_position_pct REAL NOT NULL DEFAULT 100.0,
                    max_order_amount REAL NOT NULL DEFAULT 0.0,
                    max_total_loss_pct REAL NOT NULL DEFAULT 0.0,
                    block_buy_on_loss INTEGER NOT NULL DEFAULT 1,
                    updated_at TEXT NOT NULL
                );
                INSERT INTO risk_config_v2 (user_id, max_single_position_pct, max_order_amount, max_total_loss_pct, block_buy_on_loss, updated_at)
                SELECT 'default', max_single_position_pct, max_order_amount, max_total_loss_pct, block_buy_on_loss, updated_at FROM risk_config;
                DROP TABLE risk_config;
                ALTER TABLE risk_config_v2 RENAME TO risk_config;
                """
            )

        ai_cols = self._table_columns(conn, "ai_config")
        if ai_cols and "user_id" not in ai_cols:
            conn.executescript(
                """
                CREATE TABLE ai_config_v2 (
                    user_id TEXT PRIMARY KEY,
                    provider_name TEXT NOT NULL DEFAULT 'Ollama 本地 (默认)',
                    base_url TEXT NOT NULL DEFAULT 'http://127.0.0.1:11434/v1',
                    model TEXT NOT NULL DEFAULT 'qwen2.5:14b',
                    api_key TEXT NOT NULL DEFAULT '',
                    temperature REAL NOT NULL DEFAULT 0.3,
                    max_tokens INTEGER NOT NULL DEFAULT 1500,
                    timeout_sec INTEGER NOT NULL DEFAULT 30,
                    system_prompt TEXT NOT NULL DEFAULT '',
                    enabled INTEGER NOT NULL DEFAULT 0,
                    last_check_status TEXT NOT NULL DEFAULT 'UNCONFIGURED',
                    last_check_message TEXT NOT NULL DEFAULT '',
                    last_checked_at TEXT,
                    updated_at TEXT NOT NULL
                );
                INSERT INTO ai_config_v2 (
                    user_id, provider_name, base_url, model, api_key, temperature, max_tokens,
                    timeout_sec, system_prompt, enabled, last_check_status, last_check_message,
                    last_checked_at, updated_at
                )
                SELECT 'default', provider_name, base_url, model, api_key, temperature, max_tokens,
                       timeout_sec, system_prompt, enabled, last_check_status, last_check_message,
                       last_checked_at, updated_at FROM ai_config;
                DROP TABLE ai_config;
                ALTER TABLE ai_config_v2 RENAME TO ai_config;
                """
            )

        nav_cols = self._table_columns(conn, "nav_history")
        if nav_cols and "user_id" not in nav_cols:
            conn.executescript(
                """
                CREATE TABLE nav_history_v2 (
                    user_id TEXT NOT NULL DEFAULT 'default',
                    trade_date TEXT NOT NULL,
                    total_assets REAL NOT NULL,
                    available_cash REAL NOT NULL,
                    market_value REAL NOT NULL,
                    daily_return_pct REAL NOT NULL,
                    recorded_at TEXT NOT NULL,
                    PRIMARY KEY (user_id, trade_date)
                );
                INSERT INTO nav_history_v2 (user_id, trade_date, total_assets, available_cash, market_value, daily_return_pct, recorded_at)
                SELECT 'default', trade_date, total_assets, available_cash, market_value, daily_return_pct, recorded_at FROM nav_history;
                DROP TABLE nav_history;
                ALTER TABLE nav_history_v2 RENAME TO nav_history;
                """
            )

    def _init_schema(self) -> None:
        with self.connect() as conn:
            self._migrate_legacy_tables_if_needed(conn)
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    display_name TEXT NOT NULL DEFAULT '',
                    email TEXT NOT NULL DEFAULT '',
                    email_verified INTEGER NOT NULL DEFAULT 0,
                    alert_email_enabled INTEGER NOT NULL DEFAULT 1,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'USER',
                    is_active INTEGER NOT NULL DEFAULT 1,
                    invite_code_used TEXT NOT NULL DEFAULT '',
                    ai_calls_today INTEGER NOT NULL DEFAULT 0,
                    ai_calls_date TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    last_login_at TEXT
                );

                CREATE TABLE IF NOT EXISTS email_codes (
                    id TEXT PRIMARY KEY,
                    email TEXT NOT NULL,
                    purpose TEXT NOT NULL,
                    code_hash TEXT NOT NULL,
                    expires_at_ts INTEGER NOT NULL,
                    consumed INTEGER NOT NULL DEFAULT 0,
                    created_at_ts INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS system_settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS invite_codes (
                    code TEXT PRIMARY KEY COLLATE NOCASE,
                    created_by TEXT NOT NULL,
                    max_uses INTEGER NOT NULL DEFAULT 1,
                    used_count INTEGER NOT NULL DEFAULT 0,
                    note TEXT NOT NULL DEFAULT '',
                    enabled INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS account (
                    user_id TEXT PRIMARY KEY,
                    initial_cash REAL NOT NULL,
                    available_cash REAL NOT NULL,
                    frozen_cash REAL NOT NULL,
                    current_trade_date TEXT NOT NULL,
                    market_mode TEXT NOT NULL DEFAULT 'REAL',
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS watchlist (
                    user_id TEXT NOT NULL DEFAULT 'default',
                    stock_code TEXT NOT NULL,
                    short_name TEXT NOT NULL,
                    is_pinned INTEGER NOT NULL DEFAULT 0,
                    group_name TEXT NOT NULL DEFAULT '默认',
                    note TEXT NOT NULL DEFAULT '',
                    added_at TEXT NOT NULL,
                    PRIMARY KEY (user_id, stock_code)
                );

                CREATE TABLE IF NOT EXISTS positions (
                    user_id TEXT NOT NULL DEFAULT 'default',
                    stock_code TEXT NOT NULL,
                    short_name TEXT NOT NULL,
                    total_qty INTEGER NOT NULL,
                    available_qty INTEGER NOT NULL,
                    today_bought_qty INTEGER NOT NULL DEFAULT 0,
                    frozen_qty INTEGER NOT NULL DEFAULT 0,
                    avg_cost REAL NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (user_id, stock_code)
                );

                CREATE TABLE IF NOT EXISTS orders (
                    order_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL DEFAULT 'default',
                    stock_code TEXT NOT NULL,
                    short_name TEXT NOT NULL,
                    side TEXT NOT NULL,
                    price REAL NOT NULL,
                    quantity INTEGER NOT NULL,
                    filled_quantity INTEGER NOT NULL DEFAULT 0,
                    avg_filled_price REAL NOT NULL DEFAULT 0.0,
                    frozen_cash REAL NOT NULL DEFAULT 0.0,
                    status TEXT NOT NULL,
                    reject_reason TEXT NOT NULL DEFAULT '',
                    strategy_name TEXT NOT NULL DEFAULT 'MANUAL',
                    trade_date TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS trades (
                    trade_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL DEFAULT 'default',
                    order_id TEXT NOT NULL,
                    stock_code TEXT NOT NULL,
                    short_name TEXT NOT NULL,
                    side TEXT NOT NULL,
                    price REAL NOT NULL,
                    quantity INTEGER NOT NULL,
                    amount REAL NOT NULL,
                    commission REAL NOT NULL,
                    transfer_fee REAL NOT NULL,
                    stamp_tax REAL NOT NULL,
                    total_fee REAL NOT NULL,
                    trade_date TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS alert_rules (
                    rule_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL DEFAULT 'default',
                    stock_code TEXT NOT NULL,
                    short_name TEXT NOT NULL,
                    condition TEXT NOT NULL,
                    threshold REAL NOT NULL,
                    cooldown_sec INTEGER NOT NULL DEFAULT 300,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    note TEXT NOT NULL DEFAULT '',
                    webhook_url TEXT NOT NULL DEFAULT '',
                    notify_email INTEGER NOT NULL DEFAULT 0,
                    last_triggered_at TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS alert_events (
                    event_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL DEFAULT 'default',
                    rule_id TEXT NOT NULL,
                    stock_code TEXT NOT NULL,
                    short_name TEXT NOT NULL,
                    message TEXT NOT NULL,
                    trigger_price REAL NOT NULL,
                    trigger_pct REAL NOT NULL,
                    webhook_status TEXT NOT NULL DEFAULT 'NONE',
                    email_status TEXT NOT NULL DEFAULT 'NONE',
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS auto_strategies (
                    strategy_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL DEFAULT 'default',
                    stock_code TEXT NOT NULL,
                    strategy_type TEXT NOT NULL,
                    lower_price REAL NOT NULL DEFAULT 0.0,
                    upper_price REAL NOT NULL DEFAULT 0.0,
                    fast_period INTEGER NOT NULL DEFAULT 5,
                    slow_period INTEGER NOT NULL DEFAULT 20,
                    order_qty INTEGER NOT NULL DEFAULT 100,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    last_signal_tag TEXT NOT NULL DEFAULT '',
                    last_action_at TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS risk_config (
                    user_id TEXT PRIMARY KEY,
                    max_single_position_pct REAL NOT NULL DEFAULT 100.0,
                    max_order_amount REAL NOT NULL DEFAULT 0.0,
                    max_total_loss_pct REAL NOT NULL DEFAULT 0.0,
                    block_buy_on_loss INTEGER NOT NULL DEFAULT 1,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS ai_config (
                    user_id TEXT PRIMARY KEY,
                    provider_name TEXT NOT NULL DEFAULT 'Ollama 本地 (默认)',
                    base_url TEXT NOT NULL DEFAULT 'http://127.0.0.1:11434/v1',
                    model TEXT NOT NULL DEFAULT 'qwen2.5:14b',
                    api_key TEXT NOT NULL DEFAULT '',
                    temperature REAL NOT NULL DEFAULT 0.3,
                    max_tokens INTEGER NOT NULL DEFAULT 1500,
                    timeout_sec INTEGER NOT NULL DEFAULT 30,
                    system_prompt TEXT NOT NULL DEFAULT '',
                    enabled INTEGER NOT NULL DEFAULT 0,
                    last_check_status TEXT NOT NULL DEFAULT 'UNCONFIGURED',
                    last_check_message TEXT NOT NULL DEFAULT '',
                    last_checked_at TEXT,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS nav_history (
                    user_id TEXT NOT NULL DEFAULT 'default',
                    trade_date TEXT NOT NULL,
                    total_assets REAL NOT NULL,
                    available_cash REAL NOT NULL,
                    market_value REAL NOT NULL,
                    daily_return_pct REAL NOT NULL,
                    recorded_at TEXT NOT NULL,
                    PRIMARY KEY (user_id, trade_date)
                );
                """
            )
            for table in ("orders", "trades", "alert_rules", "alert_events", "auto_strategies"):
                self._ensure_columns(conn, table, {"user_id": "TEXT NOT NULL DEFAULT 'default'"})
            self._ensure_columns(
                conn,
                "watchlist",
                {
                    "is_pinned": "INTEGER NOT NULL DEFAULT 0",
                    "group_name": "TEXT NOT NULL DEFAULT '默认'",
                    "note": "TEXT NOT NULL DEFAULT ''",
                },
            )
            self._ensure_columns(
                conn,
                "auto_strategies",
                {
                    "fast_period": "INTEGER NOT NULL DEFAULT 5",
                    "slow_period": "INTEGER NOT NULL DEFAULT 20",
                    "last_signal_tag": "TEXT NOT NULL DEFAULT ''",
                },
            )
            self._ensure_columns(
                conn,
                "users",
                {
                    "email": "TEXT NOT NULL DEFAULT ''",
                    "email_verified": "INTEGER NOT NULL DEFAULT 0",
                    "alert_email_enabled": "INTEGER NOT NULL DEFAULT 1",
                },
            )
            self._ensure_columns(conn, "alert_rules", {"notify_email": "INTEGER NOT NULL DEFAULT 0"})
            self._ensure_columns(conn, "alert_events", {"email_status": "TEXT NOT NULL DEFAULT 'NONE'"})
            conn.executescript(
                """
                CREATE INDEX IF NOT EXISTS idx_orders_user_created ON orders (user_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_trades_user_created ON trades (user_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_alert_rules_user ON alert_rules (user_id, enabled);
                CREATE INDEX IF NOT EXISTS idx_alert_events_user ON alert_events (user_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_auto_strategies_user ON auto_strategies (user_id, enabled);
                CREATE INDEX IF NOT EXISTS idx_email_codes_lookup ON email_codes (email, purpose, created_at_ts DESC);
                CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email_unique ON users (LOWER(email)) WHERE email != '';
                """
            )

    @staticmethod
    def _ensure_columns(conn: sqlite3.Connection, table: str, specs: dict[str, str]) -> None:
        existing = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        for col, ddl in specs.items():
            if col not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {ddl}")

    def ensure_user_workspace(self, user_id: str) -> None:
        """Create account, risk_config, ai_config, and initial watchlist for user_id if absent."""
        uid = (user_id or "default").strip() or "default"
        with self.connect() as conn:
            self._ensure_user_workspace_tx(conn, uid)

    def _ensure_user_workspace_tx(self, conn: sqlite3.Connection, uid: str) -> None:
        now = now_str()
        row = conn.execute("SELECT user_id FROM account WHERE user_id = ?", (uid,)).fetchone()
        if not row:
            tdate = today_str()
            conn.execute(
                """
                INSERT INTO account (user_id, initial_cash, available_cash, frozen_cash, current_trade_date, market_mode, updated_at)
                VALUES (?, ?, ?, 0.0, ?, ?, ?)
                """,
                (
                    uid,
                    settings.initial_cash,
                    settings.initial_cash,
                    tdate,
                    MarketMode.REAL.value,
                    now,
                ),
            )
            conn.execute(
                """
                INSERT OR IGNORE INTO nav_history (user_id, trade_date, total_assets, available_cash, market_value, daily_return_pct, recorded_at)
                VALUES (?, ?, ?, ?, 0.0, 0.0, ?)
                """,
                (uid, tdate, settings.initial_cash, settings.initial_cash, now),
            )

        conn.execute(
            """
            INSERT OR IGNORE INTO risk_config (
                user_id, max_single_position_pct, max_order_amount, max_total_loss_pct, block_buy_on_loss, updated_at
            ) VALUES (?, 100.0, 0.0, 0.0, 1, ?)
            """,
            (uid, now),
        )

        conn.execute(
            """
            INSERT OR IGNORE INTO ai_config (
                user_id, provider_name, base_url, model, api_key, temperature, max_tokens,
                timeout_sec, system_prompt, enabled, last_check_status, last_check_message,
                last_checked_at, updated_at
            ) VALUES (?, 'Ollama 本地 (默认)', 'http://127.0.0.1:11434/v1', 'qwen2.5:14b', '', 0.3, 1500, 30, '', 0, 'UNCONFIGURED', '尚未测试连通性', NULL, ?)
            """,
            (uid, now),
        )

        w_count = conn.execute("SELECT COUNT(*) AS c FROM watchlist WHERE user_id = ?", (uid,)).fetchone()["c"]
        if w_count == 0 and not row:
            for item in settings.default_watchlist:
                conn.execute(
                    "INSERT OR IGNORE INTO watchlist (user_id, stock_code, short_name, added_at) VALUES (?, ?, ?, ?)",
                    (uid, item["code"], item["name"], now),
                )

    def ensure_bootstrap_admin(self, username: str, password: str) -> Optional[str]:
        """Create initial ADMIN account from env if username does not yet exist."""
        uname = (username or "").strip()
        if not uname or not password:
            return None
        with self.connect() as conn:
            existing = conn.execute(
                "SELECT user_id FROM users WHERE LOWER(username) = LOWER(?)", (uname,)
            ).fetchone()
            if existing:
                return str(existing["user_id"])
            uid = "admin_" + uuid.uuid4().hex[:8]
            now = now_str()
            conn.execute(
                """
                INSERT INTO users (
                    user_id, username, display_name, password_hash, role, is_active,
                    invite_code_used, ai_calls_today, ai_calls_date, created_at
                ) VALUES (?, ?, ?, ?, 'ADMIN', 1, 'BOOTSTRAP', 0, '', ?)
                """,
                (uid, uname, "管理员", hash_password(password), now),
            )
            self._ensure_user_workspace_tx(conn, uid)
            return uid

    def create_user(
        self,
        username: str,
        password: str,
        display_name: str = "",
        email: str = "",
        email_verified: bool = False,
        invite_code: str = "",
        require_invite: bool = False,
        role: str = "USER",
    ) -> dict[str, Any]:
        uname = (username or "").strip()
        if not _USERNAME_RE.match(uname):
            raise ValueError("用户名需为 2~32 位字母、数字、中文、下划线或点/短横线")
        if len(password or "") < 6:
            raise ValueError("密码长度至少需要 6 位")
        clean_email = normalize_email(email)
        if clean_email and not is_valid_email(clean_email):
            raise ValueError("邮箱格式不正确")
        dname = (display_name or "").strip()[:32] or uname
        code_clean = (invite_code or "").strip().upper()
        now = now_str()

        with self.connect() as conn:
            dup = conn.execute(
                "SELECT user_id FROM users WHERE LOWER(username) = LOWER(?)", (uname,)
            ).fetchone()
            if dup:
                raise ValueError(f"用户名「{uname}」已被注册")

            if clean_email:
                dup_email = conn.execute(
                    "SELECT user_id FROM users WHERE LOWER(email) = LOWER(?)", (clean_email,)
                ).fetchone()
                if dup_email:
                    raise ValueError(f"邮箱「{clean_email}」已被注册，请直接登录或使用找回密码")

            if require_invite:
                if not code_clean:
                    raise ValueError("当前处于邀请制阶段，请填写有效邀请码")
                cur = conn.execute(
                    """
                    UPDATE invite_codes
                    SET used_count = used_count + 1
                    WHERE UPPER(code) = ? AND enabled = 1 AND (max_uses = 0 OR used_count < max_uses)
                    """,
                    (code_clean,),
                )
                if cur.rowcount == 0:
                    raise ValueError("邀请码无效、已停用或使用次数已耗尽")

            uid = uuid.uuid4().hex[:12]
            conn.execute(
                """
                INSERT INTO users (
                    user_id, username, display_name, email, email_verified, alert_email_enabled,
                    password_hash, role, is_active, invite_code_used, ai_calls_today, ai_calls_date,
                    created_at, last_login_at
                ) VALUES (?, ?, ?, ?, ?, 1, ?, ?, 1, ?, 0, ?, ?, ?)
                """,
                (
                    uid,
                    uname,
                    dname,
                    clean_email,
                    1 if (clean_email and email_verified) else 0,
                    hash_password(password),
                    role,
                    code_clean,
                    today_str(),
                    now,
                    now,
                ),
            )
            self._ensure_user_workspace_tx(conn, uid)
            row = conn.execute("SELECT * FROM users WHERE user_id = ?", (uid,)).fetchone()
            return dict(row)

    def authenticate_user(self, login_id: str, password: str) -> Optional[dict[str, Any]]:
        ident = (login_id or "").strip()
        if not ident or not password:
            return None
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM users
                WHERE LOWER(username) = LOWER(?)
                   OR (email != '' AND LOWER(email) = LOWER(?))
                """,
                (ident, ident),
            ).fetchone()
            if not row:
                return None
            if not verify_password(password, str(row["password_hash"])):
                return None
            now = now_str()
            conn.execute("UPDATE users SET last_login_at = ? WHERE user_id = ?", (now, row["user_id"]))
            refreshed = conn.execute("SELECT * FROM users WHERE user_id = ?", (row["user_id"],)).fetchone()
            return dict(refreshed)

    def get_user_by_id(self, user_id: str) -> Optional[dict[str, Any]]:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
            return dict(row) if row else None

    def get_user_by_email(self, email: str) -> Optional[dict[str, Any]]:
        clean = normalize_email(email)
        if not clean:
            return None
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE email != '' AND LOWER(email) = LOWER(?)", (clean,)
            ).fetchone()
            return dict(row) if row else None

    @staticmethod
    def _hash_email_code(email: str, purpose: str, code: str) -> str:
        raw = f"{normalize_email(email)}|{purpose.upper()}|{(code or '').strip()}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def create_email_code(
        self, email: str, purpose: str, ttl_sec: int = 600, cooldown_sec: int = 60
    ) -> str:
        clean = normalize_email(email)
        if not is_valid_email(clean):
            raise ValueError("请输入有效的邮箱地址")
        now_ts = int(time.time())
        purp = purpose.upper()
        with self.connect() as conn:
            recent = conn.execute(
                """
                SELECT created_at_ts FROM email_codes
                WHERE email = ? AND purpose = ?
                ORDER BY created_at_ts DESC LIMIT 1
                """,
                (clean, purp),
            ).fetchone()
            if recent and (now_ts - int(recent["created_at_ts"])) < cooldown_sec:
                wait = cooldown_sec - (now_ts - int(recent["created_at_ts"]))
                raise ValueError(f"发送过于频繁，请 {wait} 秒后再试")

            code = f"{secrets.randbelow(900000) + 100000}"
            code_hash = self._hash_email_code(clean, purp, code)
            conn.execute(
                """
                INSERT INTO email_codes (id, email, purpose, code_hash, expires_at_ts, consumed, created_at_ts)
                VALUES (?, ?, ?, ?, ?, 0, ?)
                """,
                (uuid.uuid4().hex[:16], clean, purp, code_hash, now_ts + ttl_sec, now_ts),
            )
            return code

    def verify_and_consume_email_code(self, email: str, purpose: str, code: str) -> bool:
        clean = normalize_email(email)
        clean_code = (code or "").strip()
        if not clean or not clean_code:
            return False
        now_ts = int(time.time())
        purp = purpose.upper()
        expected_hash = self._hash_email_code(clean, purp, clean_code)
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT id FROM email_codes
                WHERE email = ? AND purpose = ? AND code_hash = ? AND consumed = 0 AND expires_at_ts >= ?
                ORDER BY created_at_ts DESC LIMIT 1
                """,
                (clean, purp, expected_hash, now_ts),
            ).fetchone()
            if not row:
                return False
            conn.execute("UPDATE email_codes SET consumed = 1 WHERE id = ?", (row["id"],))
            return True

    def update_user_email(
        self,
        user_id: str,
        email: Optional[str] = None,
        verified: Optional[bool] = None,
        alert_email_enabled: Optional[bool] = None,
    ) -> dict[str, Any]:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
            if not row:
                raise ValueError("用户不存在")
            next_email = row["email"]
            next_verified = int(row["email_verified"] or 0)
            if email is not None:
                clean = normalize_email(email)
                if clean and not is_valid_email(clean):
                    raise ValueError("邮箱格式不正确")
                if clean:
                    dup = conn.execute(
                        "SELECT user_id FROM users WHERE LOWER(email) = LOWER(?) AND user_id != ?",
                        (clean, user_id),
                    ).fetchone()
                    if dup:
                        raise ValueError(f"邮箱「{clean}」已被其他账号绑定")
                next_email = clean
                next_verified = 1 if (clean and verified) else 0
            elif verified is not None:
                next_verified = 1 if (next_email and verified) else 0

            next_alert = (
                (1 if alert_email_enabled else 0)
                if alert_email_enabled is not None
                else int(row["alert_email_enabled"] if "alert_email_enabled" in row.keys() else 1)
            )
            conn.execute(
                "UPDATE users SET email = ?, email_verified = ?, alert_email_enabled = ? WHERE user_id = ?",
                (next_email, next_verified, next_alert, user_id),
            )
            updated = conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
            return dict(updated)

    def reset_password_by_email(self, email: str, new_password: str) -> None:
        if len(new_password or "") < 6:
            raise ValueError("新密码长度至少需要 6 位")
        clean = normalize_email(email)
        with self.connect() as conn:
            row = conn.execute(
                "SELECT user_id FROM users WHERE email != '' AND LOWER(email) = LOWER(?)", (clean,)
            ).fetchone()
            if not row:
                raise ValueError("该邮箱未绑定任何账号")
            conn.execute(
                "UPDATE users SET password_hash = ? WHERE user_id = ?",
                (hash_password(new_password), row["user_id"]),
            )

    def get_system_setting(self, key: str, default: str = "") -> str:
        with self.connect() as conn:
            row = conn.execute("SELECT value FROM system_settings WHERE key = ?", (key,)).fetchone()
            return str(row["value"]) if row else default

    def set_system_setting(self, key: str, value: str) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO system_settings (key, value, updated_at)
                VALUES (?, ?, ?)
                """,
                (key, value, now_str()),
            )

    def change_password(self, user_id: str, old_password: str, new_password: str) -> None:
        if len(new_password or "") < 6:
            raise ValueError("新密码长度至少需要 6 位")
        with self.connect() as conn:
            row = conn.execute("SELECT password_hash FROM users WHERE user_id = ?", (user_id,)).fetchone()
            if not row:
                raise ValueError("用户不存在")
            if not verify_password(old_password, str(row["password_hash"])):
                raise ValueError("原密码不正确")
            conn.execute(
                "UPDATE users SET password_hash = ? WHERE user_id = ?",
                (hash_password(new_password), user_id),
            )

    def list_users(self) -> list[dict[str, Any]]:
        today = today_str()
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT user_id, username, display_name, email, email_verified, alert_email_enabled,
                       role, is_active, invite_code_used, ai_calls_today, ai_calls_date,
                       created_at, last_login_at
                FROM users
                ORDER BY created_at DESC
                """
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            if d.get("ai_calls_date") != today:
                d["ai_calls_today"] = 0
            out.append(d)
        return out

    def toggle_user_active(self, target_user_id: str, enabled: bool) -> dict[str, Any]:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM users WHERE user_id = ?", (target_user_id,)).fetchone()
            if not row:
                raise ValueError("用户不存在")
            if row["role"] == "ADMIN" and not enabled:
                raise ValueError("不能禁用管理员账号")
            conn.execute(
                "UPDATE users SET is_active = ? WHERE user_id = ?",
                (1 if enabled else 0, target_user_id),
            )
            updated = conn.execute(
                """
                SELECT user_id, username, display_name, role, is_active, invite_code_used,
                       ai_calls_today, ai_calls_date, created_at, last_login_at
                FROM users WHERE user_id = ?
                """,
                (target_user_id,),
            ).fetchone()
            return dict(updated)

    def create_invite_codes(
        self,
        created_by: str,
        max_uses: int = 5,
        count: int = 1,
        note: str = "",
        custom_code: str = "",
    ) -> list[dict[str, Any]]:
        now = now_str()
        clean_custom = re.sub(r"[^A-Za-z0-9_-]", "", (custom_code or "").strip()).upper()[:24]
        created: list[dict[str, Any]] = []
        with self.connect() as conn:
            total = 1 if clean_custom else max(1, min(count, 20))
            for _ in range(total):
                if clean_custom:
                    code = clean_custom
                else:
                    code = "ZT-" + secrets.token_hex(3).upper() + "-" + secrets.token_hex(2).upper()
                existing = conn.execute("SELECT code FROM invite_codes WHERE UPPER(code) = ?", (code,)).fetchone()
                if existing:
                    if clean_custom:
                        raise ValueError(f"邀请码 {code} 已存在")
                    code = "ZT-" + secrets.token_hex(5).upper()
                conn.execute(
                    """
                    INSERT INTO invite_codes (code, created_by, max_uses, used_count, note, enabled, created_at)
                    VALUES (?, ?, ?, 0, ?, 1, ?)
                    """,
                    (code, created_by, max_uses, (note or "").strip()[:80], now),
                )
                row = conn.execute("SELECT * FROM invite_codes WHERE code = ?", (code,)).fetchone()
                created.append(dict(row))
        return created

    def list_invite_codes(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM invite_codes ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]

    def toggle_invite_code(self, code: str, enabled: bool) -> dict[str, Any]:
        clean = (code or "").strip().upper()
        with self.connect() as conn:
            conn.execute("UPDATE invite_codes SET enabled = ? WHERE UPPER(code) = ?", (1 if enabled else 0, clean))
            row = conn.execute("SELECT * FROM invite_codes WHERE UPPER(code) = ?", (clean,)).fetchone()
            if not row:
                raise ValueError(f"邀请码 {clean} 不存在")
            return dict(row)

    def delete_invite_code(self, code: str) -> None:
        clean = (code or "").strip().upper()
        with self.connect() as conn:
            conn.execute("DELETE FROM invite_codes WHERE UPPER(code) = ?", (clean,))

    def get_platform_ai_usage(self, user_id: str, daily_limit: int) -> tuple[int, int]:
        """Return (used_today, remaining_today)."""
        if daily_limit <= 0:
            return 0, 0
        today = today_str()
        with self.connect() as conn:
            row = conn.execute(
                "SELECT ai_calls_today, ai_calls_date FROM users WHERE user_id = ?", (user_id,)
            ).fetchone()
            if not row:
                return 0, daily_limit
            used = int(row["ai_calls_today"] or 0) if row["ai_calls_date"] == today else 0
            return used, max(0, daily_limit - used)

    def consume_platform_ai_quota(self, user_id: str, daily_limit: int) -> tuple[bool, int]:
        """Atomically consume 1 platform AI quota call for today; returns (allowed, remaining_after)."""
        if daily_limit <= 0:
            return False, 0
        today = today_str()
        with self.connect() as conn:
            row = conn.execute(
                "SELECT ai_calls_today, ai_calls_date FROM users WHERE user_id = ?", (user_id,)
            ).fetchone()
            if not row:
                return True, max(0, daily_limit - 1)
            used = int(row["ai_calls_today"] or 0) if row["ai_calls_date"] == today else 0
            if used >= daily_limit:
                return False, 0
            new_used = used + 1
            conn.execute(
                "UPDATE users SET ai_calls_today = ?, ai_calls_date = ? WHERE user_id = ?",
                (new_used, today, user_id),
            )
            return True, max(0, daily_limit - new_used)

    def list_all_account_user_ids(self) -> list[str]:
        with self.connect() as conn:
            rows = conn.execute("SELECT user_id FROM account").fetchall()
        return [str(r["user_id"]) for r in rows]

    def list_all_watched_and_held_items(self) -> list[tuple[str, str]]:
        """Return deduplicated (stock_code, short_name) union across all users' watchlists and open positions."""
        with self.connect() as conn:
            w_rows = conn.execute("SELECT stock_code, short_name FROM watchlist").fetchall()
            p_rows = conn.execute("SELECT stock_code, short_name FROM positions WHERE total_qty > 0").fetchall()
        merged: dict[str, str] = {}
        for r in list(w_rows) + list(p_rows):
            code = str(r["stock_code"])
            name = str(r["short_name"] or code)
            if code not in merged or (merged[code] == code and name != code):
                merged[code] = name
        return list(merged.items())

    def get_public_ai_config(self) -> dict[str, Any]:
        """Return merged platform trial AI config (DB system_settings overrides env defaults)."""
        provider = self.get_system_setting("public_ai_provider", "").strip() or settings.public_ai_provider
        base_url = self.get_system_setting("public_ai_base_url", "").strip() or settings.public_ai_base_url
        model = self.get_system_setting("public_ai_model", "").strip() or settings.public_ai_model
        stored_key = self.get_system_setting("public_ai_api_key", "")
        if stored_key == "__CLEARED__":
            api_key = ""
        else:
            api_key = stored_key.strip() or settings.public_ai_api_key
        raw_limit = self.get_system_setting("public_ai_daily_limit", "").strip()
        try:
            daily_limit = max(0, int(raw_limit)) if raw_limit != "" else settings.public_ai_daily_limit
        except ValueError:
            daily_limit = settings.public_ai_daily_limit
        return {
            "provider_name": provider or "智瞳公共体验 AI",
            "base_url": base_url,
            "model": model,
            "api_key": api_key,
            "daily_limit": daily_limit,
        }

    def set_public_ai_config(
        self,
        *,
        provider_name: str,
        base_url: str,
        model: str,
        daily_limit: int,
        api_key: str = "",
        clear_api_key: bool = False,
    ) -> dict[str, Any]:
        """Persist platform trial AI config in system_settings."""
        self.set_system_setting("public_ai_provider", (provider_name or "智瞳公共体验 AI").strip()[:60])
        self.set_system_setting("public_ai_base_url", (base_url or "").strip()[:300])
        self.set_system_setting("public_ai_model", (model or "").strip()[:120])
        self.set_system_setting("public_ai_daily_limit", str(max(0, min(int(daily_limit), 1000))))
        if clear_api_key:
            self.set_system_setting("public_ai_api_key", "__CLEARED__")
        elif (api_key or "").strip():
            self.set_system_setting("public_ai_api_key", api_key.strip())
        return self.get_public_ai_config()

    def get_announcement(self) -> dict[str, Any]:
        """Return foreground site announcement configuration."""
        title = self.get_system_setting("site_announcement_title", "").strip()
        content = self.get_system_setting("site_announcement_content", "").strip()
        enabled = self.get_system_setting("site_announcement_enabled", "false") == "true"
        updated_at = self.get_system_setting("site_announcement_updated_at", "")
        return {
            "title": title,
            "content": content,
            "enabled": enabled and bool(title or content),
            "raw_enabled": enabled,
            "updated_at": updated_at,
        }

    def set_announcement(self, *, title: str, content: str, enabled: bool) -> dict[str, Any]:
        """Persist foreground site announcement configuration."""
        self.set_system_setting("site_announcement_title", (title or "").strip()[:100])
        self.set_system_setting("site_announcement_content", (content or "").strip()[:600])
        self.set_system_setting("site_announcement_enabled", "true" if enabled else "false")
        self.set_system_setting("site_announcement_updated_at", now_str())
        return self.get_announcement()

    def get_admin_telemetry(
        self,
        *,
        quotes_cached: int = 0,
        market_mode: str = "REAL",
        uptime_seconds: float = 0.0,
    ) -> dict[str, Any]:
        """Aggregate platform-wide telemetry, 14-day trends, strategy breakdown, and per-user summaries."""
        today = today_str()
        now_dt = beijing_now()
        days_14 = [(now_dt - timedelta(days=offset)).strftime("%Y-%m-%d") for offset in range(13, -1, -1)]
        day_set = set(days_14)

        with self.connect() as conn:
            u_rows = conn.execute(
                """
                SELECT user_id, username, display_name, role, is_active, invite_code_used,
                       ai_calls_today, ai_calls_date, created_at, last_login_at
                FROM users
                ORDER BY created_at DESC
                """
            ).fetchall()
            acct_rows = conn.execute(
                "SELECT user_id, available_cash, frozen_cash, initial_cash FROM account"
            ).fetchall()
            pos_rows = conn.execute(
                "SELECT user_id, COUNT(*) AS cnt FROM positions WHERE total_qty > 0 GROUP BY user_id"
            ).fetchall()
            ord_rows = conn.execute(
                "SELECT user_id, COUNT(*) AS cnt FROM orders GROUP BY user_id"
            ).fetchall()
            trd_rows = conn.execute(
                "SELECT user_id, COUNT(*) AS cnt FROM trades GROUP BY user_id"
            ).fetchall()
            watch_rows = conn.execute(
                "SELECT user_id, COUNT(*) AS cnt FROM watchlist GROUP BY user_id"
            ).fetchall()
            strat_rows = conn.execute(
                "SELECT user_id, COUNT(*) AS cnt, SUM(CASE WHEN enabled = 1 THEN 1 ELSE 0 END) AS active_cnt FROM auto_strategies GROUP BY user_id"
            ).fetchall()
            ai_own_rows = conn.execute(
                "SELECT user_id, enabled, api_key, base_url FROM ai_config"
            ).fetchall()

            strat_type_rows = conn.execute(
                "SELECT strategy_type, COUNT(*) AS cnt FROM auto_strategies GROUP BY strategy_type ORDER BY cnt DESC"
            ).fetchall()
            top_watch_rows = conn.execute(
                """
                SELECT stock_code, MAX(short_name) AS short_name, COUNT(*) AS watchers
                FROM watchlist
                GROUP BY stock_code
                ORDER BY watchers DESC, stock_code ASC
                LIMIT 9
                """
            ).fetchall()

            alert_rules_total = int(conn.execute("SELECT COUNT(*) FROM alert_rules").fetchone()[0])
            alert_events_total = int(conn.execute("SELECT COUNT(*) FROM alert_events").fetchone()[0])

            since_day = days_14[0]
            ord_daily = conn.execute(
                "SELECT substr(created_at, 1, 10) AS d, COUNT(*) AS cnt FROM orders WHERE substr(created_at, 1, 10) >= ? GROUP BY d",
                (since_day,),
            ).fetchall()
            trd_daily = conn.execute(
                "SELECT substr(created_at, 1, 10) AS d, COUNT(*) AS cnt FROM trades WHERE substr(created_at, 1, 10) >= ? GROUP BY d",
                (since_day,),
            ).fetchall()
            usr_daily = conn.execute(
                "SELECT substr(created_at, 1, 10) AS d, COUNT(*) AS cnt FROM users WHERE substr(created_at, 1, 10) >= ? GROUP BY d",
                (since_day,),
            ).fetchall()
            alt_daily = conn.execute(
                "SELECT substr(created_at, 1, 10) AS d, COUNT(*) AS cnt FROM alert_events WHERE substr(created_at, 1, 10) >= ? GROUP BY d",
                (since_day,),
            ).fetchall()

        acct_map = {
            str(r["user_id"]): {
                "cash": float(r["available_cash"]) + float(r["frozen_cash"]),
                "initial_cash": float(r["initial_cash"]),
            }
            for r in acct_rows
        }
        pos_map = {str(r["user_id"]): int(r["cnt"]) for r in pos_rows}
        ord_map = {str(r["user_id"]): int(r["cnt"]) for r in ord_rows}
        trd_map = {str(r["user_id"]): int(r["cnt"]) for r in trd_rows}
        watch_map = {str(r["user_id"]): int(r["cnt"]) for r in watch_rows}
        strat_map = {
            str(r["user_id"]): {"total": int(r["cnt"] or 0), "active": int(r["active_cnt"] or 0)}
            for r in strat_rows
        }
        ai_own_map = {
            str(r["user_id"]): bool(r["enabled"]) and bool(str(r["api_key"] or "").strip() or str(r["base_url"] or "").strip())
            for r in ai_own_rows
        }

        user_summaries: list[dict[str, Any]] = []
        users_admin = 0
        users_active = 0
        users_disabled = 0
        ai_calls_today_total = 0
        user_ai_configured_count = 0

        for u in u_rows:
            uid = str(u["user_id"])
            is_admin = u["role"] == "ADMIN"
            is_act = bool(u["is_active"])
            if is_admin:
                users_admin += 1
            if is_act:
                users_active += 1
            else:
                users_disabled += 1
            ai_today = int(u["ai_calls_today"] or 0) if u["ai_calls_date"] == today else 0
            ai_calls_today_total += ai_today
            has_own_ai = ai_own_map.get(uid, False)
            if has_own_ai:
                user_ai_configured_count += 1
            acct = acct_map.get(uid, {"cash": 0.0, "initial_cash": 0.0})
            st = strat_map.get(uid, {"total": 0, "active": 0})
            user_summaries.append(
                {
                    "user_id": uid,
                    "username": str(u["username"]),
                    "display_name": str(u["display_name"] or u["username"]),
                    "role": "ADMIN" if is_admin else "USER",
                    "is_active": is_act,
                    "invite_code_used": str(u["invite_code_used"] or ""),
                    "cash": round(acct["cash"], 2),
                    "positions_count": pos_map.get(uid, 0),
                    "orders_count": ord_map.get(uid, 0),
                    "trades_count": trd_map.get(uid, 0),
                    "watchlist_count": watch_map.get(uid, 0),
                    "strategies_total": st["total"],
                    "strategies_active": st["active"],
                    "has_own_ai": has_own_ai,
                    "ai_calls_today": ai_today,
                    "created_at": str(u["created_at"] or ""),
                    "last_login_at": u["last_login_at"],
                }
            )

        trend_map: dict[str, dict[str, Any]] = {
            d: {"date": d, "label": d[5:], "new_users": 0, "orders": 0, "trades": 0, "alerts": 0}
            for d in days_14
        }
        for r in ord_daily:
            if r["d"] in day_set:
                trend_map[r["d"]]["orders"] = int(r["cnt"])
        for r in trd_daily:
            if r["d"] in day_set:
                trend_map[r["d"]]["trades"] = int(r["cnt"])
        for r in usr_daily:
            if r["d"] in day_set:
                trend_map[r["d"]]["new_users"] = int(r["cnt"])
        for r in alt_daily:
            if r["d"] in day_set:
                trend_map[r["d"]]["alerts"] = int(r["cnt"])

        db_size_bytes = 0
        try:
            if self.db_path.exists():
                db_size_bytes = int(os.path.getsize(self.db_path))
                wal = Path(str(self.db_path) + "-wal")
                if wal.exists():
                    db_size_bytes += int(os.path.getsize(wal))
        except OSError:
            pass

        disk_total = 0
        disk_used = 0
        disk_free = 0
        disk_pct = 0.0
        try:
            target = self.db_path.parent if self.db_path.parent.exists() else Path(".")
            usage = shutil.disk_usage(target)
            disk_total = int(usage.total)
            disk_used = int(usage.used)
            disk_free = int(usage.free)
            disk_pct = round((disk_used / disk_total) * 100.0, 1) if disk_total > 0 else 0.0
        except OSError:
            pass

        return {
            "counts": {
                "users_total": len(u_rows),
                "users_admin": users_admin,
                "users_regular": max(0, len(u_rows) - users_admin),
                "users_active": users_active,
                "users_disabled": users_disabled,
                "watchlist_total": sum(watch_map.values()),
                "positions_total": sum(pos_map.values()),
                "orders_total": sum(ord_map.values()),
                "trades_total": sum(trd_map.values()),
                "strategies_total": sum(v["total"] for v in strat_map.values()),
                "strategies_enabled": sum(v["active"] for v in strat_map.values()),
                "alert_rules_total": alert_rules_total,
                "alert_events_total": alert_events_total,
                "ai_calls_today_total": ai_calls_today_total,
                "user_ai_configured_count": user_ai_configured_count,
                "total_cash_sum": round(sum(a["cash"] for a in acct_map.values()), 2),
                "total_initial_cash_sum": round(sum(a["initial_cash"] for a in acct_map.values()), 2),
            },
            "trends_14d": [trend_map[d] for d in days_14],
            "strategy_breakdown": [
                {"strategy_type": str(r["strategy_type"]), "count": int(r["cnt"])}
                for r in strat_type_rows
            ],
            "top_watched": [
                {
                    "stock_code": str(r["stock_code"]),
                    "short_name": str(r["short_name"] or r["stock_code"]),
                    "watchers": int(r["watchers"]),
                }
                for r in top_watch_rows
            ],
            "user_summaries": user_summaries,
            "env_stats": {
                "db_path": str(self.db_path),
                "db_size_bytes": db_size_bytes,
                "disk_total_bytes": disk_total,
                "disk_used_bytes": disk_used,
                "disk_free_bytes": disk_free,
                "disk_used_pct": disk_pct,
                "uptime_seconds": round(max(0.0, float(uptime_seconds)), 1),
                "quotes_cached": int(quotes_cached),
                "market_mode": str(market_mode),
            },
        }
