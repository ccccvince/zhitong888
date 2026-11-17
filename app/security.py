from __future__ import annotations

import base64
import hashlib
import hmac
import ipaddress
import os
import secrets
import socket
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit

try:
    from zoneinfo import ZoneInfo

    _BJ_TZ = ZoneInfo("Asia/Shanghai")
except Exception:  # pragma: no cover
    _BJ_TZ = timezone(timedelta(hours=8))


def beijing_now() -> datetime:
    """Return current Beijing (Asia/Shanghai, UTC+8) time as naive datetime for consistent local comparisons."""
    return datetime.now(_BJ_TZ).replace(tzinfo=None)


def now_str() -> str:
    return beijing_now().strftime("%Y-%m-%d %H:%M:%S")


def today_str() -> str:
    return beijing_now().strftime("%Y-%m-%d")


def hash_password(password: str) -> str:
    raw = password.encode("utf-8")
    salt = os.urandom(16)
    n, r, p = 16384, 8, 1
    dk = hashlib.scrypt(raw, salt=salt, n=n, r=r, p=p, dklen=32)
    return f"scrypt${n}${r}${p}${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        parts = (stored or "").split("$")
        if len(parts) != 6 or parts[0] != "scrypt":
            return False
        n, r, p = int(parts[1]), int(parts[2]), int(parts[3])
        salt = bytes.fromhex(parts[4])
        expected = bytes.fromhex(parts[5])
        dk = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=n, r=r, p=p, dklen=len(expected))
        return hmac.compare_digest(dk, expected)
    except Exception:
        return False


def resolve_secret_key(data_dir: Path, explicit_secret: str = "") -> str:
    sec = (explicit_secret or "").strip()
    if sec:
        return sec
    data_dir.mkdir(parents=True, exist_ok=True)
    secret_file = data_dir / ".session_secret"
    if secret_file.exists():
        saved = secret_file.read_text(encoding="utf-8").strip()
        if saved:
            return saved
    generated = secrets.token_urlsafe(48)
    secret_file.write_text(generated, encoding="utf-8")
    try:
        os.chmod(secret_file, 0o600)
    except OSError:
        pass
    return generated


def create_session_token(user_id: str, secret: str, ttl_sec: int = 86400 * 14) -> str:
    exp = int(time.time()) + ttl_sec
    nonce = secrets.token_hex(8)
    payload_raw = f"{user_id}|{exp}|{nonce}".encode("utf-8")
    payload_b64 = base64.urlsafe_b64encode(payload_raw).decode("ascii").rstrip("=")
    sig = hmac.new(secret.encode("utf-8"), payload_b64.encode("ascii"), hashlib.sha256).hexdigest()
    return f"{payload_b64}.{sig}"


def verify_session_token(token: str, secret: str) -> Optional[str]:
    if not token or "." not in token:
        return None
    try:
        payload_b64, sig = token.rsplit(".", 1)
        expected_sig = hmac.new(
            secret.encode("utf-8"), payload_b64.encode("ascii"), hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(sig, expected_sig):
            return None
        padded = payload_b64 + "=" * (-len(payload_b64) % 4)
        decoded = base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8")
        parts = decoded.split("|")
        if len(parts) != 3:
            return None
        user_id, exp_str, _nonce = parts
        if int(exp_str) < int(time.time()):
            return None
        return user_id
    except Exception:
        return None


def _is_disallowed_ip(ip_obj: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return bool(
        ip_obj.is_private
        or ip_obj.is_loopback
        or ip_obj.is_link_local
        or ip_obj.is_multicast
        or ip_obj.is_reserved
        or ip_obj.is_unspecified
    )


def is_private_or_reserved_host(host: str, resolve_dns: bool = False) -> bool:
    h = (host or "").strip().strip("[]").lower()
    if not h:
        return True
    if h in ("localhost", "localhost.localdomain", "metadata.google.internal"):
        return True
    if h.endswith((".local", ".internal", ".localhost", ".lan", ".home")):
        return True
    try:
        ip = ipaddress.ip_address(h)
        return _is_disallowed_ip(ip)
    except ValueError:
        pass

    if resolve_dns:
        try:
            infos = socket.getaddrinfo(h, None, proto=socket.IPPROTO_TCP)
            for info in infos:
                addr = info[4][0].split("%")[0]
                try:
                    ip = ipaddress.ip_address(addr)
                    if _is_disallowed_ip(ip):
                        return True
                except ValueError:
                    continue
        except OSError:
            return False
    return False


def validate_outbound_http_url(
    raw_url: str,
    *,
    allow_private: bool = False,
    resolve_dns: bool = False,
    label: str = "URL",
) -> str:
    cleaned = (raw_url or "").strip()
    if not cleaned:
        raise ValueError(f"{label} 不能为空")
    parts = urlsplit(cleaned)
    scheme = parts.scheme.lower()
    if scheme not in ("http", "https"):
        raise ValueError(f"{label} 仅支持 http:// 或 https:// 协议")
    host = parts.hostname
    if not host:
        raise ValueError(f"{label} 缺少有效主机名")
    try:
        port = parts.port
    except ValueError as exc:
        raise ValueError(f"{label} 端口号无效") from exc
    if port is not None and (port < 1 or port > 65535):
        raise ValueError(f"{label} 端口号超出范围 (1~65535)")
    if not allow_private and is_private_or_reserved_host(host, resolve_dns=resolve_dns):
        raise ValueError(f"安全限制：{label} 不允许指向本机回环、内网或保留 IP 地址 ({host})")
    return cleaned


class LoginRateLimiter:
    """In-memory sliding window limiter to throttle brute-force login attempts."""

    def __init__(self, max_failures: int = 10, window_sec: int = 300) -> None:
        self.max_failures = max_failures
        self.window_sec = window_sec
        self._lock = threading.Lock()
        self._failures: dict[str, list[float]] = {}

    def is_blocked(self, key: str) -> bool:
        now = time.time()
        with self._lock:
            hits = [ts for ts in self._failures.get(key, []) if now - ts < self.window_sec]
            self._failures[key] = hits
            return len(hits) >= self.max_failures

    def record_failure(self, key: str) -> None:
        now = time.time()
        with self._lock:
            hits = [ts for ts in self._failures.get(key, []) if now - ts < self.window_sec]
            hits.append(now)
            self._failures[key] = hits

    def clear(self, key: str) -> None:
        with self._lock:
            self._failures.pop(key, None)
