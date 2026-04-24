"""OKX v5 REST signing helper.

Credentials are read from env vars — never commit them.
    OKX_API_KEY, OKX_API_SECRET, OKX_API_PASSPHRASE
    OKX_DEMO=1  -> demo trading (simulated), 0 -> live

This module intentionally does NOT allow live trading unless
OKX_ENABLE_LIVE=1 is also set, as a double-safety against foot-guns.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests

OKX_BASE = "https://www.okx.com"


def _load_dotenv(path: str | Path = ".env") -> None:
    """Minimal .env loader — no new dependency.

    Reads KEY=VALUE lines. Strips surrounding quotes. Skips blanks + comments.
    Does NOT override existing env vars (so shell exports still win).
    """
    p = Path(path)
    if not p.exists():
        return
    try:
        for raw in p.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip()
            # strip inline comments (very simple — only if value is unquoted)
            if not (val.startswith('"') or val.startswith("'")):
                val = val.split("#", 1)[0].strip()
            # strip matching quote pair
            if len(val) >= 2 and val[0] == val[-1] and val[0] in ('"', "'"):
                val = val[1:-1]
            if key and key not in os.environ:
                os.environ[key] = val
    except Exception:
        pass  # best-effort; never crash on malformed .env


# Auto-load .env from CWD at import time.
_load_dotenv()


@dataclass
class OkxCreds:
    api_key: str
    api_secret: str
    passphrase: str
    demo: bool

    @classmethod
    def from_env(cls) -> "OkxCreds":
        demo = os.getenv("OKX_DEMO", "1") != "0"
        live_enabled = os.getenv("OKX_ENABLE_LIVE") == "1"
        if not demo and not live_enabled:
            raise RuntimeError(
                "OKX_DEMO=0 requires OKX_ENABLE_LIVE=1 as an explicit safety confirmation."
            )
        missing = [
            k for k in ("OKX_API_KEY", "OKX_API_SECRET", "OKX_API_PASSPHRASE") if not os.getenv(k)
        ]
        if missing:
            raise RuntimeError(f"missing OKX env vars: {', '.join(missing)}")
        return cls(
            api_key=os.environ["OKX_API_KEY"],
            api_secret=os.environ["OKX_API_SECRET"],
            passphrase=os.environ["OKX_API_PASSPHRASE"],
            demo=demo,
        )


def _ts_iso() -> str:
    # millisecond ISO like 2024-01-01T00:00:00.000Z
    t = time.time()
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t)) + f".{int((t%1)*1000):03d}Z"


def _sign(secret: str, ts: str, method: str, path: str, body: str) -> str:
    msg = (ts + method.upper() + path + body).encode()
    digest = hmac.new(secret.encode(), msg, hashlib.sha256).digest()
    return base64.b64encode(digest).decode()


def request(
    method: str,
    path: str,
    creds: OkxCreds,
    params: Optional[dict] = None,
    body: Optional[dict] = None,
    timeout: float = 15.0,
) -> dict:
    """Signed request to OKX v5. `path` includes /api/v5/... and does NOT include query string."""
    qs = ""
    if params:
        # OKX includes the query string in the signed path
        qs = "?" + "&".join(f"{k}={v}" for k, v in params.items() if v is not None)
    signed_path = path + qs

    body_str = "" if body is None else json.dumps(body, separators=(",", ":"))
    ts = _ts_iso()
    sign = _sign(creds.api_secret, ts, method, signed_path, body_str)

    headers = {
        "OK-ACCESS-KEY": creds.api_key,
        "OK-ACCESS-SIGN": sign,
        "OK-ACCESS-TIMESTAMP": ts,
        "OK-ACCESS-PASSPHRASE": creds.passphrase,
        "Content-Type": "application/json",
    }
    if creds.demo:
        headers["x-simulated-trading"] = "1"

    url = OKX_BASE + signed_path
    if method.upper() == "GET":
        r = requests.get(url, headers=headers, timeout=timeout)
    else:
        r = requests.request(method.upper(), url, headers=headers, data=body_str, timeout=timeout)
    r.raise_for_status()
    payload = r.json()
    if str(payload.get("code")) != "0":
        raise RuntimeError(f"OKX error {payload.get('code')}: {payload.get('msg')} | data={payload.get('data')}")
    return payload
