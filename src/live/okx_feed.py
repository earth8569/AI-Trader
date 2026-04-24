"""OKX public market data feed — no auth, usable for backtest + live."""
from __future__ import annotations

import pandas as pd
import requests

from .okx_auth import OKX_BASE

OKX_BAR = {
    "1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m", "30m": "30m",
    "1h": "1H", "2h": "2H", "4h": "4H", "1d": "1D",
}


def symbol_to_okx(symbol: str, kind: str = "spot") -> str:
    """Convert our canonical 'BTC_USDT' to OKX 'BTC-USDT' (spot) or 'BTC-USDT-SWAP'."""
    base = symbol.replace("_", "-").upper()
    return f"{base}-SWAP" if kind == "swap" else base


class OkxFeed:
    def __init__(self, timeout: float = 15.0):
        self.timeout = timeout

    def get_candles(self, instrument: str, timeframe: str, count: int = 300) -> pd.DataFrame:
        bar = OKX_BAR.get(timeframe)
        if bar is None:
            raise ValueError(f"unsupported timeframe {timeframe}")
        inst_id = symbol_to_okx(instrument)
        params = {"instId": inst_id, "bar": bar, "limit": min(count, 300)}
        r = requests.get(f"{OKX_BASE}/api/v5/market/candles", params=params, timeout=self.timeout)
        r.raise_for_status()
        payload = r.json()
        if str(payload.get("code")) != "0":
            raise RuntimeError(f"OKX candles error: {payload}")
        rows = payload.get("data") or []
        if not rows:
            raise RuntimeError(f"no candles for {inst_id}")
        # OKX row: [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm]
        parsed = [
            {
                "timestamp": pd.to_datetime(int(r[0]), unit="ms", utc=True),
                "open": float(r[1]),
                "high": float(r[2]),
                "low": float(r[3]),
                "close": float(r[4]),
                "volume": float(r[5]),
            }
            for r in rows
        ]
        df = pd.DataFrame(parsed).sort_values("timestamp").reset_index(drop=True)
        return df

    def get_ticker(self, instrument: str) -> dict:
        inst_id = symbol_to_okx(instrument)
        r = requests.get(
            f"{OKX_BASE}/api/v5/market/ticker",
            params={"instId": inst_id},
            timeout=self.timeout,
        )
        r.raise_for_status()
        payload = r.json()
        if str(payload.get("code")) != "0":
            raise RuntimeError(f"OKX ticker error: {payload}")
        data = payload.get("data") or []
        if not data:
            raise RuntimeError(f"no ticker for {inst_id}")
        t = data[0]
        return {
            "instrument": t["instId"],
            "last": float(t["last"]) if t.get("last") else None,
            "bid": float(t["bidPx"]) if t.get("bidPx") else None,
            "ask": float(t["askPx"]) if t.get("askPx") else None,
            "ts": int(t.get("ts", 0)),
        }
