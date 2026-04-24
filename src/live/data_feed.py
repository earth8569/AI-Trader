"""Live data feed — Crypto.com public exchange REST API.

No authentication required for public endpoints. Returns OHLCV candles
in the same shape the backtester/indicators module expects.
"""
from __future__ import annotations

import time
from typing import List

import pandas as pd
import requests

BASE = "https://api.crypto.com/exchange/v1/public"
# Crypto.com timeframe tokens
TF_MAP = {
    "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
    "1h": "1h", "4h": "4h", "6h": "6h", "12h": "12h",
    "1d": "1D", "1w": "7D", "1M": "1M",
}


class LiveFeed:
    def __init__(self, timeout: float = 15.0):
        self.timeout = timeout

    def get_candles(self, instrument: str, timeframe: str, count: int = 300) -> pd.DataFrame:
        tf = TF_MAP.get(timeframe, timeframe)
        params = {"instrument_name": instrument, "timeframe": tf, "count": count}
        r = requests.get(f"{BASE}/get-candlestick", params=params, timeout=self.timeout)
        r.raise_for_status()
        payload = r.json()
        data: List[dict] = (payload.get("result") or {}).get("data") or []
        if not data:
            raise RuntimeError(f"No candles for {instrument} {timeframe}: {payload}")
        rows = [
            {
                "timestamp": pd.to_datetime(int(c["t"]), unit="ms", utc=True),
                "open": float(c["o"]),
                "high": float(c["h"]),
                "low": float(c["l"]),
                "close": float(c["c"]),
                "volume": float(c["v"]),
            }
            for c in data
        ]
        df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
        return df

    def get_ticker(self, instrument: str) -> dict:
        params = {"instrument_name": instrument}
        r = requests.get(f"{BASE}/get-tickers", params=params, timeout=self.timeout)
        r.raise_for_status()
        payload = r.json()
        data = (payload.get("result") or {}).get("data") or []
        if not data:
            raise RuntimeError(f"No ticker for {instrument}: {payload}")
        t = data[0]
        return {
            "instrument": t.get("i"),
            "last": float(t["a"]) if t.get("a") else None,
            "bid": float(t["b"]) if t.get("b") else None,
            "ask": float(t["k"]) if t.get("k") else None,
            "ts": int(t.get("t", time.time() * 1000)),
        }
