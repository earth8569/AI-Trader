"""Paginated Binance public klines fetcher with on-disk cache."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

BINANCE_URL = "https://api.binance.com/api/v3/klines"
INTERVAL_MS = {
    "1m": 60_000,
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "1h": 60 * 60_000,
    "4h": 4 * 60 * 60_000,
    "1d": 24 * 60 * 60_000,
}


def _fetch_page(symbol: str, interval: str, start_ms: int, end_ms: int) -> list:
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": 1000,
    }
    r = requests.get(BINANCE_URL, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def load_ohlcv(
    symbol: str,
    interval: str,
    start: str,
    end: Optional[str] = None,
    cache_dir: str | Path = "data",
) -> pd.DataFrame:
    """Fetch OHLCV bars, cached on disk as CSV.

    start/end: ISO date strings like '2022-01-01'.
    """
    if interval not in INTERVAL_MS:
        raise ValueError(f"Unsupported interval {interval}")

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    end = end or time.strftime("%Y-%m-%d", time.gmtime())
    cache_file = cache_dir / f"{symbol}_{interval}_{start}_{end}.csv"
    if cache_file.exists():
        df = pd.read_csv(cache_file, parse_dates=["timestamp"])
        return df

    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)
    step = INTERVAL_MS[interval] * 1000

    rows: list = []
    cursor = start_ms
    while cursor < end_ms:
        batch_end = min(cursor + step, end_ms)
        page = _fetch_page(symbol, interval, cursor, batch_end)
        if not page:
            cursor = batch_end + 1
            continue
        rows.extend(page)
        last_open = page[-1][0]
        cursor = last_open + INTERVAL_MS[interval]
        time.sleep(0.15)  # be polite

    if not rows:
        raise RuntimeError(f"No data returned for {symbol} {interval}")

    cols = [
        "open_ts", "open", "high", "low", "close", "volume",
        "close_ts", "quote_vol", "trades", "tb_base", "tb_quote", "ignore",
    ]
    df = pd.DataFrame(rows, columns=cols)
    df["timestamp"] = pd.to_datetime(df["open_ts"], unit="ms", utc=True)
    num_cols = ["open", "high", "low", "close", "volume"]
    df[num_cols] = df[num_cols].astype(float)
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    df.to_csv(cache_file, index=False)
    return df
