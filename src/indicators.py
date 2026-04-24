"""Technical indicators computed once per backtest."""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(x: pd.Series, n: int) -> pd.Series:
    return x.rolling(n, min_periods=n).mean()


def ema(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(span=n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    roll_dn = down.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = roll_up / roll_dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0):
    mid = sma(close, n)
    std = close.rolling(n, min_periods=n).std()
    return mid + k * std, mid, mid - k * std


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Attach all indicators used by the agents. Returns a new dataframe."""
    out = df.copy()
    out["sma_fast"] = sma(out["close"], 20)
    out["sma_slow"] = sma(out["close"], 50)
    out["ema_fast"] = ema(out["close"], 12)
    out["ema_slow"] = ema(out["close"], 26)
    out["rsi"] = rsi(out["close"], 14)
    macd_line, macd_sig, macd_hist = macd(out["close"])
    out["macd"] = macd_line
    out["macd_signal"] = macd_sig
    out["macd_hist"] = macd_hist
    bb_up, bb_mid, bb_lo = bollinger(out["close"], 20, 2.0)
    out["bb_up"], out["bb_mid"], out["bb_lo"] = bb_up, bb_mid, bb_lo
    out["bb_pct"] = (out["close"] - bb_lo) / (bb_up - bb_lo).replace(0, np.nan)
    out["atr"] = atr(out, 14)
    out["ret_1"] = out["close"].pct_change(1)
    out["ret_5"] = out["close"].pct_change(5)
    out["vol_sma"] = sma(out["volume"], 20)
    out["vol_ratio"] = out["volume"] / out["vol_sma"].replace(0, np.nan)
    # structural swing levels used by the tactics planner for stops/targets
    out["swing_high_20"] = out["high"].rolling(20, min_periods=20).max()
    out["swing_low_20"]  = out["low"].rolling(20, min_periods=20).min()
    return out
