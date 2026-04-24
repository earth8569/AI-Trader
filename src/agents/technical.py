"""Technical agent: trend + momentum vote from classic indicators."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Signal:
    score: float       # -1 (strong sell) .. +1 (strong buy)
    confidence: float  # 0..1
    reason: str


class TechnicalAgent:
    name = "technical"

    def evaluate(self, row) -> Signal:
        votes = []
        reasons = []

        if row["sma_fast"] > row["sma_slow"]:
            votes.append(1); reasons.append("SMA20>SMA50 uptrend")
        else:
            votes.append(-1); reasons.append("SMA20<SMA50 downtrend")

        if row["ema_fast"] > row["ema_slow"]:
            votes.append(1); reasons.append("EMA12>EMA26")
        else:
            votes.append(-1); reasons.append("EMA12<EMA26")

        if row["macd_hist"] > 0:
            votes.append(1); reasons.append("MACD hist+")
        else:
            votes.append(-1); reasons.append("MACD hist-")

        rsi = row["rsi"]
        if rsi < 30:
            votes.append(1); reasons.append(f"RSI oversold {rsi:.0f}")
        elif rsi > 70:
            votes.append(-1); reasons.append(f"RSI overbought {rsi:.0f}")
        elif rsi > 55:
            votes.append(0.5); reasons.append(f"RSI bullish {rsi:.0f}")
        elif rsi < 45:
            votes.append(-0.5); reasons.append(f"RSI bearish {rsi:.0f}")
        else:
            votes.append(0); reasons.append(f"RSI neutral {rsi:.0f}")

        score = sum(votes) / len(votes)
        confidence = min(1.0, abs(score) + 0.25)
        return Signal(score=score, confidence=confidence, reason="; ".join(reasons))
