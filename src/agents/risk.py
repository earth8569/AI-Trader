"""Risk agent: gates trades on volatility and acts as a veto."""
from __future__ import annotations

from .technical import Signal


class RiskAgent:
    name = "risk"

    def __init__(self, atr_mult_stop: float = 2.0, atr_mult_target: float = 3.0):
        self.atr_mult_stop = atr_mult_stop
        self.atr_mult_target = atr_mult_target

    def evaluate(self, row) -> Signal:
        reasons = []
        score = 0.0

        atr = row["atr"]
        price = row["close"]
        if price <= 0 or atr != atr:  # NaN guard
            return Signal(score=0, confidence=0, reason="risk: insufficient data")

        atr_pct = atr / price
        # Penalise extreme volatility regimes (would blow stops immediately).
        if atr_pct > 0.05:
            score -= 0.7
            reasons.append(f"ATR/P {atr_pct*100:.2f}% too hot — reduce exposure")
        elif atr_pct < 0.003:
            score -= 0.3
            reasons.append("ATR/P dead — chop risk")
        else:
            score += 0.3
            reasons.append(f"ATR/P {atr_pct*100:.2f}% tradeable")

        # RSI extremes increase reversal risk for trend entries
        rsi = row["rsi"]
        if rsi > 80 or rsi < 20:
            score -= 0.4
            reasons.append(f"RSI extreme {rsi:.0f} — mean-reversion risk")

        confidence = min(1.0, abs(score) + 0.3)
        return Signal(score=score, confidence=confidence, reason="; ".join(reasons))

    def stop_and_target(self, entry: float, atr: float, direction: int):
        stop = entry - direction * self.atr_mult_stop * atr
        target = entry + direction * self.atr_mult_target * atr
        return stop, target
