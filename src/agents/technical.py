"""Technical agent: coherent trend-following with RSI as a confidence modulator.

Old version mixed trend signals (SMA/EMA/MACD) with mean-reversion (RSI <30
= buy). In a clear downtrend with RSI<30 the votes cancelled, producing
near-zero signal at exactly the wrong moment. This version:

  * Primary signal = trend strength = (SMA20-SMA50)/close, expressed in ATRs.
  * MACD histogram is a CONFIRMATION (only counts when it agrees with trend).
  * RSI is used to MODULATE conviction within the trend, not flip direction:
      - In uptrend, RSI 30-50 (pullback) = strongest long conviction.
      - In uptrend, RSI > 75 (extended) = caution, reduce conviction.
      - In downtrend, mirror.
  * Confidence is dampened in choppy markets (low |trend|).
"""
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
        close = float(row["close"])
        atr = float(row["atr"])
        sma_fast = float(row["sma_fast"])
        sma_slow = float(row["sma_slow"])
        sma_200 = float(row.get("sma_200", close)) or close
        macd_h = float(row["macd_hist"])
        rsi = float(row["rsi"])

        if atr <= 0 or close <= 0:
            return Signal(0.0, 0.0, "tech: insufficient data")

        # Trend strength in ATR units. >+2 ATRs = strong uptrend; <-2 = strong downtrend.
        trend_atrs = (sma_fast - sma_slow) / atr
        # HARD chop veto — abandon the trade entirely if SMAs aren't separated
        # enough. Anything under 1.0 ATR of separation is noise on intraday
        # crypto, and our historical data showed entries here are net losers.
        if abs(trend_atrs) < 1.0:
            return Signal(0.0, 0.0, f"chop trend={trend_atrs:+.2f}ATR — skip")
        # Clamp to [-1, +1] at 2.5 ATRs of separation.
        trend_score = max(-1.0, min(1.0, trend_atrs / 2.5))

        reasons = [f"trend={trend_atrs:+.2f}ATR"]

        # MACD confirmation — adds only if directionally consistent.
        macd_factor = 1.0
        if macd_h > 0 and trend_score > 0:
            macd_factor = 1.2
            reasons.append("MACD+ confirms")
        elif macd_h < 0 and trend_score < 0:
            macd_factor = 1.2
            reasons.append("MACD- confirms")
        elif macd_h * trend_score < 0:
            macd_factor = 0.6  # divergence — fade conviction
            reasons.append("MACD diverges")
        else:
            reasons.append("MACD flat")

        # RSI modulation — pullbacks in trend are the gold setup.
        rsi_factor = 1.0
        if trend_score > 0.3:                     # uptrend
            if 30 <= rsi <= 50:
                rsi_factor = 1.4                 # pullback in uptrend
                reasons.append(f"RSI {rsi:.0f} pullback")
            elif rsi > 75:
                rsi_factor = 0.5                 # extended
                reasons.append(f"RSI {rsi:.0f} extended")
            elif rsi < 30:
                rsi_factor = 1.6                 # deep pullback
                reasons.append(f"RSI {rsi:.0f} deep pullback")
            else:
                reasons.append(f"RSI {rsi:.0f}")
        elif trend_score < -0.3:                  # downtrend
            if 50 <= rsi <= 70:
                rsi_factor = 1.4
                reasons.append(f"RSI {rsi:.0f} rally in downtrend")
            elif rsi < 25:
                rsi_factor = 0.5
                reasons.append(f"RSI {rsi:.0f} oversold")
            elif rsi > 70:
                rsi_factor = 1.6
                reasons.append(f"RSI {rsi:.0f} sharp rally")
            else:
                reasons.append(f"RSI {rsi:.0f}")
        # If we made it past the chop veto, trend_score >= 0.4 — no else needed.

        # Macro-trend filter: heavy penalty for taking signals AGAINST the
        # 200-bar trend. Counter-macro trades have systematically lower edge.
        macro_factor = 1.0
        if trend_score > 0 and close < sma_200:
            macro_factor = 0.4
            reasons.append("vs macro down — fade")
        elif trend_score < 0 and close > sma_200:
            macro_factor = 0.4
            reasons.append("vs macro up — fade")
        elif trend_score > 0 and close > sma_200:
            macro_factor = 1.3
            reasons.append("with macro up")
        elif trend_score < 0 and close < sma_200:
            macro_factor = 1.3
            reasons.append("with macro down")

        score = trend_score
        confidence = min(1.0, abs(trend_score) * macd_factor * rsi_factor * macro_factor)
        return Signal(score=score, confidence=confidence, reason="; ".join(reasons))
