"""Sentiment / flow agent: regime-aware reading of volume + Bollinger + momentum.

Previous version blended Bollinger (mean-reversion: above upper = sell) with
volume-confirmed momentum (trend: green bar with volume = buy). In the same
candle these would often cancel.

This version routes by the volatility regime:
  * High ATR/price (trending): volume-confirmed momentum dominates.
  * Low ATR/price (range): Bollinger extremes dominate (fade).
"""
from __future__ import annotations

from .technical import Signal


class SentimentAgent:
    name = "sentiment"

    # ATR/price thresholds defining regimes
    TREND_ATR_PCT = 0.015     # >1.5% ATR/price → treat as trending
    RANGE_ATR_PCT = 0.008     # <0.8% ATR/price → treat as range-bound

    def evaluate(self, row) -> Signal:
        close = float(row["close"])
        atr = float(row["atr"])
        if close <= 0 or atr <= 0:
            return Signal(0.0, 0.0, "sentiment: insufficient data")

        atr_pct = atr / close
        bb_pct = float(row["bb_pct"])
        vr = float(row["vol_ratio"])
        ret_1 = float(row["ret_1"])
        ret_5 = float(row["ret_5"])

        reasons: list[str] = []

        if atr_pct >= self.TREND_ATR_PCT:
            # ---- trending regime: follow volume-confirmed momentum --------
            reasons.append(f"trending ATR%={atr_pct*100:.2f}%")
            score = 0.0
            if vr > 1.3 and ret_1 > 0:
                score += 0.6
                reasons.append(f"vol surge {vr:.1f}x +bar")
            elif vr > 1.3 and ret_1 < 0:
                score -= 0.6
                reasons.append(f"vol surge {vr:.1f}x -bar")
            # 5-bar momentum scaled
            mom = max(-0.4, min(0.4, ret_5 * 8))   # 5% in 5 bars -> 0.4
            score += mom
            reasons.append(f"5-bar {ret_5*100:+.1f}%")
            confidence = min(1.0, abs(score) + 0.25)
            return Signal(score=max(-1.0, min(1.0, score)), confidence=confidence,
                          reason="; ".join(reasons))

        if atr_pct <= self.RANGE_ATR_PCT:
            # ---- range regime: fade Bollinger extremes -----------------
            reasons.append(f"range ATR%={atr_pct*100:.2f}%")
            if bb_pct >= 0.95:
                return Signal(score=-0.6, confidence=0.65,
                              reason="; ".join(reasons + ["BB upper — fade"]))
            if bb_pct <= 0.05:
                return Signal(score=0.6, confidence=0.65,
                              reason="; ".join(reasons + ["BB lower — bid"]))
            return Signal(score=0.0, confidence=0.2,
                          reason="; ".join(reasons + [f"BB mid {bb_pct:.2f}"]))

        # ---- in-between regime: contribute nothing, let other agents lead.
        # Forcing a low-confidence signal here was creating noise in the
        # backtest — better to abstain.
        return Signal(score=0.0, confidence=0.0,
                      reason=f"transition ATR%={atr_pct*100:.2f}% — abstain")
