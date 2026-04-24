"""Sentiment/flow agent: volume surges + price-action regime."""
from __future__ import annotations

from .technical import Signal


class SentimentAgent:
    name = "sentiment"

    def evaluate(self, row) -> Signal:
        votes = []
        reasons = []

        # Bollinger position: squeeze-breakout reading
        bb_pct = row["bb_pct"]
        if bb_pct > 0.95:
            votes.append(-0.5); reasons.append("near upper band (euphoria)")
        elif bb_pct < 0.05:
            votes.append(0.5); reasons.append("near lower band (panic)")
        elif bb_pct > 0.6:
            votes.append(0.5); reasons.append("upper half — bulls driving")
        elif bb_pct < 0.4:
            votes.append(-0.5); reasons.append("lower half — bears pressing")
        else:
            votes.append(0); reasons.append("mid-band drift")

        # Volume confirmation
        vr = row["vol_ratio"]
        ret = row["ret_1"]
        if vr > 1.5 and ret > 0:
            votes.append(1); reasons.append(f"vol surge {vr:.1f}x with green bar")
        elif vr > 1.5 and ret < 0:
            votes.append(-1); reasons.append(f"vol surge {vr:.1f}x with red bar")
        elif vr < 0.7:
            votes.append(0); reasons.append("thin volume — low conviction")
        else:
            votes.append(0.2 if ret > 0 else -0.2); reasons.append("normal volume")

        # Short-term momentum
        r5 = row["ret_5"]
        if r5 > 0.02:
            votes.append(0.5); reasons.append(f"5-bar +{r5*100:.1f}%")
        elif r5 < -0.02:
            votes.append(-0.5); reasons.append(f"5-bar {r5*100:.1f}%")

        score = sum(votes) / len(votes)
        confidence = min(1.0, abs(score) + 0.2)
        return Signal(score=score, confidence=confidence, reason="; ".join(reasons))
