"""Decision agent: weighted debate over the other agents' signals."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .technical import Signal


@dataclass
class Decision:
    action: str          # 'BUY', 'SELL', 'HOLD'
    size: float          # 0..1 fraction of equity
    score: float
    rationale: str


class DecisionAgent:
    """Collective-intelligence aggregator.

    Weights each sub-agent, applies the risk agent as a veto/attenuator, and
    emits BUY/SELL/HOLD plus a position-size suggestion.
    """

    def __init__(
        self,
        weights: Dict[str, float] | None = None,
        entry_threshold: float = 0.35,
        risk_veto: float = -0.6,
    ):
        self.weights = weights or {"technical": 0.55, "sentiment": 0.45}
        self.entry_threshold = entry_threshold
        self.risk_veto = risk_veto

    def debate(self, signals: Dict[str, Signal]) -> Decision:
        risk = signals.get("risk")
        if risk is not None and risk.score <= self.risk_veto:
            return Decision(
                action="HOLD",
                size=0.0,
                score=0.0,
                rationale=f"RISK VETO: {risk.reason}",
            )

        weighted = 0.0
        conf_sum = 0.0
        lines: List[str] = []
        for name, w in self.weights.items():
            sig = signals.get(name)
            if sig is None:
                continue
            weighted += w * sig.score * sig.confidence
            conf_sum += w * sig.confidence
            lines.append(f"[{name} w={w:.2f} s={sig.score:+.2f} c={sig.confidence:.2f}] {sig.reason}")

        score = weighted / conf_sum if conf_sum > 0 else 0.0

        # Risk agent attenuates size (doesn't veto unless extreme — handled above)
        size_mult = 1.0
        if risk is not None:
            size_mult = max(0.25, 1.0 + 0.5 * risk.score)  # -0.6..+0.5 -> 0.7..1.25

        if score >= self.entry_threshold:
            action = "BUY"
        elif score <= -self.entry_threshold:
            action = "SELL"
        else:
            action = "HOLD"

        size = min(1.0, abs(score)) * size_mult if action != "HOLD" else 0.0
        rationale = " | ".join(lines)
        if risk is not None:
            rationale += f" || risk={risk.score:+.2f} size_mult={size_mult:.2f}"
        return Decision(action=action, size=size, score=score, rationale=rationale)
