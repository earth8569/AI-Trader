"""Tunable agent parameters — the thing the walk-forward optimizer fits."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional


@dataclass
class AgentParams:
    technical_weight: float = 0.45
    sentiment_weight: float = 0.35
    market_intel_weight: float = 0.20      # funding-rate contrarian voter
    entry_threshold: float = 0.35
    risk_veto: float = -0.6
    atr_mult_stop: float = 2.0
    atr_mult_target: float = 3.0
    # leverage sizing — only used by futures/swap brokers; spot ignores these
    max_leverage: int = 5
    leverage_vol_ref_pct: float = 0.02   # ATR/price regarded as "normal"
    # trailing-stop knobs — tunable by the optimizer
    # Trail late + wide so the rare 5-10R outlier moves keep going. The
    # strategy makes its year on those — tight trailing kills the tail.
    trail_activation_r: float = 3.0
    trail_distance_atr: float = 3.0

    def normalized(self) -> "AgentParams":
        """Force voter weights to sum to 1.0 so DecisionAgent's weighted
        average stays on the same scale regardless of grid choices."""
        s = self.technical_weight + self.sentiment_weight + self.market_intel_weight
        if s <= 0:
            return replace(
                self,
                technical_weight=0.45,
                sentiment_weight=0.35,
                market_intel_weight=0.20,
            )
        return replace(
            self,
            technical_weight=self.technical_weight / s,
            sentiment_weight=self.sentiment_weight / s,
            market_intel_weight=self.market_intel_weight / s,
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AgentParams":
        return cls(**{k: v for k, v in d.items() if k in cls.__annotations__})


@dataclass
class ParamSnapshot:
    """A chosen param set plus provenance — what produced it."""
    params: AgentParams
    score: float
    trades: int
    symbol: str
    interval: str
    window_start: str
    window_end: str
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "params": self.params.to_dict(),
            "score": self.score,
            "trades": self.trades,
            "symbol": self.symbol,
            "interval": self.interval,
            "window_start": self.window_start,
            "window_end": self.window_end,
            "updated_at": self.updated_at,
        }


def save_best(snap: ParamSnapshot, path: str | Path = "state/best_params.json", keep_history: int = 20):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    history: List[dict] = []
    if p.exists():
        try:
            prev = json.loads(p.read_text(encoding="utf-8"))
            history = prev.get("history", [])
            if "current" in prev:
                history.insert(0, prev["current"])
            history = history[:keep_history]
        except Exception:
            history = []
    payload = {"current": snap.to_dict(), "history": history}
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_best(path: str | Path = "state/best_params.json") -> Optional[AgentParams]:
    p = Path(path)
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        cur = d.get("current") or {}
        return AgentParams.from_dict(cur.get("params") or {})
    except Exception:
        return None


def load_snapshot(path: str | Path = "state/best_params.json") -> Optional[dict]:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
