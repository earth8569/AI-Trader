"""MarketIntelAgent — funding-rate as a contrarian crowding signal.

OKX perpetual swaps charge funding every 8 hours. The sign and magnitude tell
you which side of the market is paying the other:

  funding > 0   longs pay shorts  -> longs are crowded
  funding < 0   shorts pay longs  -> shorts are crowded

Crowded sides tend to mean-revert. We map funding to a CONTRARIAN signal —
extreme positive funding biases short, extreme negative biases long. Mild
readings are near-neutral with low confidence so the agent doesn't dilute
the technical/sentiment vote when there's nothing interesting to say.

Funding rates only update every 8h on OKX, so we cache per symbol with a
~5min TTL to avoid hammering the public endpoint.
"""
from __future__ import annotations

import math
import time
from typing import Callable, Dict, Optional, Tuple

from .technical import Signal


class MarketIntelAgent:
    name = "market_intel"

    def __init__(
        self,
        fetcher: Optional[Callable[[str], Optional[float]]] = None,
        cache_ttl: int = 300,
    ):
        self.fetcher = fetcher
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, Tuple[float, Optional[float]]] = {}

    # ---- funding cache --------------------------------------------------
    def _funding(self, symbol: str) -> Optional[float]:
        if not self.fetcher or not symbol:
            return None
        now = time.time()
        cached = self._cache.get(symbol)
        if cached and now - cached[0] < self.cache_ttl:
            return cached[1]
        try:
            rate = self.fetcher(symbol)
        except Exception:
            rate = None
        self._cache[symbol] = (now, rate)
        return rate

    # ---- agent surface --------------------------------------------------
    def evaluate(self, row, symbol: Optional[str] = None) -> Signal:
        rate = self._funding(symbol) if symbol else None
        if rate is None or math.isnan(rate):
            # no data -> contribute zero (DecisionAgent re-weights without us)
            return Signal(score=0.0, confidence=0.0, reason="funding: unavailable")

        bps = rate * 10_000  # 0.0001 -> 1 bp; 8h-period bps
        # contrarian mapping calibrated for ranges typical of major perps
        if bps >= 5.0:        # ≥ 0.05% per 8h — heavy long crowding
            score, conf = -0.7, 0.85
            label = "EXTREME long crowding"
        elif bps >= 2.0:      # ≥ 0.02% — moderate
            score, conf = -0.3, 0.55
            label = "long-leaning"
        elif bps <= -5.0:
            score, conf = 0.7, 0.85
            label = "EXTREME short crowding"
        elif bps <= -2.0:
            score, conf = 0.3, 0.55
            label = "short-leaning"
        else:
            score, conf = 0.0, 0.2
            label = "neutral"
        return Signal(
            score=score,
            confidence=conf,
            reason=f"funding {bps:+.2f}bps/8h {label} (contrarian)",
        )
