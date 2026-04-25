"""Abstract broker surface — PaperBroker and OkxBroker both implement this."""
from __future__ import annotations

from typing import Dict, List, Optional, Protocol, runtime_checkable

from .broker import LivePosition, LiveTrade


@runtime_checkable
class BrokerBase(Protocol):
    """Minimum surface the LiveTrader calls into.

    Both paper and OKX-demo brokers implement this so the trader loop is
    broker-agnostic. State mutation happens inside the broker.
    """

    # --- read-only views -------------------------------------------------
    @property
    def open_positions(self) -> Dict[str, LivePosition]: ...
    @property
    def closed_trades(self) -> List[LiveTrade]: ...
    @property
    def cash(self) -> float: ...
    @property
    def starting_equity(self) -> float: ...

    # --- mark-to-market --------------------------------------------------
    def equity(self, marks: Dict[str, float]) -> float: ...
    def record_equity(self, marks: Dict[str, float]) -> None: ...

    # --- order routing ---------------------------------------------------
    def open(
        self,
        symbol: str,
        direction: int,
        price: float,
        stop: float,
        target: float,
        equity_now: float,
        rationale: str,
        leverage: int = 1,
    ) -> Optional[LivePosition]: ...

    def close(self, symbol: str, price: float, reason: str) -> Optional[LiveTrade]: ...

    def update_stop(self, symbol: str, new_stop: float) -> bool:
        """Tighten the stop on an open position. Implementations should
        update local state and (if applicable) replace the exchange-side
        stop order. Return True on success.
        """
        ...
