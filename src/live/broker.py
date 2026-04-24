"""Paper broker with persistent JSON state.

Tracks cash, open positions, closed trades, and an equity curve. All state is
persisted atomically so the trader can be killed/restarted without corruption.
"""
from __future__ import annotations

import json
import os
import tempfile
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class LivePosition:
    id: str
    symbol: str
    direction: int          # +1 long, -1 short
    entry_ts: str
    entry_price: float
    size_units: float
    stop: float
    target: float
    notional: float
    rationale: str = ""
    leverage: int = 1       # 1x = spot / no leverage; >1 = futures
    margin_used: float = 0.0  # quote currency margin locked (= notional / leverage)


@dataclass
class LiveTrade:
    id: str
    symbol: str
    direction: int
    entry_ts: str
    exit_ts: str
    entry_price: float
    exit_price: float
    size_units: float
    pnl_quote: float
    pnl_pct: float
    exit_reason: str
    rationale: str = ""


@dataclass
class PortfolioState:
    starting_equity: float = 10_000.0
    cash: float = 10_000.0
    fee_bps: float = 5.0
    slippage_bps: float = 2.0
    risk_per_trade: float = 0.02
    max_position_pct: float = 0.35   # cap notional per symbol to 35% of equity
    positions: Dict[str, LivePosition] = field(default_factory=dict)  # keyed by symbol
    closed_trades: List[LiveTrade] = field(default_factory=list)
    equity_history: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "starting_equity": self.starting_equity,
            "cash": self.cash,
            "fee_bps": self.fee_bps,
            "slippage_bps": self.slippage_bps,
            "risk_per_trade": self.risk_per_trade,
            "max_position_pct": self.max_position_pct,
            "positions": {k: asdict(v) for k, v in self.positions.items()},
            "closed_trades": [asdict(t) for t in self.closed_trades],
            "equity_history": self.equity_history,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PortfolioState":
        s = cls(
            starting_equity=d.get("starting_equity", 10_000.0),
            cash=d.get("cash", d.get("starting_equity", 10_000.0)),
            fee_bps=d.get("fee_bps", 5.0),
            slippage_bps=d.get("slippage_bps", 2.0),
            risk_per_trade=d.get("risk_per_trade", 0.02),
            max_position_pct=d.get("max_position_pct", 0.35),
        )
        s.positions = {k: LivePosition(**v) for k, v in (d.get("positions") or {}).items()}
        s.closed_trades = [LiveTrade(**t) for t in (d.get("closed_trades") or [])]
        s.equity_history = d.get("equity_history") or []
        return s


class PaperBroker:
    """Paper broker — all orders are simulated against the mid/close price."""

    def __init__(self, state_path: str | Path = "state/portfolio.json"):
        self.state_path = Path(state_path)
        self.state = self._load()

    # ---- protocol accessors ---------------------------------------------
    @property
    def open_positions(self) -> Dict[str, LivePosition]:
        return self.state.positions

    @property
    def closed_trades(self) -> List[LiveTrade]:
        return self.state.closed_trades

    @property
    def cash(self) -> float:
        return self.state.cash

    @property
    def starting_equity(self) -> float:
        return self.state.starting_equity

    # ---- persistence -----------------------------------------------------
    def _load(self) -> PortfolioState:
        if self.state_path.exists():
            with open(self.state_path, "r", encoding="utf-8") as f:
                return PortfolioState.from_dict(json.load(f))
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        state = PortfolioState()
        self._save(state)
        return state

    def _save(self, state: Optional[PortfolioState] = None):
        state = state or self.state
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix="portfolio_", suffix=".json", dir=self.state_path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(state.to_dict(), f, indent=2, default=str)
            os.replace(tmp, self.state_path)
        except Exception:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise

    # ---- helpers ---------------------------------------------------------
    def _apply_fees(self, price: float, side: int) -> float:
        slip = self.state.slippage_bps / 10_000
        fee = self.state.fee_bps / 10_000
        return price * (1 + side * (slip + fee))

    def equity(self, marks: Dict[str, float]) -> float:
        eq = self.state.cash
        for sym, pos in self.state.positions.items():
            mark = marks.get(sym, pos.entry_price)
            eq += pos.direction * (mark - pos.entry_price) * pos.size_units + pos.notional
        return eq

    # ---- trading ---------------------------------------------------------
    def position_size(self, equity: float, entry: float, stop: float, leverage: int = 1) -> float:
        """Return base-currency units to trade.

        Risk (stop-distance × units) is NOT multiplied by leverage — a held
        stop caps loss regardless. Leverage expands the notional and cash caps,
        since margin required is notional / leverage.
        """
        risk_quote = equity * self.state.risk_per_trade
        per_unit = abs(entry - stop)
        if per_unit <= 0:
            return 0.0
        lev = max(1, int(leverage))
        units_by_risk = risk_quote / per_unit
        units_by_cap = (equity * self.state.max_position_pct * lev) / entry
        units_by_cash = (self.state.cash * lev) / entry
        return max(0.0, min(units_by_risk, units_by_cap, units_by_cash))

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
    ) -> Optional[LivePosition]:
        if symbol in self.state.positions:
            return None
        lev = max(1, int(leverage))
        entry_price = self._apply_fees(price, direction)
        units = self.position_size(equity_now, entry_price, stop, leverage=lev)
        if units <= 0:
            return None
        notional = units * entry_price
        margin = notional / lev
        pos = LivePosition(
            id=str(uuid.uuid4()),
            symbol=symbol,
            direction=direction,
            entry_ts=_now_iso(),
            entry_price=entry_price,
            size_units=units,
            stop=stop,
            target=target,
            notional=notional,
            rationale=rationale,
            leverage=lev,
            margin_used=margin,
        )
        # in a paper/spot world we lock the whole notional; in a leveraged
        # world we only lock the margin. Use margin here so leverage frees
        # cash for other positions.
        self.state.cash -= margin
        self.state.positions[symbol] = pos
        self._save()
        return pos

    def close(self, symbol: str, price: float, reason: str) -> Optional[LiveTrade]:
        pos = self.state.positions.get(symbol)
        if pos is None:
            return None
        exit_side = -pos.direction
        exit_price = self._apply_fees(price, exit_side)
        pnl = pos.direction * (exit_price - pos.entry_price) * pos.size_units
        margin = pos.margin_used or pos.notional  # back-compat
        pnl_pct = pnl / margin if margin > 0 else 0.0
        trade = LiveTrade(
            id=pos.id,
            symbol=pos.symbol,
            direction=pos.direction,
            entry_ts=pos.entry_ts,
            exit_ts=_now_iso(),
            entry_price=pos.entry_price,
            exit_price=exit_price,
            size_units=pos.size_units,
            pnl_quote=pnl,
            pnl_pct=pnl_pct,
            exit_reason=reason,
            rationale=pos.rationale,
        )
        # release margin + pnl
        self.state.cash += margin + pnl
        del self.state.positions[symbol]
        self.state.closed_trades.append(trade)
        self._save()
        return trade

    def record_equity(self, marks: Dict[str, float]):
        eq = self.equity(marks)
        self.state.equity_history.append({"ts": _now_iso(), "equity": eq})
        # cap history length to keep state file reasonable
        if len(self.state.equity_history) > 10_000:
            self.state.equity_history = self.state.equity_history[-10_000:]
        self._save()

    def reset(self, starting_equity: Optional[float] = None):
        if starting_equity is not None:
            self.state = PortfolioState(starting_equity=starting_equity, cash=starting_equity)
        else:
            self.state = PortfolioState(
                starting_equity=self.state.starting_equity,
                cash=self.state.starting_equity,
            )
        self._save()
