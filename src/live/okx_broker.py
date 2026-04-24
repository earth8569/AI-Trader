"""OKX demo-mode broker — spot, long-only.

Design notes
------------
* Uses OKX v5 REST. Demo mode is enabled via the `x-simulated-trading: 1`
  header set in `okx_auth.request()`. Real capital is not at risk.
* Spot only for Scope A — shorts require SWAP/margin and bring
  leverage + liquidation risk. Disable `allow_shorts` when using this broker.
* Positions and trades are kept in a LOCAL JSON state file because OKX
  doesn't natively track "open/close pairs" — only orders and fills.
  On startup we query OKX for spot balances and reconcile equity; open
  positions opened outside the bot won't appear here.
"""
from __future__ import annotations

import json
import logging
import math
import os
import tempfile
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from .broker import LivePosition, LiveTrade, PortfolioState
from .okx_auth import OkxCreds, request
from .okx_feed import symbol_to_okx

log = logging.getLogger("ai-trader.okx")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class OkxBroker:
    """Spot, long-only, demo mode. Implements the BrokerBase protocol."""

    def __init__(
        self,
        state_path: str | Path = "state/okx_portfolio.json",
        quote_ccy: str = "USDT",
        creds: Optional[OkxCreds] = None,
    ):
        self.state_path = Path(state_path)
        self.quote_ccy = quote_ccy
        self.creds = creds or OkxCreds.from_env()
        self._inst_cache: Dict[str, dict] = {}
        self.state = self._load()
        if not self.creds.demo:
            log.warning("OkxBroker running in LIVE mode — real money at risk")
        else:
            log.info("OkxBroker running in DEMO mode (simulated trading)")
        self._sync_cash_from_exchange()

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
        fd, tmp = tempfile.mkstemp(prefix="okx_portfolio_", suffix=".json", dir=self.state_path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(state.to_dict(), f, indent=2, default=str)
            os.replace(tmp, self.state_path)
        except Exception:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise

    # ---- OKX calls -------------------------------------------------------
    def _get_instrument(self, instrument: str) -> dict:
        inst_id = symbol_to_okx(instrument)
        if inst_id in self._inst_cache:
            return self._inst_cache[inst_id]
        payload = request(
            "GET", "/api/v5/public/instruments",
            self.creds, params={"instType": "SPOT", "instId": inst_id},
        )
        data = payload.get("data") or []
        if not data:
            raise RuntimeError(f"OKX: no instrument info for {inst_id}")
        self._inst_cache[inst_id] = data[0]
        return data[0]

    def _round_size(self, instrument: str, units: float) -> float:
        info = self._get_instrument(instrument)
        lot = float(info.get("lotSz") or "0") or float(info.get("minSz") or "0.00000001")
        min_sz = float(info.get("minSz") or "0")
        if lot > 0:
            units = math.floor(units / lot) * lot
        if units < min_sz:
            return 0.0
        # de-noise floating precision
        return float(f"{units:.10f}")

    def _sync_cash_from_exchange(self):
        """Pull USDT available balance from OKX and set it as our 'cash'."""
        try:
            payload = request("GET", "/api/v5/account/balance", self.creds, params={"ccy": self.quote_ccy})
            details = (payload["data"][0] or {}).get("details") if payload.get("data") else []
            for d in details:
                if d.get("ccy") == self.quote_ccy:
                    avail = float(d.get("availBal") or d.get("cashBal") or 0)
                    self.state.cash = avail
                    if self.state.starting_equity <= 0:
                        self.state.starting_equity = avail
                    self._save()
                    log.info("OKX cash synced: %.2f %s", avail, self.quote_ccy)
                    return
        except Exception as e:
            log.warning("failed to sync OKX cash: %s", e)

    def _place_market(self, instrument: str, side: str, size_base: float) -> dict:
        """Market order, size denominated in BASE currency (tgtCcy=base_ccy)."""
        inst_id = symbol_to_okx(instrument)
        body = {
            "instId": inst_id,
            "tdMode": "cash",
            "side": side,
            "ordType": "market",
            "sz": f"{size_base:.10f}".rstrip("0").rstrip("."),
            "tgtCcy": "base_ccy",
        }
        log.info("OKX order -> %s", body)
        payload = request("POST", "/api/v5/trade/order", self.creds, body=body)
        row = (payload.get("data") or [{}])[0]
        ord_id = row.get("ordId")
        # wait briefly for fill then pull order details for avg price
        avg_px = None
        for _ in range(5):
            time.sleep(0.3)
            try:
                det = request(
                    "GET", "/api/v5/trade/order",
                    self.creds, params={"instId": inst_id, "ordId": ord_id},
                )
                d = (det.get("data") or [{}])[0]
                if d.get("state") == "filled" and d.get("avgPx"):
                    avg_px = float(d["avgPx"])
                    break
            except Exception as e:
                log.warning("order poll failed: %s", e)
        return {"ord_id": ord_id, "avg_px": avg_px, "raw": row}

    # ---- broker surface --------------------------------------------------
    def equity(self, marks: Dict[str, float]) -> float:
        eq = self.state.cash
        for sym, pos in self.state.positions.items():
            mark = marks.get(sym, pos.entry_price)
            eq += pos.direction * (mark - pos.entry_price) * pos.size_units + pos.notional
        return eq

    def record_equity(self, marks: Dict[str, float]):
        eq = self.equity(marks)
        self.state.equity_history.append({"ts": _now_iso(), "equity": eq})
        if len(self.state.equity_history) > 10_000:
            self.state.equity_history = self.state.equity_history[-10_000:]
        self._save()

    def position_size(self, equity: float, entry: float, stop: float) -> float:
        risk_quote = equity * self.state.risk_per_trade
        per_unit = abs(entry - stop)
        if per_unit <= 0:
            return 0.0
        units_by_risk = risk_quote / per_unit
        units_by_cap = (equity * self.state.max_position_pct) / entry
        units_by_cash = self.state.cash / entry
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
        leverage: int = 1,  # ignored — spot is always 1x
    ) -> Optional[LivePosition]:
        if direction != 1:
            log.info("[%s] SHORT ignored — spot broker is long-only", symbol)
            return None
        if symbol in self.state.positions:
            return None
        units_raw = self.position_size(equity_now, price, stop)
        units = self._round_size(symbol, units_raw)
        if units <= 0:
            log.info("[%s] size %.6f below min after rounding, skipping", symbol, units_raw)
            return None
        fill = self._place_market(symbol, "buy", units)
        entry_price = fill["avg_px"] or price
        notional = units * entry_price
        pos = LivePosition(
            id=fill["ord_id"] or str(uuid.uuid4()),
            symbol=symbol,
            direction=1,
            entry_ts=_now_iso(),
            entry_price=entry_price,
            size_units=units,
            stop=stop,
            target=target,
            notional=notional,
            rationale=rationale,
        )
        self.state.cash -= notional
        self.state.positions[symbol] = pos
        self._save()
        log.info("[%s] OPENED LONG units=%.6f @ %.4f (order=%s)", symbol, units, entry_price, pos.id)
        return pos

    def close(self, symbol: str, price: float, reason: str) -> Optional[LiveTrade]:
        pos = self.state.positions.get(symbol)
        if pos is None:
            return None
        units = self._round_size(symbol, pos.size_units)
        if units <= 0:
            log.warning("[%s] close size rounds to 0, orphaning", symbol)
            del self.state.positions[symbol]
            self._save()
            return None
        fill = self._place_market(symbol, "sell", units)
        exit_price = fill["avg_px"] or price
        pnl = (exit_price - pos.entry_price) * pos.size_units
        pnl_pct = pnl / pos.notional if pos.notional > 0 else 0.0
        trade = LiveTrade(
            id=pos.id,
            symbol=pos.symbol,
            direction=1,
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
        self.state.cash += pos.notional + pnl
        del self.state.positions[symbol]
        self.state.closed_trades.append(trade)
        self._save()
        log.info(
            "[%s] CLOSED LONG units=%.6f @ %.4f reason=%s pnl=%+.2f",
            symbol, units, exit_price, reason, pnl,
        )
        return trade

    def reset(self, starting_equity: Optional[float] = None):
        """Reset LOCAL state only — does NOT liquidate OKX positions."""
        self.state = PortfolioState(
            starting_equity=starting_equity or self.state.starting_equity,
            cash=starting_equity or self.state.starting_equity,
        )
        self._save()
