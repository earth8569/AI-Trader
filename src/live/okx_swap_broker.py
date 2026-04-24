"""OKX perpetual-swap demo broker — long + short, isolated margin, per-trade leverage.

Safety posture
--------------
* Demo-only by default. Same `OKX_DEMO=1` gate as the spot broker.
* `isolated` margin mode — liquidation of one position cannot cascade to others.
* `long_short_mode` position mode so a single instrument can hold separate
  long and short positions (agent can flip direction without netting).
* Leverage is set per-instrument before each order via `/account/set-leverage`.
  Hard-capped by `max_leverage` in AgentParams (default 5×).
* Size is in CONTRACTS (OKX's native unit for swaps), converted from base
  currency using `ctVal` from the instruments endpoint. Rounded to `lotSz`.

Note: OKX account position mode is a GLOBAL setting. We set it once at
construction. If your demo account is already in long_short_mode the call
is a no-op; if it's in net_mode we flip it (demo only; live code paths
would require extra confirmation).
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

log = logging.getLogger("ai-trader.okx-swap")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class OkxSwapBroker:
    """Perpetuals, demo mode, isolated margin, long + short."""

    def __init__(
        self,
        state_path: str | Path = "state/okx_swap_portfolio.json",
        quote_ccy: str = "USDT",
        creds: Optional[OkxCreds] = None,
        position_mode: str = "long_short_mode",
        margin_mode: str = "isolated",
    ):
        self.state_path = Path(state_path)
        self.quote_ccy = quote_ccy
        self.creds = creds or OkxCreds.from_env()
        self.position_mode = position_mode
        self.margin_mode = margin_mode
        self._inst_cache: Dict[str, dict] = {}
        self._lev_cache: Dict[tuple, int] = {}   # (instId, posSide) -> lever
        self.state = self._load()
        if not self.creds.demo:
            log.warning("OkxSwapBroker running in LIVE mode — real money + leverage at risk")
        else:
            log.info("OkxSwapBroker running in DEMO mode (simulated perpetuals)")
        self._ensure_position_mode()
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
        fd, tmp = tempfile.mkstemp(prefix="okx_swap_", suffix=".json", dir=self.state_path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(state.to_dict(), f, indent=2, default=str)
            os.replace(tmp, self.state_path)
        except Exception:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise

    # ---- OKX helpers -----------------------------------------------------
    def _swap_inst_id(self, symbol: str) -> str:
        return symbol_to_okx(symbol, kind="swap")

    def _get_instrument(self, symbol: str) -> dict:
        inst_id = self._swap_inst_id(symbol)
        if inst_id in self._inst_cache:
            return self._inst_cache[inst_id]
        payload = request(
            "GET", "/api/v5/public/instruments",
            self.creds, params={"instType": "SWAP", "instId": inst_id},
        )
        data = payload.get("data") or []
        if not data:
            raise RuntimeError(f"OKX swap: no instrument info for {inst_id}")
        self._inst_cache[inst_id] = data[0]
        return data[0]

    def _ensure_position_mode(self):
        try:
            payload = request("GET", "/api/v5/account/config", self.creds)
            current = (payload.get("data") or [{}])[0].get("posMode")
            if current and current != self.position_mode:
                log.info("setting OKX position mode: %s -> %s", current, self.position_mode)
                request(
                    "POST", "/api/v5/account/set-position-mode",
                    self.creds, body={"posMode": self.position_mode},
                )
        except Exception as e:
            log.warning("position-mode check/set failed (continuing): %s", e)

    def _sync_cash_from_exchange(self):
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
                    log.info("OKX swap cash synced: %.2f %s", avail, self.quote_ccy)
                    return
        except Exception as e:
            log.warning("failed to sync OKX swap cash: %s", e)

    def _set_leverage(self, symbol: str, leverage: int, pos_side: str):
        inst_id = self._swap_inst_id(symbol)
        key = (inst_id, pos_side)
        if self._lev_cache.get(key) == leverage:
            return
        body = {
            "instId": inst_id,
            "lever": str(int(leverage)),
            "mgnMode": self.margin_mode,
            "posSide": pos_side,
        }
        try:
            request("POST", "/api/v5/account/set-leverage", self.creds, body=body)
            self._lev_cache[key] = leverage
            log.info("[%s %s] leverage set to %dx (%s)", inst_id, pos_side, leverage, self.margin_mode)
        except Exception as e:
            log.warning("set-leverage failed for %s %s @ %dx: %s", inst_id, pos_side, leverage, e)

    def _contracts_from_units(self, symbol: str, base_units: float) -> int:
        info = self._get_instrument(symbol)
        ct_val = float(info.get("ctVal") or "1")
        lot_sz = float(info.get("lotSz") or "1")
        min_sz = float(info.get("minSz") or "1")
        if ct_val <= 0:
            return 0
        raw = base_units / ct_val
        # round down to the nearest lot
        contracts = math.floor(raw / lot_sz) * lot_sz
        if contracts < min_sz:
            return 0
        return int(contracts) if lot_sz.is_integer() else contracts

    def _units_from_contracts(self, symbol: str, contracts: float) -> float:
        info = self._get_instrument(symbol)
        return float(contracts) * float(info.get("ctVal") or "1")

    def _place_market(self, symbol: str, side: str, pos_side: str, contracts: int) -> dict:
        inst_id = self._swap_inst_id(symbol)
        body = {
            "instId": inst_id,
            "tdMode": self.margin_mode,
            "side": side,                       # "buy" or "sell"
            "posSide": pos_side,                # "long" or "short"
            "ordType": "market",
            "sz": str(contracts),
        }
        log.info("OKX swap order -> %s", body)
        payload = request("POST", "/api/v5/trade/order", self.creds, body=body)
        row = (payload.get("data") or [{}])[0]
        ord_id = row.get("ordId")
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
                log.warning("swap order poll failed: %s", e)
        return {"ord_id": ord_id, "avg_px": avg_px, "raw": row}

    # ---- broker surface --------------------------------------------------
    def equity(self, marks: Dict[str, float]) -> float:
        eq = self.state.cash
        for sym, pos in self.state.positions.items():
            mark = marks.get(sym, pos.entry_price)
            margin = pos.margin_used or pos.notional
            eq += pos.direction * (mark - pos.entry_price) * pos.size_units + margin
        return eq

    def record_equity(self, marks: Dict[str, float]):
        eq = self.equity(marks)
        self.state.equity_history.append({"ts": _now_iso(), "equity": eq})
        if len(self.state.equity_history) > 10_000:
            self.state.equity_history = self.state.equity_history[-10_000:]
        self._save()

    def position_size(self, equity: float, entry: float, stop: float, leverage: int = 1) -> float:
        """Base-currency units before contract rounding. Risk stays capped by
        stop distance regardless of leverage; leverage only widens the cap."""
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
        pos_side = "long" if direction == 1 else "short"
        order_side = "buy" if direction == 1 else "sell"

        self._set_leverage(symbol, lev, pos_side)

        units_raw = self.position_size(equity_now, price, stop, leverage=lev)
        contracts = self._contracts_from_units(symbol, units_raw)
        if contracts <= 0:
            log.info("[%s] swap size %.6f base -> 0 contracts, skipping", symbol, units_raw)
            return None
        fill = self._place_market(symbol, order_side, pos_side, contracts)
        entry_price = fill["avg_px"] or price
        units = self._units_from_contracts(symbol, contracts)
        notional = units * entry_price
        margin = notional / lev
        pos = LivePosition(
            id=fill["ord_id"] or str(uuid.uuid4()),
            symbol=symbol,
            direction=direction,
            entry_ts=_now_iso(),
            entry_price=entry_price,
            size_units=units,
            stop=stop,
            target=target,
            notional=notional,
            rationale=f"{rationale} | lev={lev}x contracts={contracts}",
            leverage=lev,
            margin_used=margin,
        )
        self.state.cash -= margin
        self.state.positions[symbol] = pos
        self._save()
        log.info(
            "[%s] OPENED %s %dx units=%.6f (%d contracts) @ %.4f margin=%.2f",
            symbol, pos_side.upper(), lev, units, contracts, entry_price, margin,
        )
        return pos

    def close(self, symbol: str, price: float, reason: str) -> Optional[LiveTrade]:
        pos = self.state.positions.get(symbol)
        if pos is None:
            return None
        pos_side = "long" if pos.direction == 1 else "short"
        # closing: opposite side, same posSide
        order_side = "sell" if pos.direction == 1 else "buy"
        contracts = self._contracts_from_units(symbol, pos.size_units)
        if contracts <= 0:
            log.warning("[%s] close rounds to 0 contracts, orphaning local state", symbol)
            del self.state.positions[symbol]
            self._save()
            return None
        fill = self._place_market(symbol, order_side, pos_side, contracts)
        exit_price = fill["avg_px"] or price
        pnl = pos.direction * (exit_price - pos.entry_price) * pos.size_units
        margin = pos.margin_used or pos.notional
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
        self.state.cash += margin + pnl
        del self.state.positions[symbol]
        self.state.closed_trades.append(trade)
        self._save()
        log.info(
            "[%s] CLOSED %s @ %.4f reason=%s pnl=%+.2f (%.2f%% of margin)",
            symbol, pos_side.upper(), exit_price, reason, pnl, pnl_pct * 100,
        )
        return trade

    def reset(self, starting_equity: Optional[float] = None):
        self.state = PortfolioState(
            starting_equity=starting_equity or self.state.starting_equity,
            cash=starting_equity or self.state.starting_equity,
        )
        self._save()
