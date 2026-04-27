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

import requests

from .broker import LivePosition, LiveTrade, PortfolioState
from .okx_auth import OKX_BASE, OkxCreds, request
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
        margin_buffer: float = 0.85,   # use ≤85% of reported availBal for sizing
    ):
        self.state_path = Path(state_path)
        self.quote_ccy = quote_ccy
        self.creds = creds or OkxCreds.from_env()
        self.position_mode = position_mode
        self.margin_mode = margin_mode
        self.margin_buffer = margin_buffer
        self._inst_cache: Dict[str, dict] = {}
        self._lev_cache: Dict[tuple, int] = {}   # (instId, posSide) -> lever
        self._margin_exhausted: bool = False     # set by 51008, reset each tick
        self.active_params_hash: str = ""
        self.state = self._load()

    def set_active_params_hash(self, h: str) -> None:
        self.active_params_hash = h or ""
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

    def _sync_cash_from_exchange(self, quiet: bool = False):
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
                    if not quiet:
                        log.info("OKX swap cash synced: %.2f %s", avail, self.quote_ccy)
                    return
        except Exception as e:
            log.warning("failed to sync OKX swap cash: %s", e)

    def refresh_balance(self):
        """Pull latest availBal from OKX and reset the per-tick exhausted flag."""
        self._margin_exhausted = False
        self._sync_cash_from_exchange(quiet=True)

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

    # ---- TP/SL (native OCO algo) ----------------------------------------
    def _place_oco_tp_sl(
        self,
        symbol: str,
        pos_side: str,
        contracts: int,
        tp_price: float,
        sl_price: float,
        trigger_type: str = "last",
    ) -> Optional[str]:
        """Attach an OCO algo order that will close the position at TP or SL.

        Returns the OKX algoId, or None on failure (caller decides whether
        to abort or proceed without exchange-side protection).
        """
        inst_id = self._swap_inst_id(symbol)
        # closing side is opposite of position direction
        order_side = "sell" if pos_side == "long" else "buy"

        # round to instrument tick size to avoid -51000 errors. Important:
        # decode decimals from the ORIGINAL tickSz string — float() turns
        # small ticks like "0.00001" into "1e-05" which breaks naive parsing.
        tick_str = str(self._get_instrument(symbol).get("tickSz") or "0.01")
        tick_sz = float(tick_str)
        if "e-" in tick_str.lower():
            decimals = int(tick_str.lower().split("e-")[1])
        elif "." in tick_str:
            decimals = len(tick_str.split(".")[-1].rstrip("0")) or len(tick_str.split(".")[-1])
        else:
            decimals = 0
        def _round_tick(p: float) -> str:
            n = round(p / tick_sz)
            return f"{n * tick_sz:.{decimals}f}"

        body = {
            "instId": inst_id,
            "tdMode": self.margin_mode,
            "side": order_side,
            "posSide": pos_side,
            "ordType": "oco",
            "sz": str(contracts),
            "reduceOnly": "true",
            "tpTriggerPx": _round_tick(tp_price),
            "tpOrdPx": "-1",                 # market on trigger
            "tpTriggerPxType": trigger_type,
            "slTriggerPx": _round_tick(sl_price),
            "slOrdPx": "-1",
            "slTriggerPxType": trigger_type,
        }
        try:
            payload = request("POST", "/api/v5/trade/order-algo", self.creds, body=body)
            algo_id = (payload.get("data") or [{}])[0].get("algoId") or ""
            log.info(
                "[%s %s] OCO placed algo=%s tp=%s sl=%s",
                inst_id, pos_side, algo_id, body["tpTriggerPx"], body["slTriggerPx"],
            )
            return algo_id or None
        except Exception as e:
            # 51250 (price out of range) often clears with mark-price triggers
            if "51250" in str(e) and trigger_type == "last":
                log.info("[%s %s] OCO retry with mark-price trigger", inst_id, pos_side)
                body["tpTriggerPxType"] = "mark"
                body["slTriggerPxType"] = "mark"
                try:
                    payload = request("POST", "/api/v5/trade/order-algo", self.creds, body=body)
                    algo_id = (payload.get("data") or [{}])[0].get("algoId") or ""
                    log.info("[%s %s] OCO placed (mark) algo=%s", inst_id, pos_side, algo_id)
                    return algo_id or None
                except Exception as e2:
                    log.warning("[%s %s] OCO TP/SL retry FAILED: %s", inst_id, pos_side, e2)
                    return None
            log.warning("[%s %s] OCO TP/SL placement FAILED: %s", inst_id, pos_side, e)
            return None

    def _cancel_algo(self, symbol: str, algo_id: str) -> bool:
        if not algo_id:
            return True
        inst_id = self._swap_inst_id(symbol)
        try:
            request(
                "POST", "/api/v5/trade/cancel-algos",
                self.creds, body=[{"algoId": algo_id, "instId": inst_id}],
            )
            log.info("[%s] cancelled algo %s", inst_id, algo_id)
            return True
        except Exception as e:
            # 51400 = algo already cancelled / triggered — fine
            log.info("[%s] algo cancel returned: %s (assuming already gone)", inst_id, e)
            return False

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
        """Base-currency units before contract rounding.

        Available cash is shrunk by `margin_buffer` to leave headroom for
        maintenance margin, fees, and unrealized PnL on existing positions.
        """
        risk_quote = equity * self.state.risk_per_trade
        per_unit = abs(entry - stop)
        if per_unit <= 0:
            return 0.0
        lev = max(1, int(leverage))
        usable_cash = max(0.0, self.state.cash * self.margin_buffer)
        units_by_risk = risk_quote / per_unit
        units_by_cap = (equity * self.state.max_position_pct * lev) / entry
        units_by_cash = (usable_cash * lev) / entry
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
        if self._margin_exhausted:
            log.info("[%s] skipping new entry — margin exhausted this tick", symbol)
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
        try:
            fill = self._place_market(symbol, order_side, pos_side, contracts)
        except RuntimeError as e:
            # OKX 51008 = insufficient margin/USDT balance — no point trying
            # more new entries this tick. Pull a fresh balance for next tick.
            if "51008" in str(e):
                self._margin_exhausted = True
                log.warning("[%s] MARGIN EXHAUSTED (51008) — skipping further opens this tick", symbol)
                return None
            raise
        entry_price = fill["avg_px"] or price
        units = self._units_from_contracts(symbol, contracts)
        notional = units * entry_price
        margin = notional / lev

        # Place exchange-side TP/SL so they survive bot downtime / fast moves.
        algo_id = self._place_oco_tp_sl(
            symbol=symbol,
            pos_side=pos_side,
            contracts=contracts,
            tp_price=target,
            sl_price=stop,
        ) or ""

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
            algo_id=algo_id,
            params_hash=self.active_params_hash,
        )
        self.state.cash -= margin
        self.state.positions[symbol] = pos
        self._save()
        log.info(
            "[%s] OPENED %s %dx units=%.6f (%d contracts) @ %.4f margin=%.2f algo=%s",
            symbol, pos_side.upper(), lev, units, contracts, entry_price, margin, algo_id or "NONE",
        )
        return pos

    def close(self, symbol: str, price: float, reason: str) -> Optional[LiveTrade]:
        pos = self.state.positions.get(symbol)
        if pos is None:
            return None
        # Cancel the resting OCO TP/SL first — otherwise it would fire on
        # whatever happens after our close and accidentally re-open us.
        if pos.algo_id:
            self._cancel_algo(symbol, pos.algo_id)
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
            leverage=pos.leverage,
            params_hash=pos.params_hash,
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

    # ---- trailing stop ---------------------------------------------------
    def _get_mark(self, symbol: str) -> Optional[float]:
        inst_id = self._swap_inst_id(symbol)
        try:
            r = requests.get(
                f"{OKX_BASE}/api/v5/market/ticker",
                params={"instId": inst_id}, timeout=10,
            )
            r.raise_for_status()
            data = (r.json().get("data") or [{}])[0]
            return float(data.get("last") or data.get("markPx") or 0) or None
        except Exception:
            return None

    def _clamp_levels(self, direction: int, stop: float, target: float,
                      mark: float, max_dev: float = 0.18):
        """Force stop/target to lie within ±max_dev of the current mark and
        on the correct side of it. Required because OKX rejects OCO triggers
        outside ~25% of the mark (51250).
        """
        if mark <= 0:
            return stop, target
        if direction == 1:                       # long
            stop = max(stop, mark * (1 - max_dev))
            stop = min(stop, mark * 0.999)
            target = min(target, mark * (1 + max_dev))
            target = max(target, mark * 1.001)
        else:                                    # short
            stop = min(stop, mark * (1 + max_dev))
            stop = max(stop, mark * 1.001)
            target = max(target, mark * (1 - max_dev))
            target = min(target, mark * 0.999)
        return stop, target

    def update_stop(self, symbol: str, new_stop: float) -> bool:
        """Place / replace the position's exchange-side OCO with a fresh
        TP+SL pair. Used for both orphan adoption (no prior algo) and
        trailing ratchet. Levels are clamped relative to the current mark
        so OKX won't reject them with 51250.
        """
        pos = self.state.positions.get(symbol)
        if pos is None:
            return False
        pos_side = "long" if pos.direction == 1 else "short"
        contracts = self._contracts_from_units(symbol, pos.size_units)
        if contracts <= 0:
            log.warning("[%s] update_stop: 0 contracts, skipping", symbol)
            return False

        mark = self._get_mark(symbol) or pos.entry_price
        clamped_stop, clamped_target = self._clamp_levels(
            pos.direction, new_stop, pos.target, mark,
        )
        had_algo = bool(pos.algo_id)
        if had_algo:
            self._cancel_algo(symbol, pos.algo_id)
        new_algo = self._place_oco_tp_sl(
            symbol=symbol,
            pos_side=pos_side,
            contracts=contracts,
            tp_price=clamped_target,
            sl_price=clamped_stop,
        )
        if new_algo is None:
            note = "OCO REPLACE FAILED — position is now NAKED" if had_algo \
                else "ORPHAN ADOPTION FAILED — position remains UNPROTECTED"
            log.warning("[%s] %s (mark=%.6f stop=%.6f tgt=%.6f)",
                        symbol, note, mark, clamped_stop, clamped_target)
            pos.algo_id = ""
            self._save()
            return False
        pos.stop = clamped_stop
        pos.target = clamped_target
        pos.algo_id = new_algo
        pos.trailing_active = True
        self._save()
        log.info(
            "[%s] OCO placed algo=%s tp=%.6f sl=%.6f (mark=%.6f)",
            symbol, new_algo, clamped_target, clamped_stop, mark,
        )
        return True

    # ---- reconciliation --------------------------------------------------
    def reconcile_exchange(self, marks: Dict[str, float]) -> List[LiveTrade]:
        """Detect positions closed on the exchange (TP/SL fired, manual close,
        liquidation) and update local state to match.

        Called at the start of each tick. Returns the list of newly-closed
        trades so the caller can log them.
        """
        try:
            payload = request("GET", "/api/v5/account/positions", self.creds, params={"instType": "SWAP"})
        except Exception as e:
            log.warning("reconcile failed (skipping): %s", e)
            return []
        live = (payload.get("data") or [])
        # OKX returns positions keyed by (instId, posSide); pos == 0 means flat
        live_set: Dict[str, float] = {}
        for p in live:
            inst = p.get("instId", "")
            pos_qty = float(p.get("pos") or 0)
            if pos_qty != 0:
                live_set[inst] = pos_qty

        closed_now: List[LiveTrade] = []
        for sym in list(self.state.positions.keys()):
            pos = self.state.positions[sym]
            inst_id = self._swap_inst_id(sym)
            if inst_id not in live_set:
                # exchange says we're flat — TP/SL fired or got liquidated.
                # Resolve exit_price in priority order:
                #   1. algo fill price (most accurate — exact OCO fill)
                #   2. caller-supplied mark
                #   3. live ticker (closest available substitute)
                #   4. entry_price (last resort — produces 0 PnL, marker of failure)
                algo_fill = self._fetch_algo_fill_price(sym, pos.algo_id) if pos.algo_id else None
                if algo_fill is not None:
                    exit_price = algo_fill
                elif sym in marks and marks[sym] > 0:
                    exit_price = marks[sym]
                else:
                    live_mark = self._get_mark(sym)
                    exit_price = live_mark if live_mark and live_mark > 0 else pos.entry_price
                pnl = pos.direction * (exit_price - pos.entry_price) * pos.size_units
                margin = pos.margin_used or pos.notional
                pnl_pct = pnl / margin if margin > 0 else 0.0
                # decide TP vs SL by which side of entry the fill was
                if pos.direction == 1:
                    reason = "tp_hit" if exit_price >= pos.entry_price else "sl_hit"
                else:
                    reason = "tp_hit" if exit_price <= pos.entry_price else "sl_hit"
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
                    rationale=pos.rationale + " | EXCHANGE-CLOSED",
                    leverage=pos.leverage,
                    params_hash=pos.params_hash,
                )
                self.state.cash += margin + pnl
                del self.state.positions[sym]
                self.state.closed_trades.append(trade)
                closed_now.append(trade)
                log.info(
                    "[%s] RECONCILED %s @ %.4f reason=%s pnl=%+.2f",
                    sym, "LONG" if pos.direction == 1 else "SHORT",
                    exit_price, reason, pnl,
                )
        if closed_now:
            self._save()
        return closed_now

    def _fetch_algo_fill_price(self, symbol: str, algo_id: str) -> Optional[float]:
        if not algo_id:
            return None
        inst_id = self._swap_inst_id(symbol)
        try:
            payload = request(
                "GET", "/api/v5/trade/orders-algo-history",
                self.creds, params={"instType": "SWAP", "algoId": algo_id, "ordType": "oco"},
            )
            data = payload.get("data") or []
            if not data:
                return None
            d = data[0]
            # whichever side fired, OKX populates fillPx on the corresponding order
            for k in ("actualPx", "tpOrdPx", "slOrdPx"):
                if d.get(k) and d.get(k) not in ("", "-1"):
                    try:
                        return float(d[k])
                    except (TypeError, ValueError):
                        pass
        except Exception as e:
            log.info("[%s] algo history fetch failed: %s", inst_id, e)
        return None

    def reset(self, starting_equity: Optional[float] = None):
        self.state = PortfolioState(
            starting_equity=starting_equity or self.state.starting_equity,
            cash=starting_equity or self.state.starting_equity,
        )
        self._save()
