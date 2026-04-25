"""Live multi-agent trader loop.

One `tick()` = fetch latest candles for each symbol, evaluate agents, manage
open positions (stop/target/reversal), and open new positions if the debate
produces a BUY/SELL decision.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd

from ..agents import DecisionAgent, RiskAgent, SentimentAgent, TechnicalAgent
from ..indicators import enrich
from ..logging_setup import banner, GREEN, RED, YELLOW, CYAN, GRAY, RESET
from ..params import AgentParams, load_best
from ..tactics import compute_trailing_update, plan_position
from .broker_base import BrokerBase
from .data_feed import LiveFeed

log = logging.getLogger("ai-trader.live")


@dataclass
class LiveConfig:
    symbols: List[str] = field(default_factory=lambda: ["BTC_USDT"])
    timeframe: str = "1h"
    candle_count: int = 300
    max_hold_bars: int = 96
    allow_shorts: bool = True
    poll_seconds: int = 300   # sleep between ticks in daemon mode
    # walk-forward autonomy
    optimize_every_ticks: int = 0           # 0 = disabled
    optimize_on_startup: bool = False
    optimize_symbols: List[str] = field(default_factory=list)   # empty → use `symbols`
    optimize_interval: str = "1h"
    optimize_lookback_days: int = 90
    optimize_samples: int = 40
    optimize_min_trades: int = 30


class LiveTrader:
    def __init__(
        self,
        config: LiveConfig,
        broker: BrokerBase,
        feed=None,
        params: AgentParams | None = None,
        params_path: str = "state/best_params.json",
    ):
        self.cfg = config
        self.broker = broker
        self.feed = feed or LiveFeed()
        self.params_path = params_path
        # preference order: explicit arg → state/best_params.json → defaults
        chosen = params or load_best(params_path) or AgentParams()
        chosen = chosen.normalized()
        self.params = chosen
        log.info(
            "agent params: tw=%.2f sw=%.2f thr=%.2f veto=%.2f atr_stop=%.2f atr_tgt=%.2f",
            chosen.technical_weight, chosen.sentiment_weight,
            chosen.entry_threshold, chosen.risk_veto,
            chosen.atr_mult_stop, chosen.atr_mult_target,
        )
        self._rebuild_agents(chosen)
        # when the position was opened (bar count), per symbol — used for max-hold
        self._bar_counter: Dict[str, int] = {}

    # ---- agent construction / hot-reload --------------------------------
    def _rebuild_agents(self, p: AgentParams):
        self.technical = TechnicalAgent()
        self.sentiment = SentimentAgent()
        self.risk = RiskAgent(
            atr_mult_stop=p.atr_mult_stop,
            atr_mult_target=p.atr_mult_target,
        )
        self.decision = DecisionAgent(
            weights={"technical": p.technical_weight, "sentiment": p.sentiment_weight},
            entry_threshold=p.entry_threshold,
            risk_veto=p.risk_veto,
        )

    def reload_params(self) -> bool:
        """Re-read params file. Returns True if anything changed."""
        fresh = load_best(self.params_path)
        if fresh is None:
            return False
        fresh = fresh.normalized()
        if fresh.to_dict() == self.params.to_dict():
            return False
        log.info(
            "params changed: tw=%.2f sw=%.2f thr=%.2f veto=%.2f stop=%.1f tgt=%.1f",
            fresh.technical_weight, fresh.sentiment_weight,
            fresh.entry_threshold, fresh.risk_veto,
            fresh.atr_mult_stop, fresh.atr_mult_target,
        )
        self.params = fresh
        self._rebuild_agents(fresh)
        return True

    def _maybe_optimize(self):
        """Run walk-forward, save winner, hot-reload. Swallows errors."""
        syms = self.cfg.optimize_symbols or self.cfg.symbols
        try:
            # imported here to avoid a circular dep at module load
            from ..optimizer import walk_forward_optimize
            log.info(
                "auto-optimize starting (symbols=%s lookback=%dd samples=%d)",
                syms, self.cfg.optimize_lookback_days, self.cfg.optimize_samples,
            )
            walk_forward_optimize(
                symbols=syms,
                interval=self.cfg.optimize_interval,
                lookback_days=self.cfg.optimize_lookback_days,
                n_samples=self.cfg.optimize_samples,
                min_trades=self.cfg.optimize_min_trades,
                out_path=self.params_path,
            )
            self.reload_params()
        except Exception as e:
            log.exception("auto-optimize failed: %s", e)

    # ---- tick ------------------------------------------------------------
    def tick(self) -> dict:
        marks: Dict[str, float] = {}
        actions: List[dict] = []
        errors: List[str] = []
        # tick header — one short line, easy to scan
        ts = pd.Timestamp.utcnow().strftime("%H:%M:%S")
        print(banner(f"TICK {ts}  symbols={len(self.cfg.symbols)}"))

        # 1. refresh broker-side balance + reconcile exchange-fired exits
        #    (TP/SL/liquidation) before we look at signals.
        if hasattr(self.broker, "refresh_balance"):
            try:
                self.broker.refresh_balance()
            except Exception as e:
                log.warning("balance refresh FAILED: %s", e)
                errors.append(f"refresh: {e}")
        if hasattr(self.broker, "reconcile_exchange"):
            try:
                self.broker.reconcile_exchange({})
            except Exception as e:
                log.warning("reconcile FAILED: %s", e)
                errors.append(f"reconcile: {e}")

        for sym in self.cfg.symbols:
            try:
                info = self._tick_symbol(sym)
            except Exception as e:
                log.error("[%s] FAILED: %s", sym, e)
                errors.append(f"{sym}: {e}")
                continue
            marks[sym] = info["mark"]
            actions.append(info)
        if marks:
            self.broker.record_equity(marks)

        equity = self.broker.equity(marks)
        starting = self.broker.starting_equity or 1.0
        ret_pct = (equity / starting - 1.0) * 100
        summary = {
            "ts": pd.Timestamp.utcnow().isoformat(),
            "equity": equity,
            "cash": self.broker.cash,
            "open_positions": len(self.broker.open_positions),
            "closed_trades": len(self.broker.closed_trades),
            "actions": actions,
            "errors": errors,
        }
        # build a one-line summary, color-coded by error count
        opens = sum(1 for a in actions if a.get("status", "").startswith("OPEN_"))
        closes = sum(1 for a in actions if a.get("status", "").startswith("CLOSE_"))
        ret_color = GREEN if ret_pct >= 0 else RED
        err_seg = f"{RED}err={len(errors)}{RESET}" if errors else f"{GRAY}err=0{RESET}"
        print(
            f"{GRAY}{ts}{RESET} {CYAN}|{RESET} "
            f"equity={equity:,.2f} ({ret_color}{ret_pct:+.2f}%{RESET}) {CYAN}|{RESET} "
            f"cash={self.broker.cash:,.2f} {CYAN}|{RESET} "
            f"open={summary['open_positions']} {CYAN}|{RESET} "
            f"opened={opens} closed={closes} {CYAN}|{RESET} "
            f"{err_seg}"
        )
        # full machine-readable line goes to file only
        log.debug(
            "tick summary equity=%.2f cash=%.2f open=%d closed=%d opened=%d closed_now=%d errors=%d",
            equity, self.broker.cash, summary["open_positions"],
            summary["closed_trades"], opens, closes, len(errors),
        )
        return summary

    # ---- per-symbol ------------------------------------------------------
    def _tick_symbol(self, symbol: str) -> dict:
        df = self.feed.get_candles(symbol, self.cfg.timeframe, self.cfg.candle_count)
        df = enrich(df)
        df = df.dropna(
            subset=["sma_slow", "rsi", "macd_hist", "atr", "bb_pct", "vol_ratio", "ret_5"]
        ).reset_index(drop=True)
        if df.empty:
            log.debug("%s: not enough bars yet", symbol)
            return {"symbol": symbol, "status": "insufficient_data", "mark": 0.0}

        row = df.iloc[-1]
        mark = float(row["close"])
        action = "HOLD"
        detail = ""

        # --- 1. manage any open position -----------------------------------
        pos = self.broker.open_positions.get(symbol)
        if pos is not None:
            bars_held = self._bar_counter.get(symbol, 0) + 1
            self._bar_counter[symbol] = bars_held
            high, low = float(row["high"]), float(row["low"])

            # 1a. Adopt orphans + run trailing ratchet — one update_stop call.
            #
            # An "orphan" is an open position with no exchange-side OCO
            # (algo_id == ""), typically because it was opened before the
            # OCO feature existed or because OCO placement failed earlier.
            # On EVERY tick we want such positions protected: place an OCO
            # using the stored stop+target. If trailing also wants to
            # tighten the stop, fold both into one cancel+replace.
            atr = float(row["atr"])
            new_stop, extreme = compute_trailing_update(pos, mark, atr, self.params)
            # water marks update every tick (in-memory; persists when broker saves)
            if pos.direction == 1:
                pos.high_water_mark = extreme
            else:
                pos.low_water_mark = extreme

            needs_protect = not pos.algo_id
            trail_tightened = new_stop is not None
            if (needs_protect or trail_tightened) and hasattr(self.broker, "update_stop"):
                old_stop = pos.stop
                target_stop = new_stop if trail_tightened else pos.stop
                if self.broker.update_stop(symbol, target_stop):
                    if trail_tightened:
                        log.info(
                            "[%s] TRAIL stop %.6f -> %.6f (extreme=%.6f, mark=%.6f)",
                            symbol, old_stop, target_stop, extreme, mark,
                        )
                    else:
                        log.info(
                            "[%s] PROTECTED orphan with OCO stop=%.6f tgt=%.6f",
                            symbol, target_stop, pos.target,
                        )
                    pos = self.broker.open_positions.get(symbol)
                    if pos is None:
                        return {"symbol": symbol, "status": "TRAILED", "detail": "", "mark": mark}
            hit_stop = (pos.direction == 1 and low <= pos.stop) or (
                pos.direction == -1 and high >= pos.stop
            )
            hit_target = (pos.direction == 1 and high >= pos.target) or (
                pos.direction == -1 and low <= pos.target
            )
            closed = None
            if hit_stop and hit_target:
                closed = self.broker.close(symbol, pos.stop, "stop")
            elif hit_stop:
                closed = self.broker.close(symbol, pos.stop, "stop")
            elif hit_target:
                closed = self.broker.close(symbol, pos.target, "target")
            elif bars_held >= self.cfg.max_hold_bars:
                closed = self.broker.close(symbol, mark, "timeout")
            else:
                # reversal check via fresh debate
                sigs = self._debate(row)
                if (pos.direction == 1 and sigs["decision"].action == "SELL") or (
                    pos.direction == -1 and sigs["decision"].action == "BUY"
                ):
                    closed = self.broker.close(symbol, mark, "reversal")
            if closed is not None:
                self._bar_counter.pop(symbol, None)
                action = f"CLOSE_{closed.exit_reason.upper()}"
                detail = f"pnl={closed.pnl_quote:+.2f} ({closed.pnl_pct*100:+.2f}%)"
                log.info("[%s] closed %s %s", symbol, action, detail)
                return {"symbol": symbol, "status": action, "detail": detail, "mark": mark}

        # --- 2. look for a fresh entry -------------------------------------
        if symbol not in self.broker.open_positions:
            sigs = self._debate(row)
            dec = sigs["decision"]
            if dec.action in ("BUY", "SELL") and (dec.action == "BUY" or self.cfg.allow_shorts):
                direction = 1 if dec.action == "BUY" else -1
                plan = plan_position(
                    row=row,
                    symbol=symbol,
                    direction=direction,
                    decision_score=dec.score,
                    params=self.params,
                )
                if plan is not None:
                    equity_now = self.broker.equity({symbol: mark})
                    pos = self.broker.open(
                        symbol=symbol,
                        direction=direction,
                        price=mark,
                        stop=plan.stop,
                        target=plan.target,
                        equity_now=equity_now,
                        rationale=f"{dec.rationale} || tactics: {plan.rationale}",
                        leverage=plan.leverage,
                    )
                    if pos is not None:
                        self._bar_counter[symbol] = 0
                        action = f"OPEN_{dec.action}"
                        detail = (
                            f"{pos.leverage}x units={pos.size_units:.6f} entry={pos.entry_price:.2f} "
                            f"stop={pos.stop:.2f} target={pos.target:.2f} "
                            f"R={plan.r_multiple:.2f} score={dec.score:+.2f}"
                        )
                        log.info("[%s] %s %s", symbol, action, detail)
        return {"symbol": symbol, "status": action, "detail": detail, "mark": mark}

    # ---- agents debate ---------------------------------------------------
    def _debate(self, row) -> dict:
        st = self.technical.evaluate(row)
        ss = self.sentiment.evaluate(row)
        sr = self.risk.evaluate(row)
        dec = self.decision.debate({"technical": st, "sentiment": ss, "risk": sr})
        return {"technical": st, "sentiment": ss, "risk": sr, "decision": dec}

    # ---- daemon loop -----------------------------------------------------
    def run_forever(self, max_ticks: int | None = None):
        log.info(
            "starting live trader: symbols=%s tf=%s poll=%ss auto_optimize=%s",
            self.cfg.symbols, self.cfg.timeframe, self.cfg.poll_seconds,
            self.cfg.optimize_every_ticks or "off",
        )
        if self.cfg.optimize_on_startup and load_best(self.params_path) is None:
            log.info("no saved params — running first-time optimize")
            self._maybe_optimize()
        n = 0
        try:
            while True:
                self.tick()
                n += 1
                if self.cfg.optimize_every_ticks and n % self.cfg.optimize_every_ticks == 0:
                    self._maybe_optimize()
                if max_ticks is not None and n >= max_ticks:
                    break
                time.sleep(self.cfg.poll_seconds)
        except KeyboardInterrupt:
            log.info("interrupted by user — shutting down cleanly")
