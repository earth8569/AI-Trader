"""Bar-by-bar backtester for the AI-Trader multi-agent system.

Supports long + short positions, ATR-based stops/targets, configurable fees, and
records every closed position for downstream analytics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import math
import pandas as pd

from .agents import DecisionAgent, RiskAgent, SentimentAgent, TechnicalAgent
from .params import AgentParams


@dataclass
class Position:
    direction: int          # +1 long, -1 short
    entry_ts: pd.Timestamp
    entry_price: float
    size_units: float       # base units
    stop: float
    target: float
    notional: float         # quote value at entry
    rationale: str = ""


@dataclass
class ClosedTrade:
    direction: int
    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    entry_price: float
    exit_price: float
    size_units: float
    pnl_quote: float
    pnl_pct: float
    exit_reason: str
    bars_held: int
    rationale: str


@dataclass
class BacktestConfig:
    starting_equity: float = 10_000.0
    fee_bps: float = 5.0          # 0.05% per side
    slippage_bps: float = 2.0
    risk_per_trade: float = 0.02  # 2% of equity risked per trade
    max_hold_bars: int = 96
    allow_shorts: bool = True
    min_trades: int = 1000


@dataclass
class BacktestResult:
    trades: List[ClosedTrade] = field(default_factory=list)
    equity_curve: List[dict] = field(default_factory=list)


class Backtester:
    def __init__(
        self,
        config: BacktestConfig | None = None,
        params: AgentParams | None = None,
    ):
        self.cfg = config or BacktestConfig()
        p = (params or AgentParams()).normalized()
        self.params = p
        self.technical = TechnicalAgent()
        self.sentiment = SentimentAgent()
        self.risk = RiskAgent(atr_mult_stop=p.atr_mult_stop, atr_mult_target=p.atr_mult_target)
        self.decision = DecisionAgent(
            weights={"technical": p.technical_weight, "sentiment": p.sentiment_weight},
            entry_threshold=p.entry_threshold,
            risk_veto=p.risk_veto,
        )

    # ---- helpers ---------------------------------------------------------
    def _apply_fees(self, price: float, side: int) -> float:
        # side = +1 buy (pay more), -1 sell (receive less)
        slip = self.cfg.slippage_bps / 10_000
        fee = self.cfg.fee_bps / 10_000
        return price * (1 + side * (slip + fee))

    def _position_size(self, equity: float, entry: float, stop: float) -> float:
        risk_quote = equity * self.cfg.risk_per_trade
        per_unit_risk = abs(entry - stop)
        if per_unit_risk <= 0:
            return 0.0
        units = risk_quote / per_unit_risk
        # cap so we never use more than 100% equity notionally
        max_units = equity / entry
        return min(units, max_units)

    def _close_position(
        self,
        pos: Position,
        exit_ts: pd.Timestamp,
        exit_price_raw: float,
        reason: str,
        bars_held: int,
    ) -> ClosedTrade:
        # apply exit fees (closing a long => sell; closing a short => buy back)
        exit_side = -pos.direction
        exit_price = self._apply_fees(exit_price_raw, exit_side)
        pnl = pos.direction * (exit_price - pos.entry_price) * pos.size_units
        pnl_pct = pnl / pos.notional if pos.notional > 0 else 0.0
        return ClosedTrade(
            direction=pos.direction,
            entry_ts=pos.entry_ts,
            exit_ts=exit_ts,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            size_units=pos.size_units,
            pnl_quote=pnl,
            pnl_pct=pnl_pct,
            exit_reason=reason,
            bars_held=bars_held,
            rationale=pos.rationale,
        )

    # ---- main loop -------------------------------------------------------
    def run(self, df: pd.DataFrame) -> BacktestResult:
        result = BacktestResult()
        equity = self.cfg.starting_equity
        pos: Optional[Position] = None
        bars_held = 0

        needed = ["sma_slow", "rsi", "macd_hist", "atr", "bb_pct", "vol_ratio", "ret_5"]
        df_ready = df.dropna(subset=needed).reset_index(drop=True)

        for i, row in df_ready.iterrows():
            ts = row["timestamp"]
            price_open = row["open"]
            high = row["high"]
            low = row["low"]
            close = row["close"]

            # --- 1. manage open position --------------------------------
            if pos is not None:
                bars_held += 1
                hit_stop = (pos.direction == 1 and low <= pos.stop) or (
                    pos.direction == -1 and high >= pos.stop
                )
                hit_target = (pos.direction == 1 and high >= pos.target) or (
                    pos.direction == -1 and low <= pos.target
                )
                closed = None
                if hit_stop and hit_target:
                    # pessimistic: assume stop hit first
                    closed = self._close_position(pos, ts, pos.stop, "stop", bars_held)
                elif hit_stop:
                    closed = self._close_position(pos, ts, pos.stop, "stop", bars_held)
                elif hit_target:
                    closed = self._close_position(pos, ts, pos.target, "target", bars_held)
                elif bars_held >= self.cfg.max_hold_bars:
                    closed = self._close_position(pos, ts, close, "timeout", bars_held)
                if closed is None:
                    # Check reversal signal from agents → early exit.
                    sig_t = self.technical.evaluate(row)
                    sig_s = self.sentiment.evaluate(row)
                    sig_r = self.risk.evaluate(row)
                    dec = self.decision.debate(
                        {"technical": sig_t, "sentiment": sig_s, "risk": sig_r}
                    )
                    if (pos.direction == 1 and dec.action == "SELL") or (
                        pos.direction == -1 and dec.action == "BUY"
                    ):
                        closed = self._close_position(pos, ts, close, "reversal", bars_held)
                if closed is not None:
                    equity += closed.pnl_quote
                    result.trades.append(closed)
                    pos = None
                    bars_held = 0

            # --- 2. look for new entry ----------------------------------
            if pos is None:
                sig_t = self.technical.evaluate(row)
                sig_s = self.sentiment.evaluate(row)
                sig_r = self.risk.evaluate(row)
                dec = self.decision.debate(
                    {"technical": sig_t, "sentiment": sig_s, "risk": sig_r}
                )
                if dec.action == "BUY" or (dec.action == "SELL" and self.cfg.allow_shorts):
                    direction = 1 if dec.action == "BUY" else -1
                    atr = row["atr"]
                    if not math.isnan(atr) and atr > 0:
                        entry_price = self._apply_fees(close, direction)
                        stop, target = self.risk.stop_and_target(entry_price, atr, direction)
                        units = self._position_size(equity, entry_price, stop)
                        if units > 0:
                            notional = units * entry_price
                            pos = Position(
                                direction=direction,
                                entry_ts=ts,
                                entry_price=entry_price,
                                size_units=units,
                                stop=stop,
                                target=target,
                                notional=notional,
                                rationale=dec.rationale,
                            )
                            bars_held = 0

            # --- 3. record equity ---------------------------------------
            unreal = 0.0
            if pos is not None:
                unreal = pos.direction * (close - pos.entry_price) * pos.size_units
            result.equity_curve.append(
                {"timestamp": ts, "equity": equity + unreal, "realized": equity}
            )

            # --- 4. stop early if we've hit the trade target ------------
            if len(result.trades) >= self.cfg.min_trades and pos is None:
                # finished — enough closed positions
                break

        return result
