"""Tactics planner — smart stop, target, and leverage per trade.

Called by LiveTrader (and Backtester in future) to compute a PositionPlan
given the latest bar + agent decision. Separates RISK decisions from SIGNAL
decisions so they can be tuned/audited independently.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from .params import AgentParams


@dataclass
class PositionPlan:
    symbol: str
    direction: int          # +1 long, -1 short
    entry: float            # reference price used for sizing
    stop: float
    target: float
    leverage: int           # 1..max_leverage (integer; OKX requires integer)
    risk_per_unit: float    # |entry - stop|
    r_multiple: float       # (target-entry)/(entry-stop) in absolute terms
    conviction: float       # 0..1 — feed-forward from decision score
    rationale: str


def _conviction(decision_score: float) -> float:
    # decision_score is already in [-1, 1]; convert magnitude to [0, 1]
    return max(0.0, min(1.0, abs(decision_score)))


def _pick_leverage(conviction: float, atr_pct: float, params: AgentParams) -> int:
    """Leverage scales with conviction and INVERSELY with volatility.

    A high-conviction trade in a calm market can safely use more leverage
    without dropping the stop-out probability. Hard-capped at max_leverage.
    Minimum is always 1x.
    """
    if params.max_leverage <= 1:
        return 1
    # volatility dampener in [0.3, 1.0]: if ATR% is 2x the reference, factor ~ 0.5
    vol_ref = max(0.0005, params.leverage_vol_ref_pct)
    vol_factor = vol_ref / max(atr_pct, vol_ref * 0.25)
    vol_factor = max(0.3, min(1.0, vol_factor))
    raw = 1.0 + conviction * (params.max_leverage - 1) * vol_factor
    return max(1, min(params.max_leverage, int(round(raw))))


def compute_trailing_update(pos, mark: float, atr: float, params: AgentParams):
    """Decide if the position's stop should ratchet to a tighter level.

    Returns (new_stop or None, updated_extreme). The trader writes
    `updated_extreme` into the position's high/low water mark on every tick;
    `new_stop` is non-None only when the trail would actually tighten the
    stop AND keep it at-or-better than breakeven.

    Ratchet rule: stops only move in the favor direction (locks profit).
    Once activated, stop is forced to ≥ entry (long) / ≤ entry (short),
    so the trade is at worst risk-free.
    """
    if atr <= 0 or atr != atr:  # NaN guard
        return None, pos.high_water_mark or pos.low_water_mark or pos.entry_price

    direction = pos.direction
    initial_risk = abs(pos.entry_price - pos.stop)
    if initial_risk <= 0:
        return None, pos.high_water_mark or pos.low_water_mark or pos.entry_price

    # update favorable extreme
    if direction == 1:
        prev_extreme = pos.high_water_mark if pos.high_water_mark > 0 else pos.entry_price
        extreme = max(prev_extreme, mark)
    else:
        prev_extreme = pos.low_water_mark if pos.low_water_mark > 0 else pos.entry_price
        extreme = min(prev_extreme, mark) if prev_extreme > 0 else mark

    # has unrealized profit reached the activation threshold?
    profit = direction * (extreme - pos.entry_price)
    if profit < params.trail_activation_r * initial_risk:
        return None, extreme

    # propose a new stop based on the trail distance behind the extreme
    trail = params.trail_distance_atr * atr
    if direction == 1:
        candidate = extreme - trail
        # only ever tighten, and at minimum lock breakeven once active
        candidate = max(candidate, pos.entry_price)
        if candidate > pos.stop * 1.0001:           # require meaningful move
            return candidate, extreme
    else:
        candidate = extreme + trail
        candidate = min(candidate, pos.entry_price)
        if candidate < pos.stop * 0.9999:
            return candidate, extreme
    return None, extreme


def plan_position(
    row,
    symbol: str,
    direction: int,
    decision_score: float,
    params: AgentParams,
) -> Optional[PositionPlan]:
    """Return a PositionPlan, or None if prerequisites are missing.

    Stops blend ATR distance with recent swing levels (structural stops);
    targets are the larger of ATR R-multiple and capped by Bollinger extremes.
    """
    atr = float(row.get("atr", float("nan")))
    price = float(row["close"])
    if not (atr > 0 and price > 0) or math.isnan(atr):
        return None

    atr_pct = atr / price
    conviction = _conviction(decision_score)
    leverage = _pick_leverage(conviction, atr_pct, params)

    # --- stop: combine ATR distance with structural swing -----------------
    atr_stop = params.atr_mult_stop * atr
    swing_hi = float(row.get("swing_high_20", price + atr_stop) or (price + atr_stop))
    swing_lo = float(row.get("swing_low_20", price - atr_stop) or (price - atr_stop))

    if direction == 1:
        stop = min(price - atr_stop, swing_lo - 0.3 * atr)
        # never above entry
        stop = min(stop, price * 0.999)
    else:
        stop = max(price + atr_stop, swing_hi + 0.3 * atr)
        stop = max(stop, price * 1.001)

    risk_per_unit = abs(price - stop)
    if risk_per_unit <= 0:
        return None

    # --- target: R multiple bounded by BB extreme + small ATR buffer.
    # Targets must actually be REACHABLE — empirically we saw tradesys with
    # 2.0R+ targets on 1h crypto hit only 1% of the time, leaving exits to
    # stops/reversals. A 1.5R floor with BB+0.5ATR ceiling lands ~30% target
    # hit rate which keeps avg_win healthy.
    r_ratio = params.atr_mult_target / params.atr_mult_stop if params.atr_mult_stop > 0 else 2.0
    r_ratio = max(1.5, r_ratio)
    bb_up = float(row.get("bb_up", price + 3 * atr) or (price + 3 * atr))
    bb_lo = float(row.get("bb_lo", price - 3 * atr) or (price - 3 * atr))

    if direction == 1:
        target_raw = price + r_ratio * risk_per_unit
        target = min(target_raw, bb_up + 0.5 * atr)
        target = max(target, price + 1.5 * risk_per_unit)
    else:
        target_raw = price - r_ratio * risk_per_unit
        target = max(target_raw, bb_lo - 0.5 * atr)
        target = min(target, price - 1.5 * risk_per_unit)

    # OKX rejects OCO triggers outside ~25% of mark; cap target/stop to ±20%
    # to stay safely within range while preserving the agent's intent.
    max_dev = 0.20
    if direction == 1:
        target = min(target, price * (1 + max_dev))
        stop = max(stop, price * (1 - max_dev))
    else:
        target = max(target, price * (1 - max_dev))
        stop = min(stop, price * (1 + max_dev))
    risk_per_unit = abs(price - stop)
    if risk_per_unit <= 0:
        return None

    realized_r = abs(target - price) / risk_per_unit

    rationale = (
        f"conv={conviction:.2f} atr%={atr_pct*100:.2f}% lev={leverage}x "
        f"R={realized_r:.2f} stop=ATR{params.atr_mult_stop:.1f}+swing "
        f"tgt=min(R{r_ratio:.1f}, BB)"
    )
    return PositionPlan(
        symbol=symbol,
        direction=direction,
        entry=price,
        stop=stop,
        target=target,
        leverage=leverage,
        risk_per_unit=risk_per_unit,
        r_multiple=realized_r,
        conviction=conviction,
        rationale=rationale,
    )
