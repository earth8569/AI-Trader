"""Walk-forward parameter optimizer — self-improvement, interpretable edition.

How it works
------------
1. Pull `lookback_days` of recent hourly history.
2. Randomly sample N parameter candidates from a bounded grid.
3. For each candidate, backtest on the window and compute a score.
4. Disqualify any run below `min_trades` (too few samples to trust).
5. Save the top candidate to `state/best_params.json` with provenance.
6. LiveTrader reloads it on the next start / manual reload.

The score rewards BOTH risk-adjusted return AND sample size so a fluky 3-trade
run can't win over a statistically stable 50-trade run.
"""
from __future__ import annotations

import logging
import math
import random
import time
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .backtester import BacktestConfig, Backtester
from .data_loader import load_ohlcv
from .indicators import enrich
from .params import AgentParams, ParamSnapshot, save_best

log = logging.getLogger("ai-trader.optimizer")


# --- search space -----------------------------------------------------------
# Keep bounded and economical — walk-forward, not a kitchen-sink grid.
DEFAULT_SPACE: Dict[str, List[float]] = {
    "technical_weight":     [0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85],
    "sentiment_weight":     [0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85],
    "market_intel_weight":  [0.10, 0.15, 0.20, 0.30, 0.40],   # funding-rate voter
    "entry_threshold":      [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55],
    "risk_veto":            [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3],
    "atr_mult_stop":        [1.0, 1.5, 2.0, 2.5, 3.0],
    "atr_mult_target":      [2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0],
    # trailing stop knobs — learned alongside the entry/exit thresholds
    "trail_activation_r":   [0.5, 0.75, 1.0, 1.5, 2.0],
    "trail_distance_atr":   [0.75, 1.0, 1.5, 2.0, 2.5, 3.0],
}


def _sample_params(rng: random.Random, space: Dict[str, List[float]]) -> AgentParams:
    # resample target/stop until target > stop (positive R-multiple required)
    for _ in range(20):
        pick = {k: rng.choice(v) for k, v in space.items()}
        if pick["atr_mult_target"] > pick["atr_mult_stop"]:
            return AgentParams(**pick).normalized()
    # fallback: force target = stop + 1.0
    pick["atr_mult_target"] = pick["atr_mult_stop"] + 1.0
    return AgentParams(**pick).normalized()


def _score(result, min_trades: int) -> Tuple[float, dict]:
    """Composite score: penalises low sample size and high drawdown.

    Primary term is expectancy (avg pnl% per trade). We multiply by
    sqrt(min(n, min_trades*3)/min_trades) so scoring rises with sample size up
    to 3× the minimum, then plateaus. Then we subtract a drawdown penalty.
    """
    n = len(result.trades)
    if n < min_trades:
        return (float("-inf"), {"trades": n, "reason": "too few trades"})
    import pandas as pd
    from dataclasses import asdict

    trades = pd.DataFrame([asdict(t) for t in result.trades])
    exp = float(trades.pnl_pct.mean())
    eq = pd.DataFrame(result.equity_curve)
    if not eq.empty:
        eq["equity"] = eq["equity"].astype(float)
        running = eq["equity"].cummax()
        dd = float((eq["equity"] / running - 1.0).min())
    else:
        dd = 0.0
    sample_factor = math.sqrt(min(n, min_trades * 3) / min_trades)
    dd_penalty = max(0.0, -dd - 0.25) * 0.5   # drawdowns beyond 25% bite
    score = exp * sample_factor - dd_penalty
    return score, {
        "trades": n, "expectancy_pct": round(exp * 100, 4),
        "max_dd": round(dd, 4), "score": round(score, 6),
    }


def _evaluate(dfs: Dict[str, pd.DataFrame], candidate: AgentParams, min_trades: int) -> Tuple[float, dict]:
    """Run backtest on each symbol's frame. Return (mean_score, per_symbol_detail).

    Any symbol that fails min_trades disqualifies the whole candidate — we want
    params that work across the book, not specialised to one asset.
    """
    per: Dict[str, dict] = {}
    scores: List[float] = []
    total_trades = 0
    for sym, df in dfs.items():
        bt = Backtester(BacktestConfig(min_trades=10_000), params=candidate)
        result = bt.run(df)
        s, det = _score(result, min_trades)
        per[sym] = det
        total_trades += det.get("trades", 0)
        if s == float("-inf"):
            return (float("-inf"), {"per_symbol": per, "trades": total_trades, "reason": f"{sym} below min_trades"})
        scores.append(s)
    mean_s = sum(scores) / len(scores) if scores else float("-inf")
    return mean_s, {"per_symbol": per, "trades": total_trades, "score": round(mean_s, 6)}


def random_search(
    dfs: Dict[str, pd.DataFrame] | pd.DataFrame,
    n_samples: int = 40,
    min_trades: int = 30,
    seed: int = 42,
    space: Dict[str, List[float]] | None = None,
    progress: bool = True,
) -> Tuple[AgentParams, float, List[dict]]:
    """Try N random configs across one or more dataframes.

    `dfs` can be a single DataFrame (legacy) or {symbol: DataFrame} for
    multi-symbol scoring (candidate must clear min_trades on every symbol).
    """
    if isinstance(dfs, pd.DataFrame):
        dfs = {"default": dfs}
    rng = random.Random(seed)
    space = space or DEFAULT_SPACE
    tried: set = set()
    attempts: List[dict] = []
    best_params: Optional[AgentParams] = None
    best_score = float("-inf")

    for i in range(n_samples):
        for _ in range(10):
            candidate = _sample_params(rng, space)
            key = tuple(round(getattr(candidate, k), 4) for k in AgentParams.__annotations__)
            if key not in tried:
                tried.add(key)
                break
        t0 = time.time()
        score, detail = _evaluate(dfs, candidate, min_trades)
        dt = time.time() - t0
        attempts.append({**candidate.to_dict(), **detail, "seconds": round(dt, 2)})
        if progress:
            log.info("[%02d/%d] score=%s trades=%s (%.1fs)",
                     i + 1, n_samples, detail.get("score", "-inf"), detail.get("trades", 0), dt)
        if score > best_score:
            best_score = score
            best_params = candidate

    return best_params or AgentParams(), best_score, attempts


def _to_binance_symbol(s: str) -> str:
    """'BTC_USDT' → 'BTCUSDT' for the Binance historical loader."""
    return s.replace("_", "").replace("-", "").upper()


def walk_forward_optimize(
    symbols: List[str] | str = "BTCUSDT",
    interval: str = "1h",
    lookback_days: int = 90,
    n_samples: int = 40,
    min_trades: int = 30,
    out_path: str = "state/best_params.json",
    seed: int = 42,
) -> ParamSnapshot:
    """Walk-forward re-fit across one or more symbols."""
    if isinstance(symbols, str):
        symbols = [symbols]
    symbols = [_to_binance_symbol(s) for s in symbols]
    end = pd.Timestamp.utcnow().normalize()
    start = end - pd.Timedelta(days=lookback_days)

    dfs: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        log.info("loading %s %s from %s to %s", sym, interval, start.date(), end.date())
        df = load_ohlcv(sym, interval, str(start.date()), str(end.date()))
        df = enrich(df)
        log.info("%s: bars=%d usable=%d", sym, len(df), df.dropna().shape[0])
        dfs[sym] = df

    best, score, attempts = random_search(
        dfs, n_samples=n_samples, min_trades=min_trades, seed=seed,
    )
    # re-run winner on first symbol for trade-count provenance
    first_df = next(iter(dfs.values()))
    provenance = Backtester(BacktestConfig(min_trades=10_000), params=best).run(first_df)
    snap = ParamSnapshot(
        params=best,
        score=float(score),
        trades=len(provenance.trades),
        symbol=",".join(symbols),
        interval=interval,
        window_start=str(first_df.timestamp.iloc[0]),
        window_end=str(first_df.timestamp.iloc[-1]),
    )
    save_best(snap, path=out_path)
    log.info(
        "winner: mean_score=%.6f | tw=%.2f sw=%.2f thr=%.2f veto=%.2f stop=%.1f tgt=%.1f",
        score,
        best.technical_weight, best.sentiment_weight,
        best.entry_threshold, best.risk_veto,
        best.atr_mult_stop, best.atr_mult_target,
    )
    return snap
