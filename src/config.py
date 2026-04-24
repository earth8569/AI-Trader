"""Tiny config loader — reads optional config.json then overlays CLI args."""
from __future__ import annotations

import json
from pathlib import Path


DEFAULTS: dict = {
    "broker": "okx",           # "paper" | "okx"
    "feed": "okx",         # "cryptocom" | "okx"
    "symbols": ["BTC_USDT", "ETH_USDT"],
    "timeframe": "1h",
    "candle_count": 300,
    "max_hold_bars": 96,
    "allow_shorts": True,
    "poll_seconds": 300,
    "starting_equity": 10_000.0,
    "risk_per_trade": 0.02,
    "max_position_pct": 0.35,
    "fee_bps": 5.0,
    "slippage_bps": 2.0,
    "state_path": "state/portfolio.json",
    "okx_state_path": "state/okx_portfolio.json",
    "okx_swap_state_path": "state/okx_swap_portfolio.json",
    "okx_position_mode": "long_short_mode",
    "okx_margin_mode": "isolated",
    "okx_quote_ccy": "USDT",
    "log_path": "logs/trader.log",
    # walk-forward optimizer — autonomous self-improvement
    "params_path": "state/best_params.json",
    "optimize_symbols": [],              # empty → use the live `symbols` list
    "optimize_interval": "1h",
    "optimize_lookback_days": 180,
    "optimize_samples": 60,
    "optimize_min_trades": 30,
    "optimize_every_ticks": 168,         # re-optimize weekly at 1h/900s poll
    "optimize_on_startup": True,         # fit once on first run if no params file
}


def load_config(path: str | Path | None = "config.json") -> dict:
    cfg = dict(DEFAULTS)
    if path and Path(path).exists():
        with open(path, "r", encoding="utf-8") as f:
            cfg.update(json.load(f))
    return cfg
