"""Performance report: trades.csv, equity.csv, summary.json."""
from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .backtester import BacktestResult


def _safe(x: float) -> float:
    return 0.0 if (x is None or (isinstance(x, float) and math.isnan(x))) else float(x)


def summarize(result: BacktestResult, starting_equity: float) -> dict:
    if not result.trades:
        return {"trades": 0, "note": "no trades"}

    trades_df = pd.DataFrame([asdict(t) for t in result.trades])
    wins = trades_df[trades_df.pnl_quote > 0]
    losses = trades_df[trades_df.pnl_quote <= 0]

    eq = pd.DataFrame(result.equity_curve)
    eq["timestamp"] = pd.to_datetime(eq["timestamp"])
    eq = eq.sort_values("timestamp").set_index("timestamp")
    eq["ret"] = eq["equity"].pct_change().fillna(0.0)

    ann_factor = 24 * 365  # hourly bars default
    sharpe = 0.0
    if eq["ret"].std() > 0:
        sharpe = (eq["ret"].mean() / eq["ret"].std()) * math.sqrt(ann_factor)

    running_max = eq["equity"].cummax()
    drawdown = eq["equity"] / running_max - 1.0
    max_dd = drawdown.min()

    gross_profit = wins.pnl_quote.sum()
    gross_loss = -losses.pnl_quote.sum()
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    final_equity = float(eq["equity"].iloc[-1])
    total_return = final_equity / starting_equity - 1.0

    summary = {
        "trades": int(len(trades_df)),
        "wins": int(len(wins)),
        "losses": int(len(losses)),
        "win_rate": _safe(len(wins) / len(trades_df)),
        "avg_win_pct": _safe(wins.pnl_pct.mean()) if len(wins) else 0.0,
        "avg_loss_pct": _safe(losses.pnl_pct.mean()) if len(losses) else 0.0,
        "expectancy_pct": _safe(trades_df.pnl_pct.mean()),
        "profit_factor": _safe(profit_factor) if profit_factor != float("inf") else None,
        "gross_profit": _safe(gross_profit),
        "gross_loss": _safe(gross_loss),
        "starting_equity": float(starting_equity),
        "final_equity": final_equity,
        "total_return": _safe(total_return),
        "sharpe_annualized": _safe(sharpe),
        "max_drawdown": _safe(max_dd),
        "avg_bars_held": _safe(trades_df.bars_held.mean()),
        "long_trades": int((trades_df.direction == 1).sum()),
        "short_trades": int((trades_df.direction == -1).sum()),
        "exit_breakdown": trades_df.exit_reason.value_counts().to_dict(),
    }
    return summary


def write_reports(result: BacktestResult, out_dir: str | Path, starting_equity: float):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    trades_df = pd.DataFrame([asdict(t) for t in result.trades])
    trades_df.to_csv(out / "trades.csv", index=False)

    eq = pd.DataFrame(result.equity_curve)
    eq.to_csv(out / "equity.csv", index=False)

    summary = summarize(result, starting_equity)
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    return summary
