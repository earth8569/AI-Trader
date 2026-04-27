"""Live trade analytics — what's working, what isn't.

Reads `closed_trades` from a broker state file and produces a per-dimension
breakdown so you (and eventually the optimizer) can see which directions,
leverages, symbols, exit reasons, and parameter snapshots actually win.

This is intentionally read-only. It never mutates state.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from .logging_setup import GREEN, RED, YELLOW, CYAN, GRAY, RESET


def _load_trades(state_path: str | Path) -> List[dict]:
    p = Path(state_path)
    if not p.exists():
        return []
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return list(d.get("closed_trades") or [])
    except (OSError, json.JSONDecodeError):
        return []


def _is_suspect(t: dict) -> bool:
    """Trades reconciled when no algo fill / mark was available end up with
    exit_price == entry_price and pnl == 0. Those rows are noise — flag them
    so we don't pollute stats and heuristic suggestions.
    """
    pnl = float(t.get("pnl_quote") or 0)
    pnl_pct = float(t.get("pnl_pct") or 0)
    entry = float(t.get("entry_price") or 0)
    exit_p = float(t.get("exit_price") or 0)
    return pnl == 0 and pnl_pct == 0 and entry > 0 and abs(exit_p - entry) < 1e-12


def _stats(trades: List[dict]) -> Dict[str, float]:
    n = len(trades)
    if n == 0:
        return {"n": 0}
    pnl = [float(t.get("pnl_quote") or 0) for t in trades]
    pct = [float(t.get("pnl_pct") or 0) for t in trades]
    wins = [x for x in pnl if x > 0]
    losses = [x for x in pnl if x <= 0]
    gp = sum(wins)
    gl = -sum(losses)
    return {
        "n": n,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / n if n else 0.0,
        "avg_pnl": sum(pnl) / n,
        "avg_pct": sum(pct) / n,
        "total_pnl": sum(pnl),
        "best": max(pnl) if pnl else 0,
        "worst": min(pnl) if pnl else 0,
        "profit_factor": (gp / gl) if gl > 0 else (float("inf") if gp > 0 else 0.0),
        "expectancy_pct": sum(pct) / n,
    }


def _color_pnl(v: float) -> str:
    if v > 0:
        return f"{GREEN}{v:+.2f}{RESET}"
    if v < 0:
        return f"{RED}{v:+.2f}{RESET}"
    return f"{v:+.2f}"


def _row(label: str, s: dict) -> str:
    if s["n"] == 0:
        return f"  {label:24s}  {GRAY}n=0 (no trades){RESET}"
    pf = s["profit_factor"]
    pf_s = "inf" if pf == float("inf") else f"{pf:.2f}"
    win_color = GREEN if s["win_rate"] >= 0.5 else (YELLOW if s["win_rate"] >= 0.4 else RED)
    return (
        f"  {label:24s}  n={s['n']:<4}  "
        f"win={win_color}{s['win_rate']*100:5.1f}%{RESET}  "
        f"exp={s['expectancy_pct']*100:+5.2f}%  "
        f"PnL={_color_pnl(s['total_pnl']):>20s}  PF={pf_s}"
    )


def _group(trades: List[dict], key) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {}
    for t in trades:
        k = str(key(t))
        out.setdefault(k, []).append(t)
    return out


def _suggestions(trades: List[dict]) -> List[str]:
    """Heuristic flags — small, conservative, only fire on N>=10."""
    out: List[str] = []
    if len(trades) < 10:
        return [f"{GRAY}(need >=10 trades for suggestions; have {len(trades)}){RESET}"]
    by_dir = _group(trades, lambda t: int(t.get("direction") or 0))
    for d, label in [(1, "LONG"), (-1, "SHORT")]:
        s = _stats(by_dir.get(str(d), []))
        if s["n"] >= 10 and s["win_rate"] < 0.35:
            out.append(
                f"{YELLOW}!{RESET} {label} side win rate {s['win_rate']*100:.0f}% over {s['n']} "
                f"trades — consider raising entry_threshold or reducing {label.lower()} bias."
            )
    by_reason = _group(trades, lambda t: str(t.get("exit_reason") or "?"))
    sl = _stats(by_reason.get("sl_hit", []) + by_reason.get("stop", []))
    tp = _stats(by_reason.get("tp_hit", []) + by_reason.get("target", []))
    if sl["n"] >= 10 and tp["n"] >= 5 and sl["n"] > tp["n"] * 2:
        out.append(
            f"{YELLOW}!{RESET} Stops fire {sl['n']}× vs targets {tp['n']}× — stops may be too "
            f"tight; try widening atr_mult_stop or activating trailing later."
        )
    by_lev = _group(trades, lambda t: int(t.get("leverage") or 1))
    if by_lev:
        worst_lev = min(
            by_lev.items(),
            key=lambda kv: _stats(kv[1])["expectancy_pct"] if len(kv[1]) >= 5 else 999,
        )
        s = _stats(worst_lev[1])
        if s["n"] >= 5 and s["expectancy_pct"] < -0.005:
            out.append(
                f"{YELLOW}!{RESET} Leverage {worst_lev[0]}x has expectancy "
                f"{s['expectancy_pct']*100:+.2f}% over {s['n']} trades — consider lowering "
                f"max_leverage or making the volatility filter stricter."
            )
    if not out:
        out.append(f"{GREEN}+{RESET} No red flags from heuristics. Keep gathering trades.")
    return out
    return out


def analyze(state_path: str | Path, last_n: int | None = None) -> str:
    """Return a human-readable analytics report for the given broker state."""
    raw = _load_trades(state_path)
    if last_n:
        raw = raw[-last_n:]
    if not raw:
        return f"{YELLOW}No closed trades yet at {state_path}{RESET}"

    suspect = [t for t in raw if _is_suspect(t)]
    trades = [t for t in raw if not _is_suspect(t)]

    lines: List[str] = []
    lines.append(f"{CYAN}{'='*70}{RESET}")
    lines.append(f"{CYAN}LIVE TRADE ANALYSIS -- {state_path}{RESET}")
    lines.append(f"{CYAN}{'='*70}{RESET}")

    if suspect:
        lines.append(
            f"{YELLOW}!{RESET} {len(suspect)} of {len(raw)} trades have exit==entry and "
            f"PnL=0 (reconcile-corrupted, excluded from stats)."
        )
        lines.append(
            f"  {GRAY}This was a bug in reconcile_exchange when algo_id was missing; "
            f"now fixed — new trades will be clean.{RESET}"
        )

    if not trades:
        lines.append("")
        lines.append(f"{YELLOW}No usable trades after filtering.{RESET}")
        lines.append(
            f"  {GRAY}Once new trades land (with proper exit fills), this report will populate.{RESET}"
        )
        return "\n".join(lines)

    overall = _stats(trades)
    lines.append("")
    lines.append(f"{CYAN}OVERALL{RESET}  ({trades[0].get('entry_ts','?')[:10]} -> {trades[-1].get('exit_ts','?')[:10]})")
    lines.append(_row("all", overall))

    lines.append("")
    lines.append(f"{CYAN}BY DIRECTION{RESET}")
    by_dir = _group(trades, lambda t: int(t.get("direction") or 0))
    for d, label in [(1, "LONG"), (-1, "SHORT")]:
        lines.append(_row(label, _stats(by_dir.get(str(d), []))))

    lines.append("")
    lines.append(f"{CYAN}BY LEVERAGE{RESET}")
    by_lev = _group(trades, lambda t: int(t.get("leverage") or 1))
    for lev in sorted(by_lev.keys(), key=int):
        lines.append(_row(f"{lev}x", _stats(by_lev[lev])))

    lines.append("")
    lines.append(f"{CYAN}BY EXIT REASON{RESET}")
    by_reason = _group(trades, lambda t: str(t.get("exit_reason") or "?"))
    for reason in sorted(by_reason.keys()):
        lines.append(_row(reason, _stats(by_reason[reason])))

    lines.append("")
    lines.append(f"{CYAN}TOP 5 SYMBOLS BY PnL{RESET}")
    by_sym = _group(trades, lambda t: str(t.get("symbol") or "?"))
    sym_stats = [(sym, _stats(ts)) for sym, ts in by_sym.items()]
    sym_stats.sort(key=lambda kv: kv[1]["total_pnl"], reverse=True)
    for sym, s in sym_stats[:5]:
        lines.append(_row(sym, s))
    if len(sym_stats) > 5:
        lines.append(f"  {GRAY}... bottom 5 ...{RESET}")
        for sym, s in sym_stats[-5:]:
            lines.append(_row(sym, s))

    lines.append("")
    lines.append(f"{CYAN}BY PARAMS_HASH{RESET}")
    by_h = _group(trades, lambda t: str(t.get("params_hash") or "(unknown)"))
    h_stats = [(h, _stats(ts)) for h, ts in by_h.items()]
    h_stats.sort(key=lambda kv: kv[1]["total_pnl"], reverse=True)
    for h, s in h_stats:
        lines.append(_row(h, s))

    lines.append("")
    lines.append(f"{CYAN}HEURISTIC SUGGESTIONS{RESET}")
    for s in _suggestions(trades):
        lines.append(f"  {s}")

    return "\n".join(lines)
