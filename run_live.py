"""AI-Trader CLI — live paper trading automation.

Subcommands:
  tick      run a single tick and exit (cron-friendly)
  start     daemon loop (blocking)
  status    print current portfolio snapshot
  history   print recent closed trades
  reset     wipe state and reinitialise starting equity
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from src.config import load_config
from src.live import LiveFeed, LiveTrader, OkxFeed, PaperBroker
from src.live.trader import LiveConfig
from src.logging_setup import setup_logging


log = logging.getLogger("ai-trader.cli")


def _make_broker(cfg: dict):
    broker_kind = cfg.get("broker", "paper").lower()
    if broker_kind == "okx":
        from src.live.okx_broker import OkxBroker
        broker = OkxBroker(
            state_path=cfg["okx_state_path"],
            quote_ccy=cfg.get("okx_quote_ccy", "USDT"),
        )
        broker.state.risk_per_trade = cfg["risk_per_trade"]
        broker.state.max_position_pct = cfg.get("max_position_pct", 0.35)
        broker._save()
        return broker
    if broker_kind in ("okx_swap", "okx-swap", "okxswap"):
        from src.live.okx_swap_broker import OkxSwapBroker
        broker = OkxSwapBroker(
            state_path=cfg.get("okx_swap_state_path", "state/okx_swap_portfolio.json"),
            quote_ccy=cfg.get("okx_quote_ccy", "USDT"),
            position_mode=cfg.get("okx_position_mode", "long_short_mode"),
            margin_mode=cfg.get("okx_margin_mode", "isolated"),
        )
        broker.state.risk_per_trade = cfg["risk_per_trade"]
        broker.state.max_position_pct = cfg.get("max_position_pct", 0.35)
        broker._save()
        return broker
    broker = PaperBroker(state_path=cfg["state_path"])
    if not broker.state.closed_trades and not broker.state.positions:
        broker.state.starting_equity = cfg["starting_equity"]
        broker.state.cash = cfg["starting_equity"]
        broker.state.fee_bps = cfg["fee_bps"]
        broker.state.slippage_bps = cfg["slippage_bps"]
        broker.state.risk_per_trade = cfg["risk_per_trade"]
        broker.state.max_position_pct = cfg.get("max_position_pct", 0.35)
        broker._save()
    return broker


def _make_feed(cfg: dict):
    kind = cfg.get("feed", "cryptocom").lower()
    if kind == "okx":
        return OkxFeed()
    return LiveFeed()


def _build(cfg: dict):
    broker = _make_broker(cfg)
    feed = _make_feed(cfg)
    # spot is long-only; swap + paper allow shorts (subject to config.allow_shorts)
    broker_kind = cfg.get("broker", "paper").lower()
    allow_shorts = cfg["allow_shorts"] and broker_kind != "okx"
    lcfg = LiveConfig(
        symbols=cfg["symbols"],
        timeframe=cfg["timeframe"],
        candle_count=cfg["candle_count"],
        max_hold_bars=cfg["max_hold_bars"],
        allow_shorts=allow_shorts,
        poll_seconds=cfg["poll_seconds"],
        optimize_every_ticks=cfg.get("optimize_every_ticks", 0),
        optimize_on_startup=cfg.get("optimize_on_startup", False),
        optimize_symbols=cfg.get("optimize_symbols") or [],
        optimize_interval=cfg.get("optimize_interval", "1h"),
        optimize_lookback_days=cfg.get("optimize_lookback_days", 90),
        optimize_samples=cfg.get("optimize_samples", 40),
        optimize_min_trades=cfg.get("optimize_min_trades", 30),
    )
    trader = LiveTrader(lcfg, broker, feed, params_path=cfg.get("params_path", "state/best_params.json"))
    return trader, broker


def cmd_tick(args, cfg):
    trader, _ = _build(cfg)
    summary = trader.tick()
    print(json.dumps(summary, indent=2, default=str))


def cmd_start(args, cfg):
    trader, _ = _build(cfg)
    trader.run_forever(max_ticks=args.max_ticks)


def cmd_status(args, cfg):
    broker = _make_broker(cfg)
    feed = _make_feed(cfg)
    marks = {}
    for sym in cfg["symbols"]:
        try:
            marks[sym] = feed.get_ticker(sym)["last"]
        except Exception as e:
            log.warning("ticker fetch failed for %s: %s", sym, e)
    equity = broker.equity(marks)
    starting = broker.starting_equity or 1.0
    ret = equity / starting - 1.0
    out = {
        "broker": cfg.get("broker", "paper"),
        "feed": cfg.get("feed", "cryptocom"),
        "starting_equity": broker.starting_equity,
        "cash": broker.cash,
        "equity_now": equity,
        "total_return_pct": round(ret * 100, 3),
        "open_positions": [
            {
                "symbol": p.symbol,
                "direction": "LONG" if p.direction == 1 else "SHORT",
                "entry_price": p.entry_price,
                "size_units": p.size_units,
                "stop": p.stop,
                "target": p.target,
                "mark": marks.get(p.symbol),
                "unreal_pnl": (
                    p.direction * (marks[p.symbol] - p.entry_price) * p.size_units
                    if marks.get(p.symbol)
                    else None
                ),
            }
            for p in broker.open_positions.values()
        ],
        "closed_trades_count": len(broker.closed_trades),
    }
    print(json.dumps(out, indent=2, default=str))


def cmd_history(args, cfg):
    broker = _make_broker(cfg)
    trades = broker.closed_trades[-args.n:]
    for t in trades:
        print(
            f"{t.exit_ts} {t.symbol} {'LONG' if t.direction==1 else 'SHORT':<5} "
            f"pnl={t.pnl_quote:+8.2f} ({t.pnl_pct*100:+5.2f}%) "
            f"exit={t.exit_reason:<8} entry={t.entry_price:.2f} exit_p={t.exit_price:.2f}"
        )
    if not trades:
        print("(no closed trades yet)")
    print(f"-- total closed: {len(broker.closed_trades)}")


def cmd_optimize(args, cfg):
    from src.optimizer import walk_forward_optimize
    from src.params import load_snapshot

    # resolve symbols: CLI overrides > config optimize_symbols > config symbols
    if args.symbols:
        symbols = args.symbols
    elif cfg.get("optimize_symbols"):
        symbols = cfg["optimize_symbols"]
    else:
        symbols = cfg["symbols"]
    snap = walk_forward_optimize(
        symbols=symbols,
        interval=args.interval or cfg.get("optimize_interval", "1h"),
        lookback_days=args.lookback_days or cfg.get("optimize_lookback_days", 90),
        n_samples=args.samples or cfg.get("optimize_samples", 40),
        min_trades=args.min_trades or cfg.get("optimize_min_trades", 30),
        out_path=cfg.get("params_path", "state/best_params.json"),
        seed=args.seed,
    )
    print(json.dumps(snap.to_dict(), indent=2, default=str))
    full = load_snapshot(cfg.get("params_path", "state/best_params.json"))
    if full and full.get("history"):
        print(f"\n-- history: {len(full['history'])} prior snapshots retained")


def cmd_params(args, cfg):
    from src.params import load_snapshot
    snap = load_snapshot(cfg.get("params_path", "state/best_params.json"))
    if snap is None:
        print("(no saved params yet — run `optimize` first)")
        return
    print(json.dumps(snap, indent=2, default=str))


def cmd_reset(args, cfg):
    kind = cfg.get("broker", "paper").lower()
    if kind == "okx":
        state_path = cfg["okx_state_path"]
    elif kind in ("okx_swap", "okx-swap", "okxswap"):
        state_path = cfg.get("okx_swap_state_path", "state/okx_swap_portfolio.json")
    else:
        state_path = cfg["state_path"]
    if not args.yes:
        confirm = input(f"wipe LOCAL state at {state_path}? (does not liquidate exchange positions) [y/N] ").strip().lower()
        if confirm != "y":
            print("aborted"); return
    broker = _make_broker(cfg)
    broker.reset(starting_equity=args.starting_equity or cfg["starting_equity"])
    print(f"state reset. starting_equity={broker.starting_equity}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="ai-trader")
    p.add_argument("--config", default="config.json")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("tick", help="run a single tick and exit")
    sp.set_defaults(func=cmd_tick)

    sp = sub.add_parser("start", help="blocking daemon loop")
    sp.add_argument("--max-ticks", type=int, default=None)
    sp.set_defaults(func=cmd_start)

    sp = sub.add_parser("status", help="portfolio snapshot")
    sp.set_defaults(func=cmd_status)

    sp = sub.add_parser("history", help="recent closed trades")
    sp.add_argument("-n", type=int, default=20)
    sp.set_defaults(func=cmd_history)

    sp = sub.add_parser("reset", help="wipe paper state")
    sp.add_argument("--yes", action="store_true")
    sp.add_argument("--starting-equity", type=float, default=None)
    sp.set_defaults(func=cmd_reset)

    sp = sub.add_parser("optimize", help="walk-forward re-fit agent params (self-improvement)")
    sp.add_argument("--symbols", nargs="+", default=None, help="one or more historical symbols; default = config")
    sp.add_argument("--interval", default=None)
    sp.add_argument("--lookback-days", type=int, default=None)
    sp.add_argument("--samples", type=int, default=None)
    sp.add_argument("--min-trades", type=int, default=None)
    sp.add_argument("--seed", type=int, default=42)
    sp.set_defaults(func=cmd_optimize)

    sp = sub.add_parser("params", help="show currently-saved best params")
    sp.set_defaults(func=cmd_params)

    args = p.parse_args(argv)
    cfg = load_config(args.config)
    setup_logging(cfg["log_path"])
    try:
        args.func(args, cfg)
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
