"""Entry point — fetch data, run backtest to N closed positions, write reports."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.backtester import Backtester, BacktestConfig
from src.data_loader import load_ohlcv
from src.indicators import enrich
from src.report import write_reports


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--interval", default="1h", choices=["15m", "1h", "4h", "1d"])
    p.add_argument("--start", default="2021-01-01")
    p.add_argument("--end", default=None, help="ISO date, default today")
    p.add_argument("--min-trades", type=int, default=1000)
    p.add_argument("--starting-equity", type=float, default=10_000.0)
    p.add_argument("--risk-per-trade", type=float, default=0.02)
    p.add_argument("--max-hold-bars", type=int, default=96)
    p.add_argument("--allow-shorts", action="store_true", default=True)
    p.add_argument("--no-shorts", dest="allow_shorts", action="store_false")
    p.add_argument("--out", default="reports")
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()
    print(f"[data] fetching {args.symbol} {args.interval} {args.start}..{args.end or 'today'}")
    df = load_ohlcv(args.symbol, args.interval, args.start, args.end)
    print(f"[data] {len(df):,} bars loaded ({df.timestamp.iloc[0]} -> {df.timestamp.iloc[-1]})")

    df = enrich(df)
    print(f"[data] indicators attached, usable bars: {df.dropna().shape[0]:,}")

    cfg = BacktestConfig(
        starting_equity=args.starting_equity,
        risk_per_trade=args.risk_per_trade,
        max_hold_bars=args.max_hold_bars,
        allow_shorts=args.allow_shorts,
        min_trades=args.min_trades,
    )
    bt = Backtester(cfg)
    print(f"[bt] running backtest, target {cfg.min_trades} closed positions...")
    result = bt.run(df)
    print(f"[bt] {len(result.trades):,} trades closed in {time.time()-t0:.1f}s")

    out_dir = Path(args.out) / f"{args.symbol}_{args.interval}"
    summary = write_reports(result, out_dir, cfg.starting_equity)
    print(f"[report] written to {out_dir}")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
