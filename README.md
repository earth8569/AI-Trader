<<<<<<< HEAD
# AI-Trader
=======
# AI-Trader (HKUDS-inspired)

Multi-agent "collective intelligence" trading system. Faithful to the HKUDS/AI-Trader
debate pattern: specialized agents (Technical, Sentiment, Risk) produce signals, and a
Decision agent aggregates them into a single action. Backtested on historical crypto
data until at least 1,000 closed positions are generated.

## Architecture

```
AI-Trading/
├── data/                 cached OHLCV parquet/csv (downloaded)
├── reports/              backtest output (trades.csv, summary.json, equity.csv)
├── src/
│   ├── data_loader.py    Binance public klines fetcher (paginated)
│   ├── indicators.py     RSI / MACD / BBands / ATR / SMA / EMA
│   ├── agents/
│   │   ├── technical.py  Trend + momentum vote
│   │   ├── sentiment.py  Volume + price-action regime
│   │   ├── risk.py       Volatility + drawdown gate
│   │   └── decision.py   Weighted debate → BUY/SELL/HOLD
│   ├── backtester.py     bar-by-bar engine, position tracking, metrics
│   └── report.py         summary + equity curve output
├── run_backtest.py       entry point
└── requirements.txt
```

## Usage

```bash
pip install -r requirements.txt
```

### Backtest (historical)

```bash
python run_backtest.py --symbol BTCUSDT --interval 1h --min-trades 1000
```

Output in `reports/`:
- `trades.csv` — every closed position
- `equity.csv` — bar-by-bar equity curve
- `summary.json` — win rate, Sharpe, max drawdown, profit factor, etc.

### Live paper trading (automation)

Edit `config.json` (symbols, timeframe, equity, risk) then:

```bash
python run_live.py tick          # one-shot — cron-friendly
python run_live.py start         # daemon loop (blocking)
python run_live.py status        # portfolio snapshot
python run_live.py history -n 20 # recent closed trades
python run_live.py reset --yes   # wipe paper state
```

State persists to `state/portfolio.json`; logs rotate in `logs/trader.log`.

### Cron / Task Scheduler

Run a tick every 15 minutes:

- Linux/macOS: `*/15 * * * * cd /path/to/AI-Trading && python run_live.py tick`
- Windows: Task Scheduler → program `python`, arguments `run_live.py tick`, start in project dir.

Or just run `python run_live.py start` under a process manager.

### OKX demo trading (paper via real exchange)

Zero-risk way to test against OKX infrastructure — orders are simulated, fills
use real market data, no capital moves.

**1. Mint demo API keys**
- On okx.com switch to **Demo Trading** (top-right toggle).
- Profile → API → create key with "Trade" permission. Save key, secret, passphrase.
- Demo keys are separate from live keys and only work with the `x-simulated-trading` header.

**2. Export env vars** (copy `.env.example` → `.env`, fill in, then source it)
```bash
export OKX_API_KEY=...
export OKX_API_SECRET=...
export OKX_API_PASSPHRASE=...
export OKX_DEMO=1
```
PowerShell equivalent:
```powershell
$env:OKX_API_KEY="..."; $env:OKX_API_SECRET="..."; $env:OKX_API_PASSPHRASE="..."; $env:OKX_DEMO="1"
```

**3. Flip the broker flag** in `config.json`:
```json
{ "broker": "okx", "feed": "okx", ... }
```

**4. Run the usual commands** — tick/start/status/history/reset work identically:
```bash
python run_live.py status      # pulls live demo balance + live marks
python run_live.py tick        # decides and places demo orders
python run_live.py start       # daemon
```

**Important notes**
- **Long-only on OKX:** the demo adapter uses SPOT (`tdMode: "cash"`). Shorts are disabled automatically when `broker: okx`. Enabling shorts requires a SWAP adapter (not built yet — leverage, funding, liquidation risk).
- **`reset` only wipes local state** — it does NOT sell open demo positions on OKX. Close those in the OKX UI if you want a truly fresh start.
- **Live trading is double-gated:** setting `OKX_DEMO=0` will raise unless `OKX_ENABLE_LIVE=1` is also set. Leave both alone to stay on demo.
- Secrets are loaded from env only; `.env` is in `.gitignore`. Never paste keys into config.json.

### Self-improvement — walk-forward optimization

The agents have tunable knobs (decision weights, entry threshold, risk veto, ATR stop/target).
Rather than leave them hand-picked, the optimizer re-fits them on recent history:

```bash
python run_live.py optimize                    # use config defaults (90d, 40 samples)
python run_live.py optimize --lookback-days 180 --samples 60
python run_live.py params                      # inspect current saved winner
```

What it does:
1. Pulls `optimize_lookback_days` of historical candles (Binance).
2. Random-searches `optimize_samples` parameter candidates (target > stop enforced).
3. Scores each = expectancy × sample-size factor − drawdown penalty.
4. Disqualifies runs below `optimize_min_trades` (sample size floor).
5. Writes winner to `state/best_params.json` with provenance + keeps last 20 snapshots.
6. `LiveTrader` reads this file on startup; next `tick` / `start` uses the new params.

Run it weekly (or on a cron) to adapt to regime changes. All params stay
interpretable — no black-box ML.

### OKX futures (perpetual swaps) — agent-chosen leverage + smart TP/SL

The swap broker trades OKX perpetuals with isolated margin, long + short, and
lets the tactics module pick **per-trade leverage** plus structural stop/target
levels based on volatility + recent swing.

**Enable:**
```json
{ "broker": "okx_swap", "feed": "okx" }
```

Keys set elsewhere:
- `okx_margin_mode: "isolated"` — each position has its own margin bucket
- `okx_position_mode: "long_short_mode"` — longs and shorts can coexist per symbol
- `max_leverage` (in [src/params.py](src/params.py) `AgentParams`) — hard cap, default **5×**

**How the agent decides leverage** (see [src/tactics.py](src/tactics.py) `plan_position`):
1. `conviction` = |decision score| from the debate (0..1)
2. `vol_factor` = leverage_vol_ref_pct / ATR_pct, clamped to [0.3, 1.0]
3. `leverage = round(1 + conviction × (max_leverage − 1) × vol_factor)`, floored at 1, capped at max.

So a high-conviction signal in a calm market gets 4–5×; a weak signal in a
volatile regime gets 1×. Risk per trade stays capped at `risk_per_trade` of
equity *if the stop holds* — leverage only expands the notional cap.

**How TP/SL are decided:**
- **Stop**: blended ATR distance (`atr_mult_stop × ATR`) with the **recent
  20-bar swing low/high** (structural). Whichever is further becomes the stop,
  so it respects both volatility and obvious chart levels.
- **Target**: at least 1.5R (risk-reward floor), capped by the upper/lower
  Bollinger band ±0.5 ATR. Prevents unrealistic targets beyond typical
  2-sigma range in one leg.

**Before you run:** OKX demo has *separate* wallets for spot and futures. A
fresh demo account often starts with near-$0 futures USDT. Top up on OKX:
Demo Trading → Assets → **Reset Demo Balance** (gives $100k virtual USDT).

```bash
python run_live.py status      # should show swap equity > 0
python run_live.py tick        # one tick: plan + route orders
python run_live.py start       # daemon with auto-optimize + leverage
```
>>>>>>> 0d8220d (First Commit)
