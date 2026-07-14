# Black-Scholes vs. Market Reality

**Thesis:** Black-Scholes assumes one constant volatility for every strike and
expiry. Real options markets don't price that way — they embed a volatility
smile/skew that reflects crash risk, fat tails, and supply/demand. This
project quantifies exactly how much pricing error that flat assumption
introduces, and fixes it using SVI, the industry-standard smile
parametrization, calibrated to real market quotes.

## Results (this run)

| Model | Mean abs. pricing error vs market | Median |
|---|---|---|
| Flat Black-Scholes vol | ~20% | ~10% |
| SVI-calibrated vol | ~5% | ~0.4% |

SVI fit quality per expiry (RMSE in vol points) ranged from ~0.03% (longer
expiries) to ~0.8% (shortest expiry, where noise dominates) — consistent
with SVI's known strength on liquid, longer-dated slices.

## Pipeline

1. `src/bs_pricer.py` — closed-form BS price + all 5 Greeks, with a
   put-call parity self-test.
2. `src/implied_vol.py` — Brent's method root-finder to back out implied
   vol from a market price (more robust than Newton-Raphson near
   expiry/deep ITM).
3. `src/synthetic_data.py` — **stand-in for real market data.** See note below.
4. `src/svi_calibration.py` — fits the 5-parameter SVI curve
   (Gatheral 2004) to each expiry's smile via least squares.
5. `src/main.py` — runs the full pipeline and produces all plots + the
   error comparison.

##  Important: this uses synthetic data

The pipeline logic (pricer, IV solver, SVI calibration, error analysis) is
fully real and tested. The **input data is synthetic** — generated from a
hand-built "true" vol surface with realistic skew/smile/term-structure
baked in — because this was built in a sandboxed environment with no
internet access to Yahoo Finance.

**To run this on real market data on your own machine:**

```bash
pip install yfinance
```

```python
import yfinance as yf
tk = yf.Ticker("SPY")
expiries = tk.options                    # available expiry date strings
chain = tk.option_chain(expiries[0])     # -> chain.calls, chain.puts

# Reshape into the same columns main.py expects:
# ['strike', 'expiry_days', 'expiry_years', 'moneyness', 'option_type', 'market_price']
# Use chain.calls/chain.puts' 'strike' and ('bid'+'ask')/2 as market_price
# (mid price is more reliable than lastPrice, which can be stale).
```

Then swap the `generate_synthetic_chain()` call in `main.py` for this real
pull. Everything downstream (IV solving, SVI fit, error comparison, plots)
works unchanged — that's the entire point of separating data from logic.

## Limitations / honest caveats

- Synthetic data by necessity here (see above) — swap to real data before
  presenting this as an empirical result.
- SVI is fit expiry-by-expiry with no cross-expiry arbitrage constraint
  enforced (calendar-spread arbitrage-free SVI is a further refinement —
  see Gatheral & Jacquier 2014).
- Flat-vol baseline uses a single average IV across the whole chain as the
  "naive" comparison — a real trading desk would at least use ATM vol per
  expiry, which would narrow (but not close) the gap shown here.
- Deep ITM quotes are excluded (kept only OTM per side) since they're
  typically illiquid/noisy in real markets — standard practice, but worth
  stating explicitly.

## Possible extensions

- Add Heston (stochastic vol) and/or Merton jump-diffusion, calibrate
  alongside SVI, compare all three.
- Delta-hedging backtest: simulate hedging using flat vol vs. SVI-implied
  vol and measure real hedging P&L variance.
- Enforce arbitrage-free SVI across the full surface (butterfly + calendar
  spread constraints).
