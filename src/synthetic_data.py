"""
synthetic_data.py

Generates a realistic SYNTHETIC options chain, used here only because this
sandbox can't reach Yahoo Finance's servers. It mimics real market behavior:
  - implied vol is HIGHER for downside puts (skew), consistent with
    crash-risk premium seen in real equity index options since 2008
  - implied vol curves up somewhat for far OTM calls too (smile)
  - vol is roughly higher for shorter-dated options in each slice

SWAP TO REAL DATA (run this on your own machine, not this sandbox):

    import yfinance as yf
    tk = yf.Ticker("SPY")
    expiries = tk.options                     # list of available expiry date strings
    chain = tk.option_chain(expiries[0])       # -> chain.calls, chain.puts (DataFrames)
    # each has: strike, lastPrice, bid, ask, volume, openInterest, impliedVolatility...
    # build the same ['strike','expiry_years','market_price','option_type'] shape
    # from chain.calls / chain.puts and feed it into implied_vol_dataframe().
"""

import numpy as np
import pandas as pd
from bs_pricer import bs_price

np.random.seed(42)


def true_vol_surface(moneyness, T):
    """
    A hand-built 'true' vol surface with realistic skew + smile + term structure.
    moneyness = K / S  (1.0 = at the money)
    T         = time to expiry in years
    """
    # Skew: lower strikes (moneyness < 1, puts/downside) get higher vol
    skew = 0.10 * (1.0 - moneyness)
    # Smile: convex bump away from at-the-money in both directions
    smile = 0.15 * (moneyness - 1.0) ** 2
    # Term structure: short-dated options a bit more turbulent
    term = 0.03 * np.exp(-3 * T)
    # Base level
    base = 0.18
    vol = base + skew + smile + term
    return np.clip(vol, 0.05, 1.5)


def generate_synthetic_chain(S=500.0, r=0.045, q=0.013,
                              expiries_days=(14, 30, 60, 90, 180),
                              strike_pcts=np.arange(0.85, 1.16, 0.025)):
    """
    Build a synthetic options chain (both calls and puts) across several
    expiries and strikes, priced off a realistic 'true' vol surface,
    with small bid/ask-style noise added to mimic real quote imperfection.
    """
    rows = []
    for days in expiries_days:
        T = days / 365.0
        for pct in strike_pcts:
            K = round(S * pct, 2)
            moneyness = K / S
            true_iv = true_vol_surface(moneyness, T)

            for opt_type in ("call", "put"):
                fair_price = bs_price(S, K, T, r, true_iv, q, opt_type)
                # add small realistic noise (bid/ask spread + rounding)
                noise = np.random.normal(0, 0.003 * fair_price + 0.01)
                market_price = max(fair_price + noise, 0.01)

                rows.append({
                    "expiry_days": days,
                    "expiry_years": T,
                    "strike": K,
                    "moneyness": moneyness,
                    "option_type": opt_type,
                    "true_iv": true_iv,      # kept only for validation/plots
                    "market_price": round(market_price, 2),
                })

    return pd.DataFrame(rows), S, r, q


if __name__ == "__main__":
    df, S, r, q = generate_synthetic_chain()
    print(f"Spot: {S}, r: {r}, q: {q}")
    print(f"Generated {len(df)} option quotes across "
          f"{df['expiry_days'].nunique()} expiries and "
          f"{df['strike'].nunique()} strikes.")
    print(df.head(10))
