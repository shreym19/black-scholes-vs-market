"""
implied_vol.py
Back out implied volatility from an observed market option price,
using Brent's method (robust bracketing root-finder).
"""

import numpy as np
from scipy.optimize import brentq
from bs_pricer import bs_price


def implied_vol(market_price, S, K, T, r, q=0.0, option_type="call",
                 vol_lower=1e-4, vol_upper=5.0):
    """
    Solve for sigma such that bs_price(S,K,T,r,sigma) == market_price.

    Returns np.nan if no solution is found in [vol_lower, vol_upper]
    (this happens for bad/illiquid quotes -- e.g. price violates
    no-arbitrage bounds -- and should be dropped, not forced).
    """
    def objective(sigma):
        return bs_price(S, K, T, r, sigma, q, option_type) - market_price

    try:
        # Brent's method requires the objective to have opposite signs
        # at the two endpoints (a bracket). If it doesn't, the quote
        # is outside what's achievable -- treat as bad data.
        lo, hi = objective(vol_lower), objective(vol_upper)
        if lo * hi > 0:
            return np.nan
        return brentq(objective, vol_lower, vol_upper, xtol=1e-6)
    except (ValueError, RuntimeError):
        return np.nan


def implied_vol_dataframe(df, S, r, q=0.0):
    """
    Vectorized-ish helper: given a dataframe with columns
    ['strike', 'expiry_years', 'market_price', 'option_type'],
    returns the same df with an 'iv' column added.
    """
    ivs = []
    for _, row in df.iterrows():
        iv = implied_vol(
            market_price=row["market_price"],
            S=S,
            K=row["strike"],
            T=row["expiry_years"],
            r=r,
            q=q,
            option_type=row["option_type"],
        )
        ivs.append(iv)
    df = df.copy()
    df["iv"] = ivs
    return df


if __name__ == "__main__":
    # Self-test: price an option at a known vol, then recover that vol
    S, K, T, r, true_sigma = 100, 105, 0.25, 0.03, 0.22
    price = bs_price(S, K, T, r, true_sigma, option_type="call")
    recovered = implied_vol(price, S, K, T, r, option_type="call")
    print(f"True sigma:      {true_sigma:.4f}")
    print(f"Recovered sigma: {recovered:.4f}")
    print(f"Match: {abs(true_sigma - recovered) < 1e-4}")
