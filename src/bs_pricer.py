"""
bs_pricer.py
Black-Scholes closed-form pricing and Greeks for European options.

Conventions:
    S     - current stock (spot) price
    K     - strike price
    T     - time to expiry, in YEARS (e.g. 30 days = 30/365)
    r     - risk-free interest rate (annualized, e.g. 0.05 for 5%)
    sigma - volatility (annualized, e.g. 0.20 for 20%)
    q     - continuous dividend yield (default 0)
"""

import numpy as np
from scipy.stats import norm


def _d1_d2(S, K, T, r, sigma, q=0.0):
    """Compute the d1 and d2 terms shared by price and Greek formulas."""
    if T <= 0 or sigma <= 0:
        raise ValueError("T and sigma must be positive.")
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def bs_price(S, K, T, r, sigma, q=0.0, option_type="call"):
    """Black-Scholes price of a European call or put."""
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    if option_type == "call":
        price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    return price


def bs_greeks(S, K, T, r, sigma, q=0.0, option_type="call"):
    """
    Returns a dict of the five standard Greeks.
    delta: sensitivity to $1 move in stock
    gamma: sensitivity of delta to $1 move in stock (same for call & put)
    vega:  sensitivity to a 1.00 (100%) move in vol -> we report per 1% move
    theta: sensitivity to 1 day passing (per-day, not per-year)
    rho:   sensitivity to a 1.00 (100%) move in rates -> we report per 1% move
    """
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    pdf_d1 = norm.pdf(d1)

    gamma = np.exp(-q * T) * pdf_d1 / (S * sigma * np.sqrt(T))
    vega = S * np.exp(-q * T) * pdf_d1 * np.sqrt(T) / 100  # per 1% vol move

    if option_type == "call":
        delta = np.exp(-q * T) * norm.cdf(d1)
        theta = (
            -S * np.exp(-q * T) * pdf_d1 * sigma / (2 * np.sqrt(T))
            - r * K * np.exp(-r * T) * norm.cdf(d2)
            + q * S * np.exp(-q * T) * norm.cdf(d1)
        ) / 365  # per day
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100  # per 1% rate move
    elif option_type == "put":
        delta = -np.exp(-q * T) * norm.cdf(-d1)
        theta = (
            -S * np.exp(-q * T) * pdf_d1 * sigma / (2 * np.sqrt(T))
            + r * K * np.exp(-r * T) * norm.cdf(-d2)
            - q * S * np.exp(-q * T) * norm.cdf(-d1)
        ) / 365
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


def check_put_call_parity(S, K, T, r, q=0.0, sigma=0.2, tol=1e-6):
    """
    Sanity check: C - P should equal S*exp(-qT) - K*exp(-rT).
    Returns True if the identity holds within tolerance.
    """
    c = bs_price(S, K, T, r, sigma, q, "call")
    p = bs_price(S, K, T, r, sigma, q, "put")
    lhs = c - p
    rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
    return abs(lhs - rhs) < tol, lhs, rhs


if __name__ == "__main__":
    # Quick self-test
    S, K, T, r, sigma = 100, 100, 0.5, 0.03, 0.20
    call = bs_price(S, K, T, r, sigma, option_type="call")
    put = bs_price(S, K, T, r, sigma, option_type="put")
    greeks_c = bs_greeks(S, K, T, r, sigma, option_type="call")

    print(f"Call price: {call:.4f}")
    print(f"Put price:  {put:.4f}")
    print(f"Call Greeks: {greeks_c}")

    ok, lhs, rhs = check_put_call_parity(S, K, T, r, sigma=sigma)
    print(f"\nPut-call parity check: C-P={lhs:.6f}  S-Ke^-rT={rhs:.6f}  OK={ok}")
