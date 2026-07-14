"""
real_data.py

Pulls a REAL options chain from Yahoo Finance via yfinance and reshapes it
into the exact same dataframe format that synthetic_data.py produces, so
main.py works unchanged -- just swap the import.

Run this on your own machine (needs internet access), not in a sandboxed env.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime


def fetch_real_chain(ticker="SPY", max_expiries=5, r=0.045, q=0.013,
                      min_moneyness=0.85, max_moneyness=1.15,
                      min_volume=0):
    """
    Fetches real option chain data for `ticker` and reshapes it to match
    generate_synthetic_chain()'s output: same columns, same units.

    r, q: risk-free rate and dividend yield. yfinance doesn't give you these
    directly -- either hardcode reasonable current values (check
    https://home.treasury.gov/resource-center/data-chart-center/interest-rates
    for the risk-free rate) or estimate q from the ticker's trailing
    dividend yield: tk.info.get('dividendYield', 0).

    min_volume: filters out illiquid/stale quotes. Default 0 (no filter) --
    yfinance's 'volume' field is frequently missing/stale/None even for
    liquid names, so filtering on it aggressively can silently drop almost
    everything. Raise this only if you have plenty of quotes to spare.
    """
    tk = yf.Ticker(ticker)
    spot = tk.history(period="1d")["Close"].iloc[-1]

    all_expiries = tk.options
    if not all_expiries:
        raise ValueError(f"No options data available for {ticker}")

    chosen_expiries = all_expiries[:max_expiries]
    today = datetime.now()

    rows = []
    diag = {"total_raw": 0, "dropped_moneyness": 0, "dropped_volume": 0,
            "dropped_price": 0, "kept": 0}

    for exp_str in chosen_expiries:
        exp_date = datetime.strptime(exp_str, "%Y-%m-%d")
        days = (exp_date - today).days
        if days <= 0:
            continue
        T = days / 365.0

        chain = tk.option_chain(exp_str)

        for opt_type, df in (("call", chain.calls), ("put", chain.puts)):
            for _, row in df.iterrows():
                diag["total_raw"] += 1
                strike = row["strike"]
                moneyness = strike / spot
                if not (min_moneyness <= moneyness <= max_moneyness):
                    diag["dropped_moneyness"] += 1
                    continue

                volume = row.get("volume", 0)
                volume = 0 if pd.isna(volume) else volume
                if volume < min_volume:
                    diag["dropped_volume"] += 1
                    continue

                bid = row.get("bid", np.nan)
                ask = row.get("ask", np.nan)
                last = row.get("lastPrice", np.nan)

                if not pd.isna(bid) and not pd.isna(ask) and bid > 0 and ask > 0:
                    price = (bid + ask) / 2.0
                elif not pd.isna(last) and last > 0:
                    # fall back to last traded price if bid/ask are missing
                    price = last
                else:
                    diag["dropped_price"] += 1
                    continue

                diag["kept"] += 1
                rows.append({
                    "expiry_days": days,
                    "expiry_years": T,
                    "strike": strike,
                    "moneyness": moneyness,
                    "option_type": opt_type,
                    "market_price": price,
                })

    print(f"[real_data diagnostics] raw={diag['total_raw']} "
          f"dropped_moneyness={diag['dropped_moneyness']} "
          f"dropped_volume={diag['dropped_volume']} "
          f"dropped_price={diag['dropped_price']} "
          f"kept={diag['kept']}")

    if not rows:
        raise ValueError(
            "No quotes survived filtering. Check the diagnostics line above "
            "to see which filter is eating everything, and widen "
            "min_moneyness/max_moneyness or lower min_volume accordingly."
        )

    return pd.DataFrame(rows), spot, r, q


if __name__ == "__main__":
    df, S, r, q = fetch_real_chain("SPY")
    print(f"Spot: {S:.2f}  r={r}  q={q}")
    print(f"Fetched {len(df)} liquid quotes across "
          f"{df['expiry_days'].nunique()} expiries")
    print(df.head(10))
