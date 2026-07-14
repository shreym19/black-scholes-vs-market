"""
main.py
End-to-end pipeline: generate chain -> solve implied vols -> plot smile/surface
-> calibrate SVI per expiry -> quantify flat-vol vs SVI pricing error.

Run with real data by swapping generate_synthetic_chain() for a yfinance
pull that produces the same ['strike','expiry_days','expiry_years',
'moneyness','option_type','market_price'] dataframe shape.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import os

from synthetic_data import generate_synthetic_chain
from implied_vol import implied_vol
from bs_pricer import bs_price
from svi_calibration import calibrate_svi, svi_implied_vol

# Paths relative to this script's location -- works on any machine
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
PLOTS_DIR = os.path.join(PROJECT_DIR, "plots")
DATA_DIR = os.path.join(PROJECT_DIR, "data")
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Set to True (and set TICKER) to pull real market data via yfinance instead
# of the synthetic chain. Requires internet access and `pip install yfinance`.
USE_REAL_DATA = True
TICKER = "SPY"


def main():
    # ---- Step 1: get data ----
    if USE_REAL_DATA:
        from real_data import fetch_real_chain
        df, S, r, q = fetch_real_chain(TICKER)
        print(f"Loaded REAL data for {TICKER}: {len(df)} quotes | Spot={S:.2f} r={r} q={q}")
    else:
        df, S, r, q = generate_synthetic_chain()
        print(f"Loaded SYNTHETIC data: {len(df)} quotes | Spot={S} r={r} q={q}")

    # Use only OTM options per side (standard practice: OTM options are the
    # liquid, reliable quotes; ITM options are thinly traded / noisy)
    df["use_for_iv"] = (
        ((df["option_type"] == "call") & (df["moneyness"] >= 1.0)) |
        ((df["option_type"] == "put") & (df["moneyness"] < 1.0))
    )
    liquid = df[df["use_for_iv"]].copy()

    # ---- Step 2: solve implied vol for every quote ----
    ivs = []
    for _, row in liquid.iterrows():
        iv = implied_vol(
            market_price=row["market_price"], S=S, K=row["strike"],
            T=row["expiry_years"], r=r, q=q, option_type=row["option_type"],
        )
        ivs.append(iv)
    liquid["iv"] = ivs
    liquid = liquid.dropna(subset=["iv"])
    print(f"Successfully solved IV for {len(liquid)} / {len(df[df['use_for_iv']])} liquid quotes")

    n_expiries = liquid["expiry_days"].nunique()
    if n_expiries < 2 or len(liquid) < 10:
        raise ValueError(
            f"Not enough usable data to proceed (got {len(liquid)} quotes "
            f"across {n_expiries} expiries, need at least 2 expiries and "
            f"~10+ quotes). If using real data, check the "
            f"[real_data diagnostics] line above main.py's output to see "
            f"which filter is dropping everything."
        )

    # ---- Step 3: plot the smile for one expiry ----
    one_expiry_days = sorted(liquid["expiry_days"].unique())[1]  # 2nd shortest expiry
    slice_df = liquid[liquid["expiry_days"] == one_expiry_days].sort_values("strike")

    plt.figure(figsize=(8, 5))
    plt.plot(slice_df["strike"], slice_df["iv"] * 100, "o-", color="#c0392b")
    plt.axvline(S, color="gray", linestyle="--", alpha=0.6, label="Spot price")
    plt.xlabel("Strike")
    plt.ylabel("Implied Volatility (%)")
    plt.title(f"Volatility Smile/Skew — {one_expiry_days} Days to Expiry")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/01_smile.png", dpi=150)
    plt.close()
    print("Saved 01_smile.png")

    # ---- Step 4: plot the full 3D surface ----
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(liquid["strike"], liquid["expiry_days"], liquid["iv"] * 100,
               c=liquid["iv"], cmap="viridis", s=25)
    ax.set_xlabel("Strike")
    ax.set_ylabel("Days to Expiry")
    ax.set_zlabel("Implied Vol (%)")
    ax.set_title("Implied Volatility Surface")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/02_surface.png", dpi=150)
    plt.close()
    print("Saved 02_surface.png")

    # ---- Step 5: SVI calibration per expiry ----
    forward = S * np.exp((r - q) * slice_df["expiry_years"].iloc[0])  # per-slice below
    svi_results = {}
    fig, axes = plt.subplots(1, len(liquid["expiry_days"].unique()),
                              figsize=(22, 4), sharey=True)

    for ax_i, days in zip(axes, sorted(liquid["expiry_days"].unique())):
        sl = liquid[liquid["expiry_days"] == days].sort_values("strike")
        T = sl["expiry_years"].iloc[0]
        F = S * np.exp((r - q) * T)
        k = np.log(sl["strike"].values / F)
        iv_obs = sl["iv"].values

        params, rmse = calibrate_svi(k, iv_obs, T)
        svi_results[days] = {"params": params, "rmse": rmse, "T": T, "F": F}

        k_fine = np.linspace(k.min(), k.max(), 100)
        iv_fit = svi_implied_vol(params, k_fine, T)

        ax_i.scatter(k, iv_obs * 100, color="#c0392b", s=20, label="Market IV", zorder=3)
        ax_i.plot(k_fine, iv_fit * 100, color="#2980b9", label="SVI fit")
        ax_i.set_title(f"{days}D (RMSE={rmse*100:.2f}%)")
        ax_i.set_xlabel("log-moneyness")

    axes[0].set_ylabel("Implied Vol (%)")
    axes[0].legend()
    plt.suptitle("SVI Calibration per Expiry: Market IV vs Fitted Smile")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/03_svi_fits.png", dpi=150)
    plt.close()
    print("Saved 03_svi_fits.png")

    for days, res in svi_results.items():
        print(f"  Expiry {days}d: SVI RMSE = {res['rmse']*100:.3f}% vol pts")

    # ---- Step 6: quantify flat-vol vs SVI pricing error against market ----
    flat_vol = liquid["iv"].mean()  # naive "one number for everything" approach
    errors_flat, errors_svi = [], []

    for _, row in liquid.iterrows():
        days = row["expiry_days"]
        T = row["expiry_years"]
        K = row["strike"]
        opt_type = row["option_type"]
        market_price = row["market_price"]

        F = svi_results[days]["F"]
        params = svi_results[days]["params"]
        k = np.log(K / F)
        svi_vol = svi_implied_vol(params, np.array([k]), T)[0]

        price_flat = bs_price(S, K, T, r, flat_vol, q, opt_type)
        price_svi = bs_price(S, K, T, r, svi_vol, q, opt_type)

        errors_flat.append(abs(price_flat - market_price) / market_price * 100)
        errors_svi.append(abs(price_svi - market_price) / market_price * 100)

    liquid["pct_error_flat_vol"] = errors_flat
    liquid["pct_error_svi_vol"] = errors_svi

    print("\n--- Pricing Error Comparison (% abs error vs market price) ---")
    print(f"Flat-vol model  -> mean {np.mean(errors_flat):.2f}%  median {np.median(errors_flat):.2f}%")
    print(f"SVI-fit model   -> mean {np.mean(errors_svi):.2f}%  median {np.median(errors_svi):.2f}%")

    plt.figure(figsize=(8, 5))
    plt.hist(errors_flat, bins=25, alpha=0.6, label="Flat BS vol", color="#c0392b")
    plt.hist(errors_svi, bins=25, alpha=0.6, label="SVI-fit vol", color="#2980b9")
    plt.xlabel("Absolute Pricing Error vs Market (%)")
    plt.ylabel("Number of Options")
    plt.title("Pricing Error: Flat Black-Scholes Vol vs SVI-Calibrated Vol")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/04_error_comparison.png", dpi=150)
    plt.close()
    print("Saved 04_error_comparison.png")

    liquid.to_csv(os.path.join(DATA_DIR, "results.csv"), index=False)
    print("\nSaved full results to data/results.csv")

    return liquid, svi_results


if __name__ == "__main__":
    main()
