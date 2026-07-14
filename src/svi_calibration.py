"""
svi_calibration.py

Fits the SVI (Stochastic Volatility Inspired) parametrization to a
single expiry's implied vol smile.

SVI formula (raw form, Gatheral 2004), in TOTAL VARIANCE w = sigma^2 * T:

    w(k) = a + b * ( rho*(k - m) + sqrt((k - m)^2 + s^2) )

    k = log(K/F)   -- log-moneyness relative to the forward price
    a  -- overall level of variance
    b  -- controls the angle/steepness of the wings (b >= 0)
    rho -- controls the skew/rotation of the smile (-1 < rho < 1)
    m  -- horizontal shift of the smile
    s  -- controls the smoothness/curvature at the minimum (s > 0)

We fit (a, b, rho, m, s) by least squares against the OBSERVED total
variance at each strike, then convert back to implied vol via
sigma(k) = sqrt(w(k) / T).
"""

import numpy as np
from scipy.optimize import least_squares


def svi_total_variance(params, k):
    a, b, rho, m, s = params
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + s ** 2))


def svi_implied_vol(params, k, T):
    w = svi_total_variance(params, k)
    w = np.clip(w, 1e-8, None)  # guard against negative variance from a bad fit
    return np.sqrt(w / T)


def _initial_guess(k, w_obs):
    a0 = max(np.min(w_obs), 1e-4)
    b0 = 0.1
    rho0 = -0.3  # negative = typical equity skew (puts pricier)
    m0 = 0.0
    s0 = 0.1
    return np.array([a0, b0, rho0, m0, s0])


def calibrate_svi(k, iv_obs, T):
    """
    k       : array of log-moneyness values, log(K/F)
    iv_obs  : array of OBSERVED implied vols at those strikes
    T       : time to expiry (years) for this slice

    Returns: (fitted_params, rmse_iv)
    """
    k = np.asarray(k)
    iv_obs = np.asarray(iv_obs)
    w_obs = (iv_obs ** 2) * T

    x0 = _initial_guess(k, w_obs)

    def residuals(params):
        w_model = svi_total_variance(params, k)
        return w_model - w_obs

    # bounds keep the fit well-behaved: b>=0, -1<rho<1, s>0
    lower = [-1.0, 0.0, -0.999, -2.0, 1e-4]
    upper = [5.0, 5.0, 0.999, 2.0, 5.0]

    result = least_squares(residuals, x0, bounds=(lower, upper))
    fitted_iv = svi_implied_vol(result.x, k, T)
    rmse = np.sqrt(np.mean((fitted_iv - iv_obs) ** 2))
    return result.x, rmse


if __name__ == "__main__":
    # Self-test on synthetic smile data
    rng = np.random.default_rng(0)
    k_test = np.linspace(-0.3, 0.3, 15)
    T_test = 0.25
    true_params = np.array([0.02, 0.15, -0.4, 0.0, 0.1])
    w_true = svi_total_variance(true_params, k_test)
    iv_true = np.sqrt(w_true / T_test)
    iv_noisy = iv_true + rng.normal(0, 0.003, size=len(k_test))

    fitted, rmse = calibrate_svi(k_test, iv_noisy, T_test)
    print("Fitted SVI params (a, b, rho, m, s):", np.round(fitted, 4))
    print(f"RMSE (vol points): {rmse:.5f}")
