import numpy as np
from scipy.stats import norm


def call_price(S, K, tau, sigma, r=0.0):
    tau = np.maximum(tau, 1e-12)       # at expiry tau = 0 and I'd divide by zero
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    return S * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)


def call_delta(S, K, tau, sigma, r=0.0):
    # delta = N(d1): how many units of the asset to hold to cancel the
    # option's sensitivity to small price moves
    tau = np.maximum(tau, 1e-12)
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * tau) / (sigma * np.sqrt(tau))
    return norm.cdf(d1)
