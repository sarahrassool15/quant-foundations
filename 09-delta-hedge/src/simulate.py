"""
Sell a 30-day at-the-money call, then delta hedge it.
Bitcoin-ish numbers: 60% volatility, zero interest to keep it simple.

P&L of the hedged short call = premium received
                              + gains on the shares I hold
                              - what I pay out at expiry
If Black-Scholes were perfect and I hedged continuously, this would be
exactly zero. It isn't, for three reasons, one at a time below.
"""
import numpy as np
import matplotlib.pyplot as plt
from bs import call_price, call_delta

S0 = K = 100.0
T = 30 / 365
SIGMA = 0.60
rng = np.random.default_rng(0)


def hedge(n_rebal, sigma_real=SIGMA, sigma_hedge=SIGMA, cost=0.0, n_paths=20_000):
    dt = T / n_rebal
    # real price path: GBM with the REAL volatility (zero drift, r = 0)
    z = rng.standard_normal((n_paths, n_rebal))
    log_s = np.log(S0) + np.cumsum(-0.5 * sigma_real ** 2 * dt + sigma_real * np.sqrt(dt) * z, axis=1)
    S = np.hstack([np.full((n_paths, 1), S0), np.exp(log_s)])

    premium = call_price(S0, K, T, sigma_hedge)       # I sell at the vol I believe in
    pnl = np.full(n_paths, premium)
    held = np.zeros(n_paths)
    for i in range(n_rebal):
        tau = T - i * dt
        target = call_delta(S[:, i], K, tau, sigma_hedge)
        pnl -= cost * S[:, i] * np.abs(target - held)  # pay a fraction of what I trade
        held = target
        pnl += held * (S[:, i + 1] - S[:, i])          # gain/loss on the shares over the step
    pnl -= np.maximum(S[:, -1] - K, 0)                 # pay the option holder at expiry
    return pnl / premium                               # as a fraction of the premium


# ---- 1. how often I rebalance ----
# each time I wait, the price moves and my delta is stale. the option's
# curvature (gamma) means a stale hedge loses on big moves and wins on
# small ones. those don't cancel exactly, they average out, so the error
# shrinks like 1/sqrt(n), not 1/n. (sqrt again, this keeps happening.)
print("1. rebalancing frequency (true vol = hedge vol = 60%)")
ns = [4, 8, 16, 32, 64, 128, 256, 512, 720]
sds = []
for n in ns:
    p = hedge(n)
    sds.append(p.std())
    print(f"   {n:4d} times  ({30 * 24 / n:5.1f} h apart)   mean {p.mean():+.3f}   std {p.std():.3f}")
slope = np.polyfit(np.log(ns), np.log(sds), 1)[0]
print(f"   slope {slope:.2f}  (theory -0.5)")

# ---- 2. the wrong volatility ----
# I sell and hedge at 60%, but the market actually moves at some other vol.
# then I lose or gain about the difference between the two option prices,
# and the hedge can't save me: it only removes the direction of the price,
# not how much it wiggles.
print("\n2. real vol different from my vol (hedge 60%, rebalance hourly)")
real_vols = [0.40, 0.50, 0.60, 0.70, 0.80]
wrong_mean, wrong_theory = [], []
for sv in real_vols:
    p = hedge(720, sigma_real=sv)
    th = (call_price(S0, K, T, SIGMA) - call_price(S0, K, T, sv)) / call_price(S0, K, T, SIGMA)
    wrong_mean.append(p.mean())
    wrong_theory.append(th)
    print(f"   real vol {sv:.0%}:  mean P&L {p.mean():+.3f}   price difference {th:+.3f}")

# ---- 3. trading costs ----
# more rebalancing = smaller hedging error but more trading. with a cost
# per trade there's a sweet spot.
print("\n3. with a 0.05% trading cost")
cost_mean, cost_sd = [], []
for n in ns:
    p = hedge(n, cost=0.0005)
    cost_mean.append(p.mean())
    cost_sd.append(p.std())
    rmse = np.sqrt(p.mean() ** 2 + p.std() ** 2)
    print(f"   {n:4d} times   mean {p.mean():+.3f}   std {p.std():.3f}   root mean square {rmse:.3f}")
rmse = np.sqrt(np.array(cost_mean) ** 2 + np.array(cost_sd) ** 2)
best = ns[int(np.argmin(rmse))]
print(f"   best: {best} rebalances ({30 * 24 / best:.0f} hours apart)")

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].loglog(ns, sds, "o-", label="std of hedged P&L")
ax[0].loglog(ns, sds[0] * np.sqrt(ns[0] / np.array(ns)), "k--", label="1/√n")
ax[0].set_xlabel("rebalances over 30 days")
ax[0].set_ylabel("std of P&L / premium")
ax[0].set_title("Hedge more often, leak less")
ax[0].legend()
ax[1].plot(np.array(real_vols) * 100, wrong_mean, "o", label="simulated mean P&L")
ax[1].plot(np.array(real_vols) * 100, wrong_theory, "k-", label="price at 60% minus price at real vol")
ax[1].axhline(0, color="grey", lw=0.8)
ax[1].set_xlabel("real volatility (%)")
ax[1].set_ylabel("mean P&L / premium")
ax[1].set_title("Sold at 60%: the hedge can't fix the wrong vol")
ax[1].legend(fontsize=8)
ax[2].semilogx(ns, rmse, "o-", color="tab:red", label="with 0.05% cost")
ax[2].semilogx(ns, sds, "o--", color="grey", label="no cost")
ax[2].set_xlabel("rebalances over 30 days")
ax[2].set_ylabel("root mean square P&L / premium")
ax[2].set_title("With costs there's a sweet spot")
ax[2].legend()
for a in ax:
    a.grid(alpha=0.3, which="both")
plt.tight_layout()
plt.savefig("figures/simulation.png", dpi=150)
print("saved figures/simulation.png")
