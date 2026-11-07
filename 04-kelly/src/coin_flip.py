"""
A coin that lands heads 55% of the time. Even-money bets: win what I
bet on heads, lose it on tails. Each round I bet a fraction f of my
current wealth. What f should I pick?
"""
import numpy as np
import matplotlib.pyplot as plt

P = 0.55
rng = np.random.default_rng(0)


def growth(f, p=P):
    # wealth MULTIPLIES each round: by (1+f) on a win, (1-f) on a loss.
    # after n rounds with k wins: W = W0 (1+f)^k (1-f)^(n-k)
    # take logs to turn the product into a sum (same trick as log returns
    # in project 03): log W = log W0 + k log(1+f) + (n-k) log(1-f)
    # divide by n and let n get big: k/n -> p (law of large numbers), so
    # the growth rate per round is
    #   g(f) = p log(1+f) + (1-p) log(1-f)
    return p * np.log1p(f) + (1 - p) * np.log1p(-f)


# Kelly: set g'(f) = 0
#   p/(1+f) - (1-p)/(1-f) = 0
#   p(1-f) = (1-p)(1+f)
#   p - pf = 1 - p + f - pf      <- careful: (1-p)(1+f) = 1 + f - p - pf
#   f = 2p - 1
# so with p = 0.55 the Kelly bet is 10% of wealth. the edge IS the bet.
F_STAR = 2 * P - 1
print(f"Kelly fraction f* = 2p - 1 = {F_STAR:.2f}")
print(f"growth at f*: {growth(F_STAR):.5f} per round")
print(f"growth at f*/2: {growth(F_STAR / 2):.5f}  ({growth(F_STAR / 2) / growth(F_STAR):.0%} of the max)")
print(f"growth at 2f*: {growth(2 * F_STAR):.5f}  (basically zero: double Kelly goes nowhere)")

# ---- simulate 10,000 gamblers for 1,000 rounds at a few fractions ----
# NOTE: the mean is a trap here. expected wealth after one round is
# W(1 + f(2p-1)), which grows with f, so "maximise the mean" says bet
# everything (f = 1). but then one tails wipes you out, and over 1000
# rounds you're broke with probability ~1. the mean is held up by a
# handful of absurdly lucky gamblers. the median is what a typical
# person actually experiences, and the median follows exp(n g(f)).
n_people, n_rounds = 10_000, 1_000
wins = rng.random((n_people, n_rounds)) < P
print("\n   f     median wealth   mean wealth     P(ever down 50%)")
fs = [0.05, 0.10, 0.20, 0.30]
paths = {}
for f in fs:
    mult = np.where(wins, 1 + f, 1 - f)
    w = np.cumprod(mult, axis=1)                 # wealth path, starting from 1
    peak = np.maximum.accumulate(w, axis=1)
    drawdown_50 = np.mean((w / peak).min(axis=1) <= 0.5)   # fell to half of a previous peak
    paths[f] = np.median(w, axis=0)
    print(f"  {f:.2f}   {np.median(w[:, -1]):12.2f}   {np.mean(w[:, -1]):12.2f}     {drawdown_50:.1%}")

# ---- figures ----
f_grid = np.linspace(0, 0.25, 300)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(f_grid, growth(f_grid), "k-")
ax[0].axvline(F_STAR, color="tab:green", ls="--", label="Kelly f* = 0.10")
ax[0].axvline(F_STAR / 2, color="tab:blue", ls=":", label="half Kelly")
ax[0].axhline(0, color="grey", lw=0.8)
ax[0].set_xlabel("fraction of wealth bet each round, f")
ax[0].set_ylabel("growth rate per round, g(f)")
ax[0].set_title("Bet too much and growth goes negative")
ax[0].legend()
for f in fs:
    ax[1].semilogy(paths[f], label=f"f = {f:.2f}")
ax[1].set_xlabel("round")
ax[1].set_ylabel("median wealth (log scale)")
ax[1].set_title("The typical gambler, 10,000 of them")
ax[1].legend()
for a in ax:
    a.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures/coin_flip.png", dpi=150)
print("saved figures/coin_flip.png")
