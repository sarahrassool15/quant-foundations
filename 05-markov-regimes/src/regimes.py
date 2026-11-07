"""
Is Bitcoin's volatility a Markov chain?

I label every day calm, normal or turbulent, then treat the labels like
Monopoly squares: count how often each state follows each other state,
and build a 3 x 3 transition matrix.

Then the real test. A Markov chain only remembers TODAY. If yesterday's
state still matters once I know today's, it isn't Markov.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = sys.argv[1] if len(sys.argv) > 1 else "data/btc_daily.csv"
df = pd.read_csv(path, parse_dates=["open_time"])

# daily volatility from the high and low (Parkinson's estimator).
# the range carries more information than open-to-close alone: a day can
# go up 10%, crash 10% and close flat. the 4 ln 2 makes it unbiased for
# Brownian motion. (it's E[(log H/L)^2] = 4 ln2 sigma^2 for BM.)
vol = np.sqrt(np.log(df["high"] / df["low"]) ** 2 / (4 * np.log(2)))

# three equal-sized groups by volatility: 0 calm, 1 normal, 2 turbulent.
# NOTE: the cut-offs use the WHOLE sample, so on day 1 I'm using knowledge of
# future volatility to label it. fine for describing the past, NOT fine if I
# wanted to forecast with this (that's leak 2 from project 02 again).
state = pd.qcut(vol, 3, labels=False).to_numpy()
names = ["calm", "normal", "turbulent"]
n = len(state)
print(f"{n} days")

# ---- transition matrix: count, then divide each row by its total ----
counts = np.zeros((3, 3))
for a, b in zip(state[:-1], state[1:]):
    counts[a, b] += 1
P = counts / counts.sum(axis=1, keepdims=True)
print("\ntransition matrix (row = today, column = tomorrow):")
print("              " + "  ".join(f"{s:>9s}" for s in names))
for i in range(3):
    print(f"  {names[i]:10s}  " + "  ".join(f"{P[i, j]:9.2f}" for j in range(3)))
print("  if days were independent, every entry would be 0.33")

# eigenvalues, same idea as Monopoly: the second one says how fast the
# market forgets which regime it was in
vals = np.sort(np.abs(np.linalg.eigvals(P)))[::-1]
print(f"\nsecond largest |eigenvalue| {vals[1]:.3f}: regime half-forgotten after "
      f"{np.log(0.5) / np.log(vals[1]):.1f} days")

# ---- how long does turbulence last? ----
# a Markov chain stays put with prob p each day, so a spell lasts k days with
# prob p^(k-1)(1-p): geometric, mean 1/(1-p). check that against the real spells.
def spells(s, target):
    out, run = [], 0
    for x in s:
        if x == target:
            run += 1
        elif run:
            out.append(run)
            run = 0
    if run:
        out.append(run)
    return np.array(out)

turb = spells(state, 2)
p_stay = P[2, 2]
print(f"\nturbulent spells: {len(turb)}")
print(f"  mean length: real {turb.mean():.2f} days, Markov predicts {1 / (1 - p_stay):.2f}")
print(f"  longest: {turb.max()} days")
for k in [3, 5, 10, 15]:
    print(f"  P(spell >= {k:2d} days): real {np.mean(turb >= k):.3f}   Markov {p_stay ** (k - 1):.3f}")

# ---- the Markov test: does the day BEFORE yesterday matter? ----
# compare P(turbulent tomorrow | turbulent today, turbulent yesterday)
# with    P(turbulent tomorrow | turbulent today, NOT turbulent yesterday).
# if Markov, these are the same number.
y, t0, t1 = state[:-2], state[1:-1], state[2:]
a = t1[(t0 == 2) & (y == 2)] == 2
b = t1[(t0 == 2) & (y != 2)] == 2
se = np.sqrt(a.mean() * (1 - a.mean()) / len(a) + b.mean() * (1 - b.mean()) / len(b))
print("\nP(turbulent tomorrow | turbulent today and ...)")
print(f"  ... turbulent yesterday:     {a.mean():.3f}  ({len(a)} cases)")
print(f"  ... not turbulent yesterday: {b.mean():.3f}  ({len(b)} cases)")
print(f"  difference {a.mean() - b.mean():+.3f}, about {(a.mean() - b.mean()) / se:.1f} standard errors")
# NOTE: standard error of a difference of two independent proportions:
# sqrt(p1(1-p1)/n1 + p2(1-p2)/n2). variances add, again.

# ---- figures ----
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
cmap = {0: "tab:green", 1: "tab:orange", 2: "tab:red"}
ax[0].scatter(df["open_time"], df["close"], c=[cmap[s] for s in state], s=2)
ax[0].set_yscale("log")
ax[0].set_ylabel("BTC price (log scale)")
ax[0].set_title("Every day coloured calm / normal / turbulent")
k = np.arange(1, turb.max() + 1)
ax[1].semilogy(k, [np.mean(turb >= kk) for kk in k], "o", color="tab:red", ms=4, label="real turbulent spells")
ax[1].semilogy(k, p_stay ** (k - 1), "k-", label="Markov chain prediction")
ax[1].set_xlabel("spell length, days")
ax[1].set_ylabel("fraction of spells at least this long")
ax[1].set_title("How long does turbulence last?")
ax[1].legend()
for a_ in ax:
    a_.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures/regimes.png", dpi=150)
print("saved figures/regimes.png")
