"""
How much Bitcoin would Kelly have told me to hold, and how sure could I be?
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = sys.argv[1] if len(sys.argv) > 1 else "data/btc_daily.csv"
df = pd.read_csv(path, parse_dates=["open_time"])
# SIMPLE returns here, not log: Kelly is about how my wealth changes when
# I hold a fraction f, and my wealth changes by f * R where R is the
# simple return. (logs come in when I measure growth, below.)
R = df["close"].pct_change().dropna().to_numpy()
years = len(R) / 365          # crypto trades every day, so 365 not 252
print(f"{len(R)} daily returns, {years:.1f} years")

# annualise: the mean scales with time, the variance scales with time,
# so the std scales with sqrt(time). (variance = time again.)
mu = R.mean() * 365
sigma = R.std() * np.sqrt(365)
f_hat = mu / sigma ** 2
print(f"\nmu    = {mu:.1%} per year")
print(f"sigma = {sigma:.1%} per year")
print(f"Kelly f = mu / sigma^2 = {f_hat:.2f}")
# NOTE: f > 1 would mean borrowing to hold more than 100% of my wealth in
# Bitcoin. f < 0 would mean shorting it.

# ---- how sure am I? ----
# the standard error of a mean is std / sqrt(n). annualised, that's
# sigma / sqrt(years). this is the killer: it shrinks with sqrt of TIME,
# not with how many data points I have. sampling every minute instead of
# every day doesn't help estimate the drift at all.
se_mu = sigma / np.sqrt(years)
se_f = se_mu / sigma ** 2
print(f"\nstandard error of mu: {se_mu:.1%} per year")
print(f"so f is {f_hat:.2f} +/- {se_f:.2f}  (95% range roughly "
      f"{f_hat - 1.96 * se_f:.2f} to {f_hat + 1.96 * se_f:.2f})")
print(f"noise / edge = {se_f / abs(f_hat):.2f}")
# plug into the shrinkage formula from shrinkage.py. crude, because I'm
# using my noisy f_hat as if it were the true f*, but it shows the scale.
c = f_hat ** 2 / (f_hat ** 2 + se_f ** 2)
print(f"shrinkage formula says bet c = {c:.2f} of Kelly, i.e. f = {c * f_hat:.2f}")

# ---- the growth I'd actually have got (in hindsight) ----
# hold fraction f, rebalanced daily: wealth multiplies by (1 + f R) each day.
# growth = average of log(1 + f R), times 365.
# careful: if f R < -1 on any day, I've lost everything (log of a negative).
f_grid = np.linspace(-0.5, 3, 141)
g = []
for f in f_grid:
    m = 1 + f * R
    g.append(np.mean(np.log(m)) * 365 if np.all(m > 0) else np.nan)
g = np.array(g)
best = f_grid[np.nanargmax(g)]
print(f"\nhindsight-best fraction: {best:.2f}, growth {np.nanmax(g):.1%} per year")
print(f"growth holding 100% BTC: {np.mean(np.log1p(R)) * 365:.1%} per year")
print(f"worst day: {R.min():.1%}  -> any f above {-1 / R.min():.1f} would have been wiped out that day")

# ---- rolling 1-year Kelly: what would I have believed at each point? ----
s = pd.Series(R, index=df["open_time"].iloc[1:])
roll_mu = s.rolling(365).mean() * 365
roll_var = s.rolling(365).var() * 365
roll_f = (roll_mu / roll_var).dropna()
print(f"\nrolling 1-year Kelly fraction ranged from {roll_f.min():.2f} to {roll_f.max():.2f}")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
ax[0].plot(f_grid, g, "k-")
ax[0].axvline(best, color="tab:green", ls="--", label=f"hindsight best f = {best:.2f}")
ax[0].axvspan(f_hat - 1.96 * se_f, f_hat + 1.96 * se_f, color="tab:orange", alpha=0.15,
              label="95% range for f from the data")
ax[0].axhline(0, color="grey", lw=0.8)
ax[0].set_xlabel("fraction of wealth in Bitcoin, f")
ax[0].set_ylabel("growth per year")
ax[0].set_title("Bitcoin, held at fraction f, rebalanced daily")
ax[0].legend(fontsize=8)
ax[1].plot(roll_f.index, roll_f.values, color="tab:blue")
ax[1].axhline(0, color="grey", lw=0.8)
ax[1].axhline(1, color="k", ls=":", lw=1, label="f = 1 (all in, no borrowing)")
ax[1].set_ylabel("Kelly fraction from the past 365 days")
ax[1].set_title("What Kelly would have told me, day by day")
ax[1].legend(fontsize=8)
for a in ax:
    a.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures/btc_kelly.png", dpi=150)
print("saved figures/btc_kelly.png")
