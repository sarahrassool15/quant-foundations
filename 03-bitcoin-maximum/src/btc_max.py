"""
Does Bitcoin's daily maximum behave essentially ike Brownian motion's?

For each UTC day:
  up   = log(high / open)   how far it rose above where it started
  down = log(open / low)    how far it fell below
  sigma_day = sqrt(sum of squared 5-minute log returns that day)

If Bitcoin were Brownian motion up/sigma_day and down/sigma_day should
both look like |Z| (see the simulation.py for why) on every kind of day.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.special import zeta

BETA = -zeta(0.5) / np.sqrt(2 * np.pi)   # 0.5826, minus sign because zeta(1/2) < 0
path = sys.argv[1] if len(sys.argv) > 1 else "data/btc_1m.csv"

df = pd.read_csv(path, parse_dates=["open_time"])
df["day"] = df["open_time"].dt.floor("D")   # floor to midnight UTC = which day it is

days = []
for day, g in df.groupby("day"):
    # 24 * 60 = 1440 minutes in a day. a day with missing minutes has a
    # hole in the path, and the max could be hiding in the hole. skip it.
    if len(g) != 1440:
        continue
    o = g["open"].iloc[0]

    # LOGS not plain percentages: log returns ADD over time,
    # log(P2/P0) = log(P2/P1) + log(P1/P0), and BM is a running sum
    # for small moves they're basically the same anyway: log(1+x) ~ x.
    # the open goes on the front as minute 0.
    log_close = np.log(np.r_[o, g["close"].to_numpy()])

    # [::5] keeps minutes 0, 5, 10, ... and diff turns prices into
    # 5-minute returns.
    # why not one min? every price bounces between bid and ask, and that
    # noise gets added to EVERY squared return. more returns = more noise
    # piled up. 5 min is the usual compromise.
    r5 = np.diff(log_close[::5])

    # realised variance = sum of squared returns.
    # why squares: each return has mean ~0, so E[r^2] ~ Var(r) = sigma^2 dt.
    # add them up and you get sigma^2 * (one day). it's the quadratic
    # variation from my Ito project again.
    # NOTE: this is the "volatility is a clock" idea. a crazy day is the same
    # BM with the clock running faster. dividing by sigma_day puts every day
    # on the same clock (Dambis-Dubins-Schwarz, if anyone asks).
    sigma = np.sqrt(np.sum(r5 ** 2))

    row = {"day": day, "sigma": sigma,
           "up": np.log(g["high"].max() / o),
           # NOTE: careful with the minus sign: log(low/open) is NEGATIVE (low < open).
           # log(open/low) = -log(low/open) flips it, so "how far it fell"
           # comes out as a positive number, same as up.
           "down": np.log(o / g["low"].min())}

    # the max I'd have seen if I only checked the close every k minutes.
    # minutes are numbered 0..1439, so the close at the END of minute k is
    # at index k-1. [k::k] would be off by one. easy mistake to make.
    # the open goes on the front because at the start of the day I've
    # already "seen" it, so the seen max can't come out below the open.
    for k in [1, 5, 15, 30, 60, 240]:
        row[f"seen_{k}"] = np.log(np.max(np.r_[o, g["close"].to_numpy()[k - 1::k]]) / o)
    days.append(row)
d = pd.DataFrame(days)
print(f"{len(d)} complete days")

z_up, z_down = d["up"] / d["sigma"], d["down"] / d["sigma"]

# 1. the shape of the standardised max 
# under BM: P(z >= x) = 2(1 - Phi(x)), and the mean is sqrt(2/pi) = 0.798.
# the min should match the max exactly, because -W is also a BM
# (max of -W = -min of W). if the lows go further than the highs, that's
# asymmetry BM can't produce
print("\nP(z >= x):         up      down    Brownian motion")
for x in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
    print(f"  x = {x}:        {np.mean(z_up >= x):.3f}   {np.mean(z_down >= x):.3f}   {2 * norm.sf(x):.3f}")
print(f"  mean:            {z_up.mean():.3f}   {z_down.mean():.3f}   {np.sqrt(2 / np.pi):.3f}")

x = np.linspace(0, 4, 200)
plt.figure(figsize=(6.5, 4.2))
for z, lab, c in [(z_up, "daily high", "tab:green"), (z_down, "daily low", "tab:red")]:
    # survival curve: sort the values; the i-th smallest has a fraction
    # 1 - i/n of the data at or above it
    zs = np.sort(z.to_numpy())
    surv = 1 - np.arange(len(zs)) / len(zs)
    plt.semilogy(zs, surv, ".", color=c, ms=4, label=f"BTC {lab}")
plt.semilogy(x, 2 * norm.sf(x), "k-", label="Brownian motion: 2(1−Φ(x))")
plt.xlim(0, 4.5)
plt.ylim(1 / len(d) / 2, 1.1)   # log scale can't show 0, so stop just below one day
plt.xlabel("distance from the open, in units of that day's volatility")
plt.ylabel("fraction of days at least this far")
plt.title("How far does Bitcoin wander from the open?")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures/btc_max_distribution.png", dpi=150)
print("saved figures/btc_max_distribution.png")

# 2. how much of the true high do I miss by looking every k minutes?
# the candle highs see every trade, so I treat them as the "continuous" max.
# NOTE: beta = 0.583 is the limit for watching infinitely often. on real
# grids even PERFECT BM gives less (I checked: about 0.5). so comparing
# Bitcoin to 0.583 alone would be unfair. I push fake BM days through the
# exact same steps and compare to that instead.
ks = [1, 5, 15, 30, 60, 240]


def bm_benchmark(n_days=300, sub=20, seed=0):
    # BM days with sigma = 1, each minute split into `sub` tiny steps,
    # so there's movement INSIDE each minute, like real trading
    rng = np.random.default_rng(seed)
    steps = rng.standard_normal((n_days, 1440 * sub)) / np.sqrt(1440 * sub)   # std = sqrt(dt) again
    path = np.hstack([np.zeros((n_days, 1)), np.cumsum(steps, axis=1)])
    high = path.max(axis=1)
    closes = path[:, sub::sub]                      # the price at the end of each minute
    out = []
    for k in ks:
        seen = np.maximum(0, closes[:, k - 1::k].max(axis=1))   # max(0, ...) = the open counts
        out.append(np.mean(high - seen) / np.sqrt(k / 1440))
    return out


bench = bm_benchmark()
print(f"\nmissed high in units of sigma*sqrt(dt)   (limit for continuous watching: {BETA:.3f})")
print("                 Bitcoin   Brownian benchmark")
ratios = []
for k, b in zip(ks, bench):
    gap = d["up"] - d[f"seen_{k}"]
    # dt = k/1440 is a FRACTION OF A DAY, because sigma_day is per day
    # units have to match or the ratio means nothing
    ratio = np.mean(gap) / np.mean(d["sigma"] * np.sqrt(k / 1440))
    ratios.append(ratio)
    print(f"  every {k:3d} min:  {ratio:.3f}     {b:.3f}")

plt.figure(figsize=(6.2, 4.2))
plt.semilogx(ks, ratios, "o-", color="tab:blue", label="Bitcoin")
plt.semilogx(ks, bench, "s--", color="grey", label="Brownian motion, same procedure")
plt.axhline(BETA, color="k", ls=":", lw=1, label=f"continuous-watching limit β = {BETA:.3f}")
plt.xticks(ks, [str(k) for k in ks])
plt.xlabel("minutes between looks")
plt.ylabel("missed high ÷ σ√Δt")
plt.title("How much of Bitcoin's high you miss by not watching")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures/btc_monitoring_gap.png", dpi=150)
print("saved figures/btc_monitoring_gap.png")
