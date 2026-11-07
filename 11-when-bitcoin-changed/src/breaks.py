"""
When did Bitcoin's volatility change?

Two ways to find the breaks:
1. hindsight: look at all seven years at once (CUSUM + binary segmentation)
2. live: read one day at a time and only use the past (Bayesian online
   changepoint detection, Adams & MacKay 2007)
Then check both against things that actually happened.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import gammaln

path = sys.argv[1] if len(sys.argv) > 1 else "data/btc_daily.csv"
df = pd.read_csv(path, parse_dates=["open_time"]).set_index("open_time")
r = np.log(df["close"]).diff().dropna()
dates, x = r.index, r.to_numpy()
# NOTE: I'm treating the daily mean as zero. it's tiny next to the daily
# swings (a 0.1% average vs a 3% standard deviation), and it's volatility
# I care about here, not direction.
print(f"{len(x)} daily returns, {dates[0].date()} to {dates[-1].date()}")

EVENTS = {
    "Covid crash": "2020-03-12",
    "halving 2020": "2020-05-11",
    "Terra/Luna": "2022-05-09",
    "FTX": "2022-11-08",
    "spot ETFs": "2024-01-11",
    "halving 2024": "2024-04-20",
}

# ---------------------------------------------------------------- 1. hindsight
def cusum_break(seg):
    """Inclan-Tiao: where does the running total of squared returns bend?
    if the variance never changes, the cumulative sum of r^2 grows in a
    straight line. a change in variance puts a kink in it. the test is how
    far the running sum strays from the straight line, scaled by how much it
    SHOULD wobble."""
    T = len(seg)
    C = np.cumsum(seg ** 2)
    gap = C - np.arange(1, T + 1) / T * C[-1]
    k = int(np.argmax(np.abs(gap)))
    # how much it should wobble depends on Var(r^2) = E[r^4] - (E[r^2])^2.
    # the original test assumes normal returns, where E[r^4] = 3 sigma^4 and
    # this is 2 sigma^4. Bitcoin's tails are way fatter than normal (daily
    # kurtosis 7.7 in my first paper), so I use the real fourth moment instead.
    # with the normal version, every fat-tailed week looked like a break.
    omega = np.mean(seg ** 4) - np.mean(seg ** 2) ** 2
    stat = np.abs(gap[k]) / np.sqrt(T * omega)
    return k, stat


CRIT = 1.628          # 1% critical value (the sup of a Brownian bridge, which is what the
                      # running-sum gap turns into when nothing changes). 1% not 5%,
                      # because binary segmentation runs the test over and over, and
                      # at 5% it found a fake extra break on test data.
MIN_SEG = 60          # don't split anything into pieces shorter than 60 days


def binary_segmentation(lo, hi, out):
    # find the biggest break, split there, then do the same inside each half
    if hi - lo < 2 * MIN_SEG:
        return
    k, stat = cusum_break(x[lo:hi])
    k += lo
    if stat > CRIT and MIN_SEG <= k - lo and hi - k >= MIN_SEG:
        out.append(k)
        binary_segmentation(lo, k, out)
        binary_segmentation(k + 1, hi, out)


hind = []
binary_segmentation(0, len(x), hind)
hind = sorted(hind)
print(f"\n1. hindsight breaks (CUSUM): {len(hind)}")
edges = [0] + [k + 1 for k in hind] + [len(x)]
for a, b in zip(edges[:-1], edges[1:]):
    vol = x[a:b].std() * np.sqrt(365)
    print(f"   {dates[a].date()} to {dates[b - 1].date()}:  vol {vol:.0%} a year")

# ---------------------------------------------------------------- 2. live
def bocpd(x, hazard=1 / 500, a0=2.0, b0=0.03 ** 2):
    """
    Keep a probability for every possible "run length": how many days since
    the last change. Each new day, each run either grows by one or resets
    to zero (a change), with probability `hazard`.
    Within a run, returns are N(0, s^2) with s^2 unknown. A conjugate prior
    on s^2 (inverse gamma with a, b) means after seeing returns the
    prediction for tomorrow is a Student-t, and updating is just
        a -> a + 1/2,  b -> b + r^2 / 2
    hazard = 1/500: before seeing anything, I expect a change about every 500 days.
    b0 = 0.03^2 so my prior guess for daily vol is about 3%, and a0 = 2
    so that guess is only worth a couple of days of data.
    """
    T = len(x)
    R = np.zeros((T + 1, T + 1))
    R[0, 0] = 1.0
    a = np.array([a0])
    b = np.array([b0])
    map_run = np.zeros(T, dtype=int)
    recent = np.zeros(T)
    for t in range(T):
        # Student-t predictive with 2a degrees of freedom and scale sqrt(b/a)
        nu, scale2 = 2 * a, b / a
        logp = (gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(nu * np.pi * scale2)
                - (nu + 1) / 2 * np.log1p(x[t] ** 2 / (nu * scale2)))
        pred = np.exp(logp)
        growth = R[t, :t + 1] * pred * (1 - hazard)
        change = np.sum(R[t, :t + 1] * pred * hazard)
        R[t + 1, 1:t + 2] = growth
        R[t + 1, 0] = change
        R[t + 1, :t + 2] /= R[t + 1, :t + 2].sum()          # normalise so it's a probability again
        a = np.r_[a0, a + 0.5]
        b = np.r_[b0, b + x[t] ** 2 / 2]
        map_run[t] = int(np.argmax(R[t + 1, :t + 2]))
        recent[t] = R[t + 1, :31].sum()                     # P(a change in the last 30 days)
    return map_run, recent


run, recent = bocpd(x)
# a "detection" is the first day the detector is more than 80% sure there
# was a change in the last 30 days, after a stretch where it wasn't.
# NOTE: my first rule (the most likely run length suddenly drops) went off
# 14 times on fake data with 3 real breaks. far too twitchy.
# NOTE: I picked 80%, the 1-in-500-days prior and the 1% critical value on FAKE
# data with known breaks, then ran Bitcoin once. tuning them on Bitcoin until
# the breaks landed on the events I wanted would be project 02 all over again.
live = []
calm_days = 0
for t in range(len(run)):
    if recent[t] > 0.8 and calm_days >= 30:
        live.append((t, t - run[t]))        # (day I noticed, day it thinks the change started)
        calm_days = 0
    calm_days = calm_days + 1 if recent[t] <= 0.8 else 0
print(f"\n2. live detections (Bayesian online): {len(live)}")
for t, start in live:
    print(f"   noticed {dates[t].date()}, thinks it started {dates[start].date()}  "
          f"({t - start} days earlier)")

# ---------------------------------------------------------------- 3. vs events
print("\n3. nearest break to each event (days; + means after the event)")
for name, d in EVENTS.items():
    ev = pd.Timestamp(d, tz="UTC")
    if ev < dates[0] or ev > dates[-1]:
        continue
    h = min(((dates[k + 1] - ev).days for k in hind), key=abs) if hind else None
    l_start = min(((dates[s] - ev).days for _, s in live), key=abs) if live else None
    l_seen = min(((dates[t] - ev).days for t, _ in live if dates[t] >= ev), default=None)
    print(f"   {name:14s} hindsight {h:+5d}   live start {l_start:+5d}   live first noticed {l_seen if l_seen is not None else '-':>5}")

fig, ax = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
roll = pd.Series(np.abs(x), index=dates).rolling(30).mean() * np.sqrt(np.pi / 2) * np.sqrt(365)
ax[0].plot(dates, roll, color="grey", lw=1, label="30-day volatility")
for a_, b_ in zip(edges[:-1], edges[1:]):
    ax[0].hlines(x[a_:b_].std() * np.sqrt(365), dates[a_], dates[b_ - 1], color="tab:blue", lw=2.5)
for name, d in EVENTS.items():
    ev = pd.Timestamp(d, tz="UTC")
    for a_ in ax:
        a_.axvline(ev, color="tab:red", ls=":", lw=1)
    ax[0].text(ev, ax[0].get_ylim()[1] * 0.95, name, rotation=90, fontsize=7, va="top", color="tab:red")
ax[0].set_ylabel("annualised volatility")
ax[0].set_title("Hindsight: volatility in each segment between breaks")
ax[1].plot(dates, run, color="tab:green", lw=1)
ax[1].set_ylabel("most likely days since last change")
ax[1].set_title("Live: the online detector's guess, one day at a time")
for a_ in ax:
    a_.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures/breaks.png", dpi=150)
print("saved figures/breaks.png")
