"""
Why does Bitcoin wander LESS than Brownian motion?

My suspicion is the following: it's not the max but actually it's the clock. sigma_day adds up every
5 minute wiggle. if the price keeps bouncing back, the wiggles pile up in
sigma_day but the price doesn't actually GET anywhere so dividing by
sigma_day makes every day look tamer than it is

Two checks
1. the close. under BM, (close - open) / sigma_day ~ N(0, 1) exactly, so
   its std must be 1. if it's below 1, the clock is overcounting.
2. a signature plot. recompute sigma_day from 1, 5, 15, 30, 60 minute
   returns. if bounce-back is the story, sigma_day SHRINKS as I sample less
   often (fewer wiggles counted) and the standardised max climbs back
   towrds sqrt(2/pi).
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = sys.argv[1] if len(sys.argv) > 1 else "data/btc_1m.csv"
df = pd.read_csv(path, parse_dates=["open_time"])
df["day"] = df["open_time"].dt.floor("D")

intervals = [1, 5, 15, 30, 60]          # all divide 1440, so every day splits evenly
rows = []
for day, g in df.groupby("day"):
    if len(g) != 1440:
        continue
    o = g["open"].iloc[0]
    log_close = np.log(np.r_[o, g["close"].to_numpy()])
    row = {"up": np.log(g["high"].max() / o),
           "down": np.log(o / g["low"].min()),
           "ret": log_close[-1] - log_close[0]}      # open-to-close return for the day
    for m in intervals:
        # same realised variance as before, just sampled every m minutes
        row[f"sigma_{m}"] = np.sqrt(np.sum(np.diff(log_close[::m]) ** 2))
    rows.append(row)
d = pd.DataFrame(rows)
print(f"{len(d)} complete days\n")

# check 1: the close 
z_close = d["ret"] / d["sigma_5"]
print("check 1: (close - open) / sigma_day  (Brownian motion: mean 0, std 1)")
print(f"  mean {z_close.mean():+.3f}   std {z_close.std():.3f}")
# NOTE: if the std is about 0.83, that's the same 0.666/0.798 shortfall as
# the max. then the max isn't weird at all; the clock is just too fast.
s = z_close.std()
print(f"  mean standardised max after rescaling by that std:"
      f"  up {(d['up'] / d['sigma_5']).mean() / s:.3f}"
      f"   down {(d['down'] / d['sigma_5']).mean() / s:.3f}"
      f"   (Brownian motion: {np.sqrt(2 / np.pi):.3f})")

# check 2: the signature plot 
print("\ncheck 2: sampling interval for sigma_day")
print("  minutes   avg sigma_day   mean up/sigma   mean down/sigma   std close/sigma")
mean_sigma, mean_up, mean_down, sd_close = [], [], [], []
for m in intervals:
    sig = d[f"sigma_{m}"]
    mean_sigma.append(sig.mean())
    mean_up.append((d["up"] / sig).mean())
    mean_down.append((d["down"] / sig).mean())
    sd_close.append((d["ret"] / sig).std())
    print(f"  {m:5d}     {mean_sigma[-1]:.4f}          {mean_up[-1]:.3f}           "
          f"{mean_down[-1]:.3f}             {sd_close[-1]:.3f}")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(intervals, mean_sigma, "o-")
ax[0].set_xscale("log")
ax[0].set_xticks(intervals, [str(m) for m in intervals])
ax[0].set_xlabel("minutes between returns used for σ_day")
ax[0].set_ylabel("average σ_day")
ax[0].set_title("Signature plot: does the clock depend on how I read it?")
ax[1].plot(intervals, mean_up, "o-", color="tab:green", label="mean high / σ_day")
ax[1].plot(intervals, mean_down, "o-", color="tab:red", label="mean low / σ_day")
ax[1].axhline(np.sqrt(2 / np.pi), color="k", ls="--", label="Brownian motion: √(2/π)")
ax[1].set_xscale("log")
ax[1].set_xticks(intervals, [str(m) for m in intervals])
ax[1].set_xlabel("minutes between returns used for σ_day")
ax[1].set_title("The standardised max, by sampling interval")
ax[1].legend()
for a in ax:
    a.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures/signature.png", dpi=150)
print("saved figures/signature.png")
