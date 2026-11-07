"""
A leveraged Bitcoin long as a gambler's ruin problem.

I open a long at midnight UTC every day. Two exits:
  take profit if the price rises 10%
  liquidated  if the price falls by about 1/L (L = leverage)
Which one comes first?

Continuous version of the formula: in log prices, with no drift, the
chance of hitting the top barrier (distance a above) before the bottom
one (distance b below) is
    P(top first) = b / (a + b)
It's the k/N from the coin, with distances instead of chip counts.

NOTE: the cool part. this doesn't depend on volatility AT ALL. a wild
market gets to one of the barriers faster, but which one it hits first
is the same. that's the "volatility is a clock" idea from project 03
again: speeding up the clock doesn't change the order things happen in.
So if real Bitcoin gives different answers on calm and turbulent days,
that's something Brownian motion can't explain.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = sys.argv[1] if len(sys.argv) > 1 else "data/btc_hourly.csv"
df = pd.read_csv(path, parse_dates=["open_time"]).sort_values("open_time").reset_index(drop=True)
hi, lo, cl, op = (df[c].to_numpy() for c in ["high", "low", "close", "open"])
times = df["open_time"]

TAKE_PROFIT = 0.10            # +10% on the price
MAINTENANCE = 0.005           # exchanges liquidate a bit BEFORE you lose all your margin
HORIZON = 90 * 24             # give up after 90 days if neither barrier is hit
LEVERAGES = [2, 3, 5, 10, 20, 50]

entries = np.where(times.dt.hour == 0)[0]
entries = entries[entries + HORIZON < len(df)]
print(f"{len(entries)} entries, one per day, {times.iloc[entries[0]].date()} to {times.iloc[entries[-1]].date()}")


def first_exit(i, liq, use_wicks=True):
    """+1 take profit first, -1 liquidated first, 0 neither within the horizon."""
    entry = op[i]
    up_px, down_px = entry * (1 + TAKE_PROFIT), entry * (1 - liq)
    seg = slice(i, i + HORIZON)
    # with wicks: the candle's high and low see every trade in the hour.
    # closes only: I'd only notice the price at the end of each hour (project 03 again).
    up = (hi[seg] if use_wicks else cl[seg]) >= up_px
    down = (lo[seg] if use_wicks else cl[seg]) <= down_px
    t_up = np.argmax(up) if up.any() else HORIZON
    t_down = np.argmax(down) if down.any() else HORIZON
    if t_up == HORIZON and t_down == HORIZON:
        return 0
    # NOTE: if both happen inside the same hourly candle I can't tell which came
    # first. I count it as liquidated, because that's what would hurt.
    return 1 if t_up < t_down else -1


def formula_liq_first(liq):
    a = np.log(1 + TAKE_PROFIT)        # distance up, in log price
    b = -np.log(1 - liq)               # distance down, in log price (minus sign: log of
                                       # something below 1 is negative, I want a distance)
    return a / (a + b)                 # P(bottom first) = a/(a+b), mirror of b/(a+b)


def monthly_bootstrap(outcomes, entry_idx, n_boot=2000, seed=0):
    # the trades overlap in time (today's and tomorrow's see mostly the same
    # prices), so they're NOT independent and the usual sqrt(p(1-p)/n) would be
    # far too small. I resample whole calendar months instead.
    # NOTE: at low leverage (2x, 3x) a single trade can take months to resolve,
    # so even months aren't independent and these error bars are too small.
    # trust the 10x+ rows more.
    rng = np.random.default_rng(seed)
    month = times.iloc[entry_idx].dt.strftime("%Y-%m").to_numpy()   # "2024-03" etc.
    groups = [outcomes[month == m] for m in np.unique(month)]
    est = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        x = np.concatenate([groups[j] for j in pick])
        est.append(np.mean(x == -1))
    return np.std(est)


print("\nP(liquidated before +10%):")
print("  leverage  liq at   formula   BTC (wicks)    BTC (hourly closes)   unresolved")
rows = []
for L in LEVERAGES:
    liq = 1 / L - MAINTENANCE
    w = np.array([first_exit(i, liq, True) for i in entries])
    c = np.array([first_exit(i, liq, False) for i in entries])
    # trades that hit neither barrier in 90 days are left out. that's a small bias:
    # the ones left out are the slow ones, mostly at low leverage.
    rw, rc = w[w != 0], c[c != 0]
    se = monthly_bootstrap(rw, entries[w != 0])
    rows.append((L, formula_liq_first(liq), np.mean(rw == -1), se, np.mean(rc == -1)))
    print(f"  {L:5d}x   -{liq:5.1%}   {formula_liq_first(liq):7.3f}   {np.mean(rw == -1):.3f} ± {se:.3f}"
          f"        {np.mean(rc == -1):.3f}             {np.mean(w == 0):.1%}")

# ---- does the market's mood when I enter change the odds? ----
# mood = yesterday's high-low range compared with the median of the 90 days
# before it. only uses the PAST, so no look-ahead this time (unlike project 05).
day = df.set_index("open_time").resample("D").agg({"high": "max", "low": "min"})
rng_d = np.log(day["high"] / day["low"])
baseline = rng_d.rolling(90).median()
turbulent_yday = (rng_d > baseline).shift(1)        # shift(1): yesterday's label for today
entry_days = times.iloc[entries].dt.floor("D")
mood = turbulent_yday.reindex(entry_days.to_numpy()).to_numpy()

L = 10
liq = 1 / L - MAINTENANCE
w = np.array([first_exit(i, liq, True) for i in entries])
print(f"\n10x long, split by yesterday's volatility (formula says {formula_liq_first(liq):.3f} for both):")
mood_rows = []
for label, mask in [("calm yesterday", mood == False), ("turbulent yesterday", mood == True)]:
    m = mask & (w != 0)
    se = monthly_bootstrap(w[m], entries[m])
    mood_rows.append((label, np.mean(w[m] == -1), se))
    print(f"  {label:20s} {np.mean(w[m] == -1):.3f} ± {se:.3f}   ({m.sum()} entries)")

# ---- figures ----
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
Ls = [r[0] for r in rows]
ax[0].plot(Ls, [r[1] for r in rows], "k-", label="Brownian motion formula")
ax[0].errorbar(Ls, [r[2] for r in rows], yerr=[2 * r[3] for r in rows], fmt="o", color="tab:red",
               capsize=3, label="BTC, using highs and lows")
ax[0].plot(Ls, [r[4] for r in rows], "s", color="tab:blue", label="BTC, hourly closes only")
ax[0].set_xscale("log")
ax[0].set_xticks(Ls, [f"{l}x" for l in Ls])
ax[0].set_xlabel("leverage")
ax[0].set_ylabel("P(liquidated before +10%)")
ax[0].set_title("Liquidation or take profit, which comes first?")
ax[0].legend(fontsize=8)
ax[1].bar([r[0] for r in mood_rows], [r[1] for r in mood_rows],
          yerr=[2 * r[2] for r in mood_rows], capsize=4, color=["tab:green", "tab:red"])
ax[1].axhline(formula_liq_first(liq), color="k", ls="--", label="formula (same for both)")
ax[1].set_ylabel("P(liquidated before +10%), 10x long")
ax[1].set_title("Does it matter how wild yesterday was?")
ax[1].legend(fontsize=8)
for a_ in ax:
    a_.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures/btc_ruin.png", dpi=150)
print("saved figures/btc_ruin.png")
