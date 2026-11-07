"""
Does Bitcoin keep New York hours?

In my first paper I found a 24-hour cycle in Bitcoin's volatility and
guessed it followed the US stock market. I never tested it. Here's the test.

The trick: New York moves its clocks an hour twice a year, UTC doesn't.
If Bitcoin's busy hour follows New York, it should shift by one hour in UTC
between summer and winter. And the US and Europe change clocks on
DIFFERENT weekends (US: 2nd Sunday of March / 1st Sunday of November,
Europe: last Sunday of March / last Sunday of October), so for a few weeks
a year New York has moved and London hasn't. Whoever Bitcoin follows in
those weeks is who it's actually listening to.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = sys.argv[1] if len(sys.argv) > 1 else "data/btc_hourly.csv"
df = pd.read_csv(path, parse_dates=["open_time"]).set_index("open_time").sort_index()
r = np.log(df["close"]).diff().abs().dropna()      # size of each hour's move

# NOTE: Bitcoin's overall volatility changes a lot over seven years. without
# this, a few crazy months would dominate every hourly average. dividing each
# hour by that day's average hour puts every day on the same scale (the clock
# idea from project 03, done the cheap way).
day_mean = r.groupby(r.index.floor("D")).transform("mean")
x = (r / day_mean).to_frame("size")
x["utc_hour"] = x.index.hour
ny = x.index.tz_convert("America/New_York")
lon = x.index.tz_convert("Europe/London")
x["ny_hour"] = ny.hour
x["weekday"] = ny.dayofweek < 5
# is New York / London on summer time right now? utcoffset tells me:
# NY is UTC-4 in summer, UTC-5 in winter. London UTC+1 in summer, UTC+0 in winter.
x["ny_summer"] = np.array([t.utcoffset().total_seconds() == -4 * 3600 for t in ny])
x["lon_summer"] = np.array([t.utcoffset().total_seconds() == 1 * 3600 for t in lon])
print(f"{len(x):,} hours, {x.index[0].date()} to {x.index[-1].date()}")


def profile(sub, col="utc_hour"):
    return sub.groupby(col)["size"].mean().reindex(range(24))


wk = x[x["weekday"]]

# ---- 1. weekdays vs weekends, on New York's clock ----
p_wd = profile(wk, "ny_hour")
p_we = profile(x[~x["weekday"]], "ny_hour")
print("\n1. busiest hours on New York's clock")
print(f"   weekdays: {', '.join(f'{h}:00 ({p_wd[h]:.2f})' for h in p_wd.nlargest(3).index)}")
print(f"   weekends: {', '.join(f'{h}:00 ({p_we[h]:.2f})' for h in p_we.nlargest(3).index)}")
print(f"   biggest hour vs average hour: weekdays {p_wd.max():.2f}x, weekends {p_we.max():.2f}x")
# 1.00 = an average hour. the US stock market opens at 9:30 New York time,
# and a lot of US economic data comes out at 8:30.

# ---- 2. the daylight-saving shift ----
p_summer = profile(wk[wk["ny_summer"]])
p_winter = profile(wk[~wk["ny_summer"]])


def best_shift(a, b):
    # slide one 24-hour profile around the clock (np.roll wraps 23 -> 0) and
    # see which shift lines it up best with the other one
    corr = [np.corrcoef(np.roll(a.values, s), b.values)[0, 1] for s in range(-3, 4)]
    return list(range(-3, 4))[int(np.argmax(corr))], max(corr)


s, c = best_shift(p_summer, p_winter)
print("\n2. weekday profile in UTC, New York summer time vs winter time")
print(f"   peak hour (UTC): summer {p_summer.idxmax()}:00, winter {p_winter.idxmax()}:00")
print(f"   best shift to line them up: {s:+d} hour(s)  (correlation {c:.2f})")
print("   New York's clock predicts +1, a fixed UTC schedule predicts 0")

# ---- 3. the weeks when New York and London disagree ----
# both gaps go the same way: the US moves to summer time ~2 weeks BEFORE
# Europe in March, and moves back ~1 week AFTER Europe in autumn. so in every
# mismatch week, New York is on summer time and London is on winter time.
#   if Bitcoin follows New York -> looks like the normal SUMMER profile
#   if Bitcoin follows London   -> looks like the normal WINTER profile
mismatch = wk[wk["ny_summer"] & ~wk["lon_summer"]]
print(f"\n3. days when New York is on summer time but London isn't: {mismatch.index.floor('D').nunique()}")
p_mis = profile(mismatch)
corr_summer = np.corrcoef(p_mis.values, p_summer.values)[0, 1]
corr_winter = np.corrcoef(p_mis.values, p_winter.values)[0, 1]
print(f"   looks like the summer profile (New York): corr {corr_summer:.2f}")
print(f"   looks like the winter profile (London):   corr {corr_winter:.2f}")
print(f"   peak hour (UTC) in those weeks: {p_mis.idxmax()}:00")

# ---- 4. before and after the US spot ETFs (11 January 2024) ----
etf = pd.Timestamp("2024-01-11", tz="UTC")
for label, sub in [("before ETFs", wk[wk.index < etf]), ("after ETFs ", wk[wk.index >= etf])]:
    p = profile(sub, "ny_hour")
    print(f"\n4. {label}: 9:00-10:00 NY hour is {p[9]:.2f}x an average hour, "
          f"8:00-9:00 is {p[8]:.2f}x, busiest is {p.idxmax()}:00 ({p.max():.2f}x)")

fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
ax[0].plot(range(24), p_wd, "o-", label="weekdays")
ax[0].plot(range(24), p_we, "o-", label="weekends")
ax[0].axvspan(9.5, 16, color="grey", alpha=0.15, label="US stock market open")
ax[0].set_xlabel("hour, New York time")
ax[0].set_ylabel("size of move vs that day's average hour")
ax[0].set_title("Bitcoin by the New York clock")
ax[0].legend(fontsize=8)
ax[1].plot(range(24), p_summer, "o-", label="NY on summer time")
ax[1].plot(range(24), p_winter, "o-", label="NY on winter time")
ax[1].set_xlabel("hour, UTC")
ax[1].set_title(f"Weekdays in UTC: best shift {s:+d} h")
ax[1].legend(fontsize=8)
for label, sub in [("before ETFs", wk[wk.index < etf]), ("after ETFs", wk[wk.index >= etf])]:
    ax[2].plot(range(24), profile(sub, "ny_hour"), "o-", label=label)
ax[2].axvspan(9.5, 16, color="grey", alpha=0.15)
ax[2].set_xlabel("hour, New York time")
ax[2].set_title("Weekdays before and after the spot ETFs")
ax[2].legend(fontsize=8)
for a in ax:
    a.grid(alpha=0.3)
    a.set_xticks(range(0, 24, 3))
plt.tight_layout()
plt.savefig("figures/new_york_hours.png", dpi=150)
print("saved figures/new_york_hours.png")
