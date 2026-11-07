"""
Every day at midnight, sell a 30-day at-the-money Bitcoin call at Deribit's
implied volatility (DVOL), then delta hedge it every hour along the real
price path. Do option sellers actually make money?

From the simulation: if the market's realised vol comes in BELOW the
implied vol I sold at, I keep roughly the price difference. So this is
really asking: is Bitcoin's implied volatility usually too high?
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bs import call_price, call_delta

price_path = sys.argv[1] if len(sys.argv) > 1 else "data/btc_hourly.csv"
dvol_path = sys.argv[2] if len(sys.argv) > 2 else "data/dvol_daily.csv"

px = pd.read_csv(price_path, parse_dates=["open_time"]).set_index("open_time")["close"]
# fill any missing hours with the last price so every day has exactly 24 hours
full = pd.date_range(px.index[0], px.index[-1], freq="h")
px = px.reindex(full).ffill()

dvol = pd.read_csv(dvol_path, parse_dates=["time"]).set_index("time")["close"]
# NOTE: a daily DVOL candle stamped day d closes at the END of day d. to sell
# at midnight starting day d, I can only know day d-1's close. shift by one
# day so I never use a vol I couldn't have seen (project 02 habits).
iv_at_midnight = dvol.shift(1) / 100

H = 30 * 24
T = 30 / 365
rows = []
for day, sigma in iv_at_midnight.dropna().items():
    if day not in px.index:
        continue
    i = px.index.get_loc(day)
    if i + H >= len(px):
        break
    S = px.iloc[i:i + H + 1].to_numpy()
    K = S[0]
    premium = call_price(S[0], K, T, sigma)
    # delta at the start of every hour, all 720 at once, then
    # (shares held) x (price change over that hour), summed
    tau = T - np.arange(H) / (24 * 365)
    held = call_delta(S[:-1], K, tau, sigma)
    pnl = premium + np.sum(held * np.diff(S)) - max(S[-1] - K, 0)
    # realised vol over the same 30 days: std of hourly log returns,
    # scaled up to a year. hours in a year = 24 * 365 (crypto never closes).
    realised = np.std(np.diff(np.log(S))) * np.sqrt(24 * 365)
    rows.append({"day": day, "implied": sigma, "realised": realised, "pnl": pnl / premium})
d = pd.DataFrame(rows).set_index("day")
print(f"{len(d)} trades, {d.index[0].date()} to {d.index[-1].date()}")

print(f"\naverage implied vol (DVOL):  {d['implied'].mean():.1%}")
print(f"average realised vol:        {d['realised'].mean():.1%}")
print(f"implied above realised on {np.mean(d['implied'] > d['realised']):.0%} of days")

# trades overlap (each lasts 30 days, a new one starts daily), so I resample
# whole months for the error bar, like project 06
months = d.index.strftime("%Y-%m")
groups = [d["pnl"][months == m].to_numpy() for m in np.unique(months)]
rng = np.random.default_rng(0)
boot = [np.mean(np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))]))
        for _ in range(2000)]
print(f"\nP&L as a fraction of the premium received:")
print(f"  mean   {d['pnl'].mean():+.3f}  (± {np.std(boot):.3f})")
print(f"  median {d['pnl'].median():+.3f}")
print(f"  profitable {np.mean(d['pnl'] > 0):.0%} of trades")
print(f"  worst {d['pnl'].min():+.2f}   best {d['pnl'].max():+.2f}")
print("\nby year:")
for y, g in d.groupby(d.index.year):
    print(f"  {y}:  mean P&L {g['pnl'].mean():+.3f}   implied {g['implied'].mean():.0%}   realised {g['realised'].mean():.0%}   ({len(g)} trades)")

fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
ax[0].plot(d.index, d["implied"] * 100, lw=1, label="implied (DVOL) when sold")
ax[0].plot(d.index, d["realised"] * 100, lw=1, label="realised over the next 30 days")
ax[0].set_ylabel("volatility (%)")
ax[0].set_title("What the market expected vs what happened")
ax[0].legend(fontsize=8)
ax[1].hist(d["pnl"], bins=60, color="tab:blue", alpha=0.7)
ax[1].axvline(0, color="k", lw=0.8)
ax[1].axvline(d["pnl"].mean(), color="tab:red", ls="--", label=f"mean {d['pnl'].mean():+.2f}")
ax[1].set_xlabel("P&L / premium")
ax[1].set_title("Selling a hedged call every day")
ax[1].legend()
ax[2].scatter(d["implied"] - d["realised"], d["pnl"], s=4, alpha=0.4)
ax[2].axhline(0, color="k", lw=0.8)
ax[2].axvline(0, color="k", lw=0.8)
ax[2].set_xlabel("implied minus realised vol")
ax[2].set_ylabel("P&L / premium")
ax[2].set_title("You make money when implied was too high")
for a in ax:
    a.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures/real_hedge.png", dpi=150)
print("saved figures/real_hedge.png")
