"""
Deribit's DVOL index: the market's implied volatility for Bitcoin over the
next 30 days, worked out from Deribit's option prices. Like the VIX, but
for Bitcoin. Daily values from its start in 2021 to 1 November 2026.
"""
import os
import time
import requests
import pandas as pd

URL = "https://www.deribit.com/api/v2/public/get_volatility_index_data"
START = int(pd.Timestamp("2021-01-01", tz="UTC").timestamp() * 1000)
END = int(pd.Timestamp("2026-11-01", tz="UTC").timestamp() * 1000)

os.makedirs("data", exist_ok=True)
rows, end = [], END
while True:
    r = requests.get(URL, params={"currency": "BTC", "start_timestamp": START,
                                  "end_timestamp": end, "resolution": "1D"}, timeout=20)
    r.raise_for_status()
    res = r.json()["result"]
    new = [d for d in res["data"] if d[0] not in {x[0] for x in rows}]
    rows += new
    print(f"{len(rows)} days so far", end="\r")
    # Deribit sends the newest chunk first and a "continuation" timestamp
    # to ask for the next older chunk. stop when there isn't one (or nothing new came back).
    if not res.get("continuation") or not new:
        break
    end = res["continuation"]
    time.sleep(0.2)

df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close"]).sort_values("time")
df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
df.to_csv("data/dvol_daily.csv", index=False)
print(f"\nsaved data/dvol_daily.csv: {len(df)} days, {df['time'].iloc[0].date()} to {df['time'].iloc[-1].date()}")
