"""
Daily BTC/USDT candles from Binance US, from when it listed (Sept 2019)
to 1 November 2026. Fixed end date so the results are reproducible.
Only a few requests, so it takes seconds.
"""
import os
import time
import requests
import pandas as pd

URL = "https://api.binance.us/api/v3/klines"
START = pd.Timestamp("2019-01-01", tz="UTC")     # the API just starts at the first real day
END = pd.Timestamp("2026-11-01", tz="UTC")

os.makedirs("data", exist_ok=True)
rows = []
t = int(START.timestamp() * 1000)
end_ms = int(END.timestamp() * 1000)
while t < end_ms:
    r = requests.get(URL, params={"symbol": "BTCUSDT", "interval": "1d",
                                  "startTime": t, "endTime": end_ms - 1, "limit": 1000}, timeout=20)
    r.raise_for_status()
    batch = r.json()
    if not batch:
        break
    rows += [b[:5] for b in batch]
    t = batch[-1][0] + 86_400_000          # one day in milliseconds
    time.sleep(0.2)

df = pd.DataFrame(rows, columns=["open_time", "open", "high", "low", "close"])
df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].astype(float)
df.drop_duplicates("open_time").to_csv("data/btc_daily.csv", index=False)
print(f"saved data/btc_daily.csv: {len(df)} days, {df['open_time'].iloc[0].date()} to {df['open_time'].iloc[-1].date()}")
