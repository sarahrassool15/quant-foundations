"""
Hourly BTC/USDT candles from Binance US, September 2019 to 1 November 2026.
About 63 requests, so under a minute.
"""
import os
import time
import requests
import pandas as pd

URL = "https://api.binance.us/api/v3/klines"
START = pd.Timestamp("2019-01-01", tz="UTC")
END = pd.Timestamp("2026-11-01", tz="UTC")

os.makedirs("data", exist_ok=True)
rows = []
t = int(START.timestamp() * 1000)
end_ms = int(END.timestamp() * 1000)
while t < end_ms:
    for attempt in range(5):
        try:
            r = requests.get(URL, params={"symbol": "BTCUSDT", "interval": "1h",
                                          "startTime": t, "endTime": end_ms - 1, "limit": 1000}, timeout=20)
            r.raise_for_status()
            batch = r.json()
            break
        except Exception as e:
            print(f"retrying after error: {e}")
            time.sleep(2 * (attempt + 1))
    else:
        raise RuntimeError("gave up after 5 tries")
    if not batch:
        break
    rows += [b[:5] for b in batch]
    t = batch[-1][0] + 3_600_000           # one hour in milliseconds
    print(f"{pd.to_datetime(batch[-1][0], unit='ms')}  ({len(rows):,} candles)", end="\r")
    time.sleep(0.15)

df = pd.DataFrame(rows, columns=["open_time", "open", "high", "low", "close"])
df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].astype(float)
df.drop_duplicates("open_time").to_csv("data/btc_hourly.csv", index=False)
print(f"\nsaved data/btc_hourly.csv with {len(df):,} candles")
