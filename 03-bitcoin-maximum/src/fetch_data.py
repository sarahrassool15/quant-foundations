"""
One year of 1-minute BTC/USDT candles from the Binance US public API.
The end date is fixed so anyone re-running this gets the same data.
Takes a few minutes (about 530 requests).
"""
import os
import time
import requests
import pandas as pd

URL = "https://api.binance.us/api/v3/klines"
START = pd.Timestamp("2025-11-01", tz="UTC")
END = pd.Timestamp("2026-11-01", tz="UTC")

os.makedirs("data", exist_ok=True)
rows = []
t = int(START.timestamp() * 1000)
end_ms = int(END.timestamp() * 1000)
while t < end_ms:
    for attempt in range(5):
        try:
            r = requests.get(URL, params={"symbol": "BTCUSDT", "interval": "1m",
                                          "startTime": t, "endTime": end_ms - 1,
                                          "limit": 1000}, timeout=20)
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
    rows += [b[:5] for b in batch]          # open time, open, high, low, close
    t = batch[-1][0] + 60_000               # next minute after the last one I got
    print(f"{pd.to_datetime(batch[-1][0], unit='ms')}  ({len(rows):,} candles)", end="\r")
    time.sleep(0.1)

df = pd.DataFrame(rows, columns=["open_time", "open", "high", "low", "close"])
df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].astype(float)
df.drop_duplicates("open_time").to_csv("data/btc_1m.csv", index=False)
print(f"\nsaved data/btc_1m.csv with {len(df):,} candles")
