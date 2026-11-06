"""
Race: repeated convolution vs FFT, for n from 10 up to 20,000 dice.
Convolution grows like n^2, FFT like n log n.
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from dice import pmf_convolution, pmf_fft


def best_time(f, n, repeats=3):
    # I take the fastest of a few runs, since the slower ones are mostly
    # my laptop doing something else at the same time
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        f(n)
        times.append(time.perf_counter() - t0)
    return min(times)


ns = np.unique(np.logspace(1, np.log10(20000), 14).astype(int))
t_conv = [best_time(pmf_convolution, n) for n in ns]
t_fft = [best_time(pmf_fft, n) for n in ns]

for n, a, b in zip(ns, t_conv, t_fft):
    print(f"n = {n:6d}   convolution {a:9.5f} s   FFT {b:9.5f} s   speed-up {a / b:7.1f}x")

# slopes on the log-log plot over the bigger n (small n is all overhead)
big = ns >= 500
print(f"\nslope convolution: {np.polyfit(np.log(ns[big]), np.log(np.array(t_conv)[big]), 1)[0]:.2f}  (n^2 would be 2)")
print(f"slope FFT:         {np.polyfit(np.log(ns[big]), np.log(np.array(t_fft)[big]), 1)[0]:.2f}  (n log n is a bit above 1)")

plt.figure(figsize=(6.5, 4.2))
plt.loglog(ns, t_conv, "o-", label="repeated convolution")
plt.loglog(ns, t_fft, "s-", label="FFT")
plt.xlabel("number of dice n")
plt.ylabel("seconds")
plt.title("Exact distribution of the sum of n dice")
plt.legend()
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig("figures/timing.png", dpi=150)
print("saved figures/timing.png")
