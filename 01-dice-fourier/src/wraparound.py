"""
We ask the following: what happens if the FFT grid is too short?
Sum of 10 dice runs from 10 to 60, so it needs 61 points.
With only 40 everything above 39 wraps around and lands on top of
the low sums
"""
import numpy as np
import matplotlib.pyplot as plt
from dice import pmf_convolution, pmf_fft

n = 10
exact = pmf_convolution(n)            # length 61
short = pmf_fft(n, N=40)              # too short on purpose

# by hand: fold the exact answer onto 40 slots
folded = np.zeros(40)
for s, p in enumerate(exact):
    folded[s % 40] += p
print(f"max |short FFT - folded exact| = {np.max(np.abs(short - folded)):.1e}")
print("so the 'wrong' FFT answer is exactly the true answer wrapped around a circle")

plt.figure(figsize=(7, 4))
plt.bar(np.arange(61), exact, width=1.0, alpha=0.5, label="true (needs 61 points)")
plt.bar(np.arange(40), short, width=0.5, color="tab:red", alpha=0.8, label="FFT with only 40 points")
plt.axvline(39.5, color="k", ls=":", lw=1)
plt.xlabel("sum of 10 dice")
plt.ylabel("probability")
plt.title("Too short a grid: the top of the distribution wraps around")
plt.legend()
plt.tight_layout()
plt.savefig("figures/wraparound.png", dpi=150)
print("saved figures/wraparound.png")
