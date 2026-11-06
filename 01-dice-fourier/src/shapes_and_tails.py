"""
What does the sum of n dice look like, and how good is the bell curve?
Figure 1: the shape for n = 1, 2, 5, 50, against the normal approximation.
Figure 2: the right tail on a log scale, where the normal is wrong and
the FFT hits a floor.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from dice import pmf_convolution, pmf_fft

# one die: mean 3.5, variance 35/12
MU, VAR = 3.5, 35 / 12

# Fig 1: the shape appears 
fig, axes = plt.subplots(1, 4, figsize=(14, 3.2))
for ax, n in zip(axes, [1, 2, 5, 50]):
    pmf = pmf_convolution(n)
    s = np.arange(len(pmf))
    ax.bar(s[n:], pmf[n:], width=1.0, color="tab:blue", alpha=0.7, label="exact")
    x = np.linspace(n, 6 * n, 400)
    ax.plot(x, norm.pdf(x, n * MU, np.sqrt(n * VAR)), "k-", lw=1.2, label="normal")
    ax.set_title("1 die" if n == 1 else f"{n} dice")
    ax.set_xlabel("sum")
axes[0].set_ylabel("probability")
axes[0].legend(fontsize=8)
plt.tight_layout()
plt.savefig("figures/shapes.png", dpi=150)
print("saved figures/shapes.png")

# Fig 2: the right tail for 50 dice 
n = 50
exact = pmf_convolution(n)
fft = pmf_fft(n)
s = np.arange(len(exact))

# P(sum >= k), adding up from the top so tiny numbers don't get swamped
tail_exact = np.cumsum(exact[::-1])[::-1]
tail_fft = np.cumsum(fft[::-1])[::-1]
# normal with continuity correction: P(S >= k) ~ P(Z >= k - 0.5)
tail_normal = norm.sf(s - 0.5, n * MU, np.sqrt(n * VAR))

k = s[n * 4:]  # from 200 up to 300
plt.figure(figsize=(7, 4.2))
plt.semilogy(k, tail_exact[k], "b-", lw=2, label="exact (convolution)")
plt.semilogy(k, np.abs(tail_fft[k]), "r.", ms=4, label="FFT")
plt.semilogy(k, tail_normal[k], "k--", label="normal approximation")
plt.xlabel("k")
plt.ylabel("P(sum of 50 dice ≥ k)")
plt.title("The right tail of 50 dice")
plt.legend()
plt.tight_layout()
plt.savefig("figures/tails.png", dpi=150)
print("saved figures/tails.png")

# the numbers I quote in the README
print(f"\nmax |exact - FFT| over the whole pmf: {np.max(np.abs(exact - fft)):.1e}")
for kk in [225, 250, 275, 300]:
    print(f"P(S >= {kk}):  exact {tail_exact[kk]:.3e}   normal {tail_normal[kk]:.3e}"
          f"   ratio normal/exact {tail_normal[kk] / tail_exact[kk]:.2f}"
          f"   FFT {tail_fft[kk]:.3e}")
print(f"\nexact P(all 50 dice show 6) = (1/6)^50 = {(1/6)**50:.3e}")
