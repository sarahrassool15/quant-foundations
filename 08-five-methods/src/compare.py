"""
Accuracy vs compute time for all five methods. The question isn't which
one is "right" (they all are, eventually), it's which one gets you a
given accuracy fastest.
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from methods import (black_scholes, binomial, monte_carlo, finite_difference,
                     fourier, SIGMA, T)

EXACT = black_scholes()
print(f"Black-Scholes formula: {EXACT:.6f}\n")


def timed(f, *args, repeats=3):
    best = np.inf
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = f(*args)
        best = min(best, time.perf_counter() - t0)
    return out, best


results = {}

# tree
rows = []
for n in [25, 50, 100, 200, 400, 800, 1600, 3200]:
    v, t = timed(binomial, n)
    rows.append((t, abs(v - EXACT)))
results["binomial tree"] = rows

# Monte Carlo: the error is random, so I plot the standard error
# (the typical error) rather than one lucky or unlucky draw
rows = []
for n in [10 ** k for k in range(3, 8)]:
    (v, se), t = timed(monte_carlo, n, repeats=1)
    rows.append((t, se))
results["Monte Carlo (std error)"] = rows

# finite differences. explicit needs dt <= dx^2 / sigma^2 or it blows up
# (project 07 again), so its time steps are forced by the grid.
for name, theta, rann in [("explicit FD", 0, 0), ("implicit FD", 1, 0),
                          ("Crank-Nicolson", 0.5, 0), ("Crank-Nicolson + Rannacher", 0.5, 4)]:
    rows = []
    for nx in [51, 101, 201, 401, 801]:
        dx = 10 * SIGMA * np.sqrt(T) / (nx - 1)
        nt = int(np.ceil(1.05 * SIGMA ** 2 * T / dx ** 2)) if theta == 0 else nx
        v, t = timed(finite_difference, nx, nt, theta, rann)
        rows.append((t, abs(v - EXACT)))
    results[name] = rows
# NOTE: I expected Crank-Nicolson to win this group. it didn't. explicit's
# stability rule forces really small time steps, which also makes it very
# accurate, and each of its steps is just array arithmetic with no
# tridiagonal solve. the textbook ranking assumes you'd pick the time step
# freely, and here I couldn't.
# Rannacher (a few implicit steps first, to smooth the kink in the payoff)
# didn't change the price at all. apparently it matters for the Greeks.

# Fourier
rows = []
for n in [2 ** k for k in range(8, 17)]:
    v, t = timed(fourier, n)
    rows.append((t, abs(v - EXACT)))
results["Fourier (Carr-Madan)"] = rows

for name, rows in results.items():
    print(name)
    for t, e in rows:
        print(f"   {t * 1000:9.3f} ms   error {e:.2e}")

plt.figure(figsize=(8, 5))
for name, rows in results.items():
    t, e = zip(*rows)
    plt.loglog(np.array(t) * 1000, e, "o-", label=name, ms=4)
plt.axhline(0.005, color="grey", ls=":", lw=1)
plt.text(60, 0.0058, "half a cent", color="grey", fontsize=8)
plt.xlabel("compute time (ms)")
plt.ylabel("error vs the formula ($)")
plt.title("Same option, five methods: accuracy for the time spent")
plt.legend(fontsize=8)
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig("figures/error_vs_time.png", dpi=150)
print("saved figures/error_vs_time.png")

# ---- the tree's odd/even wobble ----
ns = np.arange(50, 91)
errs = [binomial(n) - EXACT for n in ns]
print("\ntree error, n = 50..55:", " ".join(f"{e:+.4f}" for e in errs[:6]))
plt.figure(figsize=(7, 3.8))
plt.plot(ns, errs, "o-", ms=3)
plt.axhline(0, color="k", lw=0.8)
plt.xlabel("number of steps n")
plt.ylabel("tree price minus formula ($)")
plt.title("The binomial tree zig-zags between odd and even n")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures/tree_wobble.png", dpi=150)
print("saved figures/tree_wobble.png")
