"""
Part 1: a crowd of coin-flip walkers spreads out exactly like heat.

Each walker steps +sqrt(dt) or -sqrt(dt) with equal chance. After t = 1
(100 steps), where are they? The heat equation u_t = (1/2) u_xx, started
from everything at 0, says the density is the normal N(0, t).
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

rng = np.random.default_rng(0)
dt, T = 0.01, 1.0
n_steps = int(T / dt)
step = np.sqrt(dt)        # step size sqrt(dt), not dt. variance per step = dt,
                          # so after t/dt steps the variance is t. same as BM.

# why the 1/2 in u_t = (1/2) u_xx? for Brownian motion the density spreads
# with variance t, and the heat kernel for u_t = D u_xx has variance 2Dt.
# 2Dt = t  ->  D = 1/2. easy to drop and then everything is off by sqrt(2).


def walk(n):
    # number of +steps out of 100 is Binomial(100, 1/2). I can draw that
    # directly instead of flipping 100 coins per walker.
    ups = rng.binomial(n_steps, 0.5, n)
    return (2 * ups - n_steps) * step       # (ups - downs) * step size


# NOTE: the walkers can only land on every OTHER grid point (100 steps
# means an even number of net steps), so the histogram bins have to be
# 2 * step wide or half of them come out empty.
edges = (np.arange(-n_steps, n_steps + 3, 2) - 1) * step
centres = (edges[:-1] + edges[1:]) / 2
exact = norm.pdf(centres, 0, np.sqrt(T))

print("walkers    L1 distance from the heat equation")
ns = [100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000]
errs = []
for n in ns:
    hist, _ = np.histogram(walk(n), bins=edges, density=True)
    errs.append(np.sum(np.abs(hist - exact)) * 2 * step)
    print(f"{n:10,d}   {errs[-1]:.4f}")
# I expected this to keep falling like 1/sqrt(n). it does, then starts to level off.
# the floor is the coin flips themselves: 100 binomial steps aren't exactly
# normal. more walkers can't fix that, only smaller steps can.

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
hist, _ = np.histogram(walk(10_000), bins=edges, density=True)
ax[0].bar(centres, hist, width=2 * step, alpha=0.5, label="10,000 walkers")
x = np.linspace(-4, 4, 400)
ax[0].plot(x, norm.pdf(x, 0, 1), "k-", label="heat equation")
ax[0].set_xlim(-4, 4)
ax[0].set_xlabel("position at t = 1")
ax[0].set_title("Coin-flip walkers vs the heat equation")
ax[0].legend()
ax[1].loglog(ns, errs, "o-", label="walkers")
ax[1].loglog(ns, errs[0] * np.sqrt(ns[0] / np.array(ns)), "k--", label="1/√n")
ax[1].set_xlabel("number of walkers")
ax[1].set_ylabel("L1 distance")
ax[1].set_title("More walkers helps, until it doesn't")
ax[1].legend()
for a in ax:
    a.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures/walkers.png", dpi=150)
print("saved figures/walkers.png")
