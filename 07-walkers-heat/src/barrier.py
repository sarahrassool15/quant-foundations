"""
Part 3: put a wall at x = 1 that kills any walker that touches it.

PDE version: same heat equation, with u = 0 at x = 1.
Solution by the "method of images": put a negative copy of the start
point at 2a (the mirror image of 0 across a), so the two cancel on the
wall:
    u(x, t) = phi(x; t) - phi(2a - x; t)
That's the reflection principle from project 03 written as a PDE. 2a - x
is the same mirror trick.
Surviving fraction at time t = 2 Phi(a / sqrt(t)) - 1.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.special import zeta

rng = np.random.default_rng(1)
A, T = 1.0, 1.0
BETA = -zeta(0.5) / np.sqrt(2 * np.pi)
exact_survival = 2 * norm.cdf(A / np.sqrt(T)) - 1
print(f"exact surviving fraction at t = 1: {exact_survival:.4f}")


def survivors(n, dt, wall):
    x = np.zeros(n)
    alive = np.ones(n, dtype=bool)
    for _ in range(int(T / dt)):
        x[alive] += rng.standard_normal(alive.sum()) * np.sqrt(dt)
        alive &= x < wall
    return x[alive], alive.mean()


print("\nwalkers checked only at each step:")
print("   dt        survived   error      survived (wall moved in by beta*sqrt(dt))")
dts = [0.04, 0.01, 0.0025, 0.000625]
plain, fixed = [], []
for dt in dts:
    _, s = survivors(200_000, dt, A)
    _, s2 = survivors(200_000, dt, A - BETA * np.sqrt(dt))
    plain.append(s)
    fixed.append(s2)
    print(f"  {dt:<9}  {s:.4f}     {s - exact_survival:+.4f}    {s2:.4f} ({s2 - exact_survival:+.4f})")
# first time I ran this the walkers survived too often and I thought the
# formula was wrong. it's the project 03 thing: between checks a walker
# can cross the wall and come back, and I never see it.
# fix: move the wall IN by 0.5826 sqrt(dt), towards the walkers, so the
# discrete check catches about as many as the real wall would.
# (careful with the direction. moving it OUT makes it worse, obviously,
# once you think about it for more than two seconds.)

x_alive, _ = survivors(200_000, 0.0025, A - BETA * np.sqrt(0.0025))
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].hist(x_alive, bins=80, density=False, weights=np.full(len(x_alive), 1 / 200_000 / ((x_alive.max() - x_alive.min()) / 80)),
           alpha=0.5, label="surviving walkers")
xs = np.linspace(-4, A, 400)
ax[0].plot(xs, norm.pdf(xs, 0, np.sqrt(T)) - norm.pdf(2 * A - xs, 0, np.sqrt(T)), "k-",
           label="method of images")
ax[0].axvline(A, color="tab:red", lw=2)
ax[0].set_xlabel("position at t = 1")
ax[0].set_title("Walkers with a wall at x = 1")
ax[0].legend()
ax[1].loglog(dts, np.array(plain) - exact_survival, "o-", label="check every step")
ax[1].loglog(dts, BETA * np.sqrt(dts) * 2 * norm.pdf(A), "k--", label="≈ 2φ(a)·β√dt")
ax[1].set_xlabel("dt")
ax[1].set_ylabel("too many survivors")
ax[1].set_title("Missing crossings between checks")
ax[1].legend()
for a in ax:
    a.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures/barrier.png", dpi=150)
print("saved figures/barrier.png")
