"""
Part 2: solving the heat equation on a grid, and watching it blow up.

Explicit scheme for u_t = (1/2) u_xx:
    u_j^{n+1} = u_j^n + lam (u_{j+1}^n - 2 u_j^n + u_{j-1}^n),
    lam = (1/2) dt / dx^2
(the bracket is the usual second-difference approximation to u_xx, which
I remember from numerical analysis as "right + left - 2 middle".)
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

L, T0, T1 = 6.0, 0.05, 1.0


def solve(dx, lam, t_end=T1):
    x = np.arange(-L, L + dx / 2, dx)
    dt = 2 * lam * dx ** 2           # rearranged from lam = (1/2) dt / dx^2
    n = int(round((t_end - T0) / dt))
    # start from the exact answer at t = 0.05 instead of a spike at 0,
    # because a spike on a grid is a mess
    u = norm.pdf(x, 0, np.sqrt(T0))
    snaps = []
    for k in range(n):
        u[1:-1] = u[1:-1] + lam * (u[2:] - 2 * u[1:-1] + u[:-2])
        if k in (int(0.50 * n), int(0.56 * n), int(0.60 * n)):
            snaps.append(u.copy())
    # (ends stay 0: at x = +/-6 there's basically no heat anyway)
    exact = norm.pdf(x, 0, np.sqrt(T0 + n * dt))
    return x, u, exact, snaps


print("dx = 0.05")
for lam in [0.40, 0.50, 0.51, 0.55]:
    x, u, exact, _ = solve(0.05, lam)
    print(f"  lambda = {lam:.2f}:  max error {np.max(np.abs(u - exact)):.2e}")
# the theory: each step multiplies the most jagged wave (+,-,+,-,...) by
# 1 - 4 lam. if |1 - 4 lam| > 1, i.e. lam > 1/2, it grows every step.
# rounding error puts a tiny bit of that wave in on step one, and it
# grows from 1e-16 to infinity.
# NOTE: 0.51 looked fine and I almost moved on. it just hadn't had time:
# 1 - 4(0.51) = -1.04, so it only grows 4% a step. 370 steps takes 1e-16
# to about 1e-10, which is invisible. run it longer:
x, u, exact, _ = solve(0.05, 0.51, t_end=10.0)
print(f"  lambda = 0.51, run to t = 10 instead:  max error {np.max(np.abs(u - exact)):.2e}")

print("\nerror vs grid size at lambda = 0.4 (should fall like dx^2)")
dxs = [0.2, 0.1, 0.05, 0.025]
errs = []
for dx in dxs:
    x, u, exact, _ = solve(dx, 0.4)
    errs.append(np.max(np.abs(u - exact)))
    print(f"  dx = {dx:5.3f}:  max error {errs[-1]:.2e}")
slope = np.polyfit(np.log(dxs), np.log(errs), 1)[0]
print(f"  slope {slope:.2f}")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
x, u, exact, snaps = solve(0.05, 0.55)
for s, lab in zip(snaps, ["50%", "56%", "60%"]):
    ax[0].plot(x, s, lw=1, label=f"{lab} of the way")
ax[0].set_xlim(-3, 3)
ax[0].set_ylim(-0.5, 1.5)
ax[0].set_title("λ = 0.55: a little past the limit")
ax[0].set_xlabel("x")
ax[0].legend()
ax[1].loglog(dxs, errs, "o-", label="explicit scheme, λ = 0.4")
ax[1].loglog(dxs, errs[0] * (np.array(dxs) / dxs[0]) ** 2, "k--", label="dx²")
ax[1].set_xlabel("dx")
ax[1].set_ylabel("max error at t = 1")
ax[1].set_title("When it works, it's second order")
ax[1].legend()
for a in ax:
    a.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures/explicit.png", dpi=150)
print("saved figures/explicit.png")
