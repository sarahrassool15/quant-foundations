"""
Checking two Brownian motion facts by simulation BEFORE I try them on Bitcoin.
(If the code can't get it right on fake data where I know the answer,
there's no point running it on real data.)

1. Reflection principle: P(max over [0,T] >= a) = 2 P(W_T >= a)
2. Looking at the price only every dt makes you miss the max by about
   beta * sigma * sqrt(dt), with beta = -zeta(1/2) / sqrt(2 pi) = 0.5826
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.special import zeta

# beta. zeta(1/2) is about -1.4604, NEGATIVE, so the minus sign in front
# makes beta positive. without it beta < 0, which would mean looking less
# often makes you see a HIGHER max. impossible, since fewer points can't
# have a bigger max.
# NOTE: zeta(1/2) is NOT 1 + 1/sqrt(2) + 1/sqrt(3) + ... (that blows up).
# it's the "analytically continued" value. I don't need to derive it, just
# know where it comes from (Asmussen, Glynn & Pitman 1995).
BETA = -zeta(0.5) / np.sqrt(2 * np.pi)
rng = np.random.default_rng(0)

# one "day" with sigma = 1 and T = 1, on a really fine grid
n_paths, n_fine = 4000, 2 ** 14
dt_fine = 1 / n_fine

# each step ~ N(0, dt), so the std of a step is sqrt(dt), NOT dt.
# (variance = time, std = sqrt(time). I keep mixing these up.)
# cumsum adds the steps up, so each row is one Brownian path.
paths = np.cumsum(rng.standard_normal((n_paths, n_fine)) * np.sqrt(dt_fine), axis=1)
paths = np.hstack([np.zeros((n_paths, 1)), paths])     # W_0 = 0, rule 1 of BM

# ---- fact 1: the max behaves like |Z| ----
# why: split P(max >= a) by where the path ENDS:
#   ends above a -> it obviously touched a  -> that's just P(W_T >= a)
#   ends below a -> flip everything after the first touch around a:
#                   new path = 2a - W  (reflection of x across a is
#                   a - (x - a) = 2a - x, and at x = a it's still a, so no jump)
#                   now it ends ABOVE a. BM is symmetric and restarts fresh at
#                   the first touch, so the flipped path is equally likely.
#                -> another P(W_T > a)
# total: 2 P(W_T >= a) = 2(1 - Phi(a / (sigma sqrt T)))
# >= vs > doesn't matter: P(W_T = a) = 0 for a continuous variable.
m = paths.max(axis=1)
print("fact 1: P(max >= a), simulated vs 2*(1 - Phi(a))")
for a in [0.5, 1.0, 1.5, 2.0, 2.5]:
    # norm.sf(a) = 1 - norm.cdf(a), but more accurate far out in the tail
    # (1 - 0.99999999... throws digits away, same lesson as the FFT floor)
    print(f"  a = {a}:  simulated {np.mean(m >= a):.4f}   formula {2 * norm.sf(a):.4f}")

# mean of |Z| = sqrt(2/pi):
#   E|Z| = 2 * integral_0^inf z phi(z) dz
#   the 2: |z| phi(z) is EVEN, so the whole line is twice the right half.
#   (E[Z] itself is 0 because z phi(z) is ODD, so the halves cancel. the
#   absolute value is literally what stops the cancelling.)
#   sub u = z^2/2, du = z dz. the z in front is exactly the derivative of
#   z^2/2, which is the ONLY reason this integral is doable.
#   = (2/sqrt(2pi)) * integral_0^inf e^-u du = 2/sqrt(2pi) = sqrt(2/pi)
# my grid isn't continuous, so the max I see is a bit low: minus beta*sqrt(dt).
print(f"  mean of max: simulated {m.mean():.4f}   formula sqrt(2/pi) - beta*sqrt(dt) = "
      f"{np.sqrt(2 / np.pi) - BETA * np.sqrt(dt_fine):.4f}")
# NOTE: 0.806 vs 0.793 looks off at first. but the max has a std
# of about 0.6, so the standard error is 0.6 / sqrt(4000) = 0.0095. I'm
# about 1.3 standard errors off. that's just noise.

# ---- fact 2: looking less often hides the max ----
# the seen max can never be ABOVE the true max (a max over fewer points),
# so the gap is always >= 0.
# size: between two looks BM moves about sigma*sqrt(dt), so I miss about that.
# my finest grid isn't truly continuous either, so I compare each coarse grid
# to the finest one: theory gap = beta*(sqrt(dt) - sqrt(dt_fine)).
print(f"\nfact 2: beta = {BETA:.4f}")
steps = [2 ** j for j in range(1, 11)]                 # look every 2, 4, ..., 1024 fine steps
gaps, theory = [], []
for k in steps:
    m_k = paths[:, ::k].max(axis=1)                    # ::k keeps every k-th point, starting at W_0
    gaps.append(np.mean(m - m_k))
    theory.append(BETA * (np.sqrt(k * dt_fine) - np.sqrt(dt_fine)))
    print(f"  look every {k:4d} steps: gap {gaps[-1]:.4f}   theory {theory[-1]:.4f}")
# NOTE: theory and simulation drift apart at the big k. that's expected:
# beta*sqrt(dt) is only the first term for SMALL dt.

dts = np.array(steps) * dt_fine
plt.figure(figsize=(6.2, 4.2))
plt.loglog(dts, gaps, "o", color="tab:red", label="simulated gap")
plt.loglog(dts, theory, "k-", label=r"theory: $\beta(\sqrt{\Delta t}-\sqrt{\Delta t_0})$")
plt.xlabel("time between looks, Δt (fraction of the day)")
plt.ylabel("average missed maximum (units of σ√T)")
plt.title("Looking less often hides the maximum")
plt.legend()
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig("figures/sim_monitoring_gap.png", dpi=150)
print("saved figures/sim_monitoring_gap.png")
