"""
Why bet half?

Kelly assumes I KNOW my edge. I never do; I estimate it from history.
For an asset with drift mu and volatility sigma, the Kelly fraction is
f* = mu / sigma^2 and the growth rate of betting f is
    g(f) = mu f - sigma^2 f^2 / 2
(same parabola idea as the coin, just continuous).

Suppose my estimate f_hat is right on average but noisy:
    E[f_hat] = f*,  Var(f_hat) = v
and I bet a fixed fraction c of it, f = c * f_hat. Then
    E[g] = mu c f* - (sigma^2 / 2) c^2 E[f_hat^2]
         = sigma^2 ( c f*^2 - (c^2 / 2)(f*^2 + v) )
    using mu = sigma^2 f* and E[f_hat^2] = Var + mean^2 = v + f*^2
Differentiate in c and set to zero:
    f*^2 - c (f*^2 + v) = 0   ->   c = f*^2 / (f*^2 + v)
So the best fraction of Kelly to bet depends on how noisy my estimate is.
If the noise is as big as the edge (v = f*^2), c = 1/2. Half Kelly.
"""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(1)
MU, SIGMA = 0.20, 0.60          # a made-up asset: 20%/year drift, 60%/year vol
F_STAR = MU / SIGMA ** 2        # 0.56: true Kelly
DAYS = 365


def realised_growth(c, years_of_history, n_worlds=100_000, test_years=10):
    # each "world": I see some years of daily returns, estimate mu,
    # then bet c * f_hat for the next 10 years and record my growth.
    dt = 1 / DAYS
    n_hist = int(years_of_history * DAYS)
    # mean of daily returns over the history window. I only need the mean,
    # so I draw it directly: mean of n normals ~ N(mu dt, sigma^2 dt / n)
    mu_hat = rng.normal(MU * dt, SIGMA * np.sqrt(dt / n_hist), n_worlds) / dt
    f_hat = mu_hat / SIGMA ** 2          # I pretend sigma is known (it's estimated
                                         # far better than mu, so this is fair)
    f = c * f_hat
    # growth over the test period is g(f) = mu f - sigma^2 f^2 / 2 per year
    # (exact in expectation for continuous rebalancing)
    return np.mean(MU * f - SIGMA ** 2 * f ** 2 / 2)


cs = np.linspace(0, 1.5, 61)
plt.figure(figsize=(7, 4.3))
print(f"true Kelly f* = {F_STAR:.2f}")
for years in [2, 5, 10, 30]:
    v = (SIGMA / np.sqrt(years)) ** 2 / SIGMA ** 4   # Var(mu_hat) / sigma^4
    c_theory = F_STAR ** 2 / (F_STAR ** 2 + v)
    g = [realised_growth(c, years) for c in cs]
    c_sim = cs[int(np.argmax(g))]
    print(f"  {years:2d} years of history:  noise/edge = {np.sqrt(v) / F_STAR:.2f}   "
          f"best c simulated {c_sim:.2f}   theory {c_theory:.2f}")
    plt.plot(cs, g, label=f"{years} years of history (best c = {c_theory:.2f})")
plt.axhline(0, color="grey", lw=0.8)
plt.axvline(1, color="k", ls=":", lw=1)
plt.xlabel("fraction of my estimated Kelly bet, c")
plt.ylabel("expected growth per year")
plt.title("Full Kelly on a noisy estimate is a bad idea")
plt.legend(fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures/shrinkage.png", dpi=150)
print("saved figures/shrinkage.png")
