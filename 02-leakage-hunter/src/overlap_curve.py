"""
How much fake R^2 does a given amount of overlap buy?

If the target shares a fraction f of its returns with the feature, and
the returns are independent with equal variance, the correlation between
the two is exactly f, so R^2 = f^2. I check that against simulation.
"""
import numpy as np
import matplotlib.pyplot as plt
from pipelines import H, overlapping_target

rng = np.random.default_rng(1)
shifts = np.arange(H + 1)               # shift = H is the correct alignment
overlap = (H - shifts) / H              # fraction of the target that is past

sim = []
for k in shifts:
    scores = [overlapping_target(rng.standard_normal(20_000), shift=k) for _ in range(10)]
    sim.append(np.mean(scores))
    print(f"shift {k:2d}  overlap {overlap[k]:.2f}  simulated R^2 {sim[-1]:+.3f}  theory {overlap[k]**2:.3f}")

f = np.linspace(0, 1, 200)
plt.figure(figsize=(6.2, 4.2))
plt.plot(f, f ** 2, "k-", label="theory: R² = f²")
plt.plot(overlap, sim, "o", color="tab:red", label="simulated (pure noise)")
plt.xlabel("fraction of the target already inside the feature, f")
plt.ylabel("out-of-sample R²")
plt.title("Fake R² from overlapping windows")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures/overlap_curve.png", dpi=150)
print("saved figures/overlap_curve.png")
