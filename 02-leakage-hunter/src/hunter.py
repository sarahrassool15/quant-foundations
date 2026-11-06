"""
The leakage hunter: a placebo test.

Feed a pipeline returns that I *know* are pure noise. If it still scores
a positive R^2, the pipeline is leaking. Real data can't tell you that,
because you never know how much signal is really there. Noise can.
"""
import numpy as np
import matplotlib.pyplot as plt
from pipelines import PIPELINES


def placebo(pipeline, n_trials=20, n=20_000, seed=0):
    rng = np.random.default_rng(seed)
    scores = [pipeline(rng.standard_normal(n)) for _ in range(n_trials)]
    return np.mean(scores), np.std(scores)


if __name__ == "__main__":
    names, means, sds = [], [], []
    for name, f in PIPELINES.items():
        m, s = placebo(f)
        names.append(name)
        means.append(m)
        sds.append(s)
        print(f"{name.replace(chr(10), ' '):35s}  R^2 on pure noise = {m:+.3f}  (sd {s:.3f})")

    colors = ["tab:green" if m < 0.02 else "tab:red" for m in means]
    plt.figure(figsize=(8.5, 4.3))
    plt.bar(names, means, yerr=sds, color=colors, capsize=4)
    plt.axhline(0, color="k", lw=0.8)
    plt.ylabel("out-of-sample R² on pure noise")
    plt.title("Every red bar is a model 'predicting' the unpredictable")
    plt.tight_layout()
    plt.savefig("figures/placebo.png", dpi=150)
    print("saved figures/placebo.png")
