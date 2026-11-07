"""
Gambler's ruin, the classic version first.

I start with k chips. Each round I win 1 chip with prob p or lose 1 with
prob q = 1 - p. I stop when I hit N (rich) or 0 (broke).
What's the chance I get to N first?
"""
import numpy as np


def p_reach_target(k, N, p):
    # let h(k) = P(reach N before 0 | I have k). after one round:
    #   h(k) = p h(k+1) + q h(k-1),  with h(0) = 0, h(N) = 1
    # try h(k) = r^k:  r = p r^2 + q  ->  p r^2 - r + q = 0  ->  r = 1 or r = q/p
    # so h(k) = A + B (q/p)^k, and the two boundary conditions give
    #   h(k) = (1 - (q/p)^k) / (1 - (q/p)^N)
    # careful: if p = 1/2 then q/p = 1 and that's 0/0. the two roots
    # collide, the second solution becomes k instead of (q/p)^k, and
    #   h(k) = k / N
    q = 1 - p
    if np.isclose(p, 0.5):
        return k / N
    r = q / p
    return (1 - r ** k) / (1 - r ** N)


def simulate(k, N, p, n_games=20_000, seed=0):
    # play all 20,000 games at once: each step, every game still going
    # moves up or down by 1. games that hit 0 or N just stop moving.
    rng = np.random.default_rng(seed)
    x = np.full(n_games, k)
    alive = np.ones(n_games, dtype=bool)
    while alive.any():
        step = np.where(rng.random(alive.sum()) < p, 1, -1)
        x[alive] += step
        alive = (x > 0) & (x < N)
    return np.mean(x == N)


print("fair coin vs slightly unfair coin, start with 50, target 100")
for p in [0.5, 0.49, 18 / 37]:
    print(f"  p = {p:.4f}:  formula {p_reach_target(50, 100, p):.4f}   simulated {simulate(50, 100, p):.4f}")

# roulette: betting red wins with prob 18/37 (the green 0 is the house edge).
# I have 50 and want 100. two ways to play:
#   timid: bet 1 at a time, hundreds of bets
#   bold: bet all 50 on one spin
p = 18 / 37
print("\nroulette, have 50, want 100:")
print(f"  bet 1 at a time:  {p_reach_target(50, 100, p):.4f}")
print(f"  bet it all once:  {p:.4f}")
# NOTE: in a game that's against you, playing timidly is the WORST thing to do.
# every extra bet is another chance for the small edge to grind you down.
# the long run is your enemy when the odds are against you.
