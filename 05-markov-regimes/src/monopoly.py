"""
Which Monopoly square gets landed on most?

A Markov chain: where I go next depends only on where I am now (plus the
dice), not on how I got here. So the whole game is one 40 x 40 matrix P,
where P[i, j] = chance of ending my turn on square j if I start on i.

The long-run fraction of time on each square is the vector pi with
    pi P = pi
i.e. pi is an eigenvector of P (transposed) with eigenvalue 1.
"""
import numpy as np
import matplotlib.pyplot as plt

NAMES = ["Go", "Mediterranean", "Community Chest", "Baltic", "Income Tax",
         "Reading RR", "Oriental", "Chance", "Vermont", "Connecticut",
         "Jail", "St. Charles", "Electric Co", "States", "Virginia",
         "Pennsylvania RR", "St. James", "Community Chest", "Tennessee", "New York",
         "Free Parking", "Kentucky", "Chance", "Indiana", "Illinois",
         "B&O RR", "Atlantic", "Ventnor", "Water Works", "Marvin Gardens",
         "Go To Jail", "Pacific", "North Carolina", "Community Chest", "Pennsylvania Ave",
         "Short Line", "Chance", "Park Place", "Luxury Tax", "Boardwalk"]
N = 40
GO, JAIL, GO_TO_JAIL = 0, 10, 30
CHANCE, CHEST = {7, 22, 36}, {2, 17, 33}
RAILROADS, UTILITIES = [5, 15, 25, 35], [12, 28]

# sum of two dice = convolution of two dice. same as project 01, n = 2.
d6 = np.array([0, 1, 1, 1, 1, 1, 1]) / 6
two_dice = np.convolve(d6, d6)        # two_dice[s] = P(sum = s), s = 0..12


def next_of(square, group):
    # first square in the group strictly ahead of me, wrapping past Go
    for g in sorted(group):
        if g > square:
            return g
    return min(group)


def where_i_end_up(square):
    """After landing on `square`, where do I actually finish? Returns {square: prob}."""
    if square == GO_TO_JAIL:
        return {JAIL: 1.0}
    if square in CHEST:
        # 16 cards, 2 move you: advance to Go, go to Jail. the other 14 don't.
        return {GO: 1 / 16, JAIL: 1 / 16, square: 14 / 16}
    if square in CHANCE:
        # 16 cards, 10 move you. the other 6 don't.
        moves = [GO, JAIL, 11, 24, 39, 5,              # Go, Jail, St. Charles, Illinois, Boardwalk, Reading
                 next_of(square, RAILROADS), next_of(square, RAILROADS),   # two "nearest railroad" cards
                 next_of(square, UTILITIES)]
        out = {}
        for m in moves:
            out[m] = out.get(m, 0) + 1 / 16
        # "go back 3 spaces". careful: from Chance at 36 that lands on 33,
        # which is Community Chest, so I have to apply THAT square's rules too.
        for s, p in where_i_end_up(square - 3).items():
            out[s] = out.get(s, 0) + p / 16
        out[square] = out.get(square, 0) + 6 / 16
        return out
    return {square: 1.0}


# build the 40 x 40 matrix: roll, move, then apply the square's rules
P = np.zeros((N, N))
for i in range(N):
    for s in range(2, 13):
        landing = (i + s) % N          # % 40 wraps around the board past Go
        for j, p in where_i_end_up(landing).items():
            P[i, j] += two_dice[s] * p
assert np.allclose(P.sum(axis=1), 1)   # every row is a probability distribution

# pi P = pi  ->  P.T pi.T = pi.T, so I want the eigenvector of P.T for eigenvalue 1.
# (the transpose is the step I always forget: P acts on ROW vectors here.)
vals, vecs = np.linalg.eig(P.T)
k = np.argmin(np.abs(vals - 1))
pi = np.real(vecs[:, k])
pi = pi / pi.sum()                     # eigenvectors come out with any scale (even negative), so normalise
pi[np.abs(pi) < 1e-12] = 0            # Go To Jail is exactly 0 (you never END a turn there),
                                      # but rounding leaves -0.000000000000001. tidy it.

# second biggest eigenvalue (by size) = how fast the board forgets where you started.
# after n turns the leftover memory shrinks like |lambda_2|^n.
lam = np.sort(np.abs(vals))[::-1]
print(f"largest eigenvalue {lam[0]:.4f} (always 1 for a Markov chain)")
print(f"second largest |eigenvalue| {lam[1]:.3f}: the start is half-forgotten after "
      f"{np.log(0.5) / np.log(lam[1]):.1f} turns")

# check it the slow way: just play a very long game
rng = np.random.default_rng(0)
pos, visits = 0, np.zeros(N)
for _ in range(1_000_000):
    pos = rng.choice(N, p=P[pos])
    visits[pos] += 1
sim = visits / visits.sum()
print(f"max |eigenvector - simulation| = {np.max(np.abs(pi - sim)):.4f}")

order = np.argsort(pi)[::-1]
print("\nmost landed on (a fair board would be 2.5% each):")
for i in order[:6]:
    print(f"  {NAMES[i]:18s} {pi[i]:.2%}")
print(f"least: {NAMES[order[-1]]} {pi[order[-1]]:.2%}")

colors = ["tab:red" if i == JAIL else "tab:blue" for i in range(N)]
plt.figure(figsize=(12, 4))
plt.bar(range(N), pi, color=colors)
plt.axhline(1 / N, color="k", ls="--", lw=1, label="uniform, 2.5%")
plt.xticks(range(N), NAMES, rotation=90, fontsize=7)
plt.ylabel("long-run share of turns")
plt.title("Where you end your turn in Monopoly, from the eigenvector")
plt.legend()
plt.tight_layout()
plt.savefig("figures/monopoly.png", dpi=150)
print("saved figures/monopoly.png")
