"""
One European call option, priced five different ways.
S = 100, K = 100, one year, 5% interest, 20% volatility.
"""
import numpy as np
from scipy.stats import norm
from scipy.linalg import solve_banded

S0, K, T, R, SIGMA = 100.0, 100.0, 1.0, 0.05, 0.20


# ---------------------------------------------------------------- 1. formula
def black_scholes():
    d1 = (np.log(S0 / K) + (R + SIGMA ** 2 / 2) * T) / (SIGMA * np.sqrt(T))
    d2 = d1 - SIGMA * np.sqrt(T)      # d2 = d1 - sigma sqrt(T), not d1 - sigma. (I've done that.)
    return S0 * norm.cdf(d1) - K * np.exp(-R * T) * norm.cdf(d2)


# ---------------------------------------------------------------- 2. binomial tree
def binomial(n):
    # each step the price goes up by u or down by d = 1/u.
    # the up-probability p is chosen so the stock earns the risk-free rate
    # on average: p u + (1-p) d = e^(r dt)  ->  p = (e^(r dt) - d) / (u - d)
    dt = T / n
    u = np.exp(SIGMA * np.sqrt(dt))   # sqrt(dt) again, same as the walkers
    d = 1 / u
    p = (np.exp(R * dt) - d) / (u - d)
    # prices at the end: S0 u^j d^(n-j) for j = 0..n up-moves
    j = np.arange(n + 1)
    v = np.maximum(S0 * u ** j * d ** (n - j) - K, 0)
    disc = np.exp(-R * dt)
    for _ in range(n):                # walk back one step at a time
        v = disc * (p * v[1:] + (1 - p) * v[:-1])
    return v[0]


# ---------------------------------------------------------------- 3. Monte Carlo
def monte_carlo(n, seed=0):
    # GBM has an exact formula at time T, so no time-stepping needed:
    #   S_T = S0 exp((r - sigma^2/2) T + sigma sqrt(T) Z)
    # the -sigma^2/2 is the Ito correction from my stochastic calculus project.
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    st = S0 * np.exp((R - SIGMA ** 2 / 2) * T + SIGMA * np.sqrt(T) * z)
    pay = np.exp(-R * T) * np.maximum(st - K, 0)
    return pay.mean(), pay.std() / np.sqrt(n)


# ---------------------------------------------------------------- 4. finite differences
def finite_difference(nx, nt, theta, rannacher=0):
    """
    Black-Scholes PDE in x = log S, going backwards from expiry (tau = time left):
        V_tau = (1/2) sigma^2 V_xx + (r - sigma^2/2) V_x - r V
    In log S the coefficients are constant, which makes it basically the
    heat equation from project 07 plus a drift term and a discount term.
    theta = 0 explicit, 1 implicit, 1/2 Crank-Nicolson.
    """
    width = 5 * SIGMA * np.sqrt(T)
    x0 = np.log(S0)
    x = np.linspace(x0 - width, x0 + width, nx)      # nx odd puts x0 exactly on the grid
    dx = x[1] - x[0]
    dt = T / nt
    v = np.maximum(np.exp(x) - K, 0)                  # payoff at expiry

    # coefficients for the neighbours j-1, j, j+1 of the spatial operator
    a = 0.5 * SIGMA ** 2 / dx ** 2
    b = (R - 0.5 * SIGMA ** 2) / (2 * dx)             # central difference for V_x
    lo, mid, hi = a - b, -2 * a - R, a + b

    def apply_op(u):
        out = np.zeros_like(u)
        out[1:-1] = lo * u[:-2] + mid * u[1:-1] + hi * u[2:]
        return out

    tau = 0.0
    for step in range(nt):
        th = 1.0 if step < rannacher else theta       # Rannacher: a few implicit steps first
        tau += dt
        rhs = v + (1 - th) * dt * apply_op(v)
        # boundaries: far below the strike the call is worthless,
        # far above it's worth S - K e^(-r tau)
        left, right = 0.0, np.exp(x[-1]) - K * np.exp(-R * tau)
        if th == 0:
            v = rhs
        else:
            # (I - th dt L) v_new = rhs, a tridiagonal system. solve_banded wants
            # the three diagonals stacked as rows: upper, main, lower.
            n_in = nx - 2
            ab = np.zeros((3, n_in))
            ab[0, 1:] = -th * dt * hi
            ab[1, :] = 1 - th * dt * mid
            ab[2, :-1] = -th * dt * lo
            r_in = rhs[1:-1].copy()
            r_in[0] += th * dt * lo * left
            r_in[-1] += th * dt * hi * right
            v[1:-1] = solve_banded((1, 1), ab, r_in)
        v[0], v[-1] = left, right
    return v[nx // 2]


# ---------------------------------------------------------------- 5. Fourier (Carr-Madan)
def fourier(n, eta=0.25, alpha=1.5):
    """
    Same idea as the dice in project 01: I don't need the density of log S_T,
    only its characteristic function, and an FFT does the rest.
    """
    def phi(u):
        # characteristic function of log S_T under Black-Scholes:
        # it's normal with mean log S0 + (r - sigma^2/2) T and variance sigma^2 T
        m = np.log(S0) + (R - SIGMA ** 2 / 2) * T
        return np.exp(1j * u * m - 0.5 * SIGMA ** 2 * T * u ** 2)

    # the call payoff isn't integrable in log-strike (it doesn't go to zero
    # as the strike goes to minus infinity), so Carr and Madan multiply the
    # price by e^(alpha k) to damp it, transform, and undo it at the end.
    v = np.arange(n) * eta
    psi = np.exp(-R * T) * phi(v - (alpha + 1) * 1j) / (alpha ** 2 + alpha - v ** 2 + 1j * (2 * alpha + 1) * v)
    lam = 2 * np.pi / (n * eta)                       # log-strike grid spacing: the FFT ties them together
    b = n * lam / 2
    k = -b + lam * np.arange(n)
    # Simpson's rule weights (1/3, 4/3, 2/3, 4/3, ...) for the integral
    w = (3 + (-1) ** (np.arange(n) + 1)) / 3
    w[0] = 1 / 3
    x = np.exp(1j * b * v) * psi * eta * w
    calls = np.exp(-alpha * k) / np.pi * np.real(np.fft.fft(x))
    return np.interp(np.log(K), k, calls)              # I want the strike K, which sits between grid points
