"""
Two ways to get the exact distribution of the sum of n dice.

A die is just a probability vector: die[k] = P(face = k).
For a fair six-sided die that's [0, 1/6, 1/6, 1/6, 1/6, 1/6, 1/6]
(index 0 is "rolling a 0", which has probability 0).
"""
import numpy as np

FAIR_D6 = np.array([0, 1, 1, 1, 1, 1, 1]) / 6.0


def pmf_convolution(n, die=FAIR_D6):
    # The direct way: the distribution of (sum so far + one more die)
    # is the convolution of the two distributions. I just do it n times.
    pmf = np.array([1.0])            # sum of zero dice is 0 with probability 1
    for _ in range(n):
        pmf = np.convolve(pmf, die)
    return pmf                       # pmf[s] = P(sum = s)


def pmf_fft(n, die=FAIR_D6, N=None):
    # The Fourier way
    # The FFT of the die is its characteristic function
    # sampled on a grid. Adding independent dice multiplies characteristic
    # functions, so n dice is just the n-th power. Then I transform back.
    max_sum = n * (len(die) - 1)
    if N is None:
        N = max_sum + 1              # the smallest grid that fits every sum
    phi = np.fft.fft(die, N)         # zero-pads the die out to length N
    pmf = np.fft.ifft(phi ** n).real
    return pmf

# NOTE to self: if N is smaller than max_sum + 1, the FFT can't tell sums apart that
# differ by a multiple of N, so they land on top of each other 
# so wraparound.py shows what that looks like
