#!/usr/bin/env python3
"""Independent NumPy implementation of White's Reality Check and Hansen's SPA test.

Reproduces the reference values pinned in tests/test_significance.cpp
(TestSignificance_Spa_MatchesNumPyReference). It re-implements the engine's generator
(SplitMix64-seeded xoshiro256**, one stream per resample) so the bootstrap draws are the same, and
computes everything else - excess returns, Hansen's kernel variance, recentring, p-values - with
NumPy from the published formulas rather than from the C++ code.

Usage: py scripts/spa_reference.py
"""

import math

import numpy as np

MASK = (1 << 64) - 1


def splitmix64(x):
    x = (x + 0x9E3779B97F4A7C15) & MASK
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & MASK
    return x ^ (x >> 31)


class PathRng:
    def __init__(self, seed, stream):
        x = splitmix64(seed) ^ splitmix64((stream + 0x632BE59BD9B4E019) & MASK)
        self.s = []
        for _ in range(4):
            x = splitmix64(x)
            self.s.append(x)

    @staticmethod
    def _rotl(x, k):
        return ((x << k) | (x >> (64 - k))) & MASK

    def next(self):
        s = self.s
        result = (self._rotl((s[1] * 5) & MASK, 7) * 9) & MASK
        t = (s[1] << 17) & MASK
        s[2] ^= s[0]
        s[3] ^= s[1]
        s[1] ^= s[2]
        s[0] ^= s[3]
        s[2] ^= t
        s[3] = self._rotl(s[3], 45)
        return result

    def uniform(self):
        return (self.next() >> 11) * 2.0**-53

    def index(self, n):
        i = int(self.uniform() * n)
        return min(i, n - 1)


def stationary_indices(rng, n, q):
    idx = np.empty(n, dtype=np.int64)
    pos = rng.index(n)
    for t in range(n):
        if t > 0:
            pos = rng.index(n) if rng.uniform() < q else (pos + 1) % n
        idx[t] = pos
    return idx


def hansen_variance(d, q):
    n = len(d)
    c = d - d.mean()
    lags = np.arange(1, n)
    kappa = (n - lags) / n * (1 - q) ** lags + lags / n * (1 - q) ** (n - lags)
    gamma = np.array([np.dot(c[i:], c[:-i]) / n for i in lags])
    return np.dot(c, c) / n + 2.0 * np.dot(kappa, gamma)


def spa(strategies, benchmark, block, resamples, seed):
    d = strategies - benchmark  # K x n
    k_count, n = d.shape
    q = 1.0 / block
    dbar = d.mean(axis=1)
    omega = np.sqrt([hansen_variance(row, q) for row in d])
    t = math.sqrt(n) * dbar / omega
    rc_stat = math.sqrt(n) * dbar.max()
    spa_stat = max(0.0, t.max())

    threshold = -math.sqrt(2.0 * math.log(math.log(n)))
    mu = {
        "lower": np.minimum(dbar, 0.0),
        "consistent": np.where(t < threshold, dbar, 0.0),
        "upper": np.zeros(k_count),
    }
    rc_draws = np.empty(resamples)
    spa_draws = {key: np.empty(resamples) for key in mu}
    for b in range(resamples):
        idx = stationary_indices(PathRng(seed, b), n, q)
        centred = d[:, idx].mean(axis=1) - dbar
        rc_draws[b] = math.sqrt(n) * centred.max()
        for key, m in mu.items():
            spa_draws[key][b] = max(0.0, (math.sqrt(n) * (centred + m) / omega).max())

    def p(draws, obs):
        return (1 + int(np.sum(draws >= obs))) / (resamples + 1)

    return {
        "mean_excess": dbar,
        "omega": omega,
        "t": t,
        "rc_stat": rc_stat,
        "rc_p": p(rc_draws, rc_stat),
        "spa_stat": spa_stat,
        **{f"spa_p_{key}": p(spa_draws[key], spa_stat) for key in mu},
    }


def fixture(n=250):
    # Same closed-form series as spa_fixture() in tests/test_significance.cpp.
    t = np.arange(n, dtype=float)
    bench = 0.0003 + 0.009 * np.sin(0.31 * t) + 0.004 * np.cos(0.017 * t * t)
    s0 = bench + 0.0002 + 0.006 * np.sin(0.9 * t + 1.0)
    s1 = bench - 0.0012 + 0.005 * np.cos(0.45 * t) + 0.003 * np.sin(0.07 * t * t)
    s2 = 0.5 * bench + 0.008 * np.sin(1.7 * t)
    return np.vstack([s0, s1, s2]), bench


if __name__ == "__main__":
    strategies, bench = fixture()
    out = spa(strategies, bench, block=4.0, resamples=500, seed=11)
    for key, value in out.items():
        if isinstance(value, np.ndarray):
            print(key, ", ".join(repr(float(v)) for v in value))
        else:
            print(key, repr(float(value)))
