"""Random matrix theory: the Marchenko-Pastur law and correlation-matrix cleaning.

If N assets have T observations of *pure noise*, the eigenvalues of their sample correlation
matrix do not all equal 1. As N, T -> infinity with q = N/T fixed, they spread over

    lambda_minus = sigma2 * (1 - sqrt(q))^2   to   lambda_plus = sigma2 * (1 + sqrt(q))^2

with the Marchenko-Pastur density. Eigenvalues above lambda_plus are evidence of real structure.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def mp_bounds(q: float, sigma2: float = 1.0) -> tuple[float, float]:
    return sigma2 * (1 - np.sqrt(q)) ** 2, sigma2 * (1 + np.sqrt(q)) ** 2


def mp_pdf(x: np.ndarray, q: float, sigma2: float = 1.0) -> np.ndarray:
    """Marchenko-Pastur density for q = N/T <= 1."""
    lo, hi = mp_bounds(q, sigma2)
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    inside = (x > lo) & (x < hi)
    out[inside] = np.sqrt((hi - x[inside]) * (x[inside] - lo)) / (2 * np.pi * q * sigma2 * x[inside])
    return out


def correlation(returns: pd.DataFrame | np.ndarray) -> np.ndarray:
    x = np.asarray(returns, dtype=float)
    x = (x - x.mean(axis=0)) / x.std(axis=0, ddof=1)
    return x.T @ x / (len(x) - 1)


def eigen(corr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigenvalues in descending order with matching eigenvectors (columns)."""
    vals, vecs = np.linalg.eigh(corr)
    order = np.argsort(vals)[::-1]
    return vals[order], vecs[:, order]


def signal_threshold(n: int, t: int, sigma2: float = 1.0, z: float = 2.02) -> float:
    """Largest eigenvalue pure noise plausibly produces at finite N, T.

    The Marchenko-Pastur edge lambda_plus is an N, T -> infinity limit. In a finite sample the
    top noise eigenvalue fluctuates around it with the Tracy-Widom distribution (Johnstone, 2001).
    We use its centring/scaling and the ~99th percentile (z = 2.02) so that pure noise is almost
    never mistaken for signal.
    """
    a = np.sqrt(t - 1) + np.sqrt(n)
    mu = a * a / t
    sd = a / t * (1 / np.sqrt(t - 1) + 1 / np.sqrt(n)) ** (1 / 3)
    return sigma2 * (mu + z * sd)


def noise_level(eigvals: np.ndarray, n: int, t: int, method: str = "market",
                max_iter: int = 50) -> tuple[float, int]:
    """Estimate the noise variance sigma2 and the number k of signal eigenvalues.

    The eigenvalues of a correlation matrix sum to N; any signal soaks up part of that total, so
    the noise bulk has variance sigma2 < 1. Two estimates:

    * "market" (default, Laloux et al. 1999): only remove the market mode, sigma2 = 1 - lambda_1/N.
      The market eigenvalue is by far the largest, so this captures most of the correction.
    * "iterative": sigma2 = mean of the non-signal eigenvalues, re-estimated until stable. Exact
      when the bulk really is Marchenko-Pastur, but on real data (fat tails, changing volatility)
      the true bulk is wider than theory, so each round finds 'new signal', lowers sigma2 again,
      and the count runs away. Kept for comparison.
    """
    if method == "market":
        sigma2 = 1 - eigvals[0] / n
        return sigma2, int((eigvals > signal_threshold(n, t, sigma2)).sum())
    if method != "iterative":
        raise ValueError(method)
    k = 0
    for _ in range(max_iter):
        sigma2 = eigvals[k:].mean()
        k_new = int((eigvals > signal_threshold(n, t, sigma2)).sum())
        if k_new == k:
            break
        k = k_new
    return sigma2, k


def clip_correlation(corr: np.ndarray, n_obs: int, method: str = "market") -> tuple[np.ndarray, int]:
    """Eigenvalue clipping: keep signal eigenvalues, replace the noise ones by their average.

    Replacing (rather than zeroing) keeps the trace equal to N and the matrix invertible.
    Returns the cleaned correlation matrix and the number of eigenvalues kept.
    """
    vals, vecs = eigen(corr)
    _, k = noise_level(vals, len(vals), n_obs, method)
    cleaned_vals = vals.copy()
    if k < len(vals):
        cleaned_vals[k:] = vals[k:].mean()
    c = (vecs * cleaned_vals) @ vecs.T
    d = np.sqrt(np.diag(c))
    c = c / np.outer(d, d)                     # restore an exact unit diagonal
    return (c + c.T) / 2, k


def inverse_participation_ratio(eigvecs: np.ndarray) -> np.ndarray:
    """IPR_k = sum_i v_ik^4, a measure of localisation borrowed from condensed-matter physics.

    An eigenvector spread evenly over N stocks has IPR = 1/N; one concentrated on a few stocks
    (a single sector, a merger pair) has a much larger IPR. Random eigenvectors give about 3/N.
    """
    return (eigvecs ** 4).sum(axis=0)
