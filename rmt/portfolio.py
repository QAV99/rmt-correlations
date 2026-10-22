"""Out-of-sample test: do cleaned correlation matrices produce genuinely lower-risk portfolios?

Global minimum-variance portfolios are the right test bed: their weights depend *only* on the
covariance matrix (no return forecasts), so any difference in realised risk comes from how
well each method estimated correlations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from .theory import clip_correlation, correlation
from .volatility import ewma_vol


def correlation_estimate(window: np.ndarray, method: str, noise: str = "market") -> tuple[np.ndarray, dict]:
    """Correlation-structure estimate from a (T x N) window of (standardised) returns."""
    t, n = window.shape
    info = {}
    if method == "ledoit_wolf":
        z = (window - window.mean(axis=0)) / window.std(axis=0, ddof=1)
        return LedoitWolf().fit(z).covariance_, info
    corr = correlation(window)
    if method == "rmt":
        corr, info["k_signal"] = clip_correlation(corr, n_obs=t, method=noise)
    elif method != "sample":
        raise ValueError(method)
    return corr, info


def min_variance_weights(cov: np.ndarray) -> np.ndarray:
    """w = C^-1 1 / (1' C^-1 1): fully invested, shorting allowed."""
    x = np.linalg.solve(cov, np.ones(len(cov)))
    return x / x.sum()


def backtest(returns: pd.DataFrame, window: int = 504, rebalance: int = 21,
             methods: tuple[str, ...] = ("sample", "rmt", "ledoit_wolf", "equal"),
             standardise: bool = True, halflife: float = 30, noise: str = "market") -> dict:
    """Rolling estimation, then hold the weights over the next `rebalance` days.

    Covariance = D C D, where C (the correlation structure) is estimated by each method and D holds
    the volatilities. With `standardise=True`, C comes from volatility-standardised returns and D
    is each stock's EWMA volatility forecast at the rebalance date; otherwise both come from the
    raw window. Weights chosen at date i use returns strictly before i, so every evaluation is
    out-of-sample.
    """
    simple = np.expm1(returns.values)            # portfolio returns aggregate simple returns
    n = returns.shape[1]
    if standardise:
        vol = ewma_vol(returns, halflife)
        z = (returns / vol).values
        first = int(vol.notna().all(axis=1).values.argmax()) + window
    else:
        z = returns.values
        first = window
    daily = {m: [] for m in methods}
    diag = []
    for i in range(first, len(returns), rebalance):
        est, hold = z[i - window:i], simple[i:i + rebalance]
        d = vol.values[i] if standardise else returns.values[i - window:i].std(axis=0, ddof=1)
        row = {"date": returns.index[i]}
        for m in methods:
            if m == "equal":
                w = np.full(n, 1 / n)
                cov = np.cov(returns.values[i - window:i], rowvar=False)
            else:
                c, info = correlation_estimate(est, m, noise)
                cov = c * np.outer(d, d)
                w = min_variance_weights(cov)
                row.update({f"{m}_{k}": v for k, v in info.items()})
            daily[m].append(pd.Series(hold @ w, index=returns.index[i:i + rebalance]))
            row[f"{m}_predicted_vol"] = np.sqrt(w @ cov @ w * 252)
            row[f"{m}_realised_vol"] = (hold @ w).std(ddof=1) * np.sqrt(252) if len(hold) > 1 else np.nan
            row[f"{m}_gross_leverage"] = np.abs(w).sum()
            row[f"{m}_effective_n"] = 1 / (w ** 2).sum()
        diag.append(row)
    return {"daily": pd.DataFrame({m: pd.concat(s) for m, s in daily.items()}),
            "diagnostics": pd.DataFrame(diag).set_index("date")}
