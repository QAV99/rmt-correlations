"""Full-sample spectrum analysis and the charts used in the README."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .theory import (correlation, eigen, inverse_participation_ratio, mp_bounds, mp_pdf, noise_level,
                     signal_threshold)

LABELS = {"sample": "Sample", "rmt": "RMT clipped", "ledoit_wolf": "Ledoit-Wolf", "equal": "Equal weight"}


def spectrum(returns: pd.DataFrame, sectors: pd.Series, noise: str = "market") -> dict:
    t, n = returns.shape
    q = n / t
    vals, vecs = eigen(correlation(returns))
    sigma2, k = noise_level(vals, n, t, noise)
    lo, hi = mp_bounds(q, sigma2)
    threshold = signal_threshold(n, t, sigma2)

    # Which sectors does each signal eigenvector load on? Raw shares of squared loadings favour big
    # sectors (Financials is ~15% of the index, so it tops many lists by sheer size). The *lift*
    # divides by each sector's share of stocks: 1 = typical, 3 = three times over-represented.
    sec = sectors.reindex(returns.columns).fillna("Unknown")
    kk = max(k, 1)
    loadings = pd.DataFrame(vecs[:, :kk] ** 2, index=returns.columns,
                            columns=[f"#{i + 1}" for i in range(kk)])
    sector_share = loadings.groupby(sec).sum().T
    sector_lift = sector_share / sec.value_counts(normalize=True)

    # Eigenvector signs are arbitrary, but *relative* signs matter: stocks on opposite sides move
    # against each other along that factor (e.g. tech up while banks fall).
    sides = {}
    for i in range(min(kk, 10)):
        v = pd.Series(vecs[:, i], index=returns.columns)
        top = v.abs().nlargest(10).index
        sides[f"#{i + 1}"] = (v[top][v[top] > 0].index.tolist(), v[top][v[top] < 0].index.tolist())

    # The top eigenvector should look like "the market": compare with the equal-weighted index.
    v1 = vecs[:, 0] * np.sign(vecs[:, 0].sum())
    z = (returns - returns.mean()) / returns.std()
    market_mode_corr = np.corrcoef(z.values @ v1, returns.mean(axis=1))[0, 1]

    return {"eigvals": vals, "eigvecs": vecs, "q": q, "sigma2": sigma2, "k_signal": k,
            "lambda_minus": lo, "lambda_plus": hi, "threshold": threshold, "sector_share": sector_share,
            "sector_lift": sector_lift, "sides": sides,
            "market_mode_corr": market_mode_corr,
            "variance_explained_by_signal": vals[:k].sum() / n,
            "n": n, "t": t}


def plot_spectrum(s: dict, path: str, title: str) -> None:
    vals, hi, thr = s["eigvals"], s["lambda_plus"], s["threshold"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [3, 2]})
    xmax = 1.8 * thr
    bulk = vals[vals < xmax]
    a1.hist(bulk, bins=min(60, max(20, len(vals) // 5)), density=True, color="#9aa5b1", alpha=0.8, label="Empirical eigenvalues")
    x = np.linspace(1e-3, xmax, 600)
    # Density of the noise part only, scaled by the fraction of eigenvalues it describes.
    noise_frac = 1 - s["k_signal"] / s["n"]
    a1.plot(x, mp_pdf(x, s["q"], s["sigma2"]) * noise_frac * len(vals) / len(bulk),
            color="#1f5fa8", lw=2, label=f"Marchenko-Pastur (q={s['q']:.2f}, $\\sigma^2$={s['sigma2']:.2f})")
    a1.axvline(thr, color="#c0392b", ls="--", lw=1, label=f"Signal threshold = {thr:.2f}")
    off = (vals >= xmax).sum()
    a1.text(0.98, 0.6, f"{off} eigenvalues beyond\nthe chart, largest = {vals[0]:.1f}",
            transform=a1.transAxes, ha="right", fontsize=9)
    a1.set_xlabel("Eigenvalue")
    a1.set_ylabel("Density")
    a1.set_title(title)
    a1.legend(fontsize=8)

    ranks = np.arange(1, len(vals) + 1)
    signal = vals > thr
    a2.scatter(ranks[~signal], vals[~signal], s=6, color="#9aa5b1", label="Noise")
    a2.scatter(ranks[signal], vals[signal], s=18, color="#1f5fa8", label="Signal")
    a2.axhline(thr, color="#c0392b", ls="--", lw=1)
    a2.set_xscale("log")
    a2.set_yscale("log")
    a2.set_xlabel("Rank")
    a2.set_ylabel("Eigenvalue")
    a2.set_title(f"{s['k_signal']} of {s['n']} eigenvalues above the noise band")
    a2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_sectors(s: dict, path: str, max_vecs: int = 10) -> None:
    lift = s["sector_lift"].iloc[:max_vecs]
    fig, ax = plt.subplots(figsize=(10, 0.5 * len(lift) + 2))
    im = ax.imshow(lift.values, cmap="Blues", vmin=0, vmax=5, aspect="auto")
    ax.set_xticks(range(lift.shape[1]), lift.columns, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(lift)), [f"Eigenvector {c}" for c in lift.index], fontsize=9)
    for i in range(lift.shape[0]):
        for j in range(lift.shape[1]):
            v = lift.values[i, j]
            if v >= 1.5:
                ax.text(j, i, f"{v:.1f}x", ha="center", va="center", fontsize=8,
                        color="white" if v > 3 else "black")
    fig.colorbar(im, ax=ax, label="Over-representation vs sector size (1 = typical)")
    ax.set_title("Which sectors drive each signal eigenvector")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_ipr(s: dict, path: str) -> None:
    ipr = inverse_participation_ratio(s["eigvecs"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sig = s["eigvals"] > s["threshold"]
    ax.scatter(s["eigvals"][~sig], ipr[~sig], s=8, color="#9aa5b1", label="Noise eigenvectors")
    ax.scatter(s["eigvals"][sig], ipr[sig], s=20, color="#1f5fa8", label="Signal eigenvectors")
    ax.axhline(3 / s["n"], color="#c0392b", ls="--", lw=1, label="Random vector (3/N)")
    ax.axhline(1 / s["n"], color="#2c3e50", ls=":", lw=1, label="Fully spread out (1/N)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Eigenvalue")
    ax.set_ylabel("Inverse participation ratio")
    ax.set_title("Localisation: how many stocks each eigenvector involves")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_oos(daily: pd.DataFrame, diag: pd.DataFrame, path: str) -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    methods = list(daily.columns)
    realised = daily.std() * np.sqrt(252)
    predicted = pd.Series({m: diag[f"{m}_predicted_vol"].mean() for m in methods})
    x = np.arange(len(methods))
    a1.bar(x - 0.2, predicted, 0.4, color="#9aa5b1", label="Predicted (in-sample)")
    a1.bar(x + 0.2, realised, 0.4, color="#1f5fa8", label="Realised (out-of-sample)")
    a1.set_xticks(x, [LABELS.get(m, m) for m in methods])
    a1.set_ylabel("Annualised volatility")
    a1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    a1.set_title("Minimum-variance portfolios: promised vs delivered risk")
    a1.legend(fontsize=8)
    for m, c in zip(methods, ["#c0392b", "#1f5fa8", "#27ae60", "#7f8c8d"]):
        (daily[m].rolling(252).std() * np.sqrt(252)).plot(ax=a2, label=LABELS.get(m, m), color=c, lw=1.2)
    a2.set_ylabel("Rolling 1-year realised volatility")
    a2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    a2.set_title("Out-of-sample risk over time")
    a2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
