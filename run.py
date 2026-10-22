"""Random matrix theory on S&P 500 correlations: find the signal, clean the noise, test it out-of-sample.

Examples
--------
python run.py                         # S&P 500 stocks, 2010-2025
python run.py --window 252            # shorter estimation window -> noisier (q closer to 1)
python run.py --raw                   # skip volatility standardisation (to compare)
python run.py --synthetic             # offline check on a factor model with known structure
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from rmt.analysis import LABELS, plot_ipr, plot_oos, plot_sectors, plot_spectrum, spectrum
from rmt.data import (download_prices, drop_near_duplicates, make_synthetic_returns, sp500_constituents,
                      to_returns)
from rmt.portfolio import backtest
from rmt.theory import correlation, eigen, noise_level
from rmt.volatility import standardise


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--window", type=int, default=504, help="estimation window in days")
    ap.add_argument("--rebalance", type=int, default=21, help="days between rebalances")
    ap.add_argument("--raw", action="store_true", help="skip volatility standardisation")
    ap.add_argument("--noise", default="market", choices=["market", "iterative"],
                    help="how to estimate the noise level (see rmt/theory.py)")
    ap.add_argument("--dup-threshold", type=float, default=0.95,
                    help="drop one of any two stocks with return correlation above this")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(exist_ok=True)

    if a.synthetic:
        returns, sectors = make_synthetic_returns(n_stocks=400, n_days=4000, n_sectors=11, realistic=True)
        label = "synthetic"
    else:
        members = sp500_constituents()
        returns = to_returns(download_prices(members["ticker"].tolist(), a.start, a.end))
        sectors = members.set_index("ticker")["sector"]
        label = "sp500"
    returns, dups = drop_near_duplicates(returns, a.dup_threshold)
    for dropped, kept, c in dups:
        print(f"Dropped {dropped}: near-duplicate of {kept} (correlation {c:.3f})")
    t, n = returns.shape
    print(f"{label}: {n} stocks x {t} days ({returns.index[0].date()} -> {returns.index[-1].date()})\n")

    # 0. How many 'signal' eigenvalues does each modelling choice find?
    print("Signal eigenvalues found by each approach:")
    for std in (False, True):
        x = standardise(returns) if std else returns
        vals, _ = eigen(correlation(x))
        counts = {m: noise_level(vals, x.shape[1], x.shape[0], m)[1] for m in ("iterative", "market")}
        print(f"  {'vol-standardised' if std else 'raw returns     '}  iterative: {counts['iterative']:4d}"
              f"   market: {counts['market']:4d}")
    print()

    # 1. Full-sample spectrum: how much of the correlation matrix is signal?
    analysed = returns if a.raw else standardise(returns)
    s = spectrum(analysed, sectors, a.noise)
    print(f"q = N/T = {s['q']:.3f}; noise band [{s['lambda_minus']:.2f}, {s['lambda_plus']:.2f}] "
          f"(sigma2 = {s['sigma2']:.2f}); finite-sample signal threshold {s['threshold']:.2f}")
    print(f"{s['k_signal']} of {n} eigenvalues are above the band, explaining "
          f"{s['variance_explained_by_signal']:.0%} of total variance")
    print(f"Largest eigenvalue {s['eigvals'][0]:.1f} (~{s['eigvals'][0] / n:.0%} of variance); "
          f"its portfolio has correlation {s['market_mode_corr']:.2f} with the equal-weight index")
    print("\nSignal eigenvectors (lift = over-representation vs sector size; sides move against each other):")
    for name, row in s["sector_lift"].iloc[:10].iterrows():
        top = row.nlargest(2)
        pos, neg = s["sides"].get(name, ([], []))
        lift_txt = ", ".join(f"{sec} {v:.1f}x" for sec, v in top.items())
        print(f"  {name:4s} {lift_txt}")
        print(f"       side A: {', '.join(pos[:6]) or '-'}")
        print(f"       side B: {', '.join(neg[:6]) or '-'}")

    plot_spectrum(s, str(out / f"{label}_spectrum.png"), f"Eigenvalue spectrum vs pure noise ({label})")
    plot_sectors(s, str(out / f"{label}_eigenvector_sectors.png"))
    plot_ipr(s, str(out / f"{label}_ipr.png"))

    # 2. Out-of-sample: does cleaning give lower-risk portfolios?
    print(f"\nRolling backtest: {a.window}-day window (q = {n / a.window:.2f}), "
          f"rebalance every {a.rebalance} days")
    res = backtest(returns, a.window, a.rebalance, standardise=not a.raw, noise=a.noise)
    daily, diag = res["daily"], res["diagnostics"]
    rows = {}
    for m in daily.columns:
        rows[LABELS[m]] = {
            "realised_vol": daily[m].std() * np.sqrt(252),
            "predicted_vol": diag[f"{m}_predicted_vol"].mean(),
            "gross_leverage": diag[f"{m}_gross_leverage"].mean(),
            "effective_n": diag[f"{m}_effective_n"].mean(),
        }
    table = pd.DataFrame(rows).T
    table["realised / predicted"] = table["realised_vol"] / table["predicted_vol"]
    print(table.round(3).to_string())
    print(f"\nRMT kept {diag['rmt_k_signal'].mean():.1f} eigenvalues on average "
          f"(range {diag['rmt_k_signal'].min()}-{diag['rmt_k_signal'].max()})")

    plot_oos(daily, diag, str(out / f"{label}_out_of_sample.png"))
    table.to_csv(out / f"{label}_portfolio_comparison.csv")
    daily.to_csv(out / f"{label}_daily_returns.csv")
    diag.to_csv(out / f"{label}_diagnostics.csv")
    s["sector_lift"].to_csv(out / f"{label}_eigenvector_sectors.csv")
    with open(out / f"{label}_summary.json", "w") as f:
        json.dump({"standardised": not a.raw, "noise_method": a.noise,
                   "dropped_duplicates": [d[0] for d in dups], "n_stocks": n, "n_days": t, "q": s["q"], "sigma2": s["sigma2"],
                   "lambda_plus": s["lambda_plus"], "k_signal": s["k_signal"],
                   "top_eigenvalues": s["eigvals"][:10].tolist(),
                   "market_mode_corr": s["market_mode_corr"],
                   "portfolios": table.to_dict(orient="index")}, f, indent=2, default=float)
    print(f"\nSaved results to {out}/")


if __name__ == "__main__":
    main()
