"""Data: S&P 500 constituents from Wikipedia, prices from Yahoo Finance, and a synthetic factor model."""
from __future__ import annotations

import time
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def sp500_constituents(cache_dir: str = "data") -> pd.DataFrame:
    """Current S&P 500 members with GICS sectors (cached). Tickers use Yahoo format (BRK-B, not BRK.B)."""
    path = Path(cache_dir) / "sp500_constituents.csv"
    if path.exists():
        return pd.read_csv(path)
    import requests
    html = requests.get(WIKI_URL, headers={"User-Agent": "Mozilla/5.0 (research project)"}, timeout=30).text
    table = pd.read_html(StringIO(html))[0]
    df = pd.DataFrame({"ticker": table["Symbol"].str.replace(".", "-", regex=False),
                       "sector": table["GICS Sector"]})
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def download_prices(tickers: list[str], start: str, end: str, cache_dir: str = "data",
                    batch_size: int = 50, pause: float = 2.0) -> pd.DataFrame:
    """Adjusted closes, downloaded in batches (Yahoo rate-limits large requests) and cached to CSV."""
    path = Path(cache_dir) / f"prices_{start}_{end}.csv"
    if path.exists():
        cached = pd.read_csv(path, index_col=0, parse_dates=True)
        if set(tickers) <= set(cached.columns):
            return cached[tickers]
    import yfinance as yf
    frames = []
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        print(f"Downloading {i + 1}-{i + len(batch)} of {len(tickers)}...")
        raw = yf.download(batch, start=start, end=end, auto_adjust=True, progress=False)
        frames.append(raw["Close"])
        time.sleep(pause)
    prices = pd.concat(frames, axis=1)
    prices.to_csv(path)
    return prices


def to_returns(prices: pd.DataFrame, max_missing: float = 0.0) -> pd.DataFrame:
    """Daily log returns, keeping only stocks with (almost) complete history over the whole sample.

    Requiring full history means every correlation is estimated on the same days, but it also
    introduces survivorship bias: only firms that existed throughout are kept (see README).
    """
    prices = prices.sort_index()
    keep = prices.columns[prices.isna().mean() <= max_missing]
    dropped = len(prices.columns) - len(keep)
    if dropped:
        print(f"Dropped {dropped} tickers without full price history")
    rets = np.log(prices[keep]).diff().iloc[1:]
    return rets.dropna()


def drop_near_duplicates(returns: pd.DataFrame, threshold: float = 0.95) -> tuple[pd.DataFrame, list]:
    """Remove one stock from every pair whose returns are almost identical.

    The S&P 500 lists several companies twice (GOOG/GOOGL, FOX/FOXA, NWS/NWSA): different share
    classes of the same firm. Their correlation is ~0.99, which makes the sample covariance matrix
    nearly singular, so a minimum-variance optimiser takes huge offsetting long/short bets on the
    pair. Keeping one share class per company removes this artefact.
    """
    corr = returns.corr().to_numpy(copy=True)
    np.fill_diagonal(corr, 0)
    cols = list(returns.columns)
    drop, log = set(), []
    for i, j in zip(*np.where(np.triu(corr) > threshold)):
        if cols[i] not in drop and cols[j] not in drop:
            drop.add(cols[j])
            log.append((cols[j], cols[i], corr[i, j]))
    return returns.drop(columns=sorted(drop)), log


def make_synthetic_returns(n_stocks: int = 200, n_days: int = 2000, n_sectors: int = 5,
                           market_beta: float = 0.5, sector_beta: float = 0.35,
                           realistic: bool = False, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """Returns from a known structure: one market factor + `n_sectors` sector factors + noise.

    RMT should find exactly `n_sectors` signal eigenvalues (the sector indicators already span the
    market direction). With `realistic=True` the data also gets two features of real markets that
    break the Marchenko-Pastur assumptions: market-wide volatility regimes (calm years, crises)
    and fat-tailed (Student-t, 4 degrees of freedom) shocks. The true structure is unchanged.
    """
    rng = np.random.default_rng(seed)

    def shocks(shape):
        if not realistic:
            return rng.normal(0, 1, shape)
        return rng.standard_t(4, shape) / np.sqrt(2)      # unit variance, fat tails

    sectors = np.arange(n_stocks) % n_sectors
    market, sector_f, noise = shocks(n_days), shocks((n_days, n_sectors)), shocks((n_days, n_stocks))
    betas = market_beta * rng.uniform(0.6, 1.4, n_stocks)
    x = market[:, None] * betas + sector_f[:, sectors] * sector_beta + noise
    if realistic:
        # Volatility clusters: a slowly varying log-volatility shared by every stock, with spikes.
        log_vol = np.zeros(n_days)
        for t in range(1, n_days):
            log_vol[t] = 0.99 * log_vol[t - 1] + rng.normal(0, 0.08)
        x = x * np.exp(log_vol)[:, None]
    vols = rng.uniform(0.01, 0.03, n_stocks)               # different volatility per stock
    x = x / x.std(axis=0) * vols
    cols = [f"S{i:03d}" for i in range(n_stocks)]
    idx = pd.bdate_range("2015-01-01", periods=n_days)
    labels = pd.Series([f"Sector {s}" for s in sectors], index=cols, name="sector")
    return pd.DataFrame(x, index=idx, columns=cols), labels
