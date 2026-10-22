"""Correctness tests: the theory matches simulation, cleaning is well-behaved, the backtest is causal."""
import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad

from rmt.data import drop_near_duplicates, make_synthetic_returns
from rmt.portfolio import backtest, min_variance_weights
from rmt.theory import clip_correlation, correlation, eigen, mp_bounds, mp_pdf, noise_level
from rmt.volatility import ewma_vol, standardise


@pytest.mark.parametrize("q", [0.1, 0.4, 0.8])
def test_mp_density_integrates_to_one(q):
    lo, hi = mp_bounds(q)
    assert quad(lambda x: mp_pdf(np.array([x]), q)[0], lo, hi, limit=200)[0] == pytest.approx(1, abs=1e-3)


@pytest.mark.parametrize("method", ["market", "iterative"])
@pytest.mark.parametrize("seed", range(5))
def test_pure_noise_has_no_signal(seed, method):
    """Independent stocks: every eigenvalue should be inside the noise band, none flagged as signal."""
    t, n = 1000, 300
    x = np.random.default_rng(seed).normal(size=(t, n))
    vals, _ = eigen(correlation(x))
    sigma2, k = noise_level(vals, n, t, method)
    lo, hi = mp_bounds(n / t)
    assert k == 0
    assert sigma2 == pytest.approx(1, abs=0.01)
    assert lo * 0.9 < vals.min() and vals.max() < hi * 1.05


@pytest.mark.parametrize("seed", [0, 1])
def test_factor_model_structure_is_recovered(seed):
    """Market + 5 sectors. The 5 sector indicators already span the all-ones (market) direction,
    so the true signal subspace has dimension 5, and RMT should find exactly that."""
    r, _ = make_synthetic_returns(n_stocks=200, n_days=2000, n_sectors=5, seed=seed)
    vals, _ = eigen(correlation(r))
    for method in ("market", "iterative"):
        assert noise_level(vals, *r.shape[::-1], method)[1] == 5


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_standardisation_fixes_realistic_data(seed):
    """With volatility regimes and fat tails (like real markets), the naive approach invents
    dozens of extra 'factors'; volatility standardisation + the market-mode noise estimate
    brings the count back near the true 11."""
    r, _ = make_synthetic_returns(n_stocks=400, n_days=4000, n_sectors=11, realistic=True, seed=seed)
    def count(x, method):
        vals, _ = eigen(correlation(x))
        return noise_level(vals, x.shape[1], x.shape[0], method)[1]
    naive, robust = count(r, "iterative"), count(standardise(r), "market")
    assert naive > 50
    assert 11 <= robust <= 20


def test_near_duplicates_are_removed():
    r, _ = make_synthetic_returns(n_stocks=30, n_days=500, seed=5)
    r["S000_B"] = r["S000"] + np.random.default_rng(0).normal(0, 1e-4, len(r))   # second share class
    cleaned, log = drop_near_duplicates(r)
    assert "S000_B" not in cleaned and "S000" in cleaned
    assert len(log) == 1 and cleaned.shape[1] == 30


def test_volatility_forecast_is_causal():
    r, _ = make_synthetic_returns(n_stocks=5, n_days=300, seed=6)
    v1 = ewma_vol(r)
    r2 = r.copy()
    r2.iloc[200:] *= 10
    v2 = ewma_vol(r2)
    pd.testing.assert_frame_equal(v1.iloc[:201], v2.iloc[:201])   # day 200 forecast uses data to day 199


def test_clipped_matrix_is_a_valid_correlation_matrix():
    r, _ = make_synthetic_returns(n_stocks=150, n_days=300, seed=2)
    c, k = clip_correlation(correlation(r), n_obs=300)
    assert k >= 1
    np.testing.assert_allclose(np.diag(c), 1, atol=1e-12)
    np.testing.assert_allclose(c, c.T, atol=1e-12)
    assert np.linalg.eigvalsh(c).min() > 0               # positive definite -> invertible


def test_min_variance_weights_are_optimal():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(50, 50))
    cov = a @ a.T / 50 + np.eye(50) * 0.1
    w = min_variance_weights(cov)
    assert w.sum() == pytest.approx(1)
    for _ in range(20):                                    # any fully-invested perturbation is worse
        d = rng.normal(size=50)
        d -= d.mean()
        assert (w + 0.01 * d) @ cov @ (w + 0.01 * d) >= w @ cov @ w


def test_backtest_is_causal():
    r, _ = make_synthetic_returns(n_stocks=60, n_days=700, seed=4)
    cutoff = 500
    scrambled = r.copy()
    scrambled.iloc[cutoff + 1:] = np.random.default_rng(9).normal(0, 0.05, scrambled.iloc[cutoff + 1:].shape)
    a = backtest(r, window=250, rebalance=21)["daily"]
    b = backtest(scrambled, window=250, rebalance=21)["daily"]
    pd.testing.assert_frame_equal(a.loc[:r.index[cutoff]], b.loc[:r.index[cutoff]])
