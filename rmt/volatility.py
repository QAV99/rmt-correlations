"""Volatility standardisation.

Marchenko-Pastur assumes every return is drawn with the same variance over time. Real returns are
not: volatility in March 2020 was several times that of 2017. Because this volatility moves all
stocks together, it creates spurious common structure and widens the noise band. Dividing each
return by a forecast of that stock's own volatility removes most of this effect.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ewma_vol(returns: pd.DataFrame, halflife: float = 30, min_periods: int = 60) -> pd.DataFrame:
    """Exponentially weighted volatility forecast. The value on day t uses returns up to t-1 only."""
    var = (returns ** 2).ewm(halflife=halflife, min_periods=min_periods).mean()
    return np.sqrt(var).shift(1)


def standardise(returns: pd.DataFrame, halflife: float = 30) -> pd.DataFrame:
    """Returns divided by their own volatility forecast (first rows, without a forecast, dropped)."""
    return (returns / ewma_vol(returns, halflife)).dropna()
