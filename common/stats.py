"""Statistical routines shared by more than one analysis script.

Why this module exists
----------------------
``newey_west_t`` was written three times -- in ``backtest/long_short.py``,
``models/event_horizon.py`` and ``models/ic_significance.py`` -- with the same
Bartlett kernel and the same variance formula, differing only in whether the
lag was chosen automatically and whether the chosen lag was returned. Three
copies of a HAC estimator is three chances for the reported significance of
three different results to quietly disagree. There is now one.

Anything added here must be genuinely shared. Per-script reporting, plotting and
orchestration stay in their own scripts, where they are easier to read next to
the analysis they serve.
"""

from __future__ import annotations

import numpy as np

__all__ = ["newey_west_lags", "newey_west_t"]


def newey_west_lags(n: int) -> int:
    """Newey-West's rule of thumb: floor(4 * (T/100)^(2/9)), at least 1."""
    return max(1, int(np.floor(4 * (n / 100) ** (2 / 9))))


def newey_west_t(x: np.ndarray, lags: int | None = None) -> tuple[float, int]:
    """t-statistic on the mean of ``x``, robust to serial correlation.

    Uses a Bartlett (triangular) kernel, which is what makes the variance
    estimate positive semi-definite -- a plain truncated sum can return a
    negative variance and no t-statistic at all.

    Parameters
    ----------
    x
        The series whose mean is being tested. Monthly IC, monthly long-short
        returns, overlapping event-window returns: all are autocorrelated, which
        is exactly why the naive t-statistic overstates significance.
    lags
        Truncation lag. ``None`` applies :func:`newey_west_lags`. Pass a lag
        explicitly when the autocorrelation is structural rather than estimated
        -- overlapping *h*-month windows induce exactly *h-1* lags of overlap,
        and that is known a priori, not something to infer from the sample size.

    Returns
    -------
    ``(t_statistic, lags_used)``. The t-statistic is ``nan`` when the sample is
    too short (fewer than 3 observations) or the HAC variance is non-positive;
    ``lags_used`` is always reported so the choice is visible in the output
    rather than buried in this function.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if lags is None:
        lags = newey_west_lags(n)
    if n < 3:
        return np.nan, lags

    mu = x.mean()
    e = x - mu
    var = float(e @ e) / n
    for j in range(1, min(lags, n - 1) + 1):
        gamma = float(e[j:] @ e[:-j]) / n
        var += 2.0 * (1.0 - j / (lags + 1.0)) * gamma
    if var <= 0:
        return np.nan, lags
    return float(mu / np.sqrt(var / n)), lags
