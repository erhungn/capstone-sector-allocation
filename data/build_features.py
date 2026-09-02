"""Per-sector price features for LLM-Enhanced Tactical Sector Allocation.

Builds ``data/sector_features.csv``: momentum and realized volatility for each
of the 11 SPDR sector ETFs, on the same month-end index as
``macro_features.csv``, every column lagged one month.

Columns are ``{TICKER}_{feature}_lag1``, four per ticker (44 total):

    {T}_ret_1m_lag1     simple return over the prior 1 month
    {T}_ret_3m_lag1     simple return over the prior 3 months
    {T}_ret_12m_lag1    simple return over the prior 12 months
    {T}_rvol_21d_lag1   std-dev of daily simple returns over 21 trading days

Usage
-----
    python data/collect.py        # writes the daily price cache
    python data/build_features.py

Inputs are ``data/raw/sector_prices_daily.csv`` (written by collect.py) for
volatility and ``data/raw/sector_prices_raw.csv`` for the untrimmed month-end
panel. If the daily cache is absent the script downloads it rather than failing.

Look-ahead bias
---------------
Features are computed on the month-end panel and then shifted forward one
month, so the row stamped month *t* is built only from prices through the close
of *t-1* -- information an allocator genuinely holds when rebalancing into *t*.
``audit_lags`` re-derives every shift and aborts if one did not take.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

import collect as C

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

RETURN_HORIZONS = {"ret_1m": 1, "ret_3m": 3, "ret_12m": 12}

# Realized vol window, in trading days (~1 calendar month).
RVOL_WINDOW = 21

LAG = 1

# Longest lookback in months, used to size the warm-up needed before ``start``.
MAX_LOOKBACK_MONTHS = max(RETURN_HORIZONS.values())

DATA_DIR = Path(__file__).resolve().parent
RAW_DIR = DATA_DIR / "raw"

log = logging.getLogger("build_features")


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_daily(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Daily adjusted closes, from the collect.py cache when it exists."""
    path = RAW_DIR / "sector_prices_daily.csv"
    if path.exists():
        daily = pd.read_csv(path, index_col=0, parse_dates=True)
        log.info("Loaded %s (%d trading days, %s -> %s)", path, len(daily),
                 daily.index.min().date(), daily.index.max().date())
        if daily.index.min() > start:
            log.warning(
                "Cache starts %s, after the required warm-up start %s; "
                "re-run collect.py if early features come back short.",
                daily.index.min().date(), start.date(),
            )
        return daily

    log.warning("%s not found - downloading daily prices (run collect.py to cache them)", path)
    daily = C.fetch_daily_closes(C.SECTOR_ETFS, start, end)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    daily.to_csv(path, float_format="%.6f")
    return daily


# --------------------------------------------------------------------------
# Feature construction
# --------------------------------------------------------------------------

def compute_returns(monthly: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Simple returns over each configured horizon.

    Simple rather than log returns: a portfolio's return is the weighted sum of
    its holdings' simple returns, so the backtest and the features stay in the
    same units. NaNs propagate naturally -- a ticker without 12 months of
    history has no 12-month return, which is the correct answer for XLC.

    ``fill_method=None`` is explicit because pandas < 3.0 defaulted to ``'pad'``,
    which would forward-fill XLC's pre-inception NaNs into fabricated returns.
    """
    return {
        name: monthly.pct_change(n, fill_method=None)
        for name, n in RETURN_HORIZONS.items()
    }


def compute_rvol(daily: pd.DataFrame, window: int = RVOL_WINDOW) -> pd.DataFrame:
    """Month-end reading of the rolling std-dev of daily simple returns.

    ``min_periods=window`` forces a full window: a ticker with fewer than
    ``window`` observations yields NaN rather than a std-dev computed from a
    handful of days. This is what keeps XLC empty before mid-2018 instead of
    quietly producing a noisy estimate off its first few sessions.

    Values are in daily units (a typical reading is ~0.008-0.015). Multiply by
    sqrt(252) for the annualized figure.
    """
    daily_ret = daily.pct_change(fill_method=None)
    rvol = daily_ret.rolling(window=window, min_periods=window).std()
    monthly = rvol.resample("ME").last()
    monthly.index = monthly.index.normalize()
    monthly.index.name = "date"
    return monthly


def assemble(
    returns: dict[str, pd.DataFrame],
    rvol: pd.DataFrame,
    tickers: list[str],
) -> pd.DataFrame:
    """Wide ticker-major frame: all of XLK's features, then all of XLF's, ..."""
    blocks: dict[str, pd.Series] = {}
    for ticker in tickers:
        for name in RETURN_HORIZONS:
            blocks[f"{ticker}_{name}"] = returns[name][ticker]
        blocks[f"{ticker}_rvol_{RVOL_WINDOW}d"] = rvol[ticker]
    out = pd.DataFrame(blocks)
    out.index.name = "date"
    return out


def lag_features(df: pd.DataFrame, lag: int = LAG) -> pd.DataFrame:
    """Shift every column forward by ``lag`` months, tagging the column names."""
    out = df.shift(lag)
    out.columns = [f"{c}_lag{lag}" for c in df.columns]
    out.index.name = "date"
    return out


def audit_lags(unlagged: pd.DataFrame, lagged: pd.DataFrame, lag: int = LAG) -> list[str]:
    """Verify every column is a true ``lag``-month shift of its source.

    Same role as the macro-side audit in collect.py: the no-look-ahead claim is
    checked on every build rather than assumed.
    """
    failures: list[str] = []
    for col in unlagged.columns:
        name = f"{col}_lag{lag}"
        src, got = unlagged[col], lagged[name]
        expected = src.shift(lag)
        mismatch = (got != expected) & ~(got.isna() & expected.isna())
        if mismatch.any():
            failures.append(f"{name}: {int(mismatch.sum())} rows are not a {lag}-month shift")
        overlap = got.notna() & src.notna()
        if overlap.sum() > 12 and (got[overlap] == src[overlap]).all():
            failures.append(f"{name}: identical to contemporaneous {col} -- lag did not apply")
    return failures


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def build(start_str: str, end_str: str, force: bool) -> int:
    start = C.month_end(start_str)
    end = C.month_end(end_str)
    if start >= end:
        log.error("--start (%s) must be before --end (%s)", start_str, end_str)
        return 2

    out_path = DATA_DIR / "sector_features.csv"
    if out_path.exists() and not force:
        log.error("%s already exists; pass --force to overwrite", out_path)
        return 1

    # Warm-up: the 12-month return plus the 1-month lag, with a month of slack.
    warmup = MAX_LOOKBACK_MONTHS + LAG + 1
    buffer_start = (start - pd.DateOffset(months=warmup)).normalize()

    daily = load_daily(buffer_start, end)
    monthly = C.to_month_end_prices(daily)

    returns = compute_returns(monthly)
    rvol = compute_rvol(daily)

    # Both paths resample independently; align before assembling.
    rvol = rvol.reindex(monthly.index)

    unlagged = assemble(returns, rvol, C.SECTOR_ETFS)
    lagged = lag_features(unlagged)

    failures = audit_lags(unlagged, lagged)
    if failures:
        for msg in failures:
            log.error("LAG CHECK FAILED: %s", msg)
        return 1
    log.info("Lag audit passed for all %d features (lag %d).", len(lagged.columns), LAG)

    win = lagged.loc[(lagged.index >= start) & (lagged.index <= end)]

    # Index must line up with the macro side or the two cannot be joined.
    macro_path = DATA_DIR / "macro_features.csv"
    if macro_path.exists():
        macro = pd.read_csv(macro_path, index_col=0, parse_dates=True)
        if not win.index.equals(macro.index):
            log.error(
                "Index does not match macro_features.csv (features=%d rows, macro=%d rows)",
                len(win), len(macro),
            )
            return 1
        log.info("Index matches macro_features.csv (%d rows).", len(macro))
    else:
        log.warning("%s not found; skipping index cross-check", macro_path)

    # 8 dp rather than 6: realized vol runs ~0.003-0.09, where 6 dp costs real
    # significant figures on the low-vol tickers.
    win.to_csv(out_path, float_format="%.8f")
    log.info("Wrote %s (%d rows x %d columns)", out_path, *win.shape)

    # Report coverage per ticker rather than per column - 44 lines is noise.
    for ticker in C.SECTOR_ETFS:
        cols = [c for c in win.columns if c.startswith(f"{ticker}_")]
        block = win[cols]
        n_missing = int(block.isna().sum().sum())
        first = block.dropna(how="any").index.min()
        log.info(
            "    %-5s all-features-present from %s, %d missing cells",
            ticker, first.date() if pd.notna(first) else "never", n_missing,
        )

    log.info(
        "XLC NaNs before its 2018-06 inception are left as NaN, not filled. "
        "Its 12-month return is first available 2019-07."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", default=C.DEFAULT_START, help="first month, YYYY-MM")
    parser.add_argument("--end", default=C.DEFAULT_END, help="last month, YYYY-MM")
    parser.add_argument("--force", action="store_true", help="overwrite existing CSV")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    return build(args.start, args.end, args.force)


if __name__ == "__main__":
    sys.exit(main())
