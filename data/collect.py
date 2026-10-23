"""Data collection for LLM-Enhanced Tactical Sector Allocation.

Downloads two datasets:

1. ``data/sector_prices.csv``  -- month-end adjusted close for the 11 SPDR
   sector ETFs.
2. ``data/macro_features.csv`` -- month-end macro indicators from FRED,
   transformed and **lagged** so that the row stamped month *t* contains only
   information already observable at *t*. Real-time market series carry a
   1-month lag, government/survey series a 2-month lag; the applied lag is
   encoded in each column name.

Usage
-----
    export FRED_API_KEY=...
    python data/collect.py
    python data/collect.py --start 2016-01 --end 2025-12 --force

Look-ahead bias
---------------
Prices are stored unlagged (they are the target side of the problem). Every
macro feature is shifted forward by its configured lag, so a model predicting
month-*t* sector returns from ``macro_features.csv`` row *t* never sees
contemporaneous macro data. ``audit_lags`` re-derives every shift on each run
and aborts if one did not take. See the caveat on release lags in README.md.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import NamedTuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))

import universe as U  # noqa: E402

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

# The full download set: every ticker any universe can contain. Declared in
# common/universe.py so collect.py, build_dataset.py and the sentiment pipeline
# cannot drift apart. Downloading the superset once means the 11-asset results
# stay reproducible from the same price files the 14-asset extension uses --
# the extra columns are simply not selected by the 11-asset build.
SECTOR_ETFS = U.ALL_TICKERS

class Spec(NamedTuple):
    """How one FRED series becomes one model feature.

    column    output column stem; the lag is appended as ``_lag{n}``
    transform "level"   : month-end level, as reported
              "yoy_pct" : 12-month percent change
              "mom_pct" : 1-month percent change
    lag       months to shift forward. 1 for series observable in real time
              (daily market data); 2 for government/survey series that are not
              published until part-way through the following month.
    """

    column: str
    transform: str
    lag: int


# Daily series collapse to their month-end reading; monthly series pass through.
FRED_SERIES = {
    # --- real-time market data -> 1-month lag ----------------------------
    "VIXCLS":     Spec("VIX", "level", 1),           # daily
    "T10Y2Y":     Spec("T10Y2Y", "level", 1),        # daily, 10y-2y slope
    "BAA10Y":     Spec("BAA10Y", "level", 1),        # daily, BBB credit spread
    "T10YIE":     Spec("T10YIE", "level", 1),        # daily, 10y breakeven inflation
    "DTWEXBGS":   Spec("DTWEXBGS", "level", 1),      # daily, USD trade-weighted
    "DCOILWTICO": Spec("WTI", "level", 1),           # daily, WTI crude
    "FEDFUNDS":   Spec("FEDFUNDS", "level", 1),      # monthly, policy rate
    # --- government / survey releases -> 2-month lag ---------------------
    # None of these are published until part-way through the following month,
    # so a 1-month lag would still leak data the allocator could not have had.
    "UNRATE":     Spec("UNRATE", "level", 2),
    "CPIAUCSL":   Spec("CPI_YOY", "yoy_pct", 2),
    "INDPRO":     Spec("INDPRO_MOM", "mom_pct", 2),
    "UMCSENT":    Spec("UMCSENT", "level", 2),       # Michigan consumer sentiment
    "M2SL":       Spec("M2_YOY", "yoy_pct", 2),      # M2 money supply
    "PERMIT":     Spec("PERMIT", "level", 2),        # building permits, SAAR
    "MANEMP":     Spec("MANEMP_MOM", "mom_pct", 2),  # mfg employment, ISM PMI proxy
}

DEFAULT_START = "2016-01"
DEFAULT_END = "2025-12"

# History pulled before ``start`` so YoY transforms (12 months) and the largest
# lag are both fully populated on the first row of the saved window. Derived
# from the config so adding a longer lag cannot silently truncate the warm-up.
WARMUP_MONTHS = 12 + max(s.lag for s in FRED_SERIES.values()) + 1

DATA_DIR = Path(__file__).resolve().parent
RAW_DIR = DATA_DIR / "raw"
REPO_ROOT = DATA_DIR.parent

log = logging.getLogger("collect")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def month_end(period: str) -> pd.Timestamp:
    """'2016-01' -> Timestamp('2016-01-31')."""
    return pd.Period(period, freq="M").to_timestamp(how="end").normalize()


def to_month_end_index(obj: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Canonicalise any datetime index to month-end timestamps.

    FRED stamps monthly series at the *first* of the month while price data
    lands on the last trading day; mapping both through a monthly Period makes
    them join correctly.
    """
    out = obj.copy()
    out.index = pd.DatetimeIndex(out.index).to_period("M").to_timestamp(how="end").normalize()
    return out


def coverage_report(df: pd.DataFrame, label: str) -> None:
    """Log first/last valid observation and missing count for each column."""
    log.info("%s: %d rows, %s -> %s", label, len(df),
             df.index.min().date(), df.index.max().date())
    for col in df.columns:
        s = df[col]
        n_missing = int(s.isna().sum())
        if s.notna().any():
            log.info(
                "    %-12s first=%s last=%s missing=%d",
                col, s.first_valid_index().date(), s.last_valid_index().date(), n_missing,
            )
        else:
            log.warning("    %-12s NO DATA", col)


# --------------------------------------------------------------------------
# 1. Sector ETF prices
# --------------------------------------------------------------------------

def fetch_daily_closes(
    tickers: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    retries: int = 3,
) -> pd.DataFrame:
    """Daily split/dividend-adjusted close prices.

    Daily bars are downloaded and resampled to month-end rather than requesting
    ``interval='1mo'`` directly: the monthly endpoint returns a partial bar for
    the current month and occasionally misaligns the period stamp. The daily
    frame is kept because realized volatility is built from it
    (see build_features.py).
    """
    import yfinance as yf

    # Pad the end date so the final month's last trading day is included.
    dl_end = (end + pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    dl_start = start.strftime("%Y-%m-%d")

    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            log.info("Downloading %d tickers from yfinance (%s -> %s), attempt %d/%d",
                     len(tickers), dl_start, dl_end, attempt, retries)
            raw = yf.download(
                tickers,
                start=dl_start,
                end=dl_end,
                interval="1d",
                auto_adjust=True,   # 'Close' is then the adjusted close
                progress=False,
                group_by="column",
                threads=True,
            )
            if raw is None or raw.empty:
                raise RuntimeError("yfinance returned an empty frame")
            break
        except Exception as err:  # noqa: BLE001 - network layer, retry on anything
            last_err = err
            log.warning("Download failed: %s", err)
            if attempt < retries:
                time.sleep(2 ** attempt)
    else:
        raise RuntimeError(f"yfinance download failed after {retries} attempts") from last_err

    # Normalise single- vs multi-ticker column shapes to a plain ticker frame.
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw.xs("Close", axis=1, level=0)
    elif len(tickers) == 1:
        close = raw[["Close"]].rename(columns={"Close": tickers[0]})
    else:
        close = raw["Close"].to_frame() if isinstance(raw["Close"], pd.Series) else raw["Close"]

    missing = [t for t in tickers if t not in close.columns]
    if missing:
        raise RuntimeError(f"yfinance returned no data for: {missing}")

    close = close[tickers].sort_index()
    close.index = pd.DatetimeIndex(close.index).normalize()
    close.index.name = "date"
    return close


def to_month_end_prices(daily: pd.DataFrame) -> pd.DataFrame:
    """Last observed close in each calendar month, stamped at month end."""
    monthly = daily.resample("ME").last()
    monthly.index = monthly.index.normalize()
    monthly.index.name = "date"
    return monthly


# --------------------------------------------------------------------------
# 2. FRED macro indicators
# --------------------------------------------------------------------------

def fetch_macro_raw(
    start: pd.Timestamp,
    end: pd.Timestamp,
    api_key: str,
) -> pd.DataFrame:
    """Untransformed FRED series on a month-end index."""
    from fredapi import Fred

    fred = Fred(api_key=api_key)
    # Daily series need extra room so the first month-end has observations.
    obs_start = (start - pd.DateOffset(months=1)).strftime("%Y-%m-%d")
    obs_end = end.strftime("%Y-%m-%d")

    frames: dict[str, pd.Series] = {}
    for series_id in FRED_SERIES:
        log.info("Fetching FRED series %s", series_id)
        s = fred.get_series(series_id, observation_start=obs_start, observation_end=obs_end)
        s = pd.Series(s).dropna()
        if s.empty:
            raise RuntimeError(f"FRED returned no observations for {series_id}")
        s.index = pd.DatetimeIndex(s.index)
        # Daily series (VIXCLS, T10Y2Y) collapse to their month-end reading;
        # monthly series pass through unchanged.
        monthly = s.resample("ME").last()
        frames[series_id] = to_month_end_index(monthly)

    df = pd.DataFrame(frames).sort_index()
    df.index.name = "date"
    return df


def transform_macro(raw: pd.DataFrame) -> pd.DataFrame:
    """Apply the per-series transform declared in FRED_SERIES."""
    out = pd.DataFrame(index=raw.index)
    for series_id, spec in FRED_SERIES.items():
        s = raw[series_id]
        if spec.transform == "level":
            out[spec.column] = s
        # fill_method=None is explicit: pandas < 3.0 defaulted to 'pad', which
        # would paper over a missing release instead of propagating the gap.
        elif spec.transform == "yoy_pct":
            out[spec.column] = s.pct_change(12, fill_method=None) * 100.0
        elif spec.transform == "mom_pct":
            out[spec.column] = s.pct_change(1, fill_method=None) * 100.0
        else:  # pragma: no cover - guarded by config
            raise ValueError(f"unknown transform {spec.transform!r} for {series_id}")
    out.index.name = "date"
    return out


def fill_release_gaps(df: pd.DataFrame, limit: int = 1) -> pd.DataFrame:
    """Carry the last published value forward across a missed release.

    The October 2025 US government shutdown cancelled the BLS household survey
    and the October CPI release outright, so UNRATE and CPIAUCSL have no October
    2025 observation and never will -- it is a permanent hole in the source, not
    a late print. Forward-filling matches what an allocator actually faced that
    month: no fresh data, so act on the September number.

    Runs on the *transformed* frame, before lagging, so each feature stays a
    clean shift of its own source and ``audit_lags`` still holds.

    ``limit`` caps the carry at one month so a longer outage cannot be absorbed
    silently; every filled cell is logged.
    """
    filled = df.ffill(limit=limit)
    newly = filled.notna() & df.isna()
    for col in df.columns:
        for ts in df.index[newly[col]]:
            log.warning(
                "Gap fill: %s at %s was unpublished; carried forward %.4f",
                col, ts.date(), filled.at[ts, col],
            )
    return filled


def lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Shift each column forward by its own configured lag.

    Row *t* of the result holds the value observed at *t - lag*, so a model
    trained on these features cannot see the present. The lag is encoded in the
    column name (``VIX_lag1``, ``UMCSENT_lag2``) so the horizon travels with the
    data instead of living only in this file.
    """
    out = pd.DataFrame(index=df.index)
    for spec in FRED_SERIES.values():
        out[f"{spec.column}_lag{spec.lag}"] = df[spec.column].shift(spec.lag)
    out.index.name = "date"
    return out


def audit_lags(transformed: pd.DataFrame, lagged: pd.DataFrame) -> list[str]:
    """Verify every column really is shifted by its configured lag.

    Returns a list of failure messages; empty means the whole frame is clean.
    This is the check that the no-look-ahead claim rests on, so it runs on every
    collection rather than living in a test file.
    """
    failures: list[str] = []
    for spec in FRED_SERIES.values():
        name = f"{spec.column}_lag{spec.lag}"
        src, got = transformed[spec.column], lagged[name]

        # Positional comparison: got[i] must equal src[i - lag].
        expected = src.shift(spec.lag)
        mismatch = (got != expected) & ~(got.isna() & expected.isna())
        if mismatch.any():
            failures.append(f"{name}: {int(mismatch.sum())} rows do not match a {spec.lag}-month shift")

        # And it must NOT equal the contemporaneous value, which is the actual
        # failure mode we care about (a lag silently becoming a no-op).
        overlap = got.notna() & src.notna()
        if overlap.sum() > 12 and (got[overlap] == src[overlap]).all():
            failures.append(f"{name}: identical to the contemporaneous {spec.column} -- lag did not apply")
    return failures


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def collect(start_str: str, end_str: str, force: bool) -> int:
    start = month_end(start_str)
    end = month_end(end_str)
    if start >= end:
        log.error("--start (%s) must be before --end (%s)", start_str, end_str)
        return 2

    buffer_start = (start - pd.DateOffset(months=WARMUP_MONTHS)).normalize()

    prices_path = DATA_DIR / "sector_prices.csv"
    macro_path = DATA_DIR / "macro_features.csv"
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    api_key = os.environ.get("FRED_API_KEY", "").strip()
    if not api_key:
        log.error(
            "FRED_API_KEY is not set. Get a free key at "
            "https://fredaccount.stlouisfed.org/apikeys and run: export FRED_API_KEY=..."
        )
        return 1

    for path in (prices_path, macro_path):
        if path.exists() and not force:
            log.error("%s already exists; pass --force to overwrite", path)
            return 1

    # --- prices -----------------------------------------------------------
    daily = fetch_daily_closes(SECTOR_ETFS, buffer_start, end)
    # Persisted for build_features.py: realized vol needs daily returns, and
    # re-downloading them there would be a second trip for data already in hand.
    daily.to_csv(RAW_DIR / "sector_prices_daily.csv", float_format="%.6f")
    log.info("Wrote %s (%d trading days)", RAW_DIR / "sector_prices_daily.csv", len(daily))

    prices = to_month_end_prices(daily)
    prices.to_csv(RAW_DIR / "sector_prices_raw.csv", float_format="%.6f")

    prices_win = prices.loc[(prices.index >= start) & (prices.index <= end)]
    prices_win.to_csv(prices_path, float_format="%.6f")
    coverage_report(prices_win, "sector_prices.csv")
    log.info("Wrote %s", prices_path)

    # --- macro ------------------------------------------------------------
    macro_raw = fetch_macro_raw(buffer_start, end, api_key)
    macro_raw.to_csv(RAW_DIR / "macro_raw.csv", float_format="%.6f")

    macro_t = transform_macro(macro_raw)
    macro_t = fill_release_gaps(macro_t)
    macro_lagged = lag_features(macro_t)

    macro_win = macro_lagged.loc[(macro_lagged.index >= start) & (macro_lagged.index <= end)]
    macro_win.to_csv(macro_path, float_format="%.6f")
    coverage_report(macro_win, "macro_features.csv")
    log.info("Wrote %s", macro_path)

    # --- sanity checks ----------------------------------------------------
    if not prices_win.index.equals(macro_win.index):
        log.warning(
            "Price and macro month-end indices differ (prices=%d rows, macro=%d rows); "
            "join on 'date' before modelling.",
            len(prices_win), len(macro_win),
        )

    # The lag is what protects against look-ahead, so verify every column held.
    failures = audit_lags(macro_t, macro_lagged)
    if failures:
        for msg in failures:
            log.error("LAG CHECK FAILED: %s", msg)
        return 1
    by_lag: dict[int, int] = {}
    for spec in FRED_SERIES.values():
        by_lag[spec.lag] = by_lag.get(spec.lag, 0) + 1
    log.info(
        "Lag audit passed for all %d features (%s).",
        len(FRED_SERIES),
        ", ".join(f"{n} at lag {lag}" for lag, n in sorted(by_lag.items())),
    )

    n_missing = int(macro_win.isna().sum().sum())
    if n_missing:
        log.warning("macro_features.csv contains %d missing values", n_missing)
    log.info(
        "Done. XLC has no history before 2018-06 and XLRE none before 2015-10 "
        "(fund inception) - expect leading NaNs in those columns."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", default=DEFAULT_START, help="first month, YYYY-MM")
    parser.add_argument("--end", default=DEFAULT_END, help="last month, YYYY-MM")
    parser.add_argument("--force", action="store_true", help="overwrite existing CSVs")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    return collect(args.start, args.end, args.force)


if __name__ == "__main__":
    sys.exit(main())
