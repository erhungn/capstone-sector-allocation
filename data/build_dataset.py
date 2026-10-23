"""Merge every feature source into one modeling-ready panel.

Joins the macro, sector, and LLM-sentiment feature files on the month-end index,
reshapes to a (date, ticker) panel, adds cross-sectional ranks, and attaches a
binary target: did this sector beat the equal-weight average of all sectors over
the following month?

Usage
-----
    python data/build_dataset.py
    python data/build_dataset.py --skip-sentiment   # macro + sector only

Output
------
    data/model_dataset.csv

Alignment
---------
Every feature file is already stamped so that row *t* contains only information
an investor holds at the close of month *t*:

    macro_features.csv    row t -> observations from t-1 (market) or t-2 (survey)
    sector_features.csv   row t -> prices through the close of t-1
    llm_sentiment.csv     row t -> FOMC text published on or before t, scored
                                   against the macro row for t

So none of the three is shifted again here. The target is the *forward* return
from t to t+1, which is the only column that looks ahead -- by construction.

The sentiment column keeps the ``_lag1`` suffix of its siblings to signal
"safe to use at t", but note the difference: the macro and sector features are
built from observations dated before t, whereas the sentiment score is formed
*at* t from information published by then. Both are known at t; only the target
is not.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))

import universe as U  # noqa: E402

# Universe-dependent; configure() rebinds these. They start at the main
# specification so an unparameterised import still describes the 11-asset build.
SECTORS = list(U.EQUITY_SECTORS)

# Per-ticker feature stems in sector_features.csv, mapped to their panel names.
SECTOR_FEATURES = {
    "ret_1m_lag1": "ret_1m_lag1",
    "ret_3m_lag1": "ret_3m_lag1",
    "ret_12m_lag1": "ret_12m_lag1",
    "rvol_21d_lag1": "rvol_21d_lag1",
}

# Columns the model must never see: they are the target or its ingredients.
TARGET_COLUMNS = ["y_fwd_return", "y_benchmark_return", "y_excess_return", "target"]

# --- sector identity ------------------------------------------------------
# XLC is the reference level, so its dummies are all zero. It is the natural
# choice: it is the youngest sector and absent from the panel until 2019-07,
# so making it the reference keeps the remaining ten columns fully populated.
REFERENCE_SECTOR = U.REFERENCE_TICKER
DUMMY_SECTORS = [t for t in SECTORS if t != REFERENCE_SECTOR]

# Economically motivated macro x sector terms. A linear model cannot otherwise
# learn that energy and utilities respond to the same VIX reading in opposite
# directions -- every sector sees an identical macro row each month, so a single
# coefficient has to serve all eleven. Trees can find these splits unaided
# given the dummies; the columns are built once and shared for comparability.
INTERACTIONS = list(U.INTERACTIONS_EQUITY)

# --- derived sentiment features -------------------------------------------
# Built from the full 120 x 11 sentiment matrix rather than the filtered panel:
# the LLM scored every sector every month, so XLC has a usable prior-month score
# at its first panel appearance (2019-07) even though its price features do not
# start until then. Computing these after the panel filter would strand XLC's
# first row with a NaN change and drop it.
SENTIMENT_DERIVED = [
    "sentiment_change",       # vs the same sector last month
    "sentiment_rank",         # cross-sectional percentile within the month
    "relative_sentiment",     # sector score minus that month's mean
    "sentiment_dispersion",   # spread across sectors, identical within a month
]

DATA_DIR = Path(__file__).resolve().parent
MACRO_CSV = DATA_DIR / "macro_features.csv"
SECTOR_CSV = DATA_DIR / "sector_features.csv"
SENTIMENT_CSV = DATA_DIR / "llm_sentiment.csv"
PRICES_CSV = DATA_DIR / "sector_prices.csv"
OUT_CSV = DATA_DIR / "model_dataset.csv"


def configure(universe: str) -> None:
    """Rebind the universe-dependent module state and output paths."""
    global SECTORS, DUMMY_SECTORS, INTERACTIONS, SECTOR_CSV, SENTIMENT_CSV, OUT_CSV
    SECTORS = U.tickers(universe)
    DUMMY_SECTORS = U.dummy_tickers(universe)
    INTERACTIONS = U.interactions(universe)
    SECTOR_CSV = U.path_for(DATA_DIR / "sector_features.csv", universe)
    SENTIMENT_CSV = U.path_for(DATA_DIR / "llm_sentiment.csv", universe)
    OUT_CSV = U.path_for(DATA_DIR / "model_dataset.csv", universe)

log = logging.getLogger("build_dataset")


def read_indexed(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.DatetimeIndex(df.index).normalize()
    df.index.name = "date"
    return df


# --------------------------------------------------------------------------
# Target
# --------------------------------------------------------------------------

def forward_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple return from month t to t+1, stamped at t.

    ``shift(-1)`` is the one deliberate look-ahead in this file: row t must hold
    the *next* month's outcome for supervised learning. The final row becomes
    NaN and is dropped downstream.
    """
    return prices.shift(-1) / prices - 1.0


def build_target(prices: pd.DataFrame) -> pd.DataFrame:
    """Long-form target: fwd return, equal-weight benchmark, excess, and label.

    The benchmark averages every sector with a forward return that month, so it
    reflects the investable universe as it actually existed -- XLC joins from
    2018-06 when its prices begin, independent of when its *features* become
    complete.
    """
    fwd = forward_returns(prices)
    benchmark = fwd.mean(axis=1, skipna=True)

    # melt rather than stack: it keeps every (date, ticker) pair including the
    # NaN forward returns we need to count and drop explicitly, and its
    # behaviour has not shifted between pandas 2.x and 3.x.
    long = fwd.reset_index().melt(
        id_vars="date", var_name="ticker", value_name="y_fwd_return"
    )
    long["y_benchmark_return"] = long["date"].map(benchmark)
    long["y_excess_return"] = long["y_fwd_return"] - long["y_benchmark_return"]
    long["target"] = (long["y_excess_return"] > 0).astype("float")
    long.loc[long["y_fwd_return"].isna(), "target"] = pd.NA
    return long


# --------------------------------------------------------------------------
# Panel assembly
# --------------------------------------------------------------------------

def sector_panel(sector: pd.DataFrame) -> pd.DataFrame:
    """Wide {TICKER}_{feature} frame -> long (date, ticker, feature...) frame."""
    frames = []
    for ticker in SECTORS:
        cols = {f"{ticker}_{src}": dst for src, dst in SECTOR_FEATURES.items()}
        missing = [c for c in cols if c not in sector.columns]
        if missing:
            raise KeyError(f"sector_features.csv missing columns: {missing}")
        block = sector[list(cols)].rename(columns=cols).copy()
        block.insert(0, "ticker", ticker)
        frames.append(block.reset_index())
    return pd.concat(frames, ignore_index=True)


def derive_sentiment_features(sentiment: pd.DataFrame) -> pd.DataFrame:
    """Long-form sentiment plus four derived views of it.

    Takes the wide 120 x 11 sentiment matrix and returns one row per
    (date, ticker) with:

      sentiment_lag1        the raw score
      sentiment_change      month-over-month change for that sector
      sentiment_rank        percentile rank across sectors within the month
      relative_sentiment    score minus the month's cross-sectional mean
      sentiment_dispersion  std across sectors that month (constant within it)

    All four use only information from month *t* or earlier, so they inherit the
    raw score's timing and need no further lag.

    Note the algebra before reading too much into them separately:
    ``relative_sentiment = sentiment_lag1 - monthly mean``, and
    ``sentiment_rank`` is a monotone transform of ``relative_sentiment`` within
    each month. The three carry overlapping information; only the split between
    the monthly *level* and the cross-sectional *deviation* is genuinely new.
    """
    wide = sentiment[SECTORS].sort_index()

    change = wide.diff()                                    # same sector, t vs t-1
    rank = wide.rank(axis=1, pct=True)                      # across sectors, per month
    mean = wide.mean(axis=1)
    relative = wide.sub(mean, axis=0)
    dispersion = wide.std(axis=1, ddof=1)

    def melt(frame: pd.DataFrame, name: str) -> pd.DataFrame:
        out = frame.reset_index().melt(id_vars="date", var_name="ticker", value_name=name)
        return out

    long = melt(wide, "sentiment_lag1")
    for frame, name in ((change, "sentiment_change"), (rank, "sentiment_rank"),
                        (relative, "relative_sentiment")):
        long = long.merge(melt(frame, name), on=["date", "ticker"], validate="one_to_one")
    long["sentiment_dispersion"] = long["date"].map(dispersion)
    return long


def dummy_name(ticker: str) -> str:
    return f"is_{ticker}"


def interaction_name(macro_col: str, ticker: str) -> str:
    return f"{macro_col}_x_is_{ticker}"


def add_sector_identity(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """One-hot sector dummies plus the macro x sector interaction terms.

    Returns (panel, dummy_columns, interaction_columns).

    The dummies carry no time variation, so they can only express a sector's
    *unconditional* tendency to beat the cross-sectional average. The
    interactions are the part that does the intended work: they let a single
    macro reading move sectors in different directions.
    """
    for ticker in DUMMY_SECTORS:
        panel[dummy_name(ticker)] = (panel["ticker"] == ticker).astype(float)
    dummies = [dummy_name(t) for t in DUMMY_SECTORS]

    interactions = []
    for macro_col, ticker in INTERACTIONS:
        if macro_col not in panel.columns:
            raise KeyError(f"interaction references missing macro column: {macro_col}")
        if ticker not in DUMMY_SECTORS:
            raise KeyError(f"interaction references the reference sector: {ticker}")
        name = interaction_name(macro_col, ticker)
        panel[name] = panel[macro_col] * panel[dummy_name(ticker)]
        interactions.append(name)

    return panel, dummies, interactions


def add_cross_sectional_ranks(panel: pd.DataFrame) -> pd.DataFrame:
    """Rank each sector against its peers within the same month.

    Percentile rank rather than an integer 1..11: the universe is only 10 sectors
    before XLC's 2018-06 inception, so an integer rank would not mean the same
    thing in 2016 as in 2020. Ranks are computed from the already-lagged features,
    so they add no new information about t+1.
    """
    for src, dst in (("ret_1m_lag1", "ret_1m_rank"), ("rvol_21d_lag1", "rvol_21d_rank")):
        panel[dst] = panel.groupby("date")[src].rank(pct=True)
    return panel


def build(skip_sentiment: bool, universe: str = U.DEFAULT_UNIVERSE) -> int:
    configure(universe)
    log.info("universe %r: %d assets, %d dummies (%s reference), %d interactions",
             universe, len(SECTORS), len(DUMMY_SECTORS), REFERENCE_SECTOR, len(INTERACTIONS))
    needed = [MACRO_CSV, SECTOR_CSV, PRICES_CSV] + ([] if skip_sentiment else [SENTIMENT_CSV])
    for path in needed:
        if not path.exists():
            log.error("%s not found -- run the upstream script first", path)
            return 1

    macro = read_indexed(MACRO_CSV)
    sector = read_indexed(SECTOR_CSV)
    # sector_prices.csv holds every ticker any universe can contain, so it must
    # be cut to this universe BEFORE the target is built: build_target averages
    # whatever columns it is handed, and an unfiltered frame would silently
    # compute the equal-weight benchmark -- and therefore the label -- over
    # assets this specification does not trade.
    prices = read_indexed(PRICES_CSV)
    extra = [c for c in prices.columns if c not in SECTORS]
    prices = prices[[c for c in SECTORS if c in prices.columns]]
    if extra:
        log.info("prices: kept %d universe column(s), dropped %s",
                 prices.shape[1], ", ".join(extra))
    missing_prices = [c for c in SECTORS if c not in prices.columns]
    if missing_prices:
        log.error("sector_prices.csv is missing %s -- re-run data/collect.py",
                  ", ".join(missing_prices))
        return 1
    log.info("macro %s | sector %s | prices %s", macro.shape, sector.shape, prices.shape)

    if not macro.index.equals(sector.index) or not macro.index.equals(prices.index):
        log.error("Feature files do not share an index; re-run the upstream scripts")
        return 1

    panel = sector_panel(sector)

    # Macro features are shared across tickers within a month.
    panel = panel.merge(macro.reset_index(), on="date", how="left", validate="many_to_one")

    if not skip_sentiment:
        sentiment = read_indexed(SENTIMENT_CSV)
        if not sentiment.index.equals(macro.index):
            log.error("llm_sentiment.csv index does not match the other feature files")
            return 1
        missing = [t for t in SECTORS if t not in sentiment.columns]
        if missing:
            log.error("llm_sentiment.csv missing sectors: %s", missing)
            return 1
        # Wide -> long, with the derived views. No shift: the score at t is
        # already formed from information published on or before t (see the
        # module docstring).
        senti = derive_sentiment_features(sentiment)
        panel = panel.merge(senti, on=["date", "ticker"], how="left", validate="one_to_one")
    else:
        log.warning("--skip-sentiment: building without llm_sentiment.csv")

    panel = add_cross_sectional_ranks(panel)
    panel, dummy_cols, interaction_cols = add_sector_identity(panel)
    panel = panel.merge(build_target(prices), on=["date", "ticker"], how="left", validate="one_to_one")

    sentiment_cols = [] if skip_sentiment else ["sentiment_lag1"] + SENTIMENT_DERIVED
    feature_cols = (
        list(macro.columns)
        + list(SECTOR_FEATURES.values())
        + sentiment_cols
        + ["ret_1m_rank", "rvol_21d_rank"]
        + dummy_cols
        + interaction_cols
    )
    log.info("features: %d macro + %d sector + %d rank + %d dummy + %d interaction + %d sentiment",
             len(macro.columns), len(SECTOR_FEATURES), 2,
             len(dummy_cols), len(interaction_cols), len(sentiment_cols))
    panel = panel[["date", "ticker"] + feature_cols + TARGET_COLUMNS]
    panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)

    # --- drop incomplete rows ---------------------------------------------
    n_start = len(panel)
    no_target = panel["target"].isna()
    log.info("Dropping %d rows with no forward return (final month %s)",
             int(no_target.sum()), panel.loc[no_target, "date"].max().date())
    panel = panel[~no_target]

    incomplete = panel[feature_cols].isna().any(axis=1)
    if incomplete.any():
        by_ticker = panel.loc[incomplete].groupby("ticker").size()
        log.info("Dropping %d rows with incomplete features:", int(incomplete.sum()))
        for ticker, n in by_ticker.items():
            last = panel.loc[incomplete & (panel["ticker"] == ticker), "date"].max()
            log.info("    %-5s %3d rows, through %s (pre-inception history)", ticker, n, last.date())
    panel = panel[~incomplete].reset_index(drop=True)

    # 8 dp to match sector_features.csv: realized vol runs ~0.003-0.09, where
    # 6 dp costs significant figures on the low-vol tickers.
    panel.to_csv(OUT_CSV, index=False, float_format="%.8f")
    log.info("Wrote %s", OUT_CSV)

    summary(panel, feature_cols, n_start)
    return 0 if verify(panel, macro, prices, feature_cols, skip_sentiment) else 1


# --------------------------------------------------------------------------
# Reporting and verification
# --------------------------------------------------------------------------

def summary(panel: pd.DataFrame, feature_cols: list[str], n_start: int) -> None:
    log.info("")
    log.info("=" * 62)
    log.info("shape          : %d rows x %d columns (from %d candidate rows)",
             *panel.shape, n_start)
    log.info("date range     : %s -> %s (%d months)",
             panel["date"].min().date(), panel["date"].max().date(), panel["date"].nunique())
    log.info("tickers        : %d", panel["ticker"].nunique())
    log.info("features       : %d", len(feature_cols))
    log.info("target columns : %s", ", ".join(TARGET_COLUMNS))
    log.info("NaNs in features: %d", int(panel[feature_cols].isna().sum().sum()))

    pos = float(panel["target"].mean())
    log.info("class balance  : %.1f%% positive (%d of %d beat the benchmark)",
             pos * 100, int(panel["target"].sum()), len(panel))
    log.info("rows per ticker:")
    for ticker, n in panel.groupby("ticker").size().sort_values().items():
        log.info("    %-5s %4d", ticker, n)
    log.info("=" * 62)


def verify(
    panel: pd.DataFrame,
    macro: pd.DataFrame,
    prices: pd.DataFrame,
    feature_cols: list[str],
    skip_sentiment: bool,
) -> bool:
    """Re-derive the alignment claims from source rather than trusting the merge."""
    ok = True
    log.info("")
    log.info("Verification")

    # 1. Target really is the t -> t+1 return, recomputed from the price panel.
    idx = list(prices.index)
    bad = 0
    for _, row in panel.sample(min(300, len(panel)), random_state=0).iterrows():
        i = idx.index(row["date"])
        expected = prices[row["ticker"]].iloc[i + 1] / prices[row["ticker"]].iloc[i] - 1.0
        if not abs(expected - row["y_fwd_return"]) < 1e-9:
            bad += 1
    if bad:
        log.error("  FAIL target: %d of 300 sampled rows are not P(t+1)/P(t)-1", bad)
        ok = False
    else:
        log.info("  PASS target is P(t+1)/P(t)-1 (300 rows recomputed from prices)")

    # 2. Macro features are the row for t itself, not a neighbouring month.
    sample = panel.sample(min(200, len(panel)), random_state=1)
    mism = sum(
        1 for _, r in sample.iterrows()
        if not abs(macro.loc[r["date"], "VIX_lag1"] - r["VIX_lag1"]) < 1e-9
    )
    if mism:
        log.error("  FAIL macro: %d rows do not match macro_features.csv for their own date", mism)
        ok = False
    else:
        log.info("  PASS macro features match macro_features.csv row for the same date")

    # 3. Sentiment is unshifted: row t carries the score stamped t.
    if not skip_sentiment:
        senti = read_indexed(SENTIMENT_CSV)
        mism = sum(
            1 for _, r in sample.iterrows()
            if not abs(senti.loc[r["date"], r["ticker"]] - r["sentiment_lag1"]) < 1e-9
        )
        if mism:
            log.error("  FAIL sentiment: %d rows do not match llm_sentiment.csv for their own date", mism)
            ok = False
        else:
            log.info("  PASS sentiment matches llm_sentiment.csv row for the same date (no extra shift)")

        # Confirm it is not accidentally shifted, measured across the whole
        # panel rather than one row -- consecutive months genuinely repeat a
        # score often enough that a single-row check fires on coincidence.
        shifted = senti.shift(1)
        same_month = prior_month = 0
        for _, r in panel.iterrows():
            val = r["sentiment_lag1"]
            if abs(senti.loc[r["date"], r["ticker"]] - val) < 1e-9:
                same_month += 1
            prev = shifted.loc[r["date"], r["ticker"]]
            if pd.notna(prev) and abs(prev - val) < 1e-9:
                prior_month += 1
        n = len(panel)
        log.info("  sentiment matches its own month %d/%d, the prior month %d/%d",
                 same_month, n, prior_month, n)
        if same_month != n:
            log.error("  FAIL sentiment is not aligned to its own month")
            ok = False
        elif prior_month > 0.9 * n:
            log.error("  FAIL sentiment looks shifted by one month")
            ok = False
        else:
            log.info("  PASS sentiment is aligned to month t, not shifted to t-1")

    # 4. The last month with a target must be the second-to-last price month.
    if panel["date"].max() != prices.index[-2]:
        log.error("  FAIL last dated row is %s, expected %s",
                  panel["date"].max().date(), prices.index[-2].date())
        ok = False
    else:
        log.info("  PASS last row is %s; %s dropped for want of a t+1 price",
                 prices.index[-2].date(), prices.index[-1].date())

    # 5. Target is consistent with its own ingredients.
    recomputed = (panel["y_fwd_return"] > panel["y_benchmark_return"]).astype(float)
    if not recomputed.equals(panel["target"].astype(float)):
        log.error("  FAIL target disagrees with y_fwd_return > y_benchmark_return")
        ok = False
    else:
        log.info("  PASS target == (y_fwd_return > y_benchmark_return)")

    # 6. No feature may be perfectly correlated with the target's ingredients --
    #    a cheap tripwire for a forward-looking column sneaking into the panel.
    leaks = [
        c for c in feature_cols
        if panel[c].notna().sum() > 30
        and abs(panel[c].corr(panel["y_fwd_return"])) > 0.99
    ]
    if leaks:
        log.error("  FAIL features almost perfectly correlated with the forward return: %s", leaks)
        ok = False
    else:
        log.info("  PASS no feature correlates > 0.99 with the forward return")

    # 7. Sector dummies are a valid one-hot with XLC as the reference.
    dummies = [dummy_name(t) for t in DUMMY_SECTORS if dummy_name(t) in panel.columns]
    if dummies:
        row_sums = panel[dummies].sum(axis=1)
        is_ref = panel["ticker"] == REFERENCE_SECTOR
        if not ((row_sums[~is_ref] == 1).all() and (row_sums[is_ref] == 0).all()):
            log.error("  FAIL sector dummies are not a clean one-hot")
            ok = False
        else:
            log.info("  PASS %d sector dummies one-hot; %s is the reference (all zero, %d rows)",
                     len(dummies), REFERENCE_SECTOR, int(is_ref.sum()))
        for ticker in DUMMY_SECTORS:
            col = dummy_name(ticker)
            if col in panel.columns and not (panel.loc[panel[col] == 1, "ticker"] == ticker).all():
                log.error("  FAIL %s is set on the wrong rows", col)
                ok = False

    # 8. Interactions really are the product of their two parents.
    bad_terms = []
    for macro_col, ticker in INTERACTIONS:
        name = interaction_name(macro_col, ticker)
        if name not in panel.columns:
            continue
        expected = panel[macro_col] * panel[dummy_name(ticker)]
        if not np.allclose(panel[name], expected, equal_nan=True):
            bad_terms.append(name)
        # and it must be zero everywhere except that sector
        if (panel.loc[panel["ticker"] != ticker, name] != 0).any():
            bad_terms.append(f"{name} (non-zero off-sector)")
    if bad_terms:
        log.error("  FAIL interaction terms incorrect: %s", bad_terms)
        ok = False
    elif INTERACTIONS:
        log.info("  PASS %d interaction terms equal macro x dummy, zero off-sector",
                 len(INTERACTIONS))

    # 9. Derived sentiment features re-derived from the source matrix.
    if not skip_sentiment and "sentiment_change" in panel.columns:
        senti = read_indexed(SENTIMENT_CSV)[SECTORS].sort_index()
        bad = []
        for _, r in panel.sample(min(200, len(panel)), random_state=2).iterrows():
            d, tk = r["date"], r["ticker"]
            month = senti.loc[d]
            prev_idx = senti.index.get_loc(d) - 1
            if prev_idx >= 0:
                expected_change = month[tk] - senti.iloc[prev_idx][tk]
                if abs(expected_change - r["sentiment_change"]) > 1e-6:
                    bad.append(f"sentiment_change {tk} {d.date()}")
            if abs((month[tk] - month.mean()) - r["relative_sentiment"]) > 1e-6:
                bad.append(f"relative_sentiment {tk} {d.date()}")
            if abs(month.std(ddof=1) - r["sentiment_dispersion"]) > 1e-6:
                bad.append(f"sentiment_dispersion {d.date()}")
            expected_rank = (month.rank(pct=True))[tk]
            if abs(expected_rank - r["sentiment_rank"]) > 1e-6:
                bad.append(f"sentiment_rank {tk} {d.date()}")
        if bad:
            log.error("  FAIL derived sentiment mismatches: %s", bad[:5])
            ok = False
        else:
            log.info("  PASS 4 derived sentiment features re-derived from llm_sentiment.csv")

        # dispersion must be constant within a month, relative must sum to ~0
        per_month = panel.groupby("date")["sentiment_dispersion"].nunique()
        if (per_month > 1).any():
            log.error("  FAIL sentiment_dispersion varies within a month")
            ok = False
        else:
            log.info("  PASS sentiment_dispersion is constant within each month")

    # 10. Class balance sanity.
    pos = float(panel["target"].mean())
    if not 0.40 <= pos <= 0.60:
        log.warning("  class balance %.1f%% is outside 40-60%%", pos * 100)
    else:
        log.info("  PASS class balance %.1f%% is close to even", pos * 100)

    log.info("  %s", "all checks passed" if ok else "SOME CHECKS FAILED")
    return ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--skip-sentiment", action="store_true",
                        help="build without llm_sentiment.csv")
    parser.add_argument("--universe", default=U.DEFAULT_UNIVERSE, choices=sorted(U.UNIVERSES),
                        help="asset universe (default %(default)s)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    return build(args.skip_sentiment, args.universe)


if __name__ == "__main__":
    sys.exit(main())
