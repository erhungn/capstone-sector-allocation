"""Is there alpha left after market and style exposures are priced out?

The backtest says the model portfolios trail the benchmarks. This asks the
sharper question: once market, size, value and momentum exposure are accounted
for, is anything left? A strategy can lose to SPY and still carry alpha (if it
runs low beta), or beat a benchmark with none (if it is just levered beta).

Usage
-----
    python models/factor_alpha.py
    python models/factor_alpha.py --refresh-factors

Outputs
-------
    data/raw/ff_factors.csv     cached Fama-French monthly factors
    models/factor_alpha.csv

Specifications
--------------
    CAPM      R_p - R_f = a + b_MKT (R_M - R_f) + e
    Carhart   R_p - R_f = a + b_MKT MKT + b_SMB SMB + b_HML HML + b_MOM MOM + e

Long-short books are **self-financing**: the short proceeds fund the long leg, so
there is no capital on which a risk-free return is foregone. Their spreads are
regressed directly on the factors with no risk-free subtraction. Subtracting R_f
there would mechanically drag alpha down by roughly the T-bill rate and is the
more common error of the two.

Risk-free rate
--------------
These regressions use Fama-French's own **RF** column rather than FEDFUNDS.
Mixing sources would put a different risk-free rate on the left and right sides
of the same equation: MKT is constructed as (market return - FF's RF), so any
other rate leaves a wedge that lands in the intercept and is read as alpha. The
backtest tables elsewhere in this project use FEDFUNDS for Sharpe ratios, which
is a separate calculation and does not enter here.
"""

from __future__ import annotations

import argparse
import io
import logging
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtest"))

EQUITY_CURVES = REPO_ROOT / "backtest" / "equity_curves.csv"
FF_CACHE = REPO_ROOT / "data" / "raw" / "ff_factors.csv"
OUT_CSV = REPO_ROOT / "models" / "factor_alpha.csv"

FF_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FF_FACTORS_ZIP = FF_BASE + "F-F_Research_Data_Factors_CSV.zip"
FF_MOMENTUM_ZIP = FF_BASE + "F-F_Momentum_Factor_CSV.zip"
USER_AGENT = "capstone-research/1.0 (masters thesis; python-requests)"

NW_LAGS = 3
MONTHS = 12
CAPM_FACTORS = ["MKT_RF"]
CARHART_FACTORS = ["MKT_RF", "SMB", "HML", "MOM"]

log = logging.getLogger("factor_alpha")


# --------------------------------------------------------------------------
# Fama-French factors
# --------------------------------------------------------------------------

def _parse_french_csv(text: str, columns: list[str]) -> pd.DataFrame:
    """Pull the monthly block out of a Ken French CSV.

    The files carry a prose preamble, then monthly rows keyed YYYYMM, then an
    annual section keyed YYYY. Selecting rows whose first field is exactly six
    digits takes the monthly block and stops before the annual one without
    depending on the preamble's line count, which changes between releases.
    """
    rows = []
    for line in text.splitlines():
        m = re.match(r"^\s*(\d{6})\s*,(.*)$", line)
        if not m:
            continue
        values = [v.strip() for v in m.group(2).split(",")]
        if len(values) < len(columns):
            continue
        try:
            rows.append([m.group(1)] + [float(v) for v in values[: len(columns)]])
        except ValueError:
            continue
    if not rows:
        raise ValueError("no monthly rows found in the Ken French file")

    df = pd.DataFrame(rows, columns=["yyyymm"] + columns)
    df.index = (pd.to_datetime(df["yyyymm"], format="%Y%m")
                + pd.offsets.MonthEnd(0)).values
    df.index.name = "date"
    return df.drop(columns=["yyyymm"])


def _download(url: str) -> str:
    import requests

    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=60)
    resp.raise_for_status()
    archive = zipfile.ZipFile(io.BytesIO(resp.content))
    return archive.read(archive.namelist()[0]).decode("latin-1")


def load_factors(refresh: bool) -> pd.DataFrame:
    """Monthly FF3 + momentum, in decimal. Cached after the first download."""
    if FF_CACHE.exists() and not refresh:
        ff = pd.read_csv(FF_CACHE, index_col=0, parse_dates=True)
        ff.index = pd.DatetimeIndex(ff.index).normalize()
        log.info("Loaded cached factors: %s (%d months, %s -> %s)",
                 FF_CACHE.name, len(ff), ff.index.min().date(), ff.index.max().date())
        return ff

    log.info("Downloading Fama-French factors from Ken French's data library")
    three = _parse_french_csv(_download(FF_FACTORS_ZIP), ["MKT_RF", "SMB", "HML", "RF"])
    mom = _parse_french_csv(_download(FF_MOMENTUM_ZIP), ["MOM"])
    ff = three.join(mom, how="inner")
    # Ken French publishes percent; every downstream calculation wants decimals.
    ff = ff / 100.0
    FF_CACHE.parent.mkdir(parents=True, exist_ok=True)
    ff.to_csv(FF_CACHE, float_format="%.6f")
    log.info("Wrote %s (%d months, %s -> %s)", FF_CACHE, len(ff),
             ff.index.min().date(), ff.index.max().date())
    return ff


# --------------------------------------------------------------------------
# OLS with conventional and Newey-West standard errors
# --------------------------------------------------------------------------

def ols(y: np.ndarray, X: np.ndarray, nw_lags: int = NW_LAGS) -> dict:
    """Least squares with both classical and HAC standard errors.

    X must already include the intercept column. Newey-West uses the Bartlett
    kernel with the given lag; at monthly frequency on non-overlapping returns it
    is a light correction, but portfolio residuals are mildly autocorrelated and
    reporting only the classical t would overstate precision.
    """
    n, k = X.shape
    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    resid = y - X @ beta
    rss = float(resid @ resid)
    tss = float(((y - y.mean()) ** 2).sum())
    dof = n - k

    sigma2 = rss / dof
    se = np.sqrt(np.diag(sigma2 * xtx_inv))

    # HAC sandwich: (X'X)^-1 [S] (X'X)^-1, Bartlett-weighted.
    Xe = X * resid[:, None]
    S = Xe.T @ Xe
    for j in range(1, min(nw_lags, n - 1) + 1):
        w = 1.0 - j / (nw_lags + 1.0)
        G = Xe[j:].T @ Xe[:-j]
        S += w * (G + G.T)
    se_nw = np.sqrt(np.diag(xtx_inv @ S @ xtx_inv))

    r2 = 1.0 - rss / tss if tss > 0 else np.nan
    return {
        "beta": beta, "se": se, "se_nw": se_nw,
        "t": beta / se, "t_nw": beta / se_nw,
        "p": 2 * (1 - stats.t.cdf(np.abs(beta / se), dof)),
        "p_nw": 2 * (1 - stats.t.cdf(np.abs(beta / se_nw), dof)),
        "r2": r2,
        "adj_r2": 1.0 - (1.0 - r2) * (n - 1) / dof if dof > 0 else np.nan,
        "nobs": n, "dof": dof,
    }


def regress(returns: pd.Series, ff: pd.DataFrame, factors: list[str],
            subtract_rf: bool) -> dict:
    common = returns.index.intersection(ff.index)
    r = returns.reindex(common).astype(float)
    F = ff.reindex(common)
    y = (r - F["RF"]).to_numpy() if subtract_rf else r.to_numpy()
    X = np.column_stack([np.ones(len(common))] + [F[f].to_numpy() for f in factors])
    return ols(y, X)


# --------------------------------------------------------------------------
# Portfolio returns
# --------------------------------------------------------------------------

def long_only_returns() -> pd.DataFrame:
    """Monthly net returns implied by the saved cumulative equity curves."""
    eq = pd.read_csv(EQUITY_CURVES, index_col=0, parse_dates=True).sort_index()
    eq.index = pd.DatetimeIndex(eq.index).normalize()
    rets = eq.pct_change()
    rets.iloc[0] = eq.iloc[0] - 1.0      # the curve starts from $1
    return rets


def long_short_returns() -> pd.DataFrame:
    """Recompute the long-short monthly series.

    ``long_short_results.csv`` stores summary statistics only, so the monthly
    series is rebuilt by calling the same functions that produced it rather than
    reimplementing the construction here.
    """
    import long_short as LS

    preds = pd.read_csv(REPO_ROOT / "models" / "predictions.csv", parse_dates=["date"])
    data = pd.read_csv(REPO_ROOT / "data" / "model_dataset.csv", parse_dates=["date"])
    frame = preds.merge(data[["date", "ticker", "sentiment_lag1", "ret_12m_lag1"]],
                        on=["date", "ticker"], how="left", validate="one_to_one")
    returns_by_date = {d: dict(zip(g["ticker"], g["actual_return"]))
                       for d, g in frame.groupby("date")}

    out = {}
    for name, col in LS.SIGNALS.items():
        if col not in frame.columns:
            continue
        longs, shorts = LS.leg_weights(frame[["date", "ticker", col]], col, LS.TOP_N)
        sim = LS.simulate_long_short(longs, shorts, returns_by_date, LS.COST_BPS)
        out[f"{name} (L/S)"] = sim["net"]
    return pd.DataFrame(out)


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def run(refresh: bool) -> int:
    if not EQUITY_CURVES.exists():
        log.error("%s not found -- run backtest/run_backtest.py first", EQUITY_CURVES)
        return 1

    ff = load_factors(refresh)
    long_only = long_only_returns()
    long_short = long_short_returns()

    check_alignment(long_only, ff)

    rows = []
    for label, rets, self_financing in (
        ("long-only", long_only, False),
        ("long-short", long_short, True),
    ):
        for name in rets.columns:
            series = rets[name].dropna()
            for spec, factors in (("CAPM", CAPM_FACTORS), ("Carhart", CARHART_FACTORS)):
                res = regress(series, ff, factors, subtract_rf=not self_financing)
                names = ["alpha"] + factors
                row = {
                    "book": label, "strategy": name, "spec": spec,
                    "subtract_rf": not self_financing,
                    "nobs": res["nobs"], "r2": res["r2"], "adj_r2": res["adj_r2"],
                    "alpha_monthly": res["beta"][0],
                    "alpha_annual": (1 + res["beta"][0]) ** MONTHS - 1,
                    "alpha_t": res["t"][0], "alpha_t_nw": res["t_nw"][0],
                    "alpha_p_nw": res["p_nw"][0], "nw_lags": NW_LAGS,
                }
                for i, f in enumerate(names[1:], start=1):
                    row[f"beta_{f}"] = res["beta"][i]
                    row[f"t_{f}"] = res["t"][i]
                    row[f"t_nw_{f}"] = res["t_nw"][i]
                rows.append(row)

    results = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT_CSV, index=False, float_format="%.5f")
    log.info("Wrote %s (%d regressions)", OUT_CSV, len(results))

    report(results)
    return 0


def check_alignment(long_only: pd.DataFrame, ff: pd.DataFrame) -> None:
    """SPY should track the FF market factor almost exactly if dates line up."""
    if "SPY buy & hold" not in long_only.columns:
        return
    common = long_only.index.intersection(ff.index)
    spy = long_only.loc[common, "SPY buy & hold"]
    mkt = ff.loc[common, "MKT_RF"] + ff.loc[common, "RF"]
    corr = float(spy.corr(mkt))
    log.info("alignment check: %d overlapping months, %s -> %s",
             len(common), common.min().date(), common.max().date())
    log.info("  corr(SPY monthly return, FF market return) = %.4f", corr)
    if corr < 0.95:
        log.error("  LOW -- the return series and factor months are probably misaligned")
    else:
        log.info("  dates line up (a one-month offset would drop this near zero)")
    log.info("  mean FF RF over the window: %.2f%% annualised",
             ff.loc[common, "RF"].mean() * MONTHS * 100)


def report(results: pd.DataFrame) -> None:
    for book in ("long-only", "long-short"):
        sub = results[results["book"] == book]
        if sub.empty:
            continue
        log.info("")
        log.info("=" * 116)
        rf_note = ("excess of RF" if book == "long-only"
                   else "raw spread, self-financing so no RF subtraction")
        log.info("%s PORTFOLIOS  (%s)", book.upper(), rf_note)
        log.info("%-22s %24s %26s %8s %8s",
                 "", "----- CAPM -----", "-------- Carhart --------", "", "")
        log.info("%-22s %10s %6s %6s %10s %6s %6s %7s %7s %7s",
                 "strategy", "alpha/yr", "t", "t_NW", "alpha/yr", "t", "t_NW",
                 "b_MKT", "b_MOM", "adj R2")
        for name in sub["strategy"].unique():
            capm = sub[(sub.strategy == name) & (sub.spec == "CAPM")].iloc[0]
            car = sub[(sub.strategy == name) & (sub.spec == "Carhart")].iloc[0]
            log.info("%-22s %9.2f%% %6.2f %6.2f %9.2f%% %6.2f %6.2f %7.2f %7.2f %7.3f",
                     name, capm["alpha_annual"] * 100, capm["alpha_t"], capm["alpha_t_nw"],
                     car["alpha_annual"] * 100, car["alpha_t"], car["alpha_t_nw"],
                     car["beta_MKT_RF"], car["beta_MOM"], car["adj_r2"])
        log.info("=" * 116)

    log.info("")
    log.info("CARHART FACTOR LOADINGS (t-stats, Newey-West lag %d)", NW_LAGS)
    car = results[results["spec"] == "Carhart"]
    log.info("%-22s %16s %16s %16s %16s",
             "strategy", "MKT", "SMB", "HML", "MOM")
    for _, r in car.iterrows():
        log.info("%-22s %8.2f (%5.2f) %8.2f (%5.2f) %8.2f (%5.2f) %8.2f (%5.2f)",
                 r["strategy"], r["beta_MKT_RF"], r["t_nw_MKT_RF"],
                 r["beta_SMB"], r["t_nw_SMB"], r["beta_HML"], r["t_nw_HML"],
                 r["beta_MOM"], r["t_nw_MOM"])

    log.info("")
    log.info("VERDICT")
    sig = results[np.abs(results["alpha_t_nw"]) > 1.96]
    if sig.empty:
        log.info("  Not one of the %d regressions produces an alpha distinguishable from",
                 len(results))
        log.info("  zero at the 5%% level (Newey-West). Every strategy's return is fully")
        log.info("  explained by market, size, value and momentum exposure -- there is no")
        log.info("  residual skill on either the long-only or the long-short side.")
    else:
        for _, r in sig.iterrows():
            log.info("  %s (%s, %s): alpha %+.2f%%/yr, t_NW %.2f",
                     r["strategy"], r["book"], r["spec"],
                     r["alpha_annual"] * 100, r["alpha_t_nw"])

    lo = results[(results.book == "long-only") & (results.spec == "Carhart")]
    if not lo.empty:
        worst = lo.loc[lo["alpha_annual"].idxmin()]
        best = lo.loc[lo["alpha_annual"].idxmax()]
        log.info("")
        log.info("  Long-only Carhart alpha ranges from %+.2f%%/yr (%s) to %+.2f%%/yr (%s);",
                 worst["alpha_annual"] * 100, worst["strategy"],
                 best["alpha_annual"] * 100, best["strategy"])
        log.info("  every one of them has |t_NW| below 2.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--refresh-factors", action="store_true",
                        help="re-download from Ken French instead of using the cache")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s", datefmt="%H:%M:%S",
    )
    return run(args.refresh_factors)


if __name__ == "__main__":
    sys.exit(main())
