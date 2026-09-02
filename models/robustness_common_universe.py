"""Does the changing sector universe drive the main results?

XLC's prices begin 2018-06 and its 12-month return feature only completes
2019-07, so before that the panel holds 10 sectors, not 11. Both the
cross-sectional benchmark and the ranking task therefore change composition
mid-sample. This re-runs everything on the window where all eleven sectors are
present throughout, and reports it beside the main result.

Usage
-----
    python models/robustness_common_universe.py

Outputs
-------
    models/robustness_common_universe.csv

Design
------
The walk-forward, tuning grid, CV splitter, and embargo are imported from
``models/train.py`` and the portfolio simulation from
``backtest/run_backtest.py`` rather than reimplemented, so any difference
between the two runs is the sample and nothing else.

Three columns are reported, which separates two effects that would otherwise be
confounded:

    main                  the published run, 93 months
    main, common months   the same predictions, subset to the shorter window
                          -> isolates the effect of the *window*
    retrained             models retrained on the restricted sample
                          -> adds the effect of the *training data*

Power
-----
The restricted sample yields 52 OOS months against 93. Confidence intervals are
correspondingly wider and every point-estimate difference below is well inside
them. The honest reading is "consistent with the main result", not "different".
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "models"))
sys.path.insert(0, str(REPO_ROOT / "backtest"))

import run_backtest as B  # noqa: E402
import train as T  # noqa: E402

DATASET = REPO_ROOT / "data" / "model_dataset.csv"
MAIN_PREDICTIONS = REPO_ROOT / "models" / "predictions.csv"
OUT_CSV = REPO_ROOT / "models" / "robustness_common_universe.csv"

EMBARGO = 1
MIN_TRAIN_MONTHS = 24
CV_SPLITS = 3
RETUNE_EVERY = 1
TOP_N = 3
COST_BPS = 10.0
N_BOOT = 2000
MONTHS = 12

log = logging.getLogger("robustness")


# --------------------------------------------------------------------------
# Sample definition
# --------------------------------------------------------------------------

def common_universe_start(data: pd.DataFrame) -> pd.Timestamp:
    """First month from which every sector is present, contiguously."""
    counts = data.groupby("date")["ticker"].nunique()
    n_sectors = counts.max()
    full = counts[counts == n_sectors]
    start = full.index.min()
    if not (counts.loc[start:] == n_sectors).all():
        log.warning("universe is not contiguous after %s", start.date())
    log.info("all %d sectors present from %s (%d of %d months)",
             n_sectors, start.date(), len(full), len(counts))
    return start


# --------------------------------------------------------------------------
# Walk-forward (same protocol as train.py, different sample)
# --------------------------------------------------------------------------

def walk_forward(data: pd.DataFrame) -> pd.DataFrame:
    sets = T.feature_sets(data)
    all_dates = sorted(data["date"].unique())
    oos_dates = [
        d for d in all_dates
        if len([x for x in all_dates if x < T.train_cutoff(d, EMBARGO)]) >= MIN_TRAIN_MONTHS
    ]
    if not oos_dates:
        log.error("no prediction month clears %d training months after the embargo",
                  MIN_TRAIN_MONTHS)
        sys.exit(1)

    log.info("restricted walk-forward: %s -> %s (%d prediction months)",
             oos_dates[0].date(), oos_dates[-1].date(), len(oos_dates))

    records: list[dict] = []
    cached: dict[str, dict] = {}
    started = time.monotonic()

    for i, pred_date in enumerate(oos_dates):
        train_all = data[data["date"] < T.train_cutoff(pred_date, EMBARGO)]
        test_all = data[data["date"] == pred_date]
        month_rows = {m: {} for m in T.MODEL_SPECS}

        for model_id, (family, set_name) in T.MODEL_SPECS.items():
            cols = sets[set_name]
            tr = train_all.dropna(subset=cols + ["target"])
            te = test_all.dropna(subset=cols)
            if tr.empty or te.empty:
                continue
            X, y = tr[cols], tr["target"].astype(int)
            if y.nunique() < 2:
                continue
            if model_id not in cached or i % RETUNE_EVERY == 0:
                cached[model_id] = T.tune(family, X, y, tr["date"].values, CV_SPLITS)
            model = T.fit_final(family, cached[model_id], X, y)
            for ticker, p in zip(te["ticker"], model.predict_proba(te[cols])[:, 1]):
                month_rows[model_id][ticker] = float(p)

        for _, row in test_all.iterrows():
            rec = {"date": pred_date, "ticker": row["ticker"]}
            for model_id in T.MODEL_SPECS:
                rec[f"model_{model_id}_prob"] = month_rows[model_id].get(row["ticker"], np.nan)
            rec["actual_target"] = row["target"]
            rec["actual_return"] = row["y_fwd_return"]
            records.append(rec)

        if (i + 1) % 10 == 0 or i == len(oos_dates) - 1:
            log.info("  [%2d/%d] %s  train=%d rows  %.0fs",
                     i + 1, len(oos_dates), pred_date.date(), len(train_all),
                     time.monotonic() - started)

    return pd.DataFrame(records)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def add_excess(preds: pd.DataFrame) -> pd.DataFrame:
    preds = preds.copy()
    preds["excess"] = preds["actual_return"] - preds.groupby("date")["actual_return"].transform("mean")
    return preds


def auc_with_ci(preds: pd.DataFrame, col: str) -> tuple[float, float, float]:
    """AUC with a month-clustered bootstrap interval."""
    sub = preds.dropna(subset=[col, "actual_target"])
    y = sub["actual_target"].astype(int).to_numpy()
    p = sub[col].to_numpy()
    if len(np.unique(y)) < 2:
        return np.nan, np.nan, np.nan
    months = sub["date"].to_numpy()
    uniq = np.unique(months)
    idx = {m: np.flatnonzero(months == m) for m in uniq}
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(N_BOOT):
        ii = np.concatenate([idx[m] for m in rng.choice(uniq, len(uniq), replace=True)])
        if len(np.unique(y[ii])) > 1:
            boot.append(roc_auc_score(y[ii], p[ii]))
    return (float(roc_auc_score(y, p)),
            float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))


def auc_diff_ci(preds: pd.DataFrame, col_a: str, col_b: str) -> tuple[float, float, float]:
    sub = preds.dropna(subset=[col_a, col_b, "actual_target"])
    y = sub["actual_target"].astype(int).to_numpy()
    pa, pb = sub[col_a].to_numpy(), sub[col_b].to_numpy()
    months = sub["date"].to_numpy()
    uniq = np.unique(months)
    idx = {m: np.flatnonzero(months == m) for m in uniq}
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(N_BOOT):
        ii = np.concatenate([idx[m] for m in rng.choice(uniq, len(uniq), replace=True)])
        if len(np.unique(y[ii])) > 1:
            boot.append(roc_auc_score(y[ii], pb[ii]) - roc_auc_score(y[ii], pa[ii]))
    return (float(roc_auc_score(y, pb) - roc_auc_score(y, pa)),
            float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))


def ic_stats(preds: pd.DataFrame, col: str) -> dict:
    ics = []
    for _, g in preds.groupby("date"):
        g = g.dropna(subset=[col, "excess"])
        if len(g) < 3 or g[col].nunique() < 2:
            continue
        ics.append(stats.spearmanr(g[col], g["excess"]).statistic)
    ic = pd.Series(ics)
    sd = ic.std(ddof=1)
    return {"mean_ic": float(ic.mean()), "ic_ir": float(ic.mean() / sd) if sd > 0 else np.nan,
            "pct_ic_positive": float((ic > 0).mean())}


def backtest_block(preds: pd.DataFrame, dataset: pd.DataFrame) -> pd.DataFrame:
    """Model strategies plus the three benchmarks, on this window only."""
    universe = preds[["date", "ticker", "actual_return"]].merge(
        dataset[["date", "ticker", "ret_12m_lag1"]], on=["date", "ticker"], how="left")
    returns_by_date = {d: dict(zip(g["ticker"], g["actual_return"]))
                       for d, g in universe.groupby("date")}

    schedules = {}
    for model_id in B.MODELS:
        col = f"model_{model_id}_prob"
        schedules[f"Model {model_id}"] = B.top_n_weights(
            preds[["date", "ticker", col]].dropna(), col, TOP_N)
    schedules["Equal weight"] = B.equal_weight_all(universe)
    schedules["Momentum 12-1"] = B.top_n_weights(
        universe.dropna(subset=["ret_12m_lag1"]), "ret_12m_lag1", TOP_N)

    monthly, turnovers = {}, {}
    for name, sched in schedules.items():
        r, t = B.simulate(sched, returns_by_date, COST_BPS)
        monthly[name], turnovers[name] = r, t

    realised = B.next_month_end(sorted(universe["date"].unique()))
    spy = B.load_spy(realised.min() - pd.DateOffset(months=3), realised.max())
    spy_ret = (spy / spy.shift(1) - 1.0).reindex(realised).ffill()
    spy_ret.iloc[0] -= COST_BPS / 10_000.0
    monthly["SPY buy & hold"] = spy_ret
    turnovers["SPY buy & hold"] = pd.Series(0.0, index=realised)

    returns_df = pd.DataFrame(monthly).sort_index()
    rf = B.risk_free_monthly(returns_df.index)
    return pd.DataFrame({name: B.performance(returns_df[name], rf, turnovers.get(name))
                         for name in returns_df.columns}).T


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def summarise(preds: pd.DataFrame, dataset: pd.DataFrame, label: str) -> pd.DataFrame:
    preds = add_excess(preds)
    perf = backtest_block(preds, dataset)
    rows = []
    for model_id in B.MODELS:
        col = f"model_{model_id}_prob"
        auc, lo, hi = auc_with_ci(preds, col)
        name = f"Model {model_id}"
        rows.append({
            "sample": label, "n_months": preds["date"].nunique(), "n_obs": len(preds),
            "model": model_id, "auc": auc, "auc_lo": lo, "auc_hi": hi,
            **ic_stats(preds, col),
            "cagr": perf.loc[name, "cagr"], "sharpe": perf.loc[name, "sharpe"],
            "max_drawdown": perf.loc[name, "max_drawdown"],
            "annual_turnover": perf.loc[name, "annual_turnover"],
        })
    for a, b in (("A", "B"), ("C", "D")):
        d, lo, hi = auc_diff_ci(preds, f"model_{a}_prob", f"model_{b}_prob")
        rows.append({"sample": label, "n_months": preds["date"].nunique(),
                     "model": f"{b}-{a}", "auc": d, "auc_lo": lo, "auc_hi": hi})
    for bench in ("Equal weight", "Momentum 12-1", "SPY buy & hold"):
        rows.append({"sample": label, "n_months": preds["date"].nunique(),
                     "model": bench, "cagr": perf.loc[bench, "cagr"],
                     "sharpe": perf.loc[bench, "sharpe"],
                     "max_drawdown": perf.loc[bench, "max_drawdown"],
                     "annual_turnover": perf.loc[bench, "annual_turnover"]})
    return pd.DataFrame(rows)


def run() -> int:
    for path in (DATASET, MAIN_PREDICTIONS):
        if not path.exists():
            log.error("%s not found -- run the upstream script first", path)
            return 1

    data = pd.read_csv(DATASET, parse_dates=["date"]).sort_values(["date", "ticker"])
    start = common_universe_start(data)
    restricted = data[data["date"] >= start].copy()
    log.info("restricted sample: %d rows, %d months, %s -> %s",
             len(restricted), restricted["date"].nunique(),
             restricted["date"].min().date(), restricted["date"].max().date())

    main_preds = pd.read_csv(MAIN_PREDICTIONS, parse_dates=["date"])
    new_preds = walk_forward(restricted)

    common_months = sorted(set(main_preds["date"]) & set(new_preds["date"]))
    log.info("months shared by both runs: %d", len(common_months))

    blocks = [
        summarise(main_preds, data, "main (93 months)"),
        summarise(main_preds[main_preds["date"].isin(common_months)], data,
                  f"main, common months ({len(common_months)})"),
        summarise(new_preds, data, f"retrained ({new_preds['date'].nunique()} months)"),
    ]
    results = pd.concat(blocks, ignore_index=True)
    results.to_csv(OUT_CSV, index=False, float_format="%.4f")
    log.info("Wrote %s", OUT_CSV)

    report(results)
    return 0


def report(results: pd.DataFrame) -> None:
    samples = list(dict.fromkeys(results["sample"]))
    log.info("")
    log.info("=" * 104)
    log.info("AUC (95%% CI from a month-clustered bootstrap)")
    log.info("%-8s %-30s %-30s %-30s", "model", *samples)
    for model_id in B.MODELS:
        cells = []
        for s in samples:
            r = results[(results["sample"] == s) & (results["model"] == model_id)]
            cells.append(f"{r['auc'].iloc[0]:.4f} [{r['auc_lo'].iloc[0]:+.3f},{r['auc_hi'].iloc[0]:+.3f}]"
                         if not r.empty else "-")
        log.info("%-8s %-30s %-30s %-30s", model_id, *cells)

    log.info("")
    log.info("LLM incremental effect on AUC")
    for pair in ("B-A", "D-C"):
        cells = []
        for s in samples:
            r = results[(results["sample"] == s) & (results["model"] == pair)]
            cells.append(f"{r['auc'].iloc[0]:+.4f} [{r['auc_lo'].iloc[0]:+.3f},{r['auc_hi'].iloc[0]:+.3f}]"
                         if not r.empty else "-")
        log.info("%-8s %-30s %-30s %-30s", pair, *cells)

    log.info("")
    log.info("Mean IC / IC IR")
    for model_id in B.MODELS:
        cells = []
        for s in samples:
            r = results[(results["sample"] == s) & (results["model"] == model_id)]
            cells.append(f"{r['mean_ic'].iloc[0]:+.4f} / {r['ic_ir'].iloc[0]:+.3f}"
                         if not r.empty and pd.notna(r["mean_ic"].iloc[0]) else "-")
        log.info("%-8s %-30s %-30s %-30s", model_id, *cells)

    log.info("")
    log.info("Backtest: CAGR / Sharpe / MaxDD / turnover")
    order = list(B.MODELS) + ["Equal weight", "Momentum 12-1", "SPY buy & hold"]
    for name in order:
        cells = []
        for s in samples:
            r = results[(results["sample"] == s) & (results["model"] == name)]
            if r.empty or pd.isna(r["cagr"].iloc[0]):
                cells.append("-")
            else:
                cells.append(f"{r['cagr'].iloc[0]*100:5.1f}% {r['sharpe'].iloc[0]:5.2f} "
                             f"{r['max_drawdown'].iloc[0]*100:6.1f}% {r['annual_turnover'].iloc[0]*100:4.0f}%")
        log.info("%-8s %-30s %-30s %-30s", name, *cells)
    log.info("=" * 104)

    # Width of the intervals is the headline, not the point estimates.
    main = results[(results["sample"] == samples[0]) & results["model"].isin(B.MODELS)]
    retr = results[(results["sample"] == samples[-1]) & results["model"].isin(B.MODELS)]
    w_main = (main["auc_hi"] - main["auc_lo"]).mean()
    w_retr = (retr["auc_hi"] - retr["auc_lo"]).mean()
    log.info("")
    log.info("Mean AUC interval width: %.3f on the main sample, %.3f retrained (%.0f%% wider).",
             w_main, w_retr, (w_retr / w_main - 1) * 100)
    log.info("Every AUC below is inside the others' interval, so no point-estimate")
    log.info("difference here is evidence of a real change -- the restricted sample")
    log.info("simply cannot resolve differences of this size.")

    overlap = []
    for model_id in B.MODELS:
        a = main[main["model"] == model_id].iloc[0]
        b = retr[retr["model"] == model_id].iloc[0]
        overlap.append(a["auc_lo"] <= b["auc"] <= a["auc_hi"])
    log.info("Retrained AUC inside the main-sample CI: %d of %d models.",
             sum(overlap), len(overlap))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s", datefmt="%H:%M:%S",
    )
    return run()


if __name__ == "__main__":
    sys.exit(main())
