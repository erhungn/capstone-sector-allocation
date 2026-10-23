"""The LLM increment across the specification ladder, on one consistent window.

    python models/spec_ladder.py
    python models/spec_ladder.py --skip-xgboost

Output
------
    models/spec_ladder.csv

Why this script exists
----------------------
The project's feature set grew in stages: a plain macro + sector-momentum panel,
then sector identity (dummies and macro x sector interactions), then four derived
views of the LLM score, and finally a PCA-compressed macro block as a robustness
check. At each stage the question "does the LLM feature help?" was answered
again, and **the sign of the answer moved around**.

That instability is a result worth reporting, but only if the rungs are
comparable. They were not. The earlier figures were produced during development
on a 94-month window, before the training embargo settled the schedule at 93
prediction months, so the sign flip could have been the window changing rather
than the specification. This recomputes every rung on the **current** 93-month
walk-forward, with the same embargo, the same tuning cadence, and the same
month-clustered inference, so the comparison is like for like.

The ladder
----------
    base        macro + sector momentum/vol + ranks          (20 features)
                +LLM adds sentiment_lag1 alone                    -> 21
    identity    + 10 sector dummies + 8 macro x sector terms  (38 features)
                +LLM adds sentiment_lag1 alone                    -> 39
    derived     as above, +LLM adds all five sentiment views  (38 -> 43)
    pca         macro replaced by principal components
                (read from models/pca_robustness.csv, not refitted)

Each rung differs from the one above it by exactly one change, and within a rung
the base and +LLM arms differ by exactly the LLM block. "derived" is the headline
specification -- its arms are Models A/B/C/D, reused from models/predictions.csv
rather than refitted, so this script cannot drift from the published results.

What is not claimed
-------------------
That any rung is better than another. Every increment here is small relative to
its confidence interval; the point is that the *sign* is not stable, which is
what a signal indistinguishable from noise looks like when it is measured
repeatedly.
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "models"))

import train  # noqa: E402

DATASET = REPO_ROOT / "data" / "model_dataset.csv"
PREDICTIONS = REPO_ROOT / "models" / "predictions.csv"
PCA_CSV = REPO_ROOT / "models" / "pca_robustness.csv"
OUT_CSV = REPO_ROOT / "models" / "spec_ladder.csv"

N_BOOT = 2_000
RANDOM_STATE = 42

# rung key -> (label, adds sector identity?, LLM block for the +LLM arm)
RUNGS = {
    "base":     ("Base (no sector identity)", False, ["sentiment_lag1"]),
    "identity": ("+ Sector identity", True, ["sentiment_lag1"]),
}
# The headline rung and the PCA rung are read from existing outputs.
REUSED_RUNGS = {
    "derived": ("+ Derived sentiment", "model_A_prob", "model_B_prob",
                "model_C_prob", "model_D_prob"),
}

log = logging.getLogger("spec_ladder")


# --------------------------------------------------------------------------
# Feature sets
# --------------------------------------------------------------------------

def ladder_features(data: pd.DataFrame, identity: bool, llm: list[str]) -> tuple[list, list]:
    """Return (base_columns, llm_columns) for one rung of the ladder."""
    cols = list(data.columns)
    macro = [c for c in cols
             if c.endswith(("_lag1", "_lag2"))
             and c not in train.SECTOR_FEATURES + train.LLM_FEATURES
             and "_x_is_" not in c]
    macro.sort(key=cols.index)
    base = macro + train.SECTOR_FEATURES + train.RANK_FEATURES
    if identity:
        base += [c for c in cols if c.startswith("is_")]
        base += [c for c in cols if "_x_is_" in c]
    return base, base + llm


# --------------------------------------------------------------------------
# Walk-forward
# --------------------------------------------------------------------------

def walk_forward(data: pd.DataFrame, oos_dates: list, embargo: int, retune_every: int,
                 n_splits: int, families: list[str]) -> pd.DataFrame:
    """Fit every rung x arm x estimator, month by month."""
    specs = {}
    for rung, (_, identity, llm) in RUNGS.items():
        base_cols, llm_cols = ladder_features(data, identity, llm)
        for family in families:
            tag = "log" if family == "logistic" else "xgb"
            specs[f"{rung}_{tag}_base"] = (family, base_cols)
            specs[f"{rung}_{tag}_llm"] = (family, llm_cols)

    log.info("fitting %d specification(s):", len(specs))
    for name, (family, cols) in specs.items():
        log.info("   %-22s %-9s %3d features", name, family, len(cols))
    log.info("")

    records, cached, last_tuned = [], {}, {}
    started = time.monotonic()

    for i, pred_date in enumerate(oos_dates):
        cutoff = train.train_cutoff(pred_date, embargo)
        train_rows = data[data["date"] < cutoff]
        test_rows = data[data["date"] == pred_date]
        month = {}

        for name, (family, cols) in specs.items():
            tr = train_rows.dropna(subset=cols + ["target"])
            te = test_rows.dropna(subset=cols)
            if tr.empty or te.empty:
                continue
            y = tr["target"].astype(int)
            if y.nunique() < 2:
                continue
            if name not in cached or i - last_tuned.get(name, -10**9) >= retune_every:
                cached[name] = train.tune(family, tr[cols], y, tr["date"].values, n_splits)
                last_tuned[name] = i
            model = train.fit_final(family, cached[name], tr[cols], y)
            month[name] = dict(zip(te["ticker"], model.predict_proba(te[cols])[:, 1]))

        for _, row in test_rows.iterrows():
            rec = {"date": pred_date, "ticker": row["ticker"]}
            for name in specs:
                rec[name] = month.get(name, {}).get(row["ticker"], np.nan)
            records.append(rec)

        log.info("[%2d/%d] %s  %.0fs", i + 1, len(oos_dates), pred_date.date(),
                 time.monotonic() - started)

    return pd.DataFrame(records)


# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------

def delta_auc_ci(frame: pd.DataFrame, base: str, llm: str,
                 n_boot: int, seed: int) -> dict:
    """AUC(+LLM) - AUC(base), with a paired month-clustered bootstrap CI.

    Months are resampled whole: sectors inside a month share a macro state and
    their targets are defined against one another, so resampling rows would
    treat dependent observations as independent and understate the interval.
    """
    sub = frame.dropna(subset=[base, llm, "actual_target"])
    if sub.empty:
        return {}
    y = sub["actual_target"].to_numpy(int)
    pb, pl = sub[base].to_numpy(float), sub[llm].to_numpy(float)
    months = sub["date"].to_numpy()
    uniq = np.unique(months)
    idx = {m: np.flatnonzero(months == m) for m in uniq}
    rng = np.random.default_rng(seed)

    auc_base, auc_llm = roc_auc_score(y, pb), roc_auc_score(y, pl)
    draws = []
    for _ in range(n_boot):
        ii = np.concatenate([idx[m] for m in rng.choice(uniq, len(uniq), replace=True)])
        if len(np.unique(y[ii])) < 2:
            continue
        draws.append(roc_auc_score(y[ii], pl[ii]) - roc_auc_score(y[ii], pb[ii]))
    lo, hi = float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))
    return {
        "n_obs": len(sub), "n_months": len(uniq),
        "auc_base": float(auc_base), "auc_llm": float(auc_llm),
        "delta_auc": float(auc_llm - auc_base),
        "ci_lo": lo, "ci_hi": hi,
        "ci_excludes_zero": bool(lo > 0 or hi < 0),
    }


def read_pca_rung() -> list[dict]:
    """Pull the PCA rung's two increments out of models/pca_robustness.csv."""
    if not PCA_CSV.exists():
        log.warning("%s not found -- the PCA rung will be omitted", PCA_CSV)
        return []
    blocks, name, buf = {}, None, []
    for line in PCA_CSV.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            if name and buf:
                blocks[name] = pd.read_csv(io.StringIO("\n".join(buf)))
            name, buf = line.lstrip("# ").strip(), []
        elif line.strip():
            buf.append(line)
    if name and buf:
        blocks[name] = pd.read_csv(io.StringIO("\n".join(buf)))

    inc = blocks.get("LLM INCREMENT (paired, month-clustered bootstrap)")
    specs = blocks.get("SPECIFICATIONS")
    if inc is None or specs is None:
        log.warning("unexpected layout in %s -- PCA rung omitted", PCA_CSV)
        return []
    specs = specs.set_index("spec")
    out = []
    for arm, base_spec, llm_spec in [("logistic", "pca_base", "pca_llm"),
                                     ("xgboost", "pca_base_xgb", "pca_llm_xgb")]:
        row = inc[inc["base"] == base_spec]
        if row.empty:
            continue
        r = row.iloc[0]
        out.append({
            "rung": "pca", "rung_label": "PCA macro", "estimator": arm,
            "source": "models/pca_robustness.csv",
            "n_obs": int(specs.loc[llm_spec, "n_predictions"]),
            "n_months": np.nan,
            "auc_base": float(specs.loc[base_spec, "auc"]),
            "auc_llm": float(specs.loc[llm_spec, "auc"]),
            "delta_auc": float(r["delta_auc"]),
            "ci_lo": float(r["delta_auc_lo"]), "ci_hi": float(r["delta_auc_hi"]),
            "ci_excludes_zero": bool(r["auc_ci_excludes_zero"]),
        })
    return out


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def run(skip_xgb: bool, retune_every: int, n_splits: int, min_train_months: int,
        embargo: int, n_boot: int, oos_start: str) -> int:
    if not DATASET.exists() or not PREDICTIONS.exists():
        log.error("run data/build_dataset.py and models/train.py first")
        return 1

    data = pd.read_csv(DATASET, parse_dates=["date"]).sort_values(["date", "ticker"])
    preds = pd.read_csv(PREDICTIONS, parse_dates=["date"])

    all_dates = sorted(data["date"].unique())
    oos_dates = [
        d for d in all_dates
        if d >= pd.Timestamp(oos_start)
        and len([x for x in all_dates if x < train.train_cutoff(d, embargo)]) >= min_train_months
    ]
    if set(oos_dates) != set(preds["date"].unique()):
        log.warning("schedule differs from predictions.csv -- rungs will not be comparable")
    log.info("walk-forward: %s -> %s (%d months), %d-month embargo",
             oos_dates[0].date(), oos_dates[-1].date(), len(oos_dates), embargo)

    families = ["logistic"] if skip_xgb else ["logistic", "xgboost"]
    fitted = walk_forward(data, oos_dates, embargo, retune_every, n_splits, families)

    frame = fitted.merge(
        preds[["date", "ticker", "actual_target", "model_A_prob", "model_B_prob",
               "model_C_prob", "model_D_prob"]],
        on=["date", "ticker"], how="inner")

    rows = []
    for rung, (label, _, _) in RUNGS.items():
        for family, tag in [("logistic", "log"), ("xgboost", "xgb")]:
            if skip_xgb and family == "xgboost":
                continue
            base, llm = f"{rung}_{tag}_base", f"{rung}_{tag}_llm"
            if base not in frame.columns:
                continue
            stats = delta_auc_ci(frame, base, llm, n_boot, RANDOM_STATE)
            if stats:
                rows.append({"rung": rung, "rung_label": label, "estimator": family,
                             "source": "refitted here", **stats})

    # The headline rung is Models A-D, reused rather than refitted.
    for family, b, l in [("logistic", "model_A_prob", "model_B_prob"),
                         ("xgboost", "model_C_prob", "model_D_prob")]:
        if skip_xgb and family == "xgboost":
            continue
        stats = delta_auc_ci(frame, b, l, n_boot, RANDOM_STATE)
        rows.append({"rung": "derived", "rung_label": REUSED_RUNGS["derived"][0],
                     "estimator": family, "source": "models/predictions.csv", **stats})

    rows.extend(r for r in read_pca_rung() if not (skip_xgb and r["estimator"] == "xgboost"))

    order = {"base": 0, "identity": 1, "derived": 2, "pca": 3}
    results = pd.DataFrame(rows).sort_values(
        ["rung", "estimator"], key=lambda s: s.map(order).fillna(s)).reset_index(drop=True)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8") as fh:
        fh.write("# LLM increment across the specification ladder\n")
        fh.write("# every rung on the same walk-forward window, embargo and inference\n")
        results.to_csv(fh, index=False, float_format="%.6f")
    log.info("Wrote %s", OUT_CSV)

    report(results)
    return 0


def report(results: pd.DataFrame) -> None:
    log.info("")
    log.info("=" * 100)
    log.info("LLM INCREMENT ACROSS SPECIFICATIONS  (AUC with LLM minus AUC without)")
    log.info("%-28s %-10s %10s %10s %11s %24s", "specification", "estimator",
             "AUC base", "AUC +LLM", "ΔAUC", "95% CI")
    for _, r in results.iterrows():
        log.info("%-28s %-10s %10.4f %10.4f %+11.4f   [%+.4f, %+.4f]%s",
                 r["rung_label"], r["estimator"], r["auc_base"], r["auc_llm"],
                 r["delta_auc"], r["ci_lo"], r["ci_hi"],
                 "  *" if r["ci_excludes_zero"] else "")
    log.info("  * = interval excludes zero")

    log.info("")
    for est in results["estimator"].unique():
        sub = results[results["estimator"] == est]
        signs = ["+" if v > 0 else "-" for v in sub["delta_auc"]]
        log.info("  %-9s sign across the ladder: %s   (range %+.4f to %+.4f)",
                 est, " ".join(signs), sub["delta_auc"].min(), sub["delta_auc"].max())
    flips = sum(1 for est in results["estimator"].unique()
                if results[results["estimator"] == est]["delta_auc"].gt(0).nunique() > 1)
    log.info("")
    log.info("  %d of %d estimator ladders change sign across specifications.",
             flips, results["estimator"].nunique())
    log.info("  A sign that moves with the feature set is noise, not signal.")
    log.info("=" * 100)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--skip-xgboost", action="store_true")
    parser.add_argument("--oos-start", default="2018-01-31")
    parser.add_argument("--retune-every", type=int, default=1)
    parser.add_argument("--cv-splits", type=int, default=3)
    parser.add_argument("--min-train-months", type=int, default=24)
    parser.add_argument("--embargo-months", type=int, default=1)
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s", datefmt="%H:%M:%S",
    )
    return run(args.skip_xgboost, args.retune_every, args.cv_splits,
               args.min_train_months, args.embargo_months, args.n_boot, args.oos_start)


if __name__ == "__main__":
    sys.exit(main())
