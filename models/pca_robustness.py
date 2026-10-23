"""PCA macro robustness: does compressing the macro block let the LLM in?

The null result so far could have a boring cause. The 14 macro features are
strongly correlated with each other -- VIX with credit spreads, the term spread
with the funds rate, industrial production with payrolls. A logistic regression
handed a collinear block spends its coefficients unstably, and a noisy,
redundant representation can drown out a small independent signal sitting beside
it. If that is what is happening, then replacing the raw macro block with
orthogonal principal components should give the LLM sentiment column room to
contribute, and the incremental delta should appear.

This script tests exactly that, and nothing else.

Usage
-----
    python models/pca_robustness.py
    python models/pca_robustness.py --skip-xgboost      # logistic pairs only
    python models/pca_robustness.py --variance 0.95     # different pre-set rule

Outputs
-------
    models/pca_robustness.csv
    viz/pca_variance.png

What is compressed, and what is not
-----------------------------------
PCA is applied to the **14 lagged FRED macro features only**. Deliberately left
out of the decomposition:

    LLM sentiment          -- compressing it with macro would destroy the very
                              separation the experiment is about
    sector momentum / vol  -- sector-level, not macro; already low-dimensional
    cross-sectional ranks  -- already normalised within each month
    sector dummies         -- indicators, not continuous macro variables
    interaction terms      -- products of macro and dummies (see below)

The raw macro block is **not modified**. Models A and B are read back from
``models/predictions.csv`` exactly as ``train.py`` produced them, so the raw arm
of this comparison is the real headline specification, not a re-fit of it.

The interaction problem, and how it is handled
----------------------------------------------
The raw specification carries eight ``macro x sector`` interaction terms. Those
are products of the *raw* macro columns, so carrying them unchanged into a "PCA"
specification would smuggle the raw macro block back in through the side door
and defeat the test. Dropping them instead confounds the comparison the other
way: the PCA arm would lose the ability to let sectors respond differently to
macro, and any performance gap could be that rather than the compression.

Both choices are therefore run, and both are fixed in advance:

    pca         PC scores + PC1 x each sector dummy   (structural parity)
    pca_noint   PC scores, no interactions            (sensitivity)

Neither is selected on results. If they disagree, that is reported.

Walk-forward discipline
-----------------------
For prediction month *P*, with the same one-month embargo used everywhere else
in this project, the scaler and the PCA rotation are fitted on training months
strictly before *P-1* and then applied unchanged to month *P*. Nothing is ever
fitted on the full sample. Because the macro block is a time series broadcast
across the 11 sectors, the scaler and PCA are fitted on the **deduplicated
monthly** macro matrix rather than on the panel: fitting on panel rows would
count every month eleven times and quietly down-weight the pre-2018 months that
have only ten sectors.

Component count is chosen by a pre-specified rule -- the smallest *k* reaching
90% cumulative explained variance **on the training window** -- never by looking
at out-of-sample performance. The count is allowed to drift as the window grows
and is reported month by month.

Component signs are stabilised (each component is flipped so its largest-
magnitude loading is positive). Sign is arbitrary in PCA and would otherwise
flip between refits; the models refit monthly so it would not change their
predictions, but it would make the PC1 interaction terms and the reported
loadings incoherent across months.

One consequence worth stating: the logistic pipeline standardises its inputs, so
the PC scores are re-scaled to unit variance before the fit. The specification is
therefore *whitened* macro, not merely rotated macro -- PC6 is penalised on the
same footing as PC1 even though it carries a fraction of the variance. That is
the same treatment the raw columns get, which is the point of keeping it, but it
means low-variance components are not automatically down-weighted.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "common"))

from plotting import DPI  # noqa: E402
sys.path.insert(0, str(REPO_ROOT / "models"))
sys.path.insert(0, str(REPO_ROOT / "backtest"))

import ranking_metrics as rank  # noqa: E402
import run_backtest as bt  # noqa: E402
import train  # noqa: E402

DATASET = REPO_ROOT / "data" / "model_dataset.csv"
PREDICTIONS = REPO_ROOT / "models" / "predictions.csv"
OUT_CSV = REPO_ROOT / "models" / "pca_robustness.csv"
PLOT_PNG = REPO_ROOT / "viz" / "pca_variance.png"

VARIANCE_THRESHOLD = 0.90
TOP_N = 3
COST_BPS = 10.0
N_BOOT = 2_000
RANDOM_STATE = 42

# spec key -> (estimator family, macro representation, carries LLM block, label)
SPECS = {
    "raw_base":       ("logistic", "raw",       False, "Raw macro (= Model A)"),
    "raw_llm":        ("logistic", "raw",       True,  "Raw macro + LLM (= Model B)"),
    "pca_base":       ("logistic", "pca",       False, "PCA macro"),
    "pca_llm":        ("logistic", "pca",       True,  "PCA macro + LLM"),
    "pca_noint_base": ("logistic", "pca_noint", False, "PCA macro, no interactions"),
    "pca_noint_llm":  ("logistic", "pca_noint", True,  "PCA macro + LLM, no interactions"),
    "raw_base_xgb":   ("xgboost",  "raw",       False, "Raw macro (= Model C)"),
    "raw_llm_xgb":    ("xgboost",  "raw",       True,  "Raw macro + LLM (= Model D)"),
    "pca_base_xgb":   ("xgboost",  "pca",       False, "PCA macro"),
    "pca_llm_xgb":    ("xgboost",  "pca",       True,  "PCA macro + LLM"),
}

# The raw arms are not re-fitted; they are the existing walk-forward output.
REUSED = {"raw_base": "model_A_prob", "raw_llm": "model_B_prob",
          "raw_base_xgb": "model_C_prob", "raw_llm_xgb": "model_D_prob"}

# (base, llm, family label) -- the incremental-LLM question, once per arm.
PAIRS = [
    ("raw_base", "raw_llm", "Logistic / raw macro"),
    ("pca_base", "pca_llm", "Logistic / PCA macro"),
    ("pca_noint_base", "pca_noint_llm", "Logistic / PCA, no interactions"),
    ("raw_base_xgb", "raw_llm_xgb", "XGBoost / raw macro"),
    ("pca_base_xgb", "pca_llm_xgb", "XGBoost / PCA macro"),
]

log = logging.getLogger("pca_robustness")


# --------------------------------------------------------------------------
# The macro block
# --------------------------------------------------------------------------

def macro_columns(data: pd.DataFrame) -> list[str]:
    """The 14 lagged FRED columns, identified the same way ``train.py`` does."""
    cols = list(data.columns)
    macro = [
        c for c in cols
        if c.endswith(("_lag1", "_lag2"))
        and c not in train.SECTOR_FEATURES + train.LLM_FEATURES
        and "_x_is_" not in c
    ]
    macro.sort(key=cols.index)
    return macro


def monthly_macro(data: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """One row per month. Errors loudly if macro is not constant within a month.

    The whole deduplication argument rests on that invariant, so it is checked
    rather than assumed.
    """
    spread = data.groupby("date")[cols].nunique(dropna=False)
    offenders = spread[(spread > 1).any(axis=1)]
    if not offenders.empty:
        raise ValueError(
            f"macro features vary within {len(offenders)} month(s) -- they are "
            f"supposed to be time series broadcast across sectors: "
            f"{offenders.index[0].date()}"
        )
    out = data.drop_duplicates(subset="date").set_index("date")[cols].sort_index()
    return out


def fit_month_pca(train_macro: np.ndarray, threshold: float):
    """Scaler + PCA fitted on training months only, with a pre-set component rule."""
    scaler = StandardScaler().fit(train_macro)
    pca = PCA(n_components=None, random_state=RANDOM_STATE).fit(scaler.transform(train_macro))

    ratios = pca.explained_variance_ratio_
    cum = np.cumsum(ratios)
    k = int(min(np.searchsorted(cum, threshold) + 1, len(ratios)))

    # Fix each component's sign by its largest-magnitude loading.
    lead = np.argmax(np.abs(pca.components_), axis=1)
    signs = np.where(pca.components_[np.arange(len(lead)), lead] < 0, -1.0, 1.0)
    return scaler, pca, ratios, cum, k, signs


def pc_scores(scaler, pca, signs, macro: pd.DataFrame, k: int) -> pd.DataFrame:
    """Project monthly macro rows onto the first k sign-stabilised components."""
    z = pca.transform(scaler.transform(macro.to_numpy(float))) * signs
    names = [f"PC{i + 1:02d}" for i in range(k)]
    return pd.DataFrame(z[:, :k], index=macro.index, columns=names)


# --------------------------------------------------------------------------
# Walk-forward
# --------------------------------------------------------------------------

def build_pca_frame(rows: pd.DataFrame, scores: pd.DataFrame, dummies: list[str],
                    with_interactions: bool) -> tuple[pd.DataFrame, list[str]]:
    """Attach PC scores (and optional PC1 x sector terms) to a set of panel rows."""
    frame = rows.join(scores, on="date")
    cols = list(scores.columns)
    if with_interactions:
        for d in dummies:
            name = f"PC01_x_{d}"
            frame[name] = frame["PC01"] * frame[d]
            cols.append(name)
    return frame, cols


def walk_forward(data: pd.DataFrame, oos_dates: list[pd.Timestamp], macro: list[str],
                 threshold: float, embargo: int, retune_every: int, n_splits: int,
                 fit_specs: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Fit every PCA specification month by month. Raw specs are not touched."""
    monthly = monthly_macro(data, macro)
    sets = train.feature_sets(data)
    dummies = [c for c in data.columns if c.startswith("is_")]
    # Everything in the raw spec that is not a macro column: sector features,
    # ranks, dummies, and (for the parity variant) the interaction slots that
    # PC1 x dummy replaces.
    non_macro = [c for c in sets["macro"] if c not in macro and "_x_is_" not in c]

    records: list[dict] = []
    variance_rows: list[dict] = []
    cached: dict[str, dict] = {}
    last_tuned: dict[str, int] = {}
    final_scree: dict = {}
    started = time.monotonic()

    for i, pred_date in enumerate(oos_dates):
        cutoff = train.train_cutoff(pred_date, embargo)
        train_months = monthly.index[monthly.index < cutoff]
        if len(train_months) < 2:
            continue

        scaler, pca, ratios, cum, k, signs = fit_month_pca(
            monthly.loc[train_months].to_numpy(float), threshold
        )
        variance_rows.append({
            "pred_date": pred_date,
            "train_months": len(train_months),
            "n_components": k,
            "cum_var_at_k": float(cum[k - 1]),
            "var_pc1": float(ratios[0]),
            "var_pc1_pc2": float(cum[1]) if len(cum) > 1 else np.nan,
        })
        final_scree = {"pred_date": pred_date, "ratios": ratios, "cum": cum,
                       "k": k, "train_months": len(train_months),
                       "loadings": pca.components_ * signs[:, None]}

        # The projection is fitted on training months and then applied
        # unchanged; month P is transformed, never fitted on.
        scores_all = pc_scores(scaler, pca, signs, monthly, k)

        train_rows = data[data["date"] < cutoff]
        test_rows = data[data["date"] == pred_date]
        month_probs = {s: {} for s in fit_specs}

        for spec in fit_specs:
            family, macro_rep, with_llm, _ = SPECS[spec]
            inter = macro_rep == "pca"
            tr, pca_cols = build_pca_frame(train_rows, scores_all, dummies, inter)
            te, _ = build_pca_frame(test_rows, scores_all, dummies, inter)
            cols = pca_cols + non_macro + (train.LLM_FEATURES if with_llm else [])

            tr = tr.dropna(subset=cols + ["target"])
            te = te.dropna(subset=cols)
            if tr.empty or te.empty:
                continue
            y_tr = tr["target"].astype(int)
            if y_tr.nunique() < 2:
                log.warning("  %s %s: single-class training set, skipped",
                            pred_date.date(), spec)
                continue

            # Retuning cadence matches train.py so the comparison is like for
            # like; the grid is retuned on the PCA columns, not reused from raw.
            if spec not in cached or i - last_tuned.get(spec, -10**9) >= retune_every:
                cached[spec] = train.tune(family, tr[cols], y_tr, tr["date"].values, n_splits)
                last_tuned[spec] = i

            model = train.fit_final(family, cached[spec], tr[cols], y_tr)
            proba = model.predict_proba(te[cols])[:, 1]
            month_probs[spec] = dict(zip(te["ticker"], (float(p) for p in proba)))

        for _, row in test_rows.iterrows():
            rec = {"date": pred_date, "ticker": row["ticker"]}
            for spec in fit_specs:
                rec[f"prob_{spec}"] = month_probs[spec].get(row["ticker"], np.nan)
            records.append(rec)

        log.info("[%2d/%d] %s  %3d train months  k=%2d PCs (%.1f%% var)  %.0fs",
                 i + 1, len(oos_dates), pred_date.date(), len(train_months), k,
                 100 * cum[k - 1], time.monotonic() - started)

    return pd.DataFrame(records), pd.DataFrame(variance_rows), final_scree


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def month_index(frame: pd.DataFrame) -> tuple[np.ndarray, dict]:
    months = frame["date"].to_numpy()
    uniq = np.unique(months)
    return uniq, {m: np.flatnonzero(months == m) for m in uniq}


def bootstrap_auc(frame: pd.DataFrame, col: str, n_boot: int, seed: int) -> tuple[float, float, float]:
    """AUC with a month-clustered bootstrap CI.

    Months are resampled whole: sectors inside a month share a macro state and
    their targets are defined against each other, so resampling rows would treat
    dependent observations as independent.
    """
    sub = frame.dropna(subset=[col, "actual_target"])
    y = sub["actual_target"].to_numpy(int)
    p = sub[col].to_numpy(float)
    uniq, idx = month_index(sub)
    rng = np.random.default_rng(seed)

    point = float(roc_auc_score(y, p))
    draws = []
    for _ in range(n_boot):
        ii = np.concatenate([idx[m] for m in rng.choice(uniq, len(uniq), replace=True)])
        if len(np.unique(y[ii])) < 2:
            continue
        draws.append(roc_auc_score(y[ii], p[ii]))
    return point, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def bootstrap_delta(frame: pd.DataFrame, base: str, llm: str,
                    n_boot: int, seed: int) -> dict:
    """Paired month-clustered CI on the LLM increment, for AUC and mean IC."""
    sub = frame.dropna(subset=[base, llm, "actual_target", "excess"])
    y = sub["actual_target"].to_numpy(int)
    pb, pl = sub[base].to_numpy(float), sub[llm].to_numpy(float)
    ex = sub["excess"].to_numpy(float)
    uniq, idx = month_index(sub)
    rng = np.random.default_rng(seed)

    from scipy import stats as _st

    def ic_of(months_drawn, scores) -> float:
        """Mean monthly Spearman IC over whichever months a draw contains.

        A resampled month appears as many times as it was drawn, which is what
        gives the mean IC its sampling variability.
        """
        vals = []
        for m in months_drawn:
            j = idx[m]
            if len(j) < 3 or np.all(scores[j] == scores[j][0]):
                continue
            vals.append(_st.spearmanr(scores[j], ex[j]).statistic)
        return float(np.mean(vals)) if vals else np.nan

    all_months = list(uniq)
    point_auc = roc_auc_score(y, pl) - roc_auc_score(y, pb)
    point_ic = ic_of(all_months, pl) - ic_of(all_months, pb)

    d_auc, d_ic = [], []
    for _ in range(n_boot):
        drawn = rng.choice(uniq, len(uniq), replace=True)
        ii = np.concatenate([idx[m] for m in drawn])
        if len(np.unique(y[ii])) >= 2:
            d_auc.append(roc_auc_score(y[ii], pl[ii]) - roc_auc_score(y[ii], pb[ii]))
        d_ic.append(ic_of(drawn, pl) - ic_of(drawn, pb))

    d_ic = [v for v in d_ic if np.isfinite(v)]
    return {
        "delta_auc": float(point_auc),
        "delta_auc_lo": float(np.percentile(d_auc, 2.5)),
        "delta_auc_hi": float(np.percentile(d_auc, 97.5)),
        "delta_ic": float(point_ic),
        "delta_ic_lo": float(np.percentile(d_ic, 2.5)),
        "delta_ic_hi": float(np.percentile(d_ic, 97.5)),
    }


def portfolio(frame: pd.DataFrame, col: str, returns_by_date: dict,
              rf: pd.Series, top_n: int, cost_bps: float) -> dict:
    scores = frame[["date", "ticker", col]].dropna()
    sched = bt.top_n_weights(scores, col, top_n)
    r, turn = bt.simulate(sched, returns_by_date, cost_bps)
    return bt.performance(r, rf, turn)


def evaluate_spec(frame: pd.DataFrame, spec: str, col: str, returns_by_date: dict,
                  rf: pd.Series, n_boot: int, top_n: int, cost_bps: float) -> dict:
    family, macro_rep, with_llm, label = SPECS[spec]
    auc, lo, hi = bootstrap_auc(frame, col, n_boot, RANDOM_STATE)
    _, rank_summary = rank.evaluate(frame.rename(columns={col: "_score"}), "_score")
    perf = portfolio(frame, col, returns_by_date, rf, top_n, cost_bps)
    return {
        "spec": spec,
        "label": label,
        "estimator": family,
        "macro_representation": macro_rep,
        "llm_block": with_llm,
        "n_predictions": int(frame[col].notna().sum()),
        "auc": auc, "auc_ci_lo": lo, "auc_ci_hi": hi,
        "mean_ic": rank_summary["mean_ic"],
        "ic_ir": rank_summary["ic_ir"],
        "precision_at_3": rank_summary["precision_at_3"],
        "ndcg_at_3": rank_summary["ndcg_at_3"],
        "sharpe": perf["sharpe"],
        "cagr": perf["cagr"],
        "max_drawdown": perf["max_drawdown"],
        "annual_turnover": perf["annual_turnover"],
    }


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def run(threshold: float, skip_xgb: bool, retune_every: int, n_splits: int,
        min_train_months: int, embargo: int, n_boot: int, oos_start: str) -> int:
    for path in (DATASET, PREDICTIONS):
        if not path.exists():
            log.error("%s not found -- run the upstream script first", path)
            return 1

    data = pd.read_csv(DATASET, parse_dates=["date"]).sort_values(["date", "ticker"])
    preds = pd.read_csv(PREDICTIONS, parse_dates=["date"])
    macro = macro_columns(data)
    log.info("macro block for PCA: %d features", len(macro))
    log.info("  %s", ", ".join(macro))
    log.info("excluded from PCA: LLM sentiment (%d), sector features (%d), ranks (%d), "
             "%d dummies, %d interactions",
             len(train.LLM_FEATURES), len(train.SECTOR_FEATURES), len(train.RANK_FEATURES),
             sum(c.startswith("is_") for c in data.columns),
             sum("_x_is_" in c for c in data.columns))

    # Same prediction months as train.py, derived the same way.
    all_dates = sorted(data["date"].unique())
    oos_dates = [
        d for d in all_dates
        if d >= pd.Timestamp(oos_start)
        and len([x for x in all_dates if x < train.train_cutoff(d, embargo)]) >= min_train_months
    ]
    existing = set(preds["date"].unique())
    if set(oos_dates) != existing:
        log.warning("prediction months differ from predictions.csv (%d vs %d) -- "
                    "the raw arm may not be directly comparable",
                    len(oos_dates), len(existing))
    log.info("walk-forward: %s -> %s (%d prediction months), %d-month embargo",
             oos_dates[0].date(), oos_dates[-1].date(), len(oos_dates), embargo)
    log.info("component rule (pre-specified): smallest k reaching %.0f%% training "
             "explained variance", 100 * threshold)
    log.info("")

    fit_specs = [s for s, v in SPECS.items()
                 if s not in REUSED and not (skip_xgb and v[0] == "xgboost")]
    log.info("fitting %d PCA specification(s): %s", len(fit_specs), ", ".join(fit_specs))
    log.info("reusing walk-forward output for the raw arm: %s",
             ", ".join(f"{k}={v}" for k, v in REUSED.items()))
    log.info("")

    fitted, variance, scree = walk_forward(
        data, oos_dates, macro, threshold, embargo, retune_every, n_splits, fit_specs
    )
    if fitted.empty:
        log.error("no predictions produced")
        return 1

    # Assemble one frame: fitted PCA columns beside the reused raw columns.
    frame = fitted.merge(
        preds[["date", "ticker", "actual_target", "actual_return"] + list(REUSED.values())],
        on=["date", "ticker"], how="inner",
    ).rename(columns={v: f"prob_{k}" for k, v in REUSED.items()})
    frame = frame.merge(
        data[["date", "ticker", "y_excess_return"]].rename(columns={"y_excess_return": "excess"}),
        on=["date", "ticker"], how="left",
    )

    active = [s for s in SPECS
              if f"prob_{s}" in frame.columns and frame[f"prob_{s}"].notna().any()]
    log.info("")
    log.info("scoring %d specification(s) on %d rows over %d months",
             len(active), len(frame), frame["date"].nunique())

    returns_by_date = {
        d: dict(zip(g["ticker"], g["actual_return"]))
        for d, g in frame.groupby("date")
    }
    rf = bt.risk_free_monthly(bt.next_month_end(sorted(frame["date"].unique())))

    results = pd.DataFrame([
        evaluate_spec(frame, s, f"prob_{s}", returns_by_date, rf, n_boot, TOP_N, COST_BPS)
        for s in active
    ])

    comps = []
    for base, llm, label in PAIRS:
        if base not in active or llm not in active:
            continue
        d = bootstrap_delta(frame, f"prob_{base}", f"prob_{llm}", n_boot, RANDOM_STATE)
        r = results.set_index("spec")
        d.update({
            "arm": label, "base": base, "llm": llm,
            "delta_sharpe": r.loc[llm, "sharpe"] - r.loc[base, "sharpe"],
            "delta_cagr": r.loc[llm, "cagr"] - r.loc[base, "cagr"],
            "delta_precision_at_3": r.loc[llm, "precision_at_3"] - r.loc[base, "precision_at_3"],
            "delta_ndcg_at_3": r.loc[llm, "ndcg_at_3"] - r.loc[base, "ndcg_at_3"],
            "auc_ci_excludes_zero": bool(d["delta_auc_lo"] > 0 or d["delta_auc_hi"] < 0),
        })
        comps.append(d)
    comps = pd.DataFrame(comps)

    write_csv(results, variance, comps, threshold)
    report(results, variance, comps, threshold, scree, macro)
    plot(variance, scree, macro, threshold)
    return 0


def write_csv(results: pd.DataFrame, variance: pd.DataFrame, comps: pd.DataFrame,
              threshold: float) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8") as fh:
        fh.write(f"# PCA applied to the 14 lagged FRED macro features only; "
                 f"{threshold:.0%} training explained variance rule\n")
        fh.write("# raw_* specs are the existing walk-forward output, not re-fitted\n")
        fh.write("# SPECIFICATIONS\n")
        results.to_csv(fh, index=False, float_format="%.6f")
        fh.write("\n# LLM INCREMENT (paired, month-clustered bootstrap)\n")
        comps.to_csv(fh, index=False, float_format="%.6f")
        fh.write("\n# COMPONENT COUNT PER PREDICTION MONTH\n")
        variance.to_csv(fh, index=False, float_format="%.6f")
    log.info("Wrote %s", OUT_CSV)


def report(results: pd.DataFrame, variance: pd.DataFrame, comps: pd.DataFrame,
           threshold: float, scree: dict, macro: list[str]) -> None:
    line = "=" * 112
    log.info("")
    log.info(line)
    log.info("COMPONENT SELECTION -- %.0f%% training explained variance, chosen before "
             "any out-of-sample look", 100 * threshold)
    log.info("  components used: mean %.2f, min %d, max %d (of %d macro features)",
             variance["n_components"].mean(), int(variance["n_components"].min()),
             int(variance["n_components"].max()), len(macro))
    log.info("  variance actually captured at k: mean %.4f, min %.4f, max %.4f",
             variance["cum_var_at_k"].mean(), variance["cum_var_at_k"].min(),
             variance["cum_var_at_k"].max())
    log.info("  PC1 alone: mean %.4f | PC1+PC2: mean %.4f",
             variance["var_pc1"].mean(), variance["var_pc1_pc2"].mean())
    counts = variance["n_components"].value_counts().sort_index()
    log.info("  distribution: %s",
             ", ".join(f"k={int(k)} in {int(v)} month(s)" for k, v in counts.items()))
    log.info("  final window (%s, %d training months): k=%d, cumulative %.4f",
             scree["pred_date"].date(), scree["train_months"], scree["k"],
             scree["cum"][scree["k"] - 1])
    log.info("  It takes %d of %d components to reach %.0f%%, which is itself the finding:",
             scree["k"], len(macro), 100 * threshold)
    log.info("  the macro block is correlated but not nearly rank-1.")

    log.info("")
    log.info(line)
    log.info("SPECIFICATIONS")
    log.info("%-16s %-34s %7s %18s %8s %7s %7s %7s %7s %8s",
             "spec", "label", "AUC", "95% CI", "mean IC", "IC IR", "P@3", "NDCG@3",
             "Sharpe", "CAGR")
    for _, r in results.iterrows():
        log.info("%-16s %-34s %7.4f  [%.4f, %.4f] %8.4f %7.3f %7.3f %7.3f %7.2f %7.2f%%",
                 r["spec"], r["label"], r["auc"], r["auc_ci_lo"], r["auc_ci_hi"],
                 r["mean_ic"], r["ic_ir"], r["precision_at_3"], r["ndcg_at_3"],
                 r["sharpe"], 100 * r["cagr"])
    log.info("  AUC 0.50 = no ranking skill. A CI spanning 0.50 cannot reject that.")

    log.info("")
    log.info(line)
    log.info("DOES THE LLM ADD VALUE?  (LLM spec minus base spec, same arm)")
    log.info("%-32s %10s %20s %10s %20s %8s %8s",
             "arm", "ΔAUC", "95% CI", "ΔIC", "95% CI", "ΔSharpe", "ΔCAGR")
    for _, r in comps.iterrows():
        log.info("%-32s %+10.4f  [%+.4f, %+.4f] %+10.4f  [%+.4f, %+.4f] %+8.2f %+7.2f%%",
                 r["arm"], r["delta_auc"], r["delta_auc_lo"], r["delta_auc_hi"],
                 r["delta_ic"], r["delta_ic_lo"], r["delta_ic_hi"],
                 r["delta_sharpe"], 100 * r["delta_cagr"])
    log.info("  positive = the LLM version is better. A CI containing zero means")
    log.info("  the increment is not distinguishable from nothing.")

    log.info("")
    log.info(line)
    log.info("VERDICT")
    signif = comps[comps["auc_ci_excludes_zero"]]
    pca_arms = comps[comps["base"].str.startswith("pca")]
    raw_arms = comps[~comps["base"].str.startswith("pca")]
    if signif.empty:
        log.info("  No arm shows an LLM increment distinguishable from zero: every ΔAUC")
        log.info("  interval contains zero, under raw macro features and under PCA alike.")
    else:
        for _, r in signif.iterrows():
            direction = "HELPS" if r["delta_auc"] > 0 else "HURTS"
            log.info("  %s: the LLM increment %s and its ΔAUC interval excludes zero "
                     "(%+.4f).", r["arm"], direction, r["delta_auc"])
    for name, arms in (("PCA", pca_arms), ("raw macro", raw_arms)):
        if arms.empty:
            continue
        best = arms.loc[arms["delta_auc"].idxmax()]
        log.info("  Best LLM increment under %s: %+.4f AUC (%s)",
                 name, best["delta_auc"], best["arm"])
    log.info("")
    log.info("  Reading: multicollinearity in the raw macro block is only an")
    log.info("  explanation for the null if orthogonalising it changes the answer.")
    log.info("  The two PCA variants (with and without PC1 x sector interactions)")
    log.info("  are a check that the result does not hinge on that design choice.")
    log.info(line)


# --------------------------------------------------------------------------
# Plot
# --------------------------------------------------------------------------

def plot(variance: pd.DataFrame, scree: dict, macro: list[str],
         threshold: float = VARIANCE_THRESHOLD) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ratios, cum, k = scree["ratios"], scree["cum"], scree["k"]
    x = np.arange(1, len(ratios) + 1)

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 9.5))

    bars = ax.bar(x, 100 * ratios, color="#B8C4D9", edgecolor="white", zorder=2)
    for b in bars[:k]:
        b.set_color("#4C78A8")
    ax.set_xticks(x)
    ax.set_xlabel("principal component", fontsize=9)
    ax.set_ylabel("explained variance (%)", fontsize=9)
    ax.set_ylim(0, max(100 * ratios) * 1.25)

    axc = ax.twinx()
    axc.plot(x, 100 * cum, "o-", color="#E45756", linewidth=2, markersize=5,
             zorder=3, label="cumulative")
    axc.axhline(100 * threshold, color="black", linestyle="--", linewidth=1.2,
                zorder=1, label=f"{threshold:.0%} threshold")
    axc.axvline(k, color="#E45756", linestyle=":", linewidth=1.4, zorder=1)
    axc.annotate(f"k = {k}\n{100 * cum[k - 1]:.1f}%",
                 xy=(k, 100 * cum[k - 1]), xytext=(10, -28),
                 textcoords="offset points", fontsize=9, color="#E45756", weight="bold")
    axc.set_ylabel("cumulative explained variance (%)", fontsize=9)
    axc.set_ylim(0, 105)
    axc.legend(fontsize=8, loc="lower right", frameon=False)
    ax.set_title(f"Scree — final training window ({scree['pred_date'].date()}, "
                 f"{scree['train_months']} months, {len(macro)} macro features)\n"
                 f"filled bars are the components retained by the "
                 f"{threshold:.0%} rule",
                 fontsize=10.5)
    ax.grid(alpha=0.2, axis="y", zorder=0)
    for side in ("top",):
        ax.spines[side].set_visible(False)
        axc.spines[side].set_visible(False)

    ax2.step(variance["pred_date"], variance["n_components"], where="post",
             color="#4C78A8", linewidth=2, label="components retained (k)")
    ax2.fill_between(variance["pred_date"], 0, variance["n_components"], step="post",
                     color="#4C78A8", alpha=0.12)
    ax2.set_ylabel("components retained", fontsize=9, color="#4C78A8")
    ax2.set_ylim(0, len(macro))
    ax2.set_xlabel("prediction month", fontsize=9)

    ax2v = ax2.twinx()
    ax2v.plot(variance["pred_date"], 100 * variance["cum_var_at_k"],
              color="#E45756", linewidth=1.6, label="variance captured at k")
    ax2v.axhline(100 * threshold, color="black", linestyle="--", linewidth=1.0)
    # The dashed line lives on the right-hand axis. Without this label it sits
    # at roughly y=4.6 on the left-hand scale and reads as a floor on k.
    ax2v.annotate(f"{threshold:.0%} rule (right axis)",
                  xy=(variance["pred_date"].iloc[1], 100 * threshold),
                  xytext=(0, 4), textcoords="offset points", fontsize=8)
    ax2v.set_ylabel("variance captured (%)", fontsize=9, color="#E45756")
    ax2v.set_ylim(85, 100)

    handles = ax2.get_legend_handles_labels()[0] + ax2v.get_legend_handles_labels()[0]
    labels = ax2.get_legend_handles_labels()[1] + ax2v.get_legend_handles_labels()[1]
    ax2.legend(handles, labels, fontsize=8, loc="lower right", frameon=False)
    ax2.set_title("Component count across the walk-forward — the rule is re-applied "
                  "on each expanding training window", fontsize=10.5)
    ax2.grid(alpha=0.2)
    for side in ("top",):
        ax2.spines[side].set_visible(False)
        ax2v.spines[side].set_visible(False)

    fig.suptitle("PCA of the macro block — fitted on training months only, at every step",
                 fontsize=13, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    PLOT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_PNG, dpi=DPI)
    plt.close(fig)
    log.info("Wrote %s", PLOT_PNG)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variance", type=float, default=VARIANCE_THRESHOLD,
                        help="training explained-variance threshold (default 0.90)")
    parser.add_argument("--skip-xgboost", action="store_true",
                        help="run the logistic arms only")
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
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    return run(args.variance, args.skip_xgboost, args.retune_every, args.cv_splits,
               args.min_train_months, args.embargo_months, args.n_boot, args.oos_start)


if __name__ == "__main__":
    sys.exit(main())
