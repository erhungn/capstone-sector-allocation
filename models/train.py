"""Walk-forward training for the 2x2 capstone design.

Four models, crossing estimator against feature set, so the only difference
between A/B and between C/D is the single LLM sentiment column:

    A  Logistic Regression  macro + sector + ranks
    B  Logistic Regression  macro + sector + ranks + sentiment_lag1
    C  XGBoost              macro + sector + ranks
    D  XGBoost              macro + sector + ranks + sentiment_lag1

Usage
-----
    python models/train.py
    python models/train.py --retune-every 6      # tune less often, runs faster
    python models/train.py --oos-start 2020-01   # shorter walk-forward

Outputs
-------
    models/predictions.csv   out-of-sample probability per (date, ticker, model)
    models/metrics.csv       accuracy / precision / recall / AUC per model
    models/feature_names.json
    models/final_models.joblib   last-window fits, for SHAP

Walk-forward protocol, and the one-month training embargo
---------------------------------------------------------
For prediction month *P*, the model trains on rows dated on or before *P-2* and
predicts the rows dated *P*. The month *P-1* is deliberately withheld.

Why: a row dated *d* carries the return from *d* to *d+1*, so the *P-1* row's
label is not settled until the close of *P*. Training on it is not a strict
look-ahead violation -- the label is known at the close of *P*, which is when
the forecast is made, so no post-*P* information is used. But it is not
*implementable*: you would have to observe P(*P*), finalise that label, retrain,
predict, and trade at P(*P*), all in the same instant at the closing bell.

Embargoing *P-1* removes that requirement. The resulting execution story is:

    close of P-1   observe prices; every training label through P-2 is settled
    (between)      retrain the model, no deadline pressure
    close of P     read the month-P features, score, and take positions

Returns are still measured close(*P*) to close(*P+1*), so this assumes execution
at the close of *P* -- the standard monthly-rebalance convention. Trading at the
next open instead would need open prices, which this dataset does not carry.

Set ``--embargo-months 0`` to reproduce the original, non-implementable
protocol; ``models/test_no_lookahead.py`` fails when it is 0.

Cross-validation splits on dates, never inside a month
------------------------------------------------------
The inner CV for hyperparameter tuning uses ``PanelDateSplit`` rather than
sklearn's ``TimeSeriesSplit``. On a panel with 11 rows per month, splitting by
row index cuts through the middle of a month and puts some tickers in train
while their same-month peers sit in validation. Because the target is defined
*cross-sectionally* (beat the equal-weight average of that month), that leaks
the answer: knowing 8 of 11 sectors' outcomes largely determines the rest.
Splitting on whole months avoids it.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "common"))

import universe as U  # noqa: E402

DATASET = REPO_ROOT / "data" / "model_dataset.csv"
OUT_DIR = REPO_ROOT / "models"
PREDICTIONS_CSV = OUT_DIR / "predictions.csv"
METRICS_CSV = OUT_DIR / "metrics.csv"
FEATURE_JSON = OUT_DIR / "feature_names.json"
FINAL_MODELS = OUT_DIR / "final_models.joblib"


def configure(universe: str) -> None:
    """Point the input dataset and every output at one universe."""
    global DATASET, PREDICTIONS_CSV, METRICS_CSV, FEATURE_JSON, FINAL_MODELS
    DATASET = U.path_for(REPO_ROOT / "data" / "model_dataset.csv", universe)
    PREDICTIONS_CSV = U.path_for(OUT_DIR / "predictions.csv", universe)
    METRICS_CSV = U.path_for(OUT_DIR / "metrics.csv", universe)
    FEATURE_JSON = U.path_for(OUT_DIR / "feature_names.json", universe)
    FINAL_MODELS = U.path_for(OUT_DIR / "final_models.joblib", universe)

RANDOM_STATE = 42

# Columns that must never reach a model: the label and its raw ingredients.
FORBIDDEN = ("y_fwd_return", "y_benchmark_return", "y_excess_return", "target")

SECTOR_FEATURES = ["ret_1m_lag1", "ret_3m_lag1", "ret_12m_lag1", "rvol_21d_lag1"]
RANK_FEATURES = ["ret_1m_rank", "rvol_21d_rank"]
# The full LLM block. sentiment_lag1 is the raw score; the other four are
# derived views built in build_dataset.py. Note relative_sentiment and
# sentiment_rank are near-redundant with each other (rank is a monotone
# transform of relative within each month), so the block is not five
# independent signals.
LLM_FEATURES = [
    "sentiment_lag1",
    "sentiment_change",
    "sentiment_rank",
    "relative_sentiment",
    "sentiment_dispersion",
]

MODEL_SPECS = {
    "A": ("logistic", "macro"),
    "B": ("logistic", "macro_llm"),
    "C": ("xgboost", "macro"),
    "D": ("xgboost", "macro_llm"),
}

LOGISTIC_GRID = {"clf__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}

XGB_GRID = {
    "learning_rate": [0.03, 0.1],
    "max_depth": [2, 3, 4],
    "n_estimators": [100, 300],
    "min_child_weight": [1, 5],
}

log = logging.getLogger("train")


# --------------------------------------------------------------------------
# Panel-aware CV
# --------------------------------------------------------------------------

class PanelDateSplit:
    """Expanding-window CV that only ever cuts between months.

    Wraps ``TimeSeriesSplit`` over the *unique dates*, then maps each date fold
    back to the row positions belonging to those dates. Every ticker for a given
    month therefore lands on the same side of the split.
    """

    def __init__(self, n_splits: int = 3):
        self.n_splits = n_splits

    def split(self, X, y=None, groups=None):
        if groups is None:
            raise ValueError("PanelDateSplit requires groups=<date array>")
        dates = np.asarray(groups)
        unique = np.unique(dates)
        inner = TimeSeriesSplit(n_splits=self.n_splits)
        for train_pos, valid_pos in inner.split(unique):
            train_dates = set(unique[train_pos])
            valid_dates = set(unique[valid_pos])
            yield (
                np.flatnonzero([d in train_dates for d in dates]),
                np.flatnonzero([d in valid_dates for d in dates]),
            )

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


# --------------------------------------------------------------------------
# Estimators
# --------------------------------------------------------------------------

def make_estimator(family: str):
    """Fresh estimator plus its grid. Scaling only for the linear model."""
    if family == "logistic":
        # L2 is the default penalty in every supported sklearn; passing
        # penalty="l2" explicitly is deprecated from 1.8 and warns loudly.
        # Regularization strength is the CV-tuned C.
        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(
                solver="lbfgs", max_iter=5000, random_state=RANDOM_STATE
            )),
        ])
        return pipe, LOGISTIC_GRID

    from xgboost import XGBClassifier

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=1,  # parallelism lives in GridSearchCV; nesting it thrashes
        verbosity=0,
    )
    return model, XGB_GRID


def tune(family: str, X: pd.DataFrame, y: pd.Series, dates: np.ndarray, n_splits: int) -> dict:
    """Grid-search hyperparameters with date-respecting inner CV."""
    estimator, grid = make_estimator(family)
    search = GridSearchCV(
        estimator,
        grid,
        cv=PanelDateSplit(n_splits),
        scoring="roc_auc",
        n_jobs=-1,
        refit=False,
    )
    search.fit(X, y, groups=dates)
    return search.best_params_


def fit_final(family: str, params: dict, X: pd.DataFrame, y: pd.Series):
    estimator, _ = make_estimator(family)
    estimator.set_params(**params)
    estimator.fit(X, y)
    return estimator


# --------------------------------------------------------------------------
# Walk-forward
# --------------------------------------------------------------------------

def feature_sets(dataset: pd.DataFrame) -> dict[str, list[str]]:
    """Build the two feature sets, differing only by ``sentiment_lag1``.

    Both now carry sector identity: ten one-hot dummies (XLC is the reference)
    and eight macro x sector interaction terms. Without them every sector sees an
    identical macro row each month, so a linear model must use one coefficient
    per macro variable for all eleven sectors -- it cannot represent energy and
    utilities reacting to the same VIX print in opposite directions.

    Interactions are handed to the trees as well as the linear models. XGBoost
    can construct them itself from the dummies, so they are redundant there, but
    keeping one feature matrix means any A-vs-C difference is the estimator
    rather than the inputs.
    """
    cols = list(dataset.columns)
    dummies = [c for c in cols if c.startswith("is_")]
    interactions = [c for c in cols if "_x_is_" in c]
    macro = [
        c for c in cols
        if c.endswith(("_lag1", "_lag2"))
        and c not in SECTOR_FEATURES + LLM_FEATURES
        and "_x_is_" not in c
    ]
    macro.sort(key=cols.index)  # keep file order, stable for SHAP
    base = macro + SECTOR_FEATURES + RANK_FEATURES + dummies + interactions
    return {"macro": base, "macro_llm": base + LLM_FEATURES}


def train_cutoff(pred_date: pd.Timestamp, embargo_months: int) -> pd.Timestamp:
    """Exclusive upper bound on training dates for a given prediction month.

    With ``embargo_months=1`` this returns *P-1*, so the training set is
    ``date < P-1`` i.e. everything through *P-2* -- whose labels consume no
    price later than the close of *P-1*.
    """
    if embargo_months <= 0:
        return pred_date
    return (pred_date - pd.DateOffset(months=embargo_months)).normalize()


def run(oos_start: str, retune_every: int, n_splits: int, min_train_months: int,
        embargo_months: int, universe: str = U.DEFAULT_UNIVERSE) -> int:
    configure(universe)
    log.info("universe %r -> %s", universe, DATASET.name)
    if not DATASET.exists():
        log.error("%s not found -- run data/build_dataset.py first", DATASET)
        return 1

    data = pd.read_csv(DATASET, parse_dates=["date"]).sort_values(["date", "ticker"])
    sets = feature_sets(data)

    for name, cols in sets.items():
        leaked = [c for c in cols if c in FORBIDDEN]
        if leaked:
            log.error("Feature set %r contains target columns: %s", name, leaked)
            return 1
        missing = [c for c in cols if c not in data.columns]
        if missing:
            log.error("Feature set %r references missing columns: %s", name, missing)
            return 1
    log.info("macro set: %d features | macro_llm set: %d features",
             len(sets["macro"]), len(sets["macro_llm"]))
    log.info("the only difference is %s", LLM_FEATURES)

    all_dates = sorted(data["date"].unique())
    oos_cut = pd.Timestamp(oos_start)
    candidates = [d for d in all_dates if d >= oos_cut]
    if not candidates:
        log.error("No prediction months on or after %s", oos_start)
        return 1

    # The embargo shortens every training window, so the minimum-months guard
    # is applied to the *embargoed* window. Prediction months that no longer
    # clear the bar are dropped from the front rather than silently trained on
    # a shorter history than requested.
    oos_dates = [
        d for d in candidates
        if len([x for x in all_dates if x < train_cutoff(d, embargo_months)]) >= min_train_months
    ]
    if not oos_dates:
        log.error("No prediction month has %d training months after a %d-month embargo",
                  min_train_months, embargo_months)
        return 1
    if len(oos_dates) < len(candidates):
        log.info("embargo drops %d leading prediction month(s): %s",
                 len(candidates) - len(oos_dates),
                 [str(d.date()) for d in candidates[:len(candidates) - len(oos_dates)]])

    cutoff0 = train_cutoff(oos_dates[0], embargo_months)
    first_train = [d for d in all_dates if d < cutoff0]
    log.info("embargo: %d month(s) -- training excludes the %d month(s) before each prediction",
             embargo_months, embargo_months)
    log.info("initial training window: %s -> %s (%d months, %d rows)",
             first_train[0].date(), first_train[-1].date(), len(first_train),
             int((data["date"] < cutoff0).sum()))
    log.info("walk-forward: %s -> %s (%d prediction months)",
             oos_dates[0].date(), oos_dates[-1].date(), len(oos_dates))
    log.info("retuning hyperparameters every %d month(s), inner CV = %d date folds",
             retune_every, n_splits)
    log.info("")

    records: list[dict] = []
    cached: dict[str, dict] = {}
    last_tuned: dict[str, int] = {}
    started = time.monotonic()

    for i, pred_date in enumerate(oos_dates):
        train_mask = data["date"] < train_cutoff(pred_date, embargo_months)
        test_mask = data["date"] == pred_date
        train_all, test_all = data[train_mask], data[test_mask]

        month_rows = {c: {} for c in MODEL_SPECS}
        n_train_reported = 0

        for model_id, (family, set_name) in MODEL_SPECS.items():
            cols = sets[set_name]

            # Defensive: the dataset should already be complete, but a NaN here
            # would silently poison the fit rather than error.
            tr = train_all.dropna(subset=cols + ["target"])
            te = test_all.dropna(subset=cols)
            if tr.empty or te.empty:
                continue
            n_train_reported = max(n_train_reported, len(tr))

            X_tr, y_tr = tr[cols], tr["target"].astype(int)
            if y_tr.nunique() < 2:
                log.warning("  %s %s: single-class training set, skipped",
                            pred_date.date(), model_id)
                continue

            if model_id not in cached or i - last_tuned.get(model_id, -10**9) >= retune_every:
                cached[model_id] = tune(family, X_tr, y_tr, tr["date"].values, n_splits)
                last_tuned[model_id] = i

            model = fit_final(family, cached[model_id], X_tr, y_tr)
            proba = model.predict_proba(te[cols])[:, 1]
            for ticker, p in zip(te["ticker"], proba):
                month_rows[model_id][ticker] = float(p)

        for _, row in test_all.iterrows():
            rec = {"date": pred_date, "ticker": row["ticker"]}
            for model_id in MODEL_SPECS:
                rec[f"model_{model_id}_prob"] = month_rows[model_id].get(row["ticker"], np.nan)
            rec["actual_target"] = row["target"]
            rec["actual_return"] = row["y_fwd_return"]
            records.append(rec)

        elapsed = time.monotonic() - started
        log.info("[%2d/%d] %s  train=%4d rows (%3d months)  predicted %2d tickers  %.0fs elapsed",
                 i + 1, len(oos_dates), pred_date.date(), n_train_reported,
                 train_all["date"].nunique(), len(test_all), elapsed)

    predictions = pd.DataFrame(records)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(PREDICTIONS_CSV, index=False, float_format="%.6f")
    log.info("")
    log.info("Wrote %s (%d rows)", PREDICTIONS_CSV, len(predictions))

    metrics = score(predictions)
    metrics.to_csv(METRICS_CSV, index=False, float_format="%.4f")
    log.info("Wrote %s", METRICS_CSV)

    FEATURE_JSON.write_text(json.dumps(
        {
            "feature_sets": sets,
            "models": {k: {"family": v[0], "feature_set": v[1], "features": sets[v[1]]}
                       for k, v in MODEL_SPECS.items()},
            "random_state": RANDOM_STATE,
        },
        indent=2,
    ))
    log.info("Wrote %s (for SHAP)", FEATURE_JSON)

    # Final-window fits, so SHAP has a model to explain.
    try:
        import joblib

        final = {}
        full = data.dropna(subset=["target"])
        for model_id, (family, set_name) in MODEL_SPECS.items():
            cols = sets[set_name]
            sub = full.dropna(subset=cols)
            final[model_id] = {
                "model": fit_final(family, cached[model_id], sub[cols], sub["target"].astype(int)),
                "features": cols,
                "family": family,
            }
        joblib.dump(final, FINAL_MODELS)
        log.info("Wrote %s (full-sample refits for SHAP)", FINAL_MODELS)
    except Exception as err:  # noqa: BLE001 - artifact is a convenience, not the deliverable
        log.warning("Could not save final models: %s", err)

    report(metrics, predictions)
    return 0


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def score(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_id, (family, set_name) in MODEL_SPECS.items():
        col = f"model_{model_id}_prob"
        sub = predictions.dropna(subset=[col, "actual_target"])
        if sub.empty:
            continue
        y = sub["actual_target"].astype(int)
        p = sub[col]
        yhat = (p >= 0.5).astype(int)
        rows.append({
            "model": model_id,
            "estimator": family,
            "feature_set": set_name,
            "n_predictions": len(sub),
            "accuracy": accuracy_score(y, yhat),
            "precision": precision_score(y, yhat, zero_division=0),
            "recall": recall_score(y, yhat, zero_division=0),
            "auc": roc_auc_score(y, p) if y.nunique() > 1 else np.nan,
        })
    return pd.DataFrame(rows)


def report(metrics: pd.DataFrame, predictions: pd.DataFrame) -> None:
    log.info("")
    log.info("=" * 78)
    log.info("Out-of-sample performance (%s -> %s, %d predictions)",
             predictions["date"].min().date(), predictions["date"].max().date(),
             len(predictions))
    log.info("%-6s %-10s %-10s %7s %9s %10s %8s %7s",
             "model", "estimator", "features", "n", "accuracy", "precision", "recall", "auc")
    for _, r in metrics.iterrows():
        log.info("%-6s %-10s %-10s %7d %9.4f %10.4f %8.4f %7.4f",
                 r["model"], r["estimator"], r["feature_set"], r["n_predictions"],
                 r["accuracy"], r["precision"], r["recall"], r["auc"])
    log.info("=" * 78)

    # The 2x2 exists to isolate the sentiment column; show that contrast directly.
    by_id = metrics.set_index("model")
    for pair, label in ((("A", "B"), "Logistic"), (("C", "D"), "XGBoost")):
        if pair[0] in by_id.index and pair[1] in by_id.index:
            d_auc = by_id.loc[pair[1], "auc"] - by_id.loc[pair[0], "auc"]
            d_acc = by_id.loc[pair[1], "accuracy"] - by_id.loc[pair[0], "accuracy"]
            log.info("%-9s adding sentiment_lag1 (%s -> %s): AUC %+.4f, accuracy %+.4f",
                     label, pair[0], pair[1], d_auc, d_acc)
    log.info("An AUC near 0.50 means no skill; monthly sector direction is a hard target.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--oos-start", default="2018-01-31",
                        help="first prediction month-end (default 2018-01-31)")
    parser.add_argument("--retune-every", type=int, default=1,
                        help="months between hyperparameter searches (1 = every month)")
    parser.add_argument("--cv-splits", type=int, default=3, help="inner CV date folds")
    parser.add_argument("--min-train-months", type=int, default=24)
    parser.add_argument("--embargo-months", type=int, default=1,
                        help="months withheld before each prediction month "
                             "(1 = implementable; 0 = reproduce the original protocol)")
    parser.add_argument("--universe", default=U.DEFAULT_UNIVERSE, choices=sorted(U.UNIVERSES),
                        help="asset universe (default %(default)s)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    if args.embargo_months == 0:
        logging.getLogger("train").warning(
            "embargo-months=0 reproduces the non-implementable protocol: the newest "
            "training label settles at the close of the prediction month itself"
        )
    return run(args.oos_start, args.retune_every, args.cv_splits, args.min_train_months,
               args.embargo_months, args.universe)


if __name__ == "__main__":
    sys.exit(main())
