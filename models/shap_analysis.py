"""SHAP attribution for the two XGBoost models.

Answers the question the 2x2 design was built to ask: when the model is *given*
the LLM sentiment feature, does it actually use it, and how does its importance
compare to the macro and price features it competes with?

Usage
-----
    python models/shap_analysis.py

Outputs
-------
    viz/shap_model_c.png        macro-only model
    viz/shap_model_d.png        macro + sentiment_lag1
    models/shap_values_c.csv    per-row SHAP values
    models/shap_values_d.csv

Notes
-----
Explains the full-sample refits saved by train.py, not the 95 walk-forward fits.
Those refits see every row, which is what you want for a question about *feature
usage* -- but it does mean these attributions are in-sample and describe the
model, not its out-of-sample skill. The backtest and AUC numbers are where skill
is measured; a feature can be heavily used and still carry no predictive value.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET = REPO_ROOT / "data" / "model_dataset.csv"
FINAL_MODELS = REPO_ROOT / "models" / "final_models.joblib"
VIZ_DIR = REPO_ROOT / "viz"
OUT_DIR = REPO_ROOT / "models"

EXPLAIN = ["C", "D"]
TOP_N = 15
SENTINEL = "sentiment_lag1"

log = logging.getLogger("shap")


def compute(model_id: str, spec: dict, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """SHAP values for one model. Returns (shap frame, feature X used)."""
    import shap

    cols = spec["features"]
    sub = data.dropna(subset=cols).reset_index(drop=True)
    X = sub[cols]

    explainer = shap.TreeExplainer(spec["model"])
    values = explainer.shap_values(X)
    if isinstance(values, list):  # older shap returns one array per class
        values = values[1]

    frame = pd.DataFrame(values, columns=cols)
    frame.insert(0, "ticker", sub["ticker"].values)
    frame.insert(0, "date", sub["date"].values)
    log.info("model %s: %d rows x %d features explained", model_id, *values.shape)
    return frame, X


def importance(shap_frame: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Global importance = mean absolute SHAP value per feature."""
    return shap_frame[cols].abs().mean().sort_values(ascending=False)


def plot(model_id: str, shap_frame: pd.DataFrame, X: pd.DataFrame, imp: pd.Series) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top = imp.head(TOP_N)[::-1]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    colors = ["#E45756" if f == SENTINEL else "#4C78A8" for f in top.index]
    ax1.barh(range(len(top)), top.values, color=colors)
    ax1.set_yticks(range(len(top)))
    ax1.set_yticklabels(top.index, fontsize=9)
    ax1.set_xlabel("mean |SHAP value|")
    ax1.set_title(f"Model {model_id}: global feature importance (top {TOP_N})")
    ax1.grid(axis="x", alpha=0.25)
    if SENTINEL in top.index:
        rank = list(imp.index).index(SENTINEL) + 1
        ax1.text(0.98, 0.02, f"{SENTINEL}: rank {rank} of {len(imp)}",
                 transform=ax1.transAxes, ha="right", va="bottom",
                 fontsize=9, color="#E45756", weight="bold")

    # Beeswarm-style: SHAP value vs feature value, coloured by feature level.
    order = list(top.index)[::-1]
    rng = np.random.default_rng(0)
    for i, feat in enumerate(order):
        y = len(order) - 1 - i + rng.normal(0, 0.08, len(shap_frame))
        v = X[feat].values
        norm = (v - np.nanpercentile(v, 1)) / (np.nanpercentile(v, 99) - np.nanpercentile(v, 1) + 1e-12)
        ax2.scatter(shap_frame[feat], y, c=np.clip(norm, 0, 1), cmap="coolwarm",
                    s=4, alpha=0.5, linewidths=0)
    ax2.set_yticks(range(len(order)))
    ax2.set_yticklabels(order[::-1], fontsize=9)
    ax2.axvline(0, color="black", linewidth=0.8)
    ax2.set_xlabel("SHAP value (impact on log-odds of beating the benchmark)")
    ax2.set_title(f"Model {model_id}: SHAP distribution (red = high feature value)")
    ax2.grid(axis="x", alpha=0.25)

    fig.tight_layout()
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    path = VIZ_DIR / f"shap_model_{model_id.lower()}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def run() -> int:
    for path in (DATASET, FINAL_MODELS):
        if not path.exists():
            log.error("%s not found -- run models/train.py first", path)
            return 1

    import joblib

    data = pd.read_csv(DATASET, parse_dates=["date"])
    models = joblib.load(FINAL_MODELS)

    results = {}
    for model_id in EXPLAIN:
        spec = models.get(model_id)
        if spec is None:
            log.error("model %s not in %s", model_id, FINAL_MODELS)
            return 1
        if spec["family"] != "xgboost":
            log.error("model %s is %s; SHAP here covers the tree models only",
                      model_id, spec["family"])
            return 1

        shap_frame, X = compute(model_id, spec, data)
        imp = importance(shap_frame, spec["features"])
        results[model_id] = (shap_frame, imp)

        out = OUT_DIR / f"shap_values_{model_id.lower()}.csv"
        shap_frame.to_csv(out, index=False, float_format="%.6f")
        log.info("Wrote %s", out)
        log.info("Wrote %s", plot(model_id, shap_frame, X, imp))

    report(results)
    return 0


def report(results: dict) -> None:
    log.info("")
    log.info("=" * 74)
    for model_id, (_, imp) in results.items():
        log.info("Model %s: top %d features by mean |SHAP|", model_id, TOP_N)
        for rank, (feat, val) in enumerate(imp.head(TOP_N).items(), 1):
            mark = "   <-- LLM feature" if feat == SENTINEL else ""
            log.info("  %2d. %-20s %.5f%s", rank, feat, val, mark)
        log.info("")

    if "D" in results:
        _, imp_d = results["D"]
        if SENTINEL in imp_d.index:
            rank = list(imp_d.index).index(SENTINEL) + 1
            n = len(imp_d)
            share = imp_d[SENTINEL] / imp_d.sum()
            log.info("%s in Model D: rank %d of %d, %.1f%% of total attribution",
                     SENTINEL, rank, n, share * 100)
            log.info("  strongest feature is %s at %.5f, %.1fx the sentiment value",
                     imp_d.index[0], imp_d.iloc[0], imp_d.iloc[0] / imp_d[SENTINEL])

    # How much does adding sentiment reshuffle the rest of the model?
    if "C" in results and "D" in results:
        imp_c, imp_d = results["C"][1], results["D"][1]
        shared = [f for f in imp_c.index if f in imp_d.index]
        rc = {f: i for i, f in enumerate(imp_c.index)}
        rd = {f: i for i, f in enumerate(imp_d.index)}
        moved = sorted(shared, key=lambda f: -abs(rc[f] - rd[f]))[:3]
        log.info("")
        log.info("Largest rank shifts C -> D among shared features:")
        for f in moved:
            log.info("  %-20s rank %2d -> %2d", f, rc[f] + 1, rd[f] + 1)
    log.info("=" * 74)
    log.info("These attributions are in-sample and describe how the model uses each")
    log.info("feature, not whether that feature predicts out of sample.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    return run()


if __name__ == "__main__":
    sys.exit(main())
