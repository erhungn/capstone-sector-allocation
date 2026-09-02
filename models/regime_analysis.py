"""Does the LLM sentiment feature earn its keep in high-volatility regimes?

The capstone guideline hypothesises that Fed communication matters most when
markets are stressed. This splits the out-of-sample window by VIX and tests
that directly: model skill, the marginal value of the LLM feature, and the raw
correlation between sentiment and realised excess return, all by regime.

Usage
-----
    python models/regime_analysis.py

Outputs
-------
    models/regime_results.csv
    viz/regime_auc_comparison.png

Regimes
-------
``VIX_lag1 < 20`` low, ``>= 25`` high, otherwise medium. VIX_lag1 at row *t* is
the month-end VIX of *t-1*, so the regime label is known when the forecast is
made -- regimes are not assigned with hindsight.

Bootstrap
---------
Resampling is at the **month** level, not the row level. Sectors within a month
share a macro state and their targets are defined against each other (beat the
cross-sectional mean), so rows are strongly dependent within a month. Resampling
rows independently would treat ~11 correlated observations as ~11 independent
ones and produce confidence intervals far too narrow.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parent.parent
PREDICTIONS = REPO_ROOT / "models" / "predictions.csv"
DATASET = REPO_ROOT / "data" / "model_dataset.csv"
RESULTS_CSV = REPO_ROOT / "models" / "regime_results.csv"
PLOT_PNG = REPO_ROOT / "viz" / "regime_auc_comparison.png"

MODELS = ["A", "B", "C", "D"]
PAIRS = [("A", "B", "Logistic"), ("C", "D", "XGBoost")]
# VIX regime cutoffs. Defined once here; backtest/run_backtest.py and
# models/llm_standalone.py import these rather than repeating the literals.
VIX_LOW = 20.0
VIX_HIGH = 25.0
REGIME_LOW = f"Low (VIX<{VIX_LOW:.0f})"
REGIME_MED = f"Medium ({VIX_LOW:.0f}-{VIX_HIGH:.0f})"
REGIME_HIGH = f"High (VIX>={VIX_HIGH:.0f})"
REGIME_ORDER = [REGIME_LOW, REGIME_MED, REGIME_HIGH, "All"]
N_BOOT = 2000

log = logging.getLogger("regime")


def classify(vix: float) -> str:
    if vix < VIX_LOW:
        return REGIME_LOW
    if vix >= VIX_HIGH:
        return REGIME_HIGH
    return REGIME_MED


def cluster_bootstrap_auc_diff(
    frame: pd.DataFrame, col_a: str, col_b: str, n_boot: int = N_BOOT, seed: int = 0
) -> tuple[float, float, float]:
    """Paired AUC difference (B - A) with a month-clustered bootstrap CI."""
    months = frame["date"].unique()
    by_month = {m: g for m, g in frame.groupby("date")}
    rng = np.random.default_rng(seed)

    observed = _auc(frame, col_b) - _auc(frame, col_a)
    diffs = []
    for _ in range(n_boot):
        picked = rng.choice(months, size=len(months), replace=True)
        sample = pd.concat([by_month[m] for m in picked], ignore_index=True)
        y = sample["actual_target"]
        if y.nunique() < 2:
            continue
        diffs.append(_auc(sample, col_b) - _auc(sample, col_a))
    if not diffs:
        return observed, np.nan, np.nan
    return observed, float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def _auc(frame: pd.DataFrame, col: str) -> float:
    y = frame["actual_target"].astype(int)
    if y.nunique() < 2:
        return np.nan
    return float(roc_auc_score(y, frame[col]))


def analyse(frame: pd.DataFrame, label: str) -> dict:
    y = frame["actual_target"].astype(int)
    row: dict = {
        "regime": label,
        "n_months": int(frame["date"].nunique()),
        "n_obs": int(len(frame)),
        "positive_rate": float(y.mean()),
        "mean_vix_lag1": float(frame.groupby("date")["VIX_lag1"].first().mean()),
    }
    for model_id in MODELS:
        col = f"model_{model_id}_prob"
        row[f"auc_{model_id}"] = _auc(frame, col)
        row[f"acc_{model_id}"] = float(accuracy_score(y, (frame[col] >= 0.5).astype(int)))

    row["mean_sentiment"] = float(frame["sentiment_lag1"].mean())
    row["std_sentiment"] = float(frame["sentiment_lag1"].std(ddof=1))
    # Dispersion across sectors within a month -- how strongly the LLM
    # differentiates sectors, which pooled std would blur with time variation.
    row["sentiment_dispersion"] = float(
        frame.groupby("date")["sentiment_lag1"].std(ddof=1).mean()
    )
    row["corr_sentiment_excess"] = float(
        frame["sentiment_lag1"].corr(frame["y_excess_return"])
    )

    for a, b, _ in PAIRS:
        obs, lo, hi = cluster_bootstrap_auc_diff(frame, f"model_{a}_prob", f"model_{b}_prob")
        row[f"d_auc_{b}_minus_{a}"] = obs
        row[f"d_auc_{b}_minus_{a}_lo"] = lo
        row[f"d_auc_{b}_minus_{a}_hi"] = hi
    return row


def run() -> int:
    for path in (PREDICTIONS, DATASET):
        if not path.exists():
            log.error("%s not found -- run the upstream script first", path)
            return 1

    preds = pd.read_csv(PREDICTIONS, parse_dates=["date"])
    data = pd.read_csv(DATASET, parse_dates=["date"])
    frame = preds.merge(
        data[["date", "ticker", "VIX_lag1", "sentiment_lag1", "y_excess_return"]],
        on=["date", "ticker"], how="left", validate="one_to_one",
    )
    if frame[["VIX_lag1", "sentiment_lag1", "y_excess_return"]].isna().any().any():
        log.error("merge left NaNs; predictions and dataset are out of sync")
        return 1

    frame["regime"] = frame["VIX_lag1"].map(classify)

    rows = []
    for label in REGIME_ORDER[:-1]:
        sub = frame[frame["regime"] == label]
        if sub.empty:
            log.warning("no observations in regime %s", label)
            continue
        rows.append(analyse(sub, label))
    rows.append(analyse(frame, "All"))

    results = pd.DataFrame(rows)
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_CSV, index=False, float_format="%.4f")
    log.info("Wrote %s", RESULTS_CSV)

    plot(results)
    report(results, frame)
    return 0


def plot(results: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    regimes = [r for r in REGIME_ORDER if r in set(results["regime"])]
    x = np.arange(len(regimes))
    width = 0.2
    colors = {"A": "#4C78A8", "B": "#72B7B2", "C": "#E45756", "D": "#F58518"}

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, model_id in enumerate(MODELS):
        vals = [results.loc[results["regime"] == r, f"auc_{model_id}"].iloc[0] for r in regimes]
        bars = ax.bar(x + (i - 1.5) * width, vals, width,
                      label=f"Model {model_id}", color=colors[model_id])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=7.5)

    ax.axhline(0.5, color="black", linestyle="--", linewidth=1.2)
    ax.text(len(regimes) - 0.45, 0.505, "0.50 = no skill", fontsize=9, va="bottom", ha="right")
    labels = [
        f"{r}\n{int(results.loc[results.regime == r, 'n_months'].iloc[0])} months, "
        f"{int(results.loc[results.regime == r, 'n_obs'].iloc[0])} obs"
        for r in regimes
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Out-of-sample AUC")
    ax.set_title("Model AUC by VIX regime\n"
                 "A/C = macro only, B/D = macro + LLM sentiment", fontsize=12)
    ax.set_ylim(0.30, max(0.65, results[[f"auc_{m}" for m in MODELS]].max().max() + 0.05))
    ax.legend(frameon=False, ncol=4, loc="upper left")
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    PLOT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_PNG, dpi=150)
    plt.close(fig)
    log.info("Wrote %s", PLOT_PNG)


def report(results: pd.DataFrame, frame: pd.DataFrame) -> None:
    log.info("")
    log.info("=" * 92)
    log.info("%-16s %7s %6s %7s | %7s %7s %7s %7s",
             "regime", "months", "obs", "meanVIX", "AUC A", "AUC B", "AUC C", "AUC D")
    for _, r in results.iterrows():
        log.info("%-16s %7d %6d %7.1f | %7.3f %7.3f %7.3f %7.3f",
                 r["regime"], r["n_months"], r["n_obs"], r["mean_vix_lag1"],
                 r["auc_A"], r["auc_B"], r["auc_C"], r["auc_D"])
    log.info("-" * 92)
    log.info("%-16s %7s %6s %7s | %7s %7s %7s %7s",
             "regime", "", "", "", "acc A", "acc B", "acc C", "acc D")
    for _, r in results.iterrows():
        log.info("%-16s %7s %6s %7s | %7.3f %7.3f %7.3f %7.3f",
                 r["regime"], "", "", "", r["acc_A"], r["acc_B"], r["acc_C"], r["acc_D"])
    log.info("=" * 92)

    log.info("")
    log.info("Sentiment behaviour by regime")
    log.info("%-16s %10s %10s %12s %14s",
             "regime", "mean", "std", "dispersion", "corr(excess)")
    for _, r in results.iterrows():
        log.info("%-16s %10.3f %10.3f %12.3f %14.4f",
                 r["regime"], r["mean_sentiment"], r["std_sentiment"],
                 r["sentiment_dispersion"], r["corr_sentiment_excess"])
    log.info("  dispersion = average within-month spread across the 11 sectors")

    log.info("")
    log.info("Marginal value of the LLM feature (month-clustered bootstrap, %d reps)", N_BOOT)
    log.info("%-16s %-10s %9s %22s  %s", "regime", "pair", "dAUC", "95% CI", "verdict")
    for _, r in results.iterrows():
        for a, b, family in PAIRS:
            obs = r[f"d_auc_{b}_minus_{a}"]
            lo, hi = r[f"d_auc_{b}_minus_{a}_lo"], r[f"d_auc_{b}_minus_{a}_hi"]
            if lo > 0:
                verdict = "HELPS (CI above 0)"
            elif hi < 0:
                verdict = "HURTS (CI below 0)"
            else:
                verdict = "no detectable effect"
            log.info("%-16s %-10s %+9.4f   [%+.4f, %+.4f]  %s",
                     r["regime"], f"{b}-{a} {family}", obs, lo, hi, verdict)

    log.info("")
    high = results[results["regime"] == REGIME_HIGH]
    low = results[results["regime"] == "Low (VIX<20)"]
    if not high.empty and not low.empty:
        h, l = high.iloc[0], low.iloc[0]
        log.info("Hypothesis: the LLM should add value when markets are stressed.")
        for a, b, family in PAIRS:
            log.info("  %-9s dAUC high-vol %+.4f [%+.4f, %+.4f] vs low-vol %+.4f",
                     family, h[f"d_auc_{b}_minus_{a}"], h[f"d_auc_{b}_minus_{a}_lo"],
                     h[f"d_auc_{b}_minus_{a}_hi"], l[f"d_auc_{b}_minus_{a}"])
        log.info("  corr(sentiment, excess return): high-vol %+.4f vs low-vol %+.4f",
                 h["corr_sentiment_excess"], l["corr_sentiment_excess"])

    # The high-vol bucket is small and episodic; say so rather than let the
    # reader assume 19 months are 19 independent events.
    hv = frame[frame["regime"] == REGIME_HIGH]
    if not hv.empty:
        years = hv.groupby(hv["date"].dt.year)["date"].nunique()
        log.info("")
        log.info("Caveat: high-vol months by year -- %s",
                 ", ".join(f"{y}:{n}" for y, n in years.items()))
        log.info("  %d of %d fall in a single year, so this is a handful of episodes,",
                 int(years.max()), int(years.sum()))
        log.info("  not %d independent observations of a stressed market.", int(years.sum()))


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
