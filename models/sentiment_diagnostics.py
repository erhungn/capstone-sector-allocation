"""How much of the LLM's sentiment is just the macro block reflected back?

The prompt hands Claude the same 14 macro variables that the model also gets as
raw features. If the score is largely a deterministic function of that block,
the model receives one signal twice -- once directly, once filtered through the
LLM -- and the "LLM adds information" claim is weaker than it looks.

Usage
-----
    python models/sentiment_diagnostics.py

Outputs
-------
    models/sentiment_redundancy.csv
    viz/sentiment_redundancy.png

The decomposition that matters
------------------------------
The macro block is identical for all eleven sectors in a given month. It can
therefore only explain the part of sentiment that moves *with the month* -- the
cross-sectional mean. Any sector-specific view is orthogonal to it by
construction. So the score splits cleanly:

    sentiment(i,t) = monthly_mean(t)  +  relative(i,t)
                     \\-- macro can explain this      \\-- macro cannot, ever

Redundancy is measured on the first term. A high R-squared there means the LLM's
overall tone is a restatement of the macro inputs; it says nothing about whether
the sector-level differentiation is informative (the AUC results address that).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "common"))

from plotting import DPI  # noqa: E402

DATASET = REPO_ROOT / "data" / "model_dataset.csv"
SENTIMENT = REPO_ROOT / "data" / "llm_sentiment.csv"
MACRO = REPO_ROOT / "data" / "macro_features.csv"
OUT_CSV = REPO_ROOT / "models" / "sentiment_redundancy.csv"
PLOT_PNG = REPO_ROOT / "viz" / "sentiment_redundancy.png"

log = logging.getLogger("sentiment_diag")


def ols_r2(X: pd.DataFrame, y: pd.Series) -> tuple[float, float]:
    """R-squared and adjusted R-squared of y on X with an intercept."""
    A = np.column_stack([np.ones(len(X)), X.values])
    coef, *_ = np.linalg.lstsq(A, y.values, rcond=None)
    resid = y.values - A @ coef
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((y.values - y.values.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    n, k = len(y), X.shape[1]
    adj = 1.0 - (1.0 - r2) * (n - 1) / (n - k - 1) if n > k + 1 else np.nan
    return r2, adj


def run() -> int:
    for path in (DATASET, SENTIMENT, MACRO):
        if not path.exists():
            log.error("%s not found", path)
            return 1

    data = pd.read_csv(DATASET, parse_dates=["date"])
    senti = pd.read_csv(SENTIMENT, index_col=0, parse_dates=True)
    macro = pd.read_csv(MACRO, index_col=0, parse_dates=True)
    macro.index = pd.DatetimeIndex(macro.index).normalize()
    senti.index = pd.DatetimeIndex(senti.index).normalize()
    macro_cols = list(macro.columns)

    # ---- 1. pooled pairwise correlations -------------------------------
    rows = []
    for col in macro_cols:
        rows.append({
            "macro_feature": col,
            "corr_with_sentiment": data["sentiment_lag1"].corr(data[col]),
            "corr_with_relative_sentiment": data["relative_sentiment"].corr(data[col]),
            "corr_with_monthly_mean": senti.mean(axis=1).corr(macro[col]),
            "corr_with_dispersion": senti.std(axis=1, ddof=1).corr(macro[col]),
        })
    table = pd.DataFrame(rows).sort_values("corr_with_monthly_mean",
                                           key=lambda s: s.abs(), ascending=False)

    # ---- 2. how much of sentiment does the whole macro block explain? ----
    monthly_mean = senti.mean(axis=1).reindex(macro.index)
    dispersion = senti.std(axis=1, ddof=1).reindex(macro.index)
    r2_mean, adj_mean = ols_r2(macro[macro_cols], monthly_mean)
    r2_disp, adj_disp = ols_r2(macro[macro_cols], dispersion)

    pooled = data.dropna(subset=["sentiment_lag1"] + macro_cols)
    r2_pooled, adj_pooled = ols_r2(pooled[macro_cols], pooled["sentiment_lag1"])
    r2_rel, adj_rel = ols_r2(pooled[macro_cols], pooled["relative_sentiment"])

    # The R^2 decomposition is the headline of this script, so it is written to
    # the CSV rather than only logged -- the report table is built from the file.
    decomposition = pd.DataFrame([
        {"component": "Monthly mean tone", "n": int(monthly_mean.notna().sum()),
         "r2_on_macro_block": r2_mean, "adj_r2": adj_mean},
        {"component": "Monthly dispersion", "n": int(dispersion.notna().sum()),
         "r2_on_macro_block": r2_disp, "adj_r2": adj_disp},
        {"component": "Pooled score", "n": len(pooled),
         "r2_on_macro_block": r2_pooled, "adj_r2": adj_pooled},
        {"component": "Relative (sector-specific)", "n": len(pooled),
         "r2_on_macro_block": r2_rel, "adj_r2": adj_rel},
    ])

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8") as fh:
        fh.write("# PAIRWISE CORRELATIONS\n")
        table.to_csv(fh, index=False, float_format="%.4f")
        fh.write("\n# VARIANCE OF SENTIMENT EXPLAINED BY THE 14-VARIABLE MACRO BLOCK\n")
        decomposition.to_csv(fh, index=False, float_format="%.4f")
    log.info("Wrote %s", OUT_CSV)

    # ---- report ---------------------------------------------------------
    log.info("")
    log.info("=" * 78)
    log.info("Strongest pairwise correlations with the LLM's monthly tone")
    log.info("%-20s %14s %14s %14s", "macro feature", "monthly mean", "pooled score", "dispersion")
    for _, r in table.head(8).iterrows():
        log.info("%-20s %14.3f %14.3f %14.3f", r["macro_feature"],
                 r["corr_with_monthly_mean"], r["corr_with_sentiment"],
                 r["corr_with_dispersion"])
    log.info("")
    log.info("Variance of sentiment explained by the full 14-variable macro block")
    log.info("%-38s %8s %8s", "", "R^2", "adj R^2")
    log.info("%-38s %8.3f %8.3f", "monthly mean tone (n=120)", r2_mean, adj_mean)
    log.info("%-38s %8.3f %8.3f", "monthly dispersion (n=120)", r2_disp, adj_disp)
    log.info("%-38s %8.3f %8.3f", f"pooled score (n={len(pooled)})", r2_pooled, adj_pooled)
    log.info("%-38s %8.3f %8.3f", "relative (sector-specific) view", r2_rel, adj_rel)
    log.info("=" * 78)

    log.info("")
    log.info("Reading:")
    log.info("  The macro block explains %.0f%% of the LLM's month-to-month tone.", r2_mean * 100)
    if adj_mean > 0.5:
        log.info("  That is substantial redundancy: the overall level of the score is")
        log.info("  largely a restatement of inputs the model already has directly.")
    elif adj_mean > 0.25:
        log.info("  That is moderate redundancy: the tone partly restates the macro inputs.")
    else:
        log.info("  That is limited redundancy: the tone is not simply a macro restatement.")
    log.info("  The sector-specific part (relative_sentiment) is orthogonal to macro by")
    log.info("  construction -- R^2 %.3f -- because every sector sees the same macro row.", r2_rel)
    log.info("  So any genuinely new information the LLM contributes has to live there,")
    log.info("  and that is the component the AUC tests found to be uninformative.")

    plot(table, {"monthly mean tone": r2_mean, "monthly dispersion": r2_disp,
                 "pooled score": r2_pooled, "relative (sector view)": r2_rel})
    return 0


def plot(table: pd.DataFrame, r2s: dict[str, float]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6),
                                   gridspec_kw={"width_ratios": [1.4, 1]})

    top = table.head(10)[::-1]
    colors = ["#E45756" if abs(v) > 0.5 else "#4C78A8" for v in top["corr_with_monthly_mean"]]
    ax1.barh(range(len(top)), top["corr_with_monthly_mean"], color=colors)
    ax1.set_yticks(range(len(top)))
    ax1.set_yticklabels(top["macro_feature"], fontsize=9)
    ax1.axvline(0, color="black", linewidth=0.8)
    ax1.set_xlabel("correlation with the LLM's monthly mean sentiment")
    ax1.set_title("Macro inputs vs the LLM's overall tone", fontsize=12)
    ax1.grid(axis="x", alpha=0.25)
    ax1.set_xlim(-1, 1)

    names = list(r2s)
    vals = [r2s[k] for k in names]
    bars = ax2.bar(range(len(names)), vals,
                   color=["#E45756", "#F58518", "#4C78A8", "#54A24B"])
    for b, v in zip(bars, vals):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}",
                 ha="center", fontsize=10, weight="bold")
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=9)
    ax2.set_ylabel("$R^2$ on the 14 macro features")
    ax2.set_ylim(0, 1.0)
    ax2.set_title("How much of the score is macro restated?", fontsize=12)
    ax2.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    PLOT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_PNG, dpi=DPI)
    plt.close(fig)
    log.info("Wrote %s", PLOT_PNG)


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
