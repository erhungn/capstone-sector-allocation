"""Portfolio-appropriate evaluation: does the signal *rank* sectors correctly?

AUC and accuracy treat every (date, ticker) row as an independent classification.
A sector strategy does not work that way -- it ranks the sectors against each
other within a month and buys the top few. These metrics score that directly, and
they evaluate the raw LLM sentiment as a standalone signal rather than only as a
feature buried inside a model.

Usage
-----
    python models/ranking_metrics.py

Outputs
-------
    models/ranking_metrics.csv
    viz/ic_timeseries.png

Metrics
-------
IC          monthly cross-sectional Spearman correlation between the signal and
            realised excess return. Reported as mean, std, IR (mean/std), the
            share of months positive, and a t-statistic on the mean.
Precision@3 of the three sectors the signal picks, the fraction that actually
            beat the equal-weight benchmark.
Top3 excess mean monthly excess return of those three picks -- the gross alpha
            of the strategy, before costs.
NDCG@3      rank-quality of the top three, with realised-return rank as
            relevance. Reported against a shuffled-signal baseline, because
            NDCG's own scale is not intuitive (random is far above zero).

Note on the benchmark: Spearman IC is invariant to subtracting a constant from
every return in a month, so the IC figures are identical whether computed on raw
or excess returns. Precision and Top3-excess are not, and use the mean of the
same universe the signal ranks over.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent
PREDICTIONS = REPO_ROOT / "models" / "predictions.csv"
DATASET = REPO_ROOT / "data" / "model_dataset.csv"
OUT_CSV = REPO_ROOT / "models" / "ranking_metrics.csv"
PLOT_PNG = REPO_ROOT / "viz" / "ic_timeseries.png"

TOP_N = 3
ROLL = 6
MONTHS_PER_YEAR = 12
N_SHUFFLE = 200
RANDOM_STATE = 42

# label -> column in the merged frame
SIGNALS = {
    "Model A (logit, macro)": "model_A_prob",
    "Model B (logit, +LLM)": "model_B_prob",
    "Model C (xgb, macro)": "model_C_prob",
    "Model D (xgb, +LLM)": "model_D_prob",
    "LLM sentiment (raw)": "sentiment_lag1",
    "LLM sentiment change": "sentiment_change",
    "LLM sentiment rank": "sentiment_rank",
    "LLM relative sentiment": "relative_sentiment",
}
PLOT_SIGNALS = ["Model C (xgb, macro)", "Model D (xgb, +LLM)", "LLM sentiment (raw)"]

log = logging.getLogger("ranking")


# --------------------------------------------------------------------------
# Per-month metrics
# --------------------------------------------------------------------------

def dcg_at_k(relevance: np.ndarray, k: int) -> float:
    r = relevance[:k]
    return float(np.sum(r / np.log2(np.arange(2, len(r) + 2))))


def ndcg_at_k(scores: np.ndarray, excess: np.ndarray, k: int) -> float:
    """NDCG@k with realised-return *rank* as graded relevance.

    Raw excess returns cannot serve as gains directly: they are signed, and a
    negative ideal DCG makes the ratio meaningless. Ranking them (0 = worst,
    n-1 = best) gives a non-negative, monotone relevance that keeps the metric
    well defined.
    """
    order = np.argsort(-scores, kind="stable")
    relevance = stats.rankdata(excess, method="average") - 1.0
    dcg = dcg_at_k(relevance[order], k)
    idcg = dcg_at_k(np.sort(relevance)[::-1], k)
    return float(dcg / idcg) if idcg > 0 else np.nan


def month_metrics(scores: np.ndarray, excess: np.ndarray) -> dict:
    if len(scores) < TOP_N or np.all(scores == scores[0]):
        return {"ic": np.nan, "precision": np.nan, "top3_excess": np.nan, "ndcg": np.nan}
    ic = stats.spearmanr(scores, excess).statistic
    picks = np.argsort(-scores, kind="stable")[:TOP_N]
    return {
        "ic": float(ic),
        "precision": float((excess[picks] > 0).mean()),
        "top3_excess": float(excess[picks].mean()),
        "ndcg": ndcg_at_k(scores, excess, TOP_N),
    }


def evaluate(frame: pd.DataFrame, score_col: str) -> tuple[pd.DataFrame, dict]:
    """Per-month series plus the aggregate summary for one signal."""
    rows = []
    for date, g in frame.groupby("date"):
        g = g.dropna(subset=[score_col, "excess"])
        m = month_metrics(g[score_col].to_numpy(float), g["excess"].to_numpy(float))
        m["date"] = date
        m["n"] = len(g)
        rows.append(m)
    per_month = pd.DataFrame(rows).set_index("date").sort_index()

    ic = per_month["ic"].dropna()
    n = len(ic)
    mean_ic, std_ic = float(ic.mean()), float(ic.std(ddof=1))
    t_stat = mean_ic / (std_ic / np.sqrt(n)) if std_ic > 0 and n > 1 else np.nan
    summary = {
        "n_months": n,
        "mean_ic": mean_ic,
        "ic_std": std_ic,
        "ic_ir": mean_ic / std_ic if std_ic > 0 else np.nan,
        "pct_months_ic_positive": float((ic > 0).mean()),
        "ic_t_stat": float(t_stat),
        "ic_p_value": float(2 * (1 - stats.t.cdf(abs(t_stat), n - 1))) if n > 1 else np.nan,
        "precision_at_3": float(per_month["precision"].mean()),
        "top3_mean_excess": float(per_month["top3_excess"].mean()),
        "top3_excess_annual": float(per_month["top3_excess"].mean() * MONTHS_PER_YEAR),
        "ndcg_at_3": float(per_month["ndcg"].mean()),
    }
    return per_month, summary


def shuffle_baselines(frame: pd.DataFrame) -> dict:
    """What a signal with no information scores, by construction.

    NDCG@3 in particular sits well above zero for random rankings, so the raw
    number is uninterpretable without this reference.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    prec, ndcg, top3 = [], [], []
    groups = [g for _, g in frame.groupby("date")]
    for _ in range(N_SHUFFLE):
        p, nd, t3 = [], [], []
        for g in groups:
            excess = g["excess"].to_numpy(float)
            scores = rng.permutation(len(excess)).astype(float)
            m = month_metrics(scores, excess)
            p.append(m["precision"]); nd.append(m["ndcg"]); t3.append(m["top3_excess"])
        prec.append(np.nanmean(p)); ndcg.append(np.nanmean(nd)); top3.append(np.nanmean(t3))
    return {
        "precision_at_3": float(np.mean(prec)),
        "ndcg_at_3": float(np.mean(ndcg)),
        "ndcg_p95": float(np.percentile(ndcg, 95)),
        "top3_mean_excess": float(np.mean(top3)),
    }


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def run() -> int:
    for path in (PREDICTIONS, DATASET):
        if not path.exists():
            log.error("%s not found -- run the upstream script first", path)
            return 1

    preds = pd.read_csv(PREDICTIONS, parse_dates=["date"])
    data = pd.read_csv(DATASET, parse_dates=["date"])
    sentiment_cols = [c for c in SIGNALS.values() if c.startswith(("sentiment", "relative"))]
    frame = preds.merge(
        data[["date", "ticker"] + sentiment_cols], on=["date", "ticker"],
        how="left", validate="one_to_one",
    )
    missing = frame[sentiment_cols].isna().sum().sum()
    if missing:
        log.warning("%d missing sentiment values after merge", int(missing))

    # Excess return against the mean of the same universe the signal ranks over,
    # so the benchmark and the ranking set never disagree.
    frame["excess"] = frame["actual_return"] - frame.groupby("date")["actual_return"].transform("mean")

    log.info("OOS window: %s -> %s, %d months, %d rows",
             frame["date"].min().date(), frame["date"].max().date(),
             frame["date"].nunique(), len(frame))
    log.info("base rate: %.1f%% of sector-months beat the equal-weight mean",
             (frame["excess"] > 0).mean() * 100)

    per_month, summaries = {}, []
    for label, col in SIGNALS.items():
        if col not in frame.columns:
            log.warning("signal %s (%s) not available, skipped", label, col)
            continue
        months, summary = evaluate(frame, col)
        per_month[label] = months
        summary["signal"] = label
        summary["column"] = col
        summaries.append(summary)

    baseline = shuffle_baselines(frame)
    log.info("random-signal baseline (%d shuffles): precision@3 %.3f, NDCG@3 %.3f, top3 excess %+.4f%%",
             N_SHUFFLE, baseline["precision_at_3"], baseline["ndcg_at_3"],
             baseline["top3_mean_excess"] * 100)

    results = pd.DataFrame(summaries)[
        ["signal", "column", "n_months", "mean_ic", "ic_std", "ic_ir",
         "pct_months_ic_positive", "ic_t_stat", "ic_p_value",
         "precision_at_3", "top3_mean_excess", "top3_excess_annual", "ndcg_at_3"]
    ]
    results = pd.concat([results, pd.DataFrame([{
        "signal": "Random baseline", "column": "shuffled", "n_months": frame["date"].nunique(),
        "mean_ic": 0.0, **{k: baseline[k] for k in
                           ("precision_at_3", "top3_mean_excess", "ndcg_at_3")},
        "top3_excess_annual": baseline["top3_mean_excess"] * MONTHS_PER_YEAR,
    }])], ignore_index=True)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT_CSV, index=False, float_format="%.4f")
    log.info("Wrote %s", OUT_CSV)

    report(results, baseline)
    plot(per_month, frame)
    return 0


def report(results: pd.DataFrame, baseline: dict) -> None:
    log.info("")
    log.info("=" * 112)
    log.info("%-24s %8s %8s %7s %8s %8s %10s %11s %8s",
             "signal", "mean IC", "IC std", "IC IR", "%IC>0", "t-stat",
             "prec@3", "top3 exc/yr", "NDCG@3")
    log.info("-" * 112)
    for _, r in results.iterrows():
        star = ""
        if pd.notna(r.get("ic_p_value")) and r["ic_p_value"] < 0.05:
            star = " *"
        log.info("%-24s %8.4f %8.4f %7.3f %7.1f%% %8.2f %10.3f %10.2f%% %8.3f%s",
                 r["signal"], r["mean_ic"], r.get("ic_std", np.nan), r.get("ic_ir", np.nan),
                 r.get("pct_months_ic_positive", np.nan) * 100, r.get("ic_t_stat", np.nan),
                 r["precision_at_3"], r["top3_excess_annual"] * 100, r["ndcg_at_3"], star)
    log.info("=" * 112)
    log.info("* mean IC significantly different from zero at 5%%")
    log.info("Random baseline: precision@3 %.3f, NDCG@3 %.3f (95th pct %.3f).",
             baseline["precision_at_3"], baseline["ndcg_at_3"], baseline["ndcg_p95"])
    log.info("NDCG near the baseline means the ranking is no better than chance --")
    log.info("the metric does not start at zero, so read it against that column.")

    real = results[results["signal"] != "Random baseline"]
    sig = real[real["ic_p_value"] < 0.05]
    log.info("")
    if sig.empty:
        log.info("No signal has a mean IC distinguishable from zero at the 5%% level.")
    else:
        for _, r in sig.iterrows():
            log.info("%s: mean IC %+.4f (t=%.2f, p=%.3f)",
                     r["signal"], r["mean_ic"], r["ic_t_stat"], r["ic_p_value"])
    best = real.loc[real["mean_ic"].idxmax()]
    log.info("Highest mean IC: %s at %+.4f (IR %.3f). For reference, a monthly IC of",
             best["signal"], best["mean_ic"], best["ic_ir"])
    log.info("0.02-0.05 is a weak-but-usable equity signal; below ~0.01 is noise.")

    # Identical rows are a property of the signals, not a bug -- flag it so the
    # table is not misread as three independent confirmations.
    dupes = real.groupby(real["mean_ic"].round(10))["signal"].apply(list)
    for group in dupes:
        if len(group) > 1:
            log.info("")
            log.info("Identical metrics for: %s", ", ".join(group))
            log.info("  Not a bug. Within a month, sentiment_rank is a monotone transform of")
            log.info("  the level and relative_sentiment subtracts a constant -- neither changes")
            log.info("  the ordering, so all three produce the same ranking every month.")
            log.info("  For a cross-sectional task they are one signal, not three.")


def plot(per_month: dict[str, pd.DataFrame], frame: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"Model C (xgb, macro)": "#E45756", "Model D (xgb, +LLM)": "#F58518",
              "LLM sentiment (raw)": "#4C78A8"}
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True,
                                  gridspec_kw={"height_ratios": [2.4, 1], "hspace": 0.1})

    for label in PLOT_SIGNALS:
        if label not in per_month:
            continue
        roll = per_month[label]["ic"].rolling(ROLL, min_periods=ROLL).mean()
        full = per_month[label]["ic"].mean()
        ax.plot(roll.index, roll, color=colors[label], linewidth=2.0,
                label=f"{label}   (full-sample mean IC {full:+.3f})")
        ax.axhline(full, color=colors[label], linestyle=":", linewidth=1.0, alpha=0.7)

    ax.axhline(0, color="black", linewidth=1.1)
    ax.set_ylabel(f"{ROLL}-month rolling mean IC", fontsize=11)
    ax.set_title("Rolling cross-sectional information coefficient\n"
                 "Spearman rank correlation between signal and realised excess return",
                 fontsize=13, weight="bold")
    ax.legend(loc="upper left", frameon=True, framealpha=0.92, fontsize=9)
    ax.grid(alpha=0.25)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    # Monthly IC for the raw LLM signal, to show the sign is a coin flip.
    if "LLM sentiment (raw)" in per_month:
        ic = per_month["LLM sentiment (raw)"]["ic"]
        ax2.bar(ic.index, ic, width=20,
                color=np.where(ic >= 0, "#4C78A8", "#E45756"), alpha=0.85)
        ax2.axhline(0, color="black", linewidth=1.0)
        ax2.set_ylabel("monthly IC\n(raw LLM)", fontsize=10)
        ax2.set_title(f"{(ic > 0).mean() * 100:.0f}% of months positive "
                      f"-- a coin flip would give 50%", fontsize=10, loc="left")
        ax2.grid(alpha=0.25, axis="y")
        for side in ("top", "right"):
            ax2.spines[side].set_visible(False)

    fig.savefig(PLOT_PNG, dpi=150, bbox_inches="tight")
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
