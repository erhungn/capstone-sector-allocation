"""Probability calibration: a quality dimension AUC cannot see.

Ranking and calibration are separate. A model can order sectors perfectly and
still say "0.9" when it means "0.6", or be beautifully calibrated while ranking
at chance. The LLM feature does not improve rank discrimination -- this asks
whether it improves the *probabilities*.

Usage
-----
    python models/calibration.py
    python models/calibration.py --no-secondary   # raw metrics only

Outputs
-------
    models/calibration_results.csv
    viz/calibration_curves.png

Raw first, always
-----------------
Every headline number here is the **raw** model output. No Platt scaling, no
isotonic regression. Post-hoc calibration fitted on the evaluation data would
improve every metric by construction and would say nothing about the models.

A secondary, clearly-labelled section does fit a calibrator, but prequentially:
for month *t* it uses only walk-forward predictions from months **before** *t*.
Those are already out-of-sample, so nothing from the evaluated month informs its
own calibration.

A caveat that governs how much any of this means
------------------------------------------------
Calibration metrics assume the predictions spread out enough to populate the
bins. If a model only ever says 0.45-0.55, every bin near the extremes is empty,
ECE is computed from two or three bins, and the Murphy resolution term is near
zero by construction. The spread is reported first for exactly that reason.
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

PREDICTIONS = REPO_ROOT / "models" / "predictions.csv"
OUT_CSV = REPO_ROOT / "models" / "calibration_results.csv"
PLOT_PNG = REPO_ROOT / "viz" / "calibration_curves.png"

MODELS = ["A", "B", "C", "D"]
LABELS = {"A": "Logistic, macro", "B": "Logistic, +LLM",
          "C": "XGBoost, macro", "D": "XGBoost, +LLM"}
N_BINS = 10
EPS = 1e-15
N_BOOT = 10_000
MIN_CALIB_MONTHS = 24
RANDOM_STATE = 42

log = logging.getLogger("calibration")


# --------------------------------------------------------------------------
# Core metrics
# --------------------------------------------------------------------------

def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    q = np.clip(p, EPS, 1.0 - EPS)
    return float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))


def bin_table(p: np.ndarray, y: np.ndarray, n_bins: int = N_BINS) -> pd.DataFrame:
    """Equal-width bins over [0, 1] with predicted vs observed frequency."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # right-closed except the first bin, so p == 1.0 lands in the last bin
    idx = np.clip(np.digitize(p, edges[1:-1], right=True), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        rows.append({
            "bin": b + 1,
            "bin_lo": edges[b], "bin_hi": edges[b + 1],
            "count": int(m.sum()),
            "mean_predicted": float(p[m].mean()) if m.any() else np.nan,
            "observed_freq": float(y[m].mean()) if m.any() else np.nan,
        })
    return pd.DataFrame(rows)


def ece(bins: pd.DataFrame, n_total: int) -> float:
    """Expected calibration error: count-weighted |predicted - observed|."""
    used = bins.dropna(subset=["mean_predicted", "observed_freq"])
    if used.empty or n_total == 0:
        return np.nan
    gap = (used["mean_predicted"] - used["observed_freq"]).abs()
    return float((used["count"] * gap).sum() / n_total)


def murphy(bins: pd.DataFrame, y: np.ndarray) -> dict:
    """Reliability / resolution / uncertainty, with the identity checked.

    The decomposition is exact for the *binned* Brier score, not the raw one --
    binning discards within-bin variation in the forecast. Both are reported plus
    the residual, rather than quietly presenting components that do not add up.
    """
    n = len(y)
    base = float(y.mean())
    used = bins.dropna(subset=["mean_predicted", "observed_freq"])
    w = used["count"].to_numpy() / n
    rel = float(np.sum(w * (used["mean_predicted"] - used["observed_freq"]) ** 2))
    res = float(np.sum(w * (used["observed_freq"] - base) ** 2))
    unc = base * (1 - base)
    return {"reliability": rel, "resolution": res, "uncertainty": unc,
            "brier_from_decomposition": rel - res + unc}


def clustered_bootstrap_brier_diff(frame: pd.DataFrame, col_a: str, col_b: str,
                                   n_boot: int, seed: int) -> tuple[float, float, float]:
    """Brier(b) - Brier(a), resampling whole months.

    Sectors within a month share a macro state and their targets are defined
    against each other, so resampling rows would treat ~11 dependent
    observations as independent and understate the interval.
    """
    y = frame["actual_target"].to_numpy(float)
    pa, pb = frame[col_a].to_numpy(float), frame[col_b].to_numpy(float)
    months = frame["date"].to_numpy()
    uniq = np.unique(months)
    idx = {m: np.flatnonzero(months == m) for m in uniq}
    rng = np.random.default_rng(seed)

    observed = brier(pb, y) - brier(pa, y)
    draws = []
    for _ in range(n_boot):
        ii = np.concatenate([idx[m] for m in rng.choice(uniq, len(uniq), replace=True)])
        draws.append(brier(pb[ii], y[ii]) - brier(pa[ii], y[ii]))
    return observed, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


# --------------------------------------------------------------------------
# Secondary: prequential calibration
# --------------------------------------------------------------------------

def prequential_platt(frame: pd.DataFrame, col: str,
                      min_months: int = MIN_CALIB_MONTHS) -> pd.Series:
    """Platt-scale month t using only walk-forward predictions from before t.

    The inputs are already out-of-sample, and only earlier months are used, so
    no month contributes to its own calibration. Returns NaN for the leading
    months that have too little history to fit on.
    """
    from sklearn.linear_model import LogisticRegression

    dates = sorted(frame["date"].unique())
    out = pd.Series(np.nan, index=frame.index)
    for i, t in enumerate(dates):
        if i < min_months:
            continue
        past = frame[frame["date"] < t]
        cur = frame[frame["date"] == t]
        y = past["actual_target"].astype(int)
        if y.nunique() < 2:
            continue
        # Fit on the logit of the raw probability -- Platt's original form.
        z = np.log(np.clip(past[col], EPS, 1 - EPS) / (1 - np.clip(past[col], EPS, 1 - EPS)))
        lr = LogisticRegression(solver="lbfgs", max_iter=2000)
        lr.fit(z.to_numpy().reshape(-1, 1), y)
        zc = np.log(np.clip(cur[col], EPS, 1 - EPS) / (1 - np.clip(cur[col], EPS, 1 - EPS)))
        out.loc[cur.index] = lr.predict_proba(zc.to_numpy().reshape(-1, 1))[:, 1]
    return out


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def run(secondary: bool, n_boot: int) -> int:
    if not PREDICTIONS.exists():
        log.error("%s not found -- run models/train.py first", PREDICTIONS)
        return 1

    frame = pd.read_csv(PREDICTIONS, parse_dates=["date"])
    cols = [f"model_{m}_prob" for m in MODELS]
    frame = frame.dropna(subset=cols + ["actual_target"]).reset_index(drop=True)
    y = frame["actual_target"].to_numpy(float)
    base = float(y.mean())

    log.info("%d predictions over %d months | base rate %.3f",
             len(frame), frame["date"].nunique(), base)
    log.info("RAW model output -- no Platt scaling or isotonic regression applied")

    # --- the caveat that governs everything below -------------------------
    log.info("")
    log.info("PREDICTED PROBABILITY SPREAD")
    log.info("%-6s %8s %8s %8s %8s %8s %10s %12s",
             "model", "min", "p25", "median", "p75", "max", "std", "within ±.05")
    spread = {}
    for m in MODELS:
        p = frame[f"model_{m}_prob"].to_numpy()
        q = np.percentile(p, [25, 50, 75])
        near = float(np.mean(np.abs(p - 0.5) <= 0.05))
        spread[m] = {"p_min": p.min(), "p_max": p.max(), "p_std": p.std(ddof=1),
                     "p_iqr": q[2] - q[0], "frac_within_005_of_half": near}
        log.info("%-6s %8.3f %8.3f %8.3f %8.3f %8.3f %10.4f %11.1f%%",
                 m, p.min(), q[0], q[1], q[2], p.max(), p.std(ddof=1), near * 100)

    # --- primary metrics ---------------------------------------------------
    rows, bins_by_model = [], {}
    bs_half = brier(np.full_like(y, 0.5), y)
    bs_base = brier(np.full_like(y, base), y)

    for m in MODELS:
        col = f"model_{m}_prob"
        p = frame[col].to_numpy()
        bins = bin_table(p, y)
        bins_by_model[m] = bins
        bs = brier(p, y)
        dec = murphy(bins, y)
        rows.append({
            "model": m, "label": LABELS[m], "n": len(y), "base_rate": base,
            "brier": bs,
            "brier_skill_vs_half": 1 - bs / bs_half,
            "brier_skill_vs_base_rate": 1 - bs / bs_base,
            "log_loss": log_loss(p, y),
            "log_loss_baseline": log_loss(np.full_like(y, base), y),
            "ece": ece(bins, len(y)),
            "n_populated_bins": int((bins["count"] > 0).sum()),
            **dec,
            "decomposition_residual": bs - dec["brier_from_decomposition"],
            **spread[m],
        })

    results = pd.DataFrame(rows)

    # --- key comparisons ---------------------------------------------------
    comparisons = []
    for a, b in (("A", "B"), ("C", "D")):
        d, lo, hi = clustered_bootstrap_brier_diff(
            frame, f"model_{a}_prob", f"model_{b}_prob", n_boot, RANDOM_STATE)
        comparisons.append({"comparison": f"{b} - {a}", "brier_diff": d,
                            "ci_lo": lo, "ci_hi": hi,
                            "improves": bool(hi < 0), "worsens": bool(lo > 0)})
    comps = pd.DataFrame(comparisons)

    # --- secondary: prequential calibration --------------------------------
    secondary_rows = []
    if secondary:
        log.info("")
        log.info("Fitting prequential Platt calibrators (month t uses only months < t)")
        for m in MODELS:
            col = f"model_{m}_prob"
            cal = prequential_platt(frame, col)
            mask = cal.notna()
            if mask.sum() < 100:
                continue
            yy = y[mask.to_numpy()]
            secondary_rows.append({
                "model": m, "n_evaluated": int(mask.sum()),
                "brier_raw": brier(frame.loc[mask, col].to_numpy(), yy),
                "brier_calibrated": brier(cal[mask].to_numpy(), yy),
                "ece_raw": ece(bin_table(frame.loc[mask, col].to_numpy(), yy), int(mask.sum())),
                "ece_calibrated": ece(bin_table(cal[mask].to_numpy(), yy), int(mask.sum())),
            })

    # --- save --------------------------------------------------------------
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8") as fh:
        fh.write("# primary: raw model calibration\n")
        results.to_csv(fh, index=False, float_format="%.6f")
        fh.write("\n# key comparisons: Brier difference, month-clustered bootstrap\n")
        comps.to_csv(fh, index=False, float_format="%.6f")
        fh.write("\n# calibration curve bins\n")
        all_bins = pd.concat([b.assign(model=m) for m, b in bins_by_model.items()])
        all_bins.to_csv(fh, index=False, float_format="%.6f")
        if secondary_rows:
            fh.write("\n# SECONDARY: prequential Platt scaling (not the headline result)\n")
            pd.DataFrame(secondary_rows).to_csv(fh, index=False, float_format="%.6f")
    log.info("Wrote %s", OUT_CSV)

    report(results, comps, bins_by_model, pd.DataFrame(secondary_rows), bs_half, bs_base)
    plot(frame, bins_by_model, results)
    return 0


def report(results, comps, bins_by_model, secondary, bs_half, bs_base) -> None:
    log.info("")
    log.info("=" * 104)
    log.info("(a,b,c) RAW CALIBRATION")
    log.info("%-6s %-18s %9s %11s %11s %10s %9s %7s",
             "model", "spec", "Brier", "BSS vs .5", "BSS vs base", "log loss", "ECE", "bins")
    for _, r in results.iterrows():
        log.info("%-6s %-18s %9.5f %10.4f %11.4f %10.5f %9.4f %7d",
                 r["model"], r["label"], r["brier"], r["brier_skill_vs_half"],
                 r["brier_skill_vs_base_rate"], r["log_loss"], r["ece"],
                 r["n_populated_bins"])
    log.info("  baselines: constant 0.5 -> Brier %.5f | base rate -> Brier %.5f, log loss %.5f",
             bs_half, bs_base, results["log_loss_baseline"].iloc[0])
    log.info("  a NEGATIVE Brier skill score means the model is worse than that constant")

    log.info("")
    log.info("(e) MURPHY DECOMPOSITION  Brier = reliability - resolution + uncertainty")
    log.info("%-6s %13s %12s %12s %14s %12s",
             "model", "reliability", "resolution", "uncertainty", "sum", "residual")
    for _, r in results.iterrows():
        log.info("%-6s %13.5f %12.5f %12.5f %14.5f %12.5f",
                 r["model"], r["reliability"], r["resolution"], r["uncertainty"],
                 r["brier_from_decomposition"], r["decomposition_residual"])
    log.info("  lower reliability is better (closer to the diagonal);")
    log.info("  higher resolution is better (bins that actually separate outcomes).")
    log.info("  the residual is the binning loss -- the identity is exact only for the")
    log.info("  binned Brier score, so it is shown rather than hidden.")

    log.info("")
    log.info("KEY COMPARISONS -- does the LLM feature improve the probabilities?")
    log.info("%-10s %14s %26s %s", "pair", "ΔBrier", "95% CI (month-clustered)", "verdict")
    for _, r in comps.iterrows():
        verdict = ("improves" if r["improves"] else
                   "worsens" if r["worsens"] else "no detectable difference")
        log.info("%-10s %+14.6f   [%+.6f, %+.6f]  %s",
                 r["comparison"], r["brier_diff"], r["ci_lo"], r["ci_hi"], verdict)
    log.info("  negative ΔBrier = the LLM version is better calibrated")

    if not secondary.empty:
        log.info("")
        log.info("-" * 104)
        log.info("SECONDARY (not the headline): prequential Platt scaling")
        log.info("  Calibrator for month t is fitted only on walk-forward predictions from")
        log.info("  months before t, so no month informs its own calibration.")
        log.info("%-6s %11s %12s %14s %11s %14s",
                 "model", "n", "Brier raw", "Brier calib.", "ECE raw", "ECE calib.")
        for _, r in secondary.iterrows():
            log.info("%-6s %11d %12.5f %14.5f %11.4f %14.4f",
                     r["model"], r["n_evaluated"], r["brier_raw"], r["brier_calibrated"],
                     r["ece_raw"], r["ece_calibrated"])
        log.info("  This shows what recalibration can repair; it does not create ranking")
        log.info("  skill, which is what the AUC and IC results measure.")
    log.info("=" * 104)

    log.info("")
    log.info("VERDICT")
    worst = results.loc[results["brier_skill_vs_half"].idxmin()]
    best = results.loc[results["brier_skill_vs_half"].idxmax()]
    log.info("  Brier skill vs a constant 0.5 runs from %+.4f (%s) to %+.4f (%s).",
             worst["brier_skill_vs_half"], worst["model"],
             best["brier_skill_vs_half"], best["model"])
    if (results["brier_skill_vs_half"] <= 0).all():
        log.info("  Every model is worse than simply predicting 0.5 for every sector-month.")
    if not ((comps["improves"]) | (comps["worsens"])).any():
        log.info("  Neither LLM comparison shows a detectable calibration difference, so")
        log.info("  the feature does not help on this dimension either -- calibration is")
        log.info("  not a hidden strength masked by the AUC results.")
    else:
        for _, r in comps.iterrows():
            if r["worsens"]:
                log.info("  %s: the LLM version is significantly WORSE calibrated "
                         "(ΔBrier %+.6f, CI excludes zero).", r["comparison"], r["brier_diff"])
            elif r["improves"]:
                log.info("  %s: the LLM version is significantly BETTER calibrated "
                         "(ΔBrier %+.6f, CI excludes zero).", r["comparison"], r["brier_diff"])
        log.info("  Two comparisons were run here; read any single significant result")
        log.info("  against that, and against the null AUC and IC results.")
    log.info("  Resolution is near zero for every model (max %.5f against uncertainty",
             results["resolution"].max())
    log.info("  %.5f): the bins barely separate outcomes, which is the same finding as",
             results["uncertainty"].iloc[0])
    log.info("  the AUC null expressed in calibration language.")


def plot(frame: pd.DataFrame, bins_by_model: dict, results: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(13, 11.5))
    # Row 2 is an empty spacer: without it the top histograms' x-label runs into
    # the two-line titles of the row below.
    gs = GridSpec(5, 2, figure=fig, height_ratios=[3, 1, 0.6, 3, 1],
                  hspace=0.32, wspace=0.22)
    colors = {"A": "#4C78A8", "B": "#72B7B2", "C": "#E45756", "D": "#F58518"}

    for k, m in enumerate(MODELS):
        r_top, c = (0 if k < 2 else 3), k % 2
        ax = fig.add_subplot(gs[r_top, c])
        axh = fig.add_subplot(gs[r_top + 1, c], sharex=ax)
        bins = bins_by_model[m]
        stats_row = results[results["model"] == m].iloc[0]
        used = bins.dropna(subset=["mean_predicted", "observed_freq"])

        ax.plot([0, 1], [0, 1], "--", color="black", linewidth=1.2, label="perfect")
        ax.plot(used["mean_predicted"], used["observed_freq"], "o-",
                color=colors[m], linewidth=2.2, markersize=7, label="observed")
        for _, b in used.iterrows():
            ax.annotate(f"{int(b['count'])}", (b["mean_predicted"], b["observed_freq"]),
                        textcoords="offset points", xytext=(0, 8), ha="center", fontsize=7.5)
        ax.axhline(stats_row["base_rate"], color="grey", linewidth=1.0, linestyle=":")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.10)   # headroom so a count label on a 1.0 bin stays inside
        ax.set_ylabel("observed frequency", fontsize=9)
        ax.set_title(f"Model {m} — {LABELS[m]}\n"
                     f"Brier {stats_row['brier']:.4f}  ECE {stats_row['ece']:.4f}  "
                     f"BSS(0.5) {stats_row['brier_skill_vs_half']:+.4f}", fontsize=10)
        ax.grid(alpha=0.25)
        # "best" rather than a fixed corner: Model A's curve runs through the
        # upper left, where the other three have empty space.
        ax.legend(fontsize=8, loc="best", frameon=False)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

        axh.hist(frame[f"model_{m}_prob"], bins=np.linspace(0, 1, 41),
                 color=colors[m], alpha=0.8)
        axh.axvline(0.5, color="black", linewidth=1.0)
        axh.set_xlim(0, 1)
        axh.set_xlabel("predicted probability", fontsize=9)
        axh.set_ylabel("count", fontsize=8)
        axh.grid(alpha=0.2, axis="y")
        for side in ("top", "right"):
            axh.spines[side].set_visible(False)

    fig.suptitle("Calibration of raw model probabilities — no post-hoc scaling applied\n"
                 "Point labels are bin counts; dotted line is the base rate",
                 fontsize=13, weight="bold")
    # GridSpec already sets the spacing; tight_layout would fight the shared axes.
    fig.subplots_adjust(top=0.91, bottom=0.06, left=0.08, right=0.97)
    PLOT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_PNG, dpi=DPI)
    plt.close(fig)
    log.info("Wrote %s", PLOT_PNG)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-secondary", action="store_true",
                        help="skip the prequential recalibration section")
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s", datefmt="%H:%M:%S",
    )
    return run(not args.no_secondary, args.n_boot)


if __name__ == "__main__":
    sys.exit(main())
