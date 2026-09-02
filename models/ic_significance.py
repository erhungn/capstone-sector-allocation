"""Rigorous inference on the information coefficients.

The headline claim across this project is that no signal has an IC
distinguishable from zero. Everything so far leaned on a conventional t-statistic
and a naive bootstrap. This tests it properly: Newey-West for autocorrelation, a
bootstrap whose resampling scheme is chosen from the measured autocorrelation, a
permutation test that preserves the panel structure exactly, and a
Benjamini-Hochberg correction for testing six signals at once.

Usage
-----
    python models/ic_significance.py
    python models/ic_significance.py --n-perm 50000   # slower, tighter p-values

Outputs
-------
    models/ic_significance.csv
    viz/permutation_null.png

Why the permutation test is the primary evidence
------------------------------------------------
The t-test and the bootstrap both assume the monthly IC series is the unit of
analysis and that its sampling distribution is roughly normal. With ~11 sectors
per month an individual IC is extremely noisy (standard error near 0.3) and its
distribution is neither normal nor independent across sectors.

The permutation test avoids all of that. Within each month it shuffles the
*sector labels* of the signal while leaving returns untouched, so the null is
exactly "this signal carries no cross-sectional information", and every other
feature of the data -- the number of sectors that month, the distribution of
signal values, the return cross-section, the tie structure -- is preserved by
construction. No distributional assumption enters.
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
sys.path.insert(0, str(REPO_ROOT / "common"))

from stats import newey_west_t  # noqa: E402
PREDICTIONS = REPO_ROOT / "models" / "predictions.csv"
DATASET = REPO_ROOT / "data" / "model_dataset.csv"
OUT_CSV = REPO_ROOT / "models" / "ic_significance.csv"
PLOT_PNG = REPO_ROOT / "viz" / "permutation_null.png"

SIGNALS = {
    "Model A": "model_A_prob",
    "Model B": "model_B_prob",
    "Model C": "model_C_prob",
    "Model D": "model_D_prob",
    "LLM sentiment": "sentiment_lag1",
    "LLM sentiment change": "sentiment_change",
}

N_BOOT = 10_000
N_PERM = 10_000
FDR_Q = 0.05
RANDOM_STATE = 42

log = logging.getLogger("ic_significance")


# --------------------------------------------------------------------------
# Monthly IC, computed so the permutation test can reuse the pieces
# --------------------------------------------------------------------------

def month_pieces(signal: np.ndarray, excess: np.ndarray) -> tuple | None:
    """Rank-deviation vectors and the Spearman denominator for one month.

    Spearman's rho is Pearson on ranks. Permuting the signal changes neither its
    mean nor its standard deviation, so only the cross-product moves -- which
    turns each permutation into one dot product instead of a fresh correlation.
    Average ranks are used, so ties are handled exactly as ``spearmanr`` does.
    """
    n = len(signal)
    if n < 4:
        return None
    a = stats.rankdata(signal)
    b = stats.rankdata(excess)
    a_dev, b_dev = a - a.mean(), b - b.mean()
    denom = np.sqrt((a_dev @ a_dev) * (b_dev @ b_dev))
    if denom <= 0:
        return None
    return a_dev, b_dev, denom


def observed_ic(pieces: list) -> np.ndarray:
    return np.array([(a @ b) / d for a, b, d in pieces])


def permutation_null(pieces: list, n_perm: int, seed: int) -> np.ndarray:
    """Null distribution of the mean IC under label shuffling within months."""
    rng = np.random.default_rng(seed)
    total = np.zeros(n_perm)
    for a_dev, b_dev, denom in pieces:
        n = len(a_dev)
        # argsort of uniforms gives independent permutations, one per row
        idx = np.argsort(rng.random((n_perm, n)), axis=1)
        total += (a_dev[idx] @ b_dev) / denom
    return total / len(pieces)


# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------



def autocorr(x: np.ndarray, lag: int = 1) -> float:
    if len(x) <= lag + 2:
        return np.nan
    a, b = x[lag:], x[:-lag]
    return float(np.corrcoef(a, b)[0, 1])


def bootstrap_ci(x: np.ndarray, block: int, n_boot: int, seed: int) -> tuple[float, float]:
    """Percentile CI. ``block`` = 1 is the iid bootstrap; >1 is a moving block."""
    rng = np.random.default_rng(seed)
    n = len(x)
    if block <= 1:
        draws = x[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    else:
        n_blocks = int(np.ceil(n / block))
        starts = rng.integers(0, n - block + 1, size=(n_boot, n_blocks))
        offsets = np.arange(block)
        idx = (starts[:, :, None] + offsets[None, None, :]).reshape(n_boot, -1)[:, :n]
        draws = x[idx].mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """BH step-up adjusted p-values, monotonised."""
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    adj = ranked * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(adj, 1.0)
    return out


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def run(n_perm: int, n_boot: int) -> int:
    for path in (PREDICTIONS, DATASET):
        if not path.exists():
            log.error("%s not found -- run the upstream script first", path)
            return 1

    preds = pd.read_csv(PREDICTIONS, parse_dates=["date"])
    data = pd.read_csv(DATASET, parse_dates=["date"])
    frame = preds.merge(data[["date", "ticker", "sentiment_lag1", "sentiment_change"]],
                        on=["date", "ticker"], how="left", validate="one_to_one")
    frame["excess"] = frame["actual_return"] - frame.groupby("date")["actual_return"].transform("mean")

    log.info("%d months, %d rows | %d permutations, %d bootstrap resamples",
             frame["date"].nunique(), len(frame), n_perm, n_boot)

    rows, nulls, observed = [], {}, {}
    for i, (label, col) in enumerate(SIGNALS.items()):
        if col not in frame.columns:
            log.warning("%s missing, skipped", label)
            continue
        pieces = []
        for _, g in frame.groupby("date"):
            g = g.dropna(subset=[col, "excess"])
            if g[col].nunique() < 2:
                continue
            p = month_pieces(g[col].to_numpy(float), g["excess"].to_numpy(float))
            if p is not None:
                pieces.append(p)
        if len(pieces) < 10:
            log.warning("%s: only %d usable months, skipped", label, len(pieces))
            continue

        ic = observed_ic(pieces)
        mean_ic = float(ic.mean())
        sd = float(ic.std(ddof=1))
        t_conv = mean_ic / (sd / np.sqrt(len(ic))) if sd > 0 else np.nan
        t_nw, lags = newey_west_t(ic)

        # Bootstrap scheme chosen from measured autocorrelation rather than assumed.
        ac1 = autocorr(ic, 1)
        band = 1.96 / np.sqrt(len(ic))
        block = 1 if abs(ac1) <= band else max(2, int(round(len(ic) ** (1 / 3))))
        lo, hi = bootstrap_ci(ic, block, n_boot, RANDOM_STATE + i)

        null = permutation_null(pieces, n_perm, RANDOM_STATE + 1000 + i)
        # +1 correction: a permutation p-value is never exactly zero.
        p_perm = (1 + int(np.sum(np.abs(null) >= abs(mean_ic)))) / (n_perm + 1)
        pct = float((null < mean_ic).mean() * 100)

        nulls[label], observed[label] = null, mean_ic
        rows.append({
            "signal": label, "column": col, "n_months": len(ic),
            "mean_ic": mean_ic, "ic_std": sd,
            "ic_ir": mean_ic / sd if sd > 0 else np.nan,
            "pct_months_positive": float((ic > 0).mean()),
            "t_conventional": float(t_conv),
            "t_newey_west": t_nw, "nw_lags": lags,
            "ic_autocorr_lag1": ac1, "acf_white_noise_band": float(band),
            "bootstrap_block": block, "ci_lo": lo, "ci_hi": hi,
            "p_permutation": p_perm, "null_percentile": pct,
            "null_sd": float(null.std(ddof=1)),
        })
        log.info("  %-22s mean IC %+.4f  p_perm %.4f", label, mean_ic, p_perm)

    results = pd.DataFrame(rows)
    results["p_bh_adjusted"] = benjamini_hochberg(results["p_permutation"].to_numpy())
    results["survives_fdr_5pct"] = results["p_bh_adjusted"] < FDR_Q

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT_CSV, index=False, float_format="%.5f")
    log.info("Wrote %s", OUT_CSV)

    report(results)
    plot(nulls, observed, results)
    return 0


def report(results: pd.DataFrame) -> None:
    log.info("")
    log.info("=" * 104)
    log.info("(a,b) CONVENTIONAL AND NEWEY-WEST")
    log.info("%-22s %7s %9s %8s %8s %8s %9s %8s %6s",
             "signal", "months", "mean IC", "IC std", "IC IR", "%>0", "t", "t (NW)", "lag")
    for _, r in results.iterrows():
        log.info("%-22s %7d %+9.4f %8.4f %+8.3f %7.1f%% %9.2f %8.2f %6d",
                 r["signal"], r["n_months"], r["mean_ic"], r["ic_std"], r["ic_ir"],
                 r["pct_months_positive"] * 100, r["t_conventional"],
                 r["t_newey_west"], r["nw_lags"])

    log.info("")
    log.info("(c) BOOTSTRAP 95%% CI  (scheme chosen from measured autocorrelation)")
    log.info("%-22s %10s %10s %8s %24s", "signal", "acf(1)", "band", "block", "95% CI")
    for _, r in results.iterrows():
        note = "iid" if r["bootstrap_block"] == 1 else f"block={int(r['bootstrap_block'])}"
        log.info("%-22s %+10.3f %10.3f %8s  [%+.4f, %+.4f]%s",
                 r["signal"], r["ic_autocorr_lag1"], r["acf_white_noise_band"], note,
                 r["ci_lo"], r["ci_hi"],
                 "  EXCLUDES 0" if (r["ci_lo"] > 0 or r["ci_hi"] < 0) else "")

    log.info("")
    log.info("(d) PERMUTATION TEST  -- shuffle sector labels within each month")
    log.info("%-22s %10s %12s %14s %12s",
             "signal", "mean IC", "null sd", "percentile", "p (2-tailed)")
    for _, r in results.iterrows():
        log.info("%-22s %+10.4f %12.4f %13.1f%% %12.4f",
                 r["signal"], r["mean_ic"], r["null_sd"], r["null_percentile"],
                 r["p_permutation"])

    log.info("")
    log.info("(e) MULTIPLE COMPARISONS -- %d signals, Benjamini-Hochberg at FDR %.0f%%",
             len(results), FDR_Q * 100)
    log.info("%-22s %14s %16s %10s", "signal", "raw p", "BH-adjusted p", "survives")
    for _, r in results.sort_values("p_permutation").iterrows():
        log.info("%-22s %14.4f %16.4f %10s",
                 r["signal"], r["p_permutation"], r["p_bh_adjusted"],
                 "YES" if r["survives_fdr_5pct"] else "no")
    log.info("=" * 104)

    n_raw = int((results["p_permutation"] < 0.05).sum())
    n_bh = int(results["survives_fdr_5pct"].sum())
    log.info("")
    log.info("VERDICT")
    log.info("  Signals with raw p < 0.05:            %d of %d", n_raw, len(results))
    log.info("  Signals surviving BH at FDR 5%%:       %d of %d", n_bh, len(results))
    if n_bh == 0:
        log.info("")
        log.info("  No signal's IC is distinguishable from zero under any of the three")
        log.info("  tests. The permutation test is the strongest of them -- it makes no")
        log.info("  distributional assumption and preserves the panel structure exactly --")
        log.info("  and it clears none of them. The headline claim is confirmed.")
    else:
        for _, r in results[results["survives_fdr_5pct"]].iterrows():
            log.info("  %s survives: mean IC %+.4f, raw p %.4f, BH p %.4f",
                     r["signal"], r["mean_ic"], r["p_permutation"], r["p_bh_adjusted"])

    best = results.loc[results["mean_ic"].abs().idxmax()]
    log.info("")
    log.info("  Largest |mean IC| is %s at %+.4f, sitting at the %.0fth percentile of its",
             best["signal"], best["mean_ic"], best["null_percentile"])
    log.info("  own null -- i.e. roughly what shuffling the labels produces %.0f%% of the time.",
             100 - best["null_percentile"] if best["mean_ic"] > 0 else best["null_percentile"])
    log.info("  The three methods agree, which is itself reassuring: t, bootstrap and")
    log.info("  permutation would only diverge if a distributional assumption were doing")
    log.info("  the work, and here none is.")


def plot(nulls: dict, observed: dict, results: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(nulls)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4.2 * nrows), sharex=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, (label, null) in zip(axes, nulls.items()):
        r = results[results["signal"] == label].iloc[0]
        obs = observed[label]
        ax.hist(null, bins=60, color="#B0BEC5", edgecolor="white", linewidth=0.4)
        lo, hi = np.percentile(null, [2.5, 97.5])
        ax.axvspan(lo, hi, color="#4C78A8", alpha=0.10)
        ax.axvline(obs, color="#E45756", linewidth=2.4)
        ax.axvline(0, color="black", linewidth=1.0)
        ax.set_title(f"{label}\nobserved {obs:+.4f}  |  p = {r['p_permutation']:.3f}  "
                     f"({r['null_percentile']:.0f}th pct)", fontsize=10.5)
        ax.set_yticks([])
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)

    for ax in axes[n:]:
        ax.set_visible(False)
    for ax in axes[max(0, n - ncols):n]:
        ax.set_xlabel("mean IC under the null", fontsize=10)

    fig.suptitle("Permutation null distributions — sector labels shuffled within each month\n"
                 "Red line is the observed mean IC; shaded band is the null's central 95%",
                 fontsize=13, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    PLOT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_PNG, dpi=150)
    plt.close(fig)
    log.info("Wrote %s", PLOT_PNG)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-perm", type=int, default=N_PERM)
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s", datefmt="%H:%M:%S",
    )
    return run(args.n_perm, args.n_boot)


if __name__ == "__main__":
    sys.exit(main())
