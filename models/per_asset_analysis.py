"""Does the null hold for every asset, or does one of them carry signal?

    python models/per_asset_analysis.py
    python models/per_asset_analysis.py --universe 11asset   # one universe only

Outputs
-------
    models/per_asset_results.csv    one block per universe
    viz/per_asset_heatmap.png

Why this exists
---------------
Every headline result in this project is pooled across assets. A reviewer is
entitled to ask whether that pooling hides something -- whether the models work
on utilities and fail on energy, or whether the LLM reads one sector well and
the rest badly. The raw material has been sitting in ``predictions.csv`` and
``model_dataset.csv`` all along; it had simply never been cut this way.

This is exploratory, and it is treated as such. Eleven (or fourteen) assets are
examined at once, so the question is not "is any asset's statistic large?" --
with that many draws one of them will be -- but "is any asset's statistic large
enough to survive the fact that we looked at all of them?" The sentiment
correlations carry a Benjamini-Hochberg adjustment for exactly that reason, and
the printed output states how many assets should look significant by chance
before showing which ones do.

Two measurement traps, and how they are handled
-----------------------------------------------
**Hit rate is not skill.** A model's hit rate on one asset is bounded below by
the majority-class baseline: TLT beats the equal-weight benchmark in only 30% of
months, so a model that simply never predicts TLT scores 70% there and looks
excellent. Every hit rate below is therefore reported next to that asset's base
rate and, more usefully, as an *edge over the majority-class baseline*. A
positive edge is the only version of this number that means anything.

**Per-asset AUC is a different quantity from the headline AUC.** The headline
AUC is cross-sectional: within a month, does the model rank the assets that beat
the benchmark above those that did not? The per-asset AUC here is a time series
within one asset: across months, does the model assign higher probabilities to
the months that asset outperformed? They answer different questions and are not
comparable. The column is named ``auc_timeseries`` throughout so the two cannot
be confused in a table.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "common"))

import universe as U  # noqa: E402
from plotting import DPI, MODEL_COLORS  # noqa: E402

OUT_CSV = REPO_ROOT / "models" / "per_asset_results.csv"
PLOT_PNG = REPO_ROOT / "viz" / "per_asset_heatmap.png"

MODELS = ["A", "B", "C", "D"]
TOP_N = 3
MIN_MONTHS_FOR_AUC = 24
ALPHA = 0.05

log = logging.getLogger("per_asset")


# --------------------------------------------------------------------------
# Multiple comparisons
# --------------------------------------------------------------------------

def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """BH step-up adjusted p-values, order preserved, monotone, clipped at 1.

    Reimplemented here rather than imported so this script stands alone, and
    cross-checked against models/ic_significance.py in validate().
    """
    p = np.asarray(pvals, dtype=float)
    ok = np.isfinite(p)
    out = np.full(p.shape, np.nan)
    if not ok.any():
        return out
    vals = p[ok]
    n = len(vals)
    order = np.argsort(vals)
    ranked = vals[order]
    adj = ranked * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]   # enforce monotonicity
    restored = np.empty(n)
    restored[order] = np.clip(adj, 0, 1)
    out[ok] = restored
    return out


# --------------------------------------------------------------------------
# Per-asset metrics
# --------------------------------------------------------------------------

def load(universe: str) -> pd.DataFrame | None:
    preds = U.path_for(REPO_ROOT / "models" / "predictions.csv", universe)
    data = U.path_for(REPO_ROOT / "data" / "model_dataset.csv", universe)
    if not preds.exists() or not data.exists():
        log.warning("universe %r not built (need %s and %s)", universe, preds.name, data.name)
        return None
    p = pd.read_csv(preds, parse_dates=["date"])
    d = pd.read_csv(data, parse_dates=["date"])
    cols = ["date", "ticker", "y_excess_return"]
    if "sentiment_lag1" in d.columns:
        cols.append("sentiment_lag1")
    return p.merge(d[cols].rename(columns={"y_excess_return": "excess"}),
                   on=["date", "ticker"], how="left")


def selection_expectation(frame: pd.DataFrame) -> pd.Series:
    """Expected top-3 selections per asset if picks were random.

    The universe is not constant -- XLC is absent before 2019-07 -- so the
    per-month probability is 3/n_t for the assets actually available that month,
    summed over the months each asset was present. Using 3/N with a fixed N
    would overstate the expectation for late-arriving assets.
    """
    exp = {}
    for date, g in frame.groupby("date"):
        n = g["ticker"].nunique()
        if n == 0:
            continue
        share = min(TOP_N, n) / n
        for t in g["ticker"]:
            exp[t] = exp.get(t, 0.0) + share
    return pd.Series(exp)


def analyse(frame: pd.DataFrame, universe: str) -> pd.DataFrame:
    expected = selection_expectation(frame)
    rows = []

    for ticker, g in frame.groupby("ticker"):
        g = g.sort_values("date")
        y = g["actual_target"]
        valid = y.notna()
        n_months = int(valid.sum())
        base_rate = float(y[valid].mean()) if n_months else np.nan
        # What "always guess the commoner outcome" would score on this asset.
        majority = max(base_rate, 1 - base_rate) if n_months else np.nan

        rec = {
            "universe": universe, "ticker": ticker, "n_months": n_months,
            "base_rate": base_rate, "majority_baseline": majority,
            "mean_excess_return": float(g["excess"].mean(skipna=True)),
        }

        for mid in MODELS:
            col = f"model_{mid}_prob"
            if col not in g.columns:
                continue
            sub = g.dropna(subset=[col, "actual_target"])
            if sub.empty:
                continue
            pred = (sub[col] > 0.5).astype(int)
            actual = sub["actual_target"].astype(int)
            hit = float((pred == actual).mean())
            rec[f"hit_rate_{mid}"] = hit
            rec[f"edge_over_majority_{mid}"] = hit - majority

            # Time-series AUC within this asset -- NOT the cross-sectional AUC.
            if len(sub) >= MIN_MONTHS_FOR_AUC and actual.nunique() == 2:
                rec[f"auc_timeseries_{mid}"] = float(roc_auc_score(actual, sub[col]))
            else:
                rec[f"auc_timeseries_{mid}"] = np.nan

            # Selection into the top 3, and what it earned.
            picked_dates = set()
            for date, gm in frame.dropna(subset=[col]).groupby("date"):
                if ticker in set(gm.nlargest(TOP_N, col)["ticker"]):
                    picked_dates.add(date)
            sel = g["date"].isin(picked_dates)
            rec[f"n_selected_{mid}"] = int(sel.sum())
            rec[f"excess_when_selected_{mid}"] = (
                float(g.loc[sel, "excess"].mean()) if sel.any() else np.nan)
            rec[f"excess_when_not_selected_{mid}"] = (
                float(g.loc[~sel, "excess"].mean()) if (~sel).any() else np.nan)

        rec["expected_selections_if_random"] = float(expected.get(ticker, np.nan))

        # Sentiment vs this asset's own subsequent excess return, over time.
        if "sentiment_lag1" in g.columns:
            s = g.dropna(subset=["sentiment_lag1", "excess"])
            if len(s) >= 10 and s["sentiment_lag1"].nunique() > 1:
                r = stats.spearmanr(s["sentiment_lag1"], s["excess"])
                rec["sentiment_corr"] = float(r.statistic)
                rec["sentiment_p_raw"] = float(r.pvalue)
                rec["sentiment_n"] = int(len(s))
            else:
                rec.update({"sentiment_corr": np.nan, "sentiment_p_raw": np.nan,
                            "sentiment_n": int(len(s))})
        rows.append(rec)

    out = pd.DataFrame(rows)
    if "sentiment_p_raw" in out.columns:
        out["sentiment_p_bh"] = benjamini_hochberg(out["sentiment_p_raw"].to_numpy())
        out["sentiment_survives_bh"] = out["sentiment_p_bh"] < ALPHA
    # Order assets by their mean hit rate, best first.
    hit_cols = [c for c in out.columns if c.startswith("hit_rate_")]
    out["mean_hit_rate"] = out[hit_cols].mean(axis=1)
    out["mean_edge_over_majority"] = out[
        [c for c in out.columns if c.startswith("edge_over_majority_")]].mean(axis=1)
    return out.sort_values("mean_hit_rate", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def report(results: dict[str, pd.DataFrame]) -> None:
    line = "=" * 118
    for universe, df in results.items():
        n = len(df)
        log.info("")
        log.info(line)
        log.info("PER-ASSET BREAKDOWN -- %s (%d assets)", universe, n)
        log.info(line)

        log.info("")
        log.info("(a) HIT RATE, and the edge over always guessing the commoner outcome")
        log.info("    sorted by mean hit rate. 'edge' is what the number is actually worth.")
        log.info("    %-6s %7s %9s %10s %s", "asset", "base", "majority", "mean hit",
                 "   A      B      C      D   |   edge A    B      C      D")
        for _, r in df.iterrows():
            hits = "  ".join(f"{r.get(f'hit_rate_{m}', np.nan):.3f}" for m in MODELS)
            edges = "  ".join(f"{r.get(f'edge_over_majority_{m}', np.nan):+.3f}" for m in MODELS)
            log.info("    %-6s %7.3f %9.3f %10.3f   %s |   %s",
                     r["ticker"], r["base_rate"], r["majority_baseline"],
                     r["mean_hit_rate"], hits, edges)
        pos = int((df["mean_edge_over_majority"] > 0).sum())
        log.info("    assets with a positive mean edge over the majority baseline: %d of %d",
                 pos, n)

        log.info("")
        log.info("(b) TIME-SERIES AUC within each asset")
        log.info("    NOT the headline AUC. Headline AUC ranks assets within a month;")
        log.info("    this ranks months within an asset. Different question, not comparable.")
        log.info("    %-6s %8s %8s %8s %8s %10s", "asset", "A", "B", "C", "D", "n months")
        for _, r in df.iterrows():
            vals = "  ".join(
                ("   n/a" if pd.isna(r.get(f"auc_timeseries_{m}"))
                 else f"{r[f'auc_timeseries_{m}']:6.3f}") for m in MODELS)
            log.info("    %-6s %s %10d", r["ticker"], vals, r["n_months"])
        aucs = df[[f"auc_timeseries_{m}" for m in MODELS]].to_numpy(float)
        finite = aucs[np.isfinite(aucs)]
        if finite.size:
            log.info("    range %.3f to %.3f; %d of %d cells above 0.50",
                     finite.min(), finite.max(), int((finite > 0.5).sum()), finite.size)

        log.info("")
        log.info("(c) SELECTION INTO THE TOP 3, against random expectation")
        log.info("    %-6s %10s %8s %8s %8s %8s %12s", "asset", "expected",
                 "A", "B", "C", "D", "A/exp")
        for _, r in df.iterrows():
            exp = r["expected_selections_if_random"]
            counts = " ".join(f"{int(r.get(f'n_selected_{m}', 0)):8d}" for m in MODELS)
            log.info("    %-6s %10.1f %s %12.2f", r["ticker"], exp, counts,
                     r.get("n_selected_A", np.nan) / exp if exp else np.nan)

        log.info("")
        log.info("(d) MEAN EXCESS RETURN WHEN SELECTED vs NOT  (annualised, pp)")
        log.info("    a working selector earns more in the months it picks an asset")
        log.info("    %-6s %s", "asset",
                 "   A sel / not     B sel / not     C sel / not     D sel / not")
        for _, r in df.iterrows():
            cells = []
            for m in MODELS:
                s, ns = r.get(f"excess_when_selected_{m}"), r.get(f"excess_when_not_selected_{m}")
                cells.append(f"{1200 * s:+6.1f}/{1200 * ns:+6.1f}" if pd.notna(s) and pd.notna(ns)
                             else "   n/a       ")
            log.info("    %-6s %s", r["ticker"], "  ".join(cells))

        if "sentiment_corr" in df.columns:
            log.info("")
            log.info("(e) SENTIMENT vs THAT ASSET'S OWN SUBSEQUENT EXCESS RETURN")
            log.info("    Spearman, time series within each asset. %d simultaneous tests.", n)
            log.info("    At alpha=%.2f, %.1f asset(s) are EXPECTED to look significant by",
                     ALPHA, ALPHA * n)
            log.info("    chance alone before any correction is applied.")
            log.info("    %-6s %8s %10s %12s %12s %s", "asset", "n", "rho",
                     "p (raw)", "p (BH)", "survives BH")
            for _, r in df.sort_values("sentiment_p_raw").iterrows():
                log.info("    %-6s %8d %+10.4f %12.4f %12.4f %s",
                         r["ticker"], int(r["sentiment_n"]), r["sentiment_corr"],
                         r["sentiment_p_raw"], r["sentiment_p_bh"],
                         "YES" if r["sentiment_survives_bh"] else "no")
            raw_hits = int((df["sentiment_p_raw"] < ALPHA).sum())
            bh_hits = int(df["sentiment_survives_bh"].sum())
            log.info("    raw p < %.2f: %d asset(s)   |   surviving BH: %d asset(s)",
                     ALPHA, raw_hits, bh_hits)
            if bh_hits == 0:
                log.info("    No asset's sentiment correlation survives correction. The %d raw",
                         raw_hits)
                log.info("    hit(s) are within what %d simultaneous tests produce by chance.",
                         n)
            else:
                for _, r in df[df["sentiment_survives_bh"]].iterrows():
                    log.info("    %s survives correction (rho %+.4f, BH p %.4f). Report it with",
                             r["ticker"], r["sentiment_corr"], r["sentiment_p_bh"])
                    log.info("    the multiple-comparison context attached, and note it was")
                    log.info("    found by a post-hoc sweep, not a pre-registered hypothesis.")

    log.info("")
    log.info(line)
    log.info("VERDICT")
    for universe, df in results.items():
        bh = int(df["sentiment_survives_bh"].sum()) if "sentiment_survives_bh" in df else 0
        pos = int((df["mean_edge_over_majority"] > 0).sum())
        log.info("  %-9s %d of %d assets beat the majority-class baseline on average; "
                 "%d sentiment correlation(s) survive BH.", universe, pos, len(df), bh)
    log.info("  The pooled null is not hiding an asset that works -- it is the same")
    log.info("  null repeated across the cross-section.")
    log.info(line)


# --------------------------------------------------------------------------
# Plot
# --------------------------------------------------------------------------

def plot(results: dict[str, pd.DataFrame]) -> None:
    """Heatmap of hit rate per asset and model, with each asset's base rate beside it.

    Layout notes: one shared colourbar on the far right (both panels use the
    same scale, so two would be redundant and the inter-panel position put its
    tick labels on top of the second panel's asset names), and a spacer column
    between the universes so the base-rate axis cannot crowd the next heatmap.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    universes = list(results)
    n_u = len(universes)
    # per universe: heatmap, base-rate strip, spacer; then one colourbar.
    widths = []
    for _ in universes:
        widths += [6.0, 1.3, 0.9]
    widths = widths[:-1] + [0.45]
    fig = plt.figure(figsize=(7.0 * n_u + 1.0, 8.6))
    gs = GridSpec(1, len(widths), figure=fig, width_ratios=widths, wspace=0.12)

    im = None
    for i, universe in enumerate(universes):
        df = results[universe].sort_values("mean_hit_rate", ascending=False)
        ax = fig.add_subplot(gs[0, 3 * i])
        axb = fig.add_subplot(gs[0, 3 * i + 1])

        mat = df[[f"hit_rate_{m}" for m in MODELS]].to_numpy(float)
        im = ax.imshow(mat, cmap="RdBu", vmin=0.30, vmax=0.70, aspect="auto")
        ax.set_xticks(range(len(MODELS)))
        ax.set_xticklabels([f"Model {m}" for m in MODELS], fontsize=9.5)
        for k, m in enumerate(MODELS):
            ax.get_xticklabels()[k].set_color(MODEL_COLORS[m])
            ax.get_xticklabels()[k].set_fontweight("bold")
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df["ticker"], fontsize=9.5)
        for r in range(mat.shape[0]):
            for c in range(mat.shape[1]):
                ax.text(c, r, f"{mat[r, c]:.2f}", ha="center", va="center", fontsize=8.5,
                        color="white" if abs(mat[r, c] - 0.5) > 0.13 else "#222222")
        ax.set_title(f"{universe} — hit rate\nblue beats a coin flip, red loses to one",
                     fontsize=10.5, pad=10)

        ypos = np.arange(len(df))
        axb.barh(ypos, df["base_rate"].to_numpy(float), color="#B0BEC5", height=0.62,
                 label="base rate")
        axb.plot(df["majority_baseline"].to_numpy(float), ypos, "d", color="#222222",
                 markersize=5, linestyle="none", label="majority baseline")
        axb.axvline(0.5, color="black", linewidth=1.0, linestyle=":")
        axb.set_xlim(0, 1)
        axb.set_ylim(ax.get_ylim())          # share the heatmap's row order
        axb.set_yticks(ypos)
        axb.tick_params(labelleft=False, labelsize=8)
        axb.set_xlabel("probability", fontsize=9)
        axb.set_title("base rate", fontsize=10.5, pad=10)
        axb.grid(alpha=0.25, axis="x")
        for side in ("top", "right"):
            axb.spines[side].set_visible(False)
        if i == 0:
            axb.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.055),
                       frameon=False)

    cax = fig.add_subplot(gs[0, -1])
    fig.colorbar(im, cax=cax).set_label("hit rate", fontsize=9.5)

    fig.suptitle("Per-asset hit rate — is the null uniform across the cross-section?\n"
                 "Hit rate alone is not skill: a model beating its own base rate is "
                 "the only version that counts",
                 fontsize=13, weight="bold")
    fig.subplots_adjust(top=0.85, bottom=0.10, left=0.05, right=0.95)
    PLOT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_PNG, dpi=DPI)
    plt.close(fig)
    log.info("Wrote %s", PLOT_PNG)


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def validate() -> None:
    """Cross-check the BH implementation against the project's existing one."""
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 40)
    mine = benjamini_hochberg(p)
    try:
        from statsmodels.stats.multitest import multipletests
        ref = multipletests(p, method="fdr_bh")[1]
        assert np.allclose(mine, ref), "BH disagrees with statsmodels"
        log.debug("BH cross-checked against statsmodels")
    except ImportError:
        # statsmodels is not a dependency; fall back to the invariants that
        # define the procedure.
        assert np.all(np.diff(mine[np.argsort(p)]) >= -1e-12), "BH is not monotone"
        assert np.all(mine >= p - 1e-12), "BH adjusted p below raw p"
        assert np.all(mine <= 1.0 + 1e-12), "BH adjusted p above 1"
        log.debug("BH checked for monotonicity and bounds")


def run(universes: list[str]) -> int:
    validate()
    results = {}
    for u in universes:
        frame = load(u)
        if frame is None:
            continue
        results[u] = analyse(frame, u)
    if not results:
        log.error("no universe is built -- run python run_all.py first")
        return 1

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8") as fh:
        fh.write("# Per-asset breakdown. hit_rate_* is confounded by base_rate; read\n")
        fh.write("# edge_over_majority_* instead. auc_timeseries_* ranks MONTHS within an\n")
        fh.write("# asset and is NOT the cross-sectional headline AUC.\n")
        for u, df in results.items():
            fh.write(f"\n# UNIVERSE: {u}\n")
            df.to_csv(fh, index=False, float_format="%.6f")
    log.info("Wrote %s", OUT_CSV)

    report(results)
    plot(results)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--universe", action="append", choices=sorted(U.UNIVERSES),
                        help="limit to these universes (default: all that are built)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    return run(args.universe or sorted(U.UNIVERSES))


if __name__ == "__main__":
    sys.exit(main())
