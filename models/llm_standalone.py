"""Standalone tests of the LLM signal, with no model wrapped around it.

`ranking_metrics.py` established that `sentiment_lag1`, `sentiment_rank`, and
`relative_sentiment` induce identical cross-sectional rankings -- within a month,
ranking is a monotone transform of the level and subtracting the monthly mean is
a constant shift. So only two distinct signals exist:

    sentiment_lag1     the level of the LLM's view
    sentiment_change   how much that view moved since last month

Four questions, all asked of the raw signal directly rather than through a
fitted model:

    a) quintile sort   does a higher score earn a higher forward excess return?
    b) regime IC       does the signal work when markets are stressed?
    c) IC persistence  is a good month predictive of the next, or is it noise?
    d) large moves     does a big shift in LLM opinion carry information?

Usage
-----
    python models/llm_standalone.py
    python models/llm_standalone.py --window full   # all 119 months, not just OOS

Outputs
-------
    models/llm_standalone_results.csv
    viz/quintile_returns.png

Window
------
Defaults to the 93 out-of-sample months so the numbers line up with
`ranking_metrics.py`. The signal is never fitted, so evaluating it over the full
119-month panel involves no leakage and gives more power -- ``--window full``
does that, and the summary prints both headline ICs either way.
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

from plotting import DPI  # noqa: E402
sys.path.insert(0, str(REPO_ROOT / "models"))

from regime_analysis import VIX_HIGH, VIX_LOW  # noqa: E402

DATASET = REPO_ROOT / "data" / "model_dataset.csv"
PREDICTIONS = REPO_ROOT / "models" / "predictions.csv"
OUT_CSV = REPO_ROOT / "models" / "llm_standalone_results.csv"
PLOT_PNG = REPO_ROOT / "viz" / "quintile_returns.png"

# Same cutoffs as regime_analysis.py, shorter labels: this script's tables are
# already headed "VIX regime", so repeating "VIX" inside every row is noise.
VIX_LO_LABEL = f"Low (<{VIX_LOW:.0f})"
VIX_MID_LABEL = f"Medium ({VIX_LOW:.0f}-{VIX_HIGH:.0f})"
VIX_HI_LABEL = f"High (>={VIX_HIGH:.0f})"

SIGNALS = {"sentiment_lag1": "LLM level", "sentiment_change": "LLM change"}
N_QUINTILES = 5
BIG_MOVE = 0.3
N_BOOT = 2000
MONTHS = 12
RANDOM_STATE = 0

log = logging.getLogger("llm_standalone")


def classify_vix(v: float) -> str:
    """Same cutoffs as models/regime_analysis.py, imported rather than repeated."""
    return (VIX_LO_LABEL if v < VIX_LOW
            else VIX_HI_LABEL if v >= VIX_HIGH
            else VIX_MID_LABEL)


def load(window: str) -> pd.DataFrame:
    data = pd.read_csv(DATASET, parse_dates=["date"])
    if window == "oos":
        if not PREDICTIONS.exists():
            log.error("%s not found; use --window full", PREDICTIONS)
            sys.exit(1)
        oos = pd.read_csv(PREDICTIONS, parse_dates=["date"])["date"].unique()
        data = data[data["date"].isin(oos)]
    data = data.dropna(subset=list(SIGNALS) + ["y_fwd_return"]).copy()
    # Excess against the mean of the same universe being ranked, matching
    # ranking_metrics.py so the two files' IC figures are comparable.
    data["excess"] = data["y_fwd_return"] - data.groupby("date")["y_fwd_return"].transform("mean")
    data["regime"] = data["VIX_lag1"].map(classify_vix)
    return data


# --------------------------------------------------------------------------
# a) quintiles
# --------------------------------------------------------------------------

def quintiles(data: pd.DataFrame, signal: str) -> tuple[pd.DataFrame, dict]:
    """Sort into 5 bins per month and measure the forward excess return of each.

    Ties are broken by ``rank(method="first")`` before binning. The LLM emits
    scores on a coarse 0.05 grid, so ties are common and ``qcut`` on the raw
    values would collapse bins in many months; ranking first guarantees five
    populated bins every month at the cost of an arbitrary tie order.
    """
    frame = data.dropna(subset=[signal]).copy()
    frame["q"] = (
        frame.groupby("date")[signal]
        .transform(lambda s: pd.qcut(s.rank(method="first"), N_QUINTILES, labels=False) + 1)
    )

    rows = []
    for q in range(1, N_QUINTILES + 1):
        sub = frame[frame["q"] == q]
        monthly = sub.groupby("date")["excess"].mean()
        n = len(monthly)
        se = monthly.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
        rows.append({
            "quintile": q, "n_obs": len(sub), "n_months": n,
            "mean_excess_monthly": float(monthly.mean()),
            "mean_excess_annual": float(monthly.mean() * MONTHS),
            "se_annual": float(se * MONTHS),
            "hit_rate": float((sub["excess"] > 0).mean()),
        })
    table = pd.DataFrame(rows)

    # Long-short spread: the standard test of whether the sort carries signal.
    q_high = frame[frame["q"] == N_QUINTILES].groupby("date")["excess"].mean()
    q_low = frame[frame["q"] == 1].groupby("date")["excess"].mean()
    spread = (q_high - q_low).dropna()
    t_stat, p_val = stats.ttest_1samp(spread, 0.0) if len(spread) > 1 else (np.nan, np.nan)
    mono = stats.spearmanr(table["quintile"], table["mean_excess_annual"]).statistic

    # Count every member of a tie group, not just the duplicates: these are the
    # sectors whose bin is decided by arbitrary tie-break rather than by signal.
    tied = frame.groupby("date")[signal].apply(lambda s: s.duplicated(keep=False).sum()).mean()
    per_month = frame.groupby("date").size().mean()
    summary = {
        "spread_annual": float(spread.mean() * MONTHS),
        "spread_t": float(t_stat), "spread_p": float(p_val),
        "monotonicity_rho": float(mono),
        "strictly_increasing": bool(table["mean_excess_annual"].is_monotonic_increasing),
        "avg_tied_per_month": float(tied),
        "avg_sectors_per_month": float(per_month),
        "tied_share": float(tied / per_month) if per_month else np.nan,
        "exact_zero_share": float((frame[signal] == 0).mean()),
    }
    return table, summary


# --------------------------------------------------------------------------
# b) regime-conditional IC
# --------------------------------------------------------------------------

def monthly_ic(frame: pd.DataFrame, signal: str) -> pd.Series:
    out = {}
    for date, g in frame.groupby("date"):
        g = g.dropna(subset=[signal, "excess"])
        if len(g) < 3 or g[signal].nunique() < 2:
            continue
        out[date] = stats.spearmanr(g[signal], g["excess"]).statistic
    return pd.Series(out).sort_index()


def ic_stats(ic: pd.Series) -> dict:
    n = len(ic)
    mean, sd = float(ic.mean()), float(ic.std(ddof=1))
    t = mean / (sd / np.sqrt(n)) if sd > 0 and n > 1 else np.nan
    return {
        "n_months": n, "mean_ic": mean, "ic_std": sd,
        "ic_ir": mean / sd if sd > 0 else np.nan,
        "pct_positive": float((ic > 0).mean()),
        "t_stat": float(t),
        "p_value": float(2 * (1 - stats.t.cdf(abs(t), n - 1))) if n > 1 else np.nan,
    }


def regime_ic(data: pd.DataFrame, signal: str) -> tuple[pd.DataFrame, dict]:
    ic_all = monthly_ic(data, signal)
    regime_of = data.groupby("date")["regime"].first()

    rows = []
    per_regime = {}
    for label in [VIX_LO_LABEL, VIX_MID_LABEL, VIX_HI_LABEL]:
        months = regime_of[regime_of == label].index
        ic = ic_all.reindex(months).dropna()
        per_regime[label] = ic
        if ic.empty:
            continue
        rows.append({"regime": label, **ic_stats(ic)})
    rows.append({"regime": "All", **ic_stats(ic_all)})

    # High vs low: disjoint month sets, so resample each independently.
    rng = np.random.default_rng(RANDOM_STATE)
    hi, lo = per_regime.get(VIX_HI_LABEL), per_regime.get(VIX_LO_LABEL)
    diff = np.nan, np.nan, np.nan
    if hi is not None and lo is not None and len(hi) > 1 and len(lo) > 1:
        boot = [
            rng.choice(hi, len(hi), replace=True).mean()
            - rng.choice(lo, len(lo), replace=True).mean()
            for _ in range(N_BOOT)
        ]
        diff = (float(hi.mean() - lo.mean()),
                float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
    return pd.DataFrame(rows), {"high_minus_low": diff[0],
                                "high_minus_low_lo": diff[1],
                                "high_minus_low_hi": diff[2]}


# --------------------------------------------------------------------------
# c) IC persistence
# --------------------------------------------------------------------------

def ic_persistence(ic: pd.Series, max_lag: int = 3) -> pd.DataFrame:
    n = len(ic)
    band = 1.96 / np.sqrt(n)  # white-noise 95% band
    rows = []
    for lag in range(1, max_lag + 1):
        a, b = ic.iloc[lag:].to_numpy(), ic.iloc[:-lag].to_numpy()
        r = float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 else np.nan
        rows.append({"lag": lag, "autocorr": r, "white_noise_band": float(band),
                     "outside_band": bool(abs(r) > band)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# d) large opinion shifts
# --------------------------------------------------------------------------

def big_moves(data: pd.DataFrame) -> pd.DataFrame:
    frame = data.dropna(subset=["sentiment_change", "excess"])
    buckets = {
        f"large positive (> +{BIG_MOVE})": frame["sentiment_change"] > BIG_MOVE,
        f"large negative (< -{BIG_MOVE})": frame["sentiment_change"] < -BIG_MOVE,
        f"small (|change| <= {BIG_MOVE})": frame["sentiment_change"].abs() <= BIG_MOVE,
    }
    rows = []
    for label, mask in buckets.items():
        sub = frame[mask]
        if sub.empty:
            continue
        t, p = stats.ttest_1samp(sub["excess"], 0.0) if len(sub) > 1 else (np.nan, np.nan)
        rows.append({
            "bucket": label, "n_obs": len(sub),
            "share_of_sample": len(sub) / len(frame),
            "mean_excess_monthly": float(sub["excess"].mean()),
            "mean_excess_annual": float(sub["excess"].mean() * MONTHS),
            "hit_rate": float((sub["excess"] > 0).mean()),
            "t_stat": float(t), "p_value": float(p),
            "corr_change_excess": float(sub["sentiment_change"].corr(sub["excess"])),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def run(window: str) -> int:
    if not DATASET.exists():
        log.error("%s not found", DATASET)
        return 1

    data = load(window)
    log.info("window=%s: %s -> %s, %d months, %d rows",
             window, data["date"].min().date(), data["date"].max().date(),
             data["date"].nunique(), len(data))

    records: list[dict] = []
    quint_tables, ic_series = {}, {}

    for signal, label in SIGNALS.items():
        table, summary = quintiles(data, signal)
        quint_tables[signal] = table
        for _, r in table.iterrows():
            records.append({"analysis": "quintile", "signal": signal,
                            "group": f"Q{int(r['quintile'])}", **r.to_dict()})
        records.append({"analysis": "quintile_spread", "signal": signal,
                        "group": "Q5-Q1", **summary})

        rtable, rsummary = regime_ic(data, signal)
        for _, r in rtable.iterrows():
            records.append({"analysis": "regime_ic", "signal": signal,
                            "group": r["regime"], **r.to_dict()})
        records.append({"analysis": "regime_ic_diff", "signal": signal,
                        "group": "High - Low", **rsummary})

        ic = monthly_ic(data, signal)
        ic_series[signal] = ic
        for _, r in ic_persistence(ic).iterrows():
            records.append({"analysis": "ic_autocorr", "signal": signal,
                            "group": f"lag {int(r['lag'])}", **r.to_dict()})

    for _, r in big_moves(data).iterrows():
        records.append({"analysis": "change_magnitude", "signal": "sentiment_change",
                        "group": r["bucket"], **r.to_dict()})

    results = pd.DataFrame(records)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT_CSV, index=False, float_format="%.5f")
    log.info("Wrote %s (%d rows)", OUT_CSV, len(results))

    report(data, quint_tables, ic_series, results, window)
    plot(quint_tables, results)
    return 0


def report(data, quint_tables, ic_series, results, window) -> None:
    log.info("")
    log.info("=" * 84)
    log.info("(a) QUINTILE SORT -- forward excess return by signal bin, annualised")
    for signal, label in SIGNALS.items():
        t = quint_tables[signal]
        s = results[(results.analysis == "quintile_spread") & (results.signal == signal)].iloc[0]
        log.info("")
        log.info("  %s (%s)", label, signal)
        log.info("    %-6s %8s %10s %10s %9s", "bin", "n", "excess/yr", "+/- 1 SE", "hit rate")
        for _, r in t.iterrows():
            log.info("    %-6s %8d %9.2f%% %9.2f%% %8.1f%%",
                     f"Q{int(r['quintile'])}", int(r["n_obs"]),
                     r["mean_excess_annual"] * 100, r["se_annual"] * 100,
                     r["hit_rate"] * 100)
        log.info("    Q5-Q1 spread %+.2f%%/yr  (t=%.2f, p=%.3f) | monotonicity rho %+.2f | "
                 "strictly increasing: %s",
                 s["spread_annual"] * 100, s["spread_t"], s["spread_p"],
                 s["monotonicity_rho"], s["strictly_increasing"])
        log.info("    resolution: %.1f of %.1f sectors per month sit in a tie group (%.0f%%); "
                 "%.0f%% of values are exactly zero",
                 s["avg_tied_per_month"], s["avg_sectors_per_month"],
                 s["tied_share"] * 100, s["exact_zero_share"] * 100)
        if s["tied_share"] > 0.4:
            log.info("    -> a majority of bin assignments are arbitrary tie-breaks, not signal;")
            log.info("       read this sort as low-resolution regardless of the spread.")

    log.info("")
    log.info("=" * 84)
    log.info("(b) REGIME-CONDITIONAL IC")
    for signal, label in SIGNALS.items():
        sub = results[(results.analysis == "regime_ic") & (results.signal == signal)]
        d = results[(results.analysis == "regime_ic_diff") & (results.signal == signal)].iloc[0]
        log.info("")
        log.info("  %s (%s)", label, signal)
        log.info("    %-16s %7s %9s %8s %9s %8s", "regime", "months", "mean IC", "IC IR", "%pos", "t")
        for _, r in sub.iterrows():
            log.info("    %-16s %7d %9.4f %8.3f %8.1f%% %8.2f",
                     r["group"], int(r["n_months"]), r["mean_ic"], r["ic_ir"],
                     r["pct_positive"] * 100, r["t_stat"])
        verdict = ("differs" if (d["high_minus_low_lo"] > 0 or d["high_minus_low_hi"] < 0)
                   else "no detectable difference")
        log.info("    high-vol minus low-vol IC: %+.4f  95%% CI [%+.4f, %+.4f] -> %s",
                 d["high_minus_low"], d["high_minus_low_lo"], d["high_minus_low_hi"], verdict)

    log.info("")
    log.info("=" * 84)
    log.info("(c) IC PERSISTENCE -- autocorrelation of the monthly IC series")
    for signal, label in SIGNALS.items():
        sub = results[(results.analysis == "ic_autocorr") & (results.signal == signal)]
        band = sub["white_noise_band"].iloc[0]
        vals = "  ".join(f"lag{int(r['lag'])} {r['autocorr']:+.3f}" for _, r in sub.iterrows())
        exceed = sub[sub["outside_band"]]
        if exceed.empty:
            verdict = "indistinguishable from white noise"
        else:
            # A negative coefficient is mean reversion, not persistence -- and with
            # three lags tested at 5%, one marginal exceedance is expected ~14% of
            # the time by chance.
            r = exceed.iloc[0]
            kind = "persistence" if r["autocorr"] > 0 else "mean reversion"
            verdict = (f"lag {int(r['lag'])} shows {kind}, but 1 of 3 lags outside the "
                       f"band is ~14% likely by chance")
        log.info("  %-12s %s   (white-noise band +/-%.3f)", label, vals, band)
        log.info("               -> %s", verdict)

    log.info("")
    log.info("=" * 84)
    log.info("(d) DOES A BIG SHIFT IN LLM OPINION PREDICT ANYTHING?")
    sub = results[results.analysis == "change_magnitude"]
    log.info("  %-28s %7s %8s %11s %9s %7s %9s",
             "bucket", "n", "share", "excess/yr", "hit rate", "t", "corr")
    for _, r in sub.iterrows():
        log.info("  %-28s %7d %7.1f%% %10.2f%% %8.1f%% %7.2f %9.3f",
                 r["group"], int(r["n_obs"]), r["share_of_sample"] * 100,
                 r["mean_excess_annual"] * 100, r["hit_rate"] * 100,
                 r["t_stat"], r["corr_change_excess"])
    log.info("=" * 84)

    # Cross-window robustness: the signal is unfitted, so the other window is a
    # free extra look rather than a second bite at a fitted model.
    other = "full" if window == "oos" else "oos"
    try:
        alt = load(other)
        log.info("")
        log.info("Robustness -- same signals over the %s window (%d months):",
                 other, alt["date"].nunique())
        for signal, label in SIGNALS.items():
            s = ic_stats(monthly_ic(alt, signal))
            log.info("  %-12s mean IC %+.4f (IR %.3f, %.0f%% positive, t=%.2f)",
                     label, s["mean_ic"], s["ic_ir"], s["pct_positive"] * 100, s["t_stat"])
    except SystemExit:
        pass


def plot(quint_tables: dict, results: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    colors = ["#B0413E", "#D98880", "#BDBDBD", "#7FB3D5", "#1F618D"]

    for ax, (signal, label) in zip(axes, SIGNALS.items()):
        t = quint_tables[signal]
        s = results[(results.analysis == "quintile_spread") & (results.signal == signal)].iloc[0]
        vals = t["mean_excess_annual"] * 100
        errs = t["se_annual"] * 100
        ax.bar(t["quintile"], vals, yerr=errs, capsize=4, color=colors,
               edgecolor="white", linewidth=0.8)
        ax.axhline(0, color="black", linewidth=1.1)
        ax.set_xticks(t["quintile"])
        ax.set_xticklabels([f"Q{int(q)}" for q in t["quintile"]])
        ax.set_xlabel("signal quintile  (Q1 = most bearish, Q5 = most bullish)", fontsize=10)
        ax.set_title(f"{label}  ({signal})\n"
                     f"Q5−Q1 {s['spread_annual']*100:+.2f}%/yr  "
                     f"(t={s['spread_t']:.2f}, p={s['spread_p']:.2f})", fontsize=11)
        ax.grid(axis="y", alpha=0.25)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    axes[0].set_ylabel("mean forward excess return, annualised (%)", fontsize=11)
    fig.suptitle("LLM signal quintile sort — does a higher score earn a higher return?",
                 fontsize=14, weight="bold")
    fig.text(0.5, 0.005,
             "Error bars are ±1 standard error. A working signal rises monotonically "
             "from Q1 to Q5; neither does.",
             ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    PLOT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_PNG, dpi=DPI)
    plt.close(fig)
    log.info("Wrote %s", PLOT_PNG)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--window", choices=["oos", "full"], default="oos")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s", datefmt="%H:%M:%S",
    )
    return run(args.window)


if __name__ == "__main__":
    sys.exit(main())
