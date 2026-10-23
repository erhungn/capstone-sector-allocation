"""11-asset vs 14-asset: does the null result survive adding bonds and gold?

    python models/universe_comparison.py

Output
------
    models/universe_comparison.csv   (multi-block)

The question
------------
FOMC communication bears on duration and on gold far more directly than on
equity sectors. If a monetary-policy signal exists anywhere in this design, the
extended universe is where it should be easiest to find -- so a null that
survives TLT, IEF and GLD is a stronger null than one measured on equities
alone.

The 11-asset specification remains the main result. This is an extension, and
its artefacts carry a ``_14asset`` suffix throughout.

What is NOT comparable, and why it is reported rather than adjusted
-------------------------------------------------------------------
Adding bonds changes what the target means. The label asks whether an asset
beat the equal-weight average of its universe that month, and IEF carries about
a third of the monthly volatility of a typical equity sector. In an up month a
low-volatility asset is close to guaranteed a below-average return whatever any
forecast says, so a raw-return ranking in the 14-asset universe partly ranks on
volatility.

That is a property of the extended universe, not an artefact to normalise away,
and normalising it (ranking on risk-adjusted rather than raw returns) would
change the target definition -- which this extension deliberately does not do.
So the volatility spread is measured and reported alongside the headline
metrics, and the headline metrics are read in its light.
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
sys.path.insert(0, str(REPO_ROOT / "models"))

import universe as U  # noqa: E402
from sentiment_diagnostics import ols_r2  # noqa: E402

OUT_CSV = REPO_ROOT / "models" / "universe_comparison.csv"
MACRO_CSV = REPO_ROOT / "data" / "macro_features.csv"
PRICES_CSV = REPO_ROOT / "data" / "sector_prices.csv"

MODELS = ["A", "B", "C", "D"]
PAIRS = [("A", "B", "Logistic (B−A)"), ("C", "D", "XGBoost (D−C)")]
TOP_N = 3
N_BOOT = 2_000
RANDOM_STATE = 42

log = logging.getLogger("universe_comparison")


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------

def load(universe: str) -> dict | None:
    """Everything one universe contributes, or None if it has not been built."""
    paths = {
        "predictions": U.path_for(REPO_ROOT / "models" / "predictions.csv", universe),
        "dataset": U.path_for(REPO_ROOT / "data" / "model_dataset.csv", universe),
        "performance": U.path_for(REPO_ROOT / "backtest" / "performance.csv", universe),
        "sentiment": U.path_for(REPO_ROOT / "data" / "llm_sentiment.csv", universe),
    }
    missing = [str(p.name) for p in paths.values() if not p.exists()]
    if missing:
        log.warning("universe %r is not built yet (missing %s)", universe, ", ".join(missing))
        return None

    preds = pd.read_csv(paths["predictions"], parse_dates=["date"])
    data = pd.read_csv(paths["dataset"], parse_dates=["date"])
    frame = preds.merge(
        data[["date", "ticker", "y_excess_return"]].rename(
            columns={"y_excess_return": "excess"}),
        on=["date", "ticker"], how="left")
    return {
        "universe": universe,
        "tickers": U.tickers(universe),
        "frame": frame,
        "dataset": data,
        "performance": pd.read_csv(paths["performance"]),
        "sentiment": pd.read_csv(paths["sentiment"], index_col=0, parse_dates=True),
    }


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------

def month_blocks(frame: pd.DataFrame):
    months = frame["date"].to_numpy()
    uniq = np.unique(months)
    return uniq, {m: np.flatnonzero(months == m) for m in uniq}


def auc_ci(frame: pd.DataFrame, col: str, n_boot: int) -> tuple[float, float, float]:
    """AUC with a month-clustered bootstrap CI, matching the rest of the project."""
    sub = frame.dropna(subset=[col, "actual_target"])
    y = sub["actual_target"].to_numpy(int)
    p = sub[col].to_numpy(float)
    uniq, idx = month_blocks(sub)
    rng = np.random.default_rng(RANDOM_STATE)
    draws = []
    for _ in range(n_boot):
        ii = np.concatenate([idx[m] for m in rng.choice(uniq, len(uniq), replace=True)])
        if len(np.unique(y[ii])) < 2:
            continue
        draws.append(roc_auc_score(y[ii], p[ii]))
    return (float(roc_auc_score(y, p)),
            float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))


def delta_auc_ci(frame: pd.DataFrame, base: str, llm: str, n_boot: int) -> dict:
    """Paired AUC increment, same month blocks for both arms."""
    sub = frame.dropna(subset=[base, llm, "actual_target"])
    y = sub["actual_target"].to_numpy(int)
    pb, pl = sub[base].to_numpy(float), sub[llm].to_numpy(float)
    uniq, idx = month_blocks(sub)
    rng = np.random.default_rng(RANDOM_STATE)
    draws = []
    for _ in range(n_boot):
        ii = np.concatenate([idx[m] for m in rng.choice(uniq, len(uniq), replace=True)])
        if len(np.unique(y[ii])) < 2:
            continue
        draws.append(roc_auc_score(y[ii], pl[ii]) - roc_auc_score(y[ii], pb[ii]))
    lo, hi = float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))
    return {"delta_auc": float(roc_auc_score(y, pl) - roc_auc_score(y, pb)),
            "ci_lo": lo, "ci_hi": hi, "ci_excludes_zero": bool(lo > 0 or hi < 0)}


def mean_ic(frame: pd.DataFrame, col: str) -> tuple[float, float]:
    """Mean monthly Spearman IC and its information ratio."""
    vals = []
    for _, g in frame.groupby("date"):
        g = g.dropna(subset=[col, "excess"])
        if len(g) < 3 or g[col].nunique() < 2:
            continue
        vals.append(stats.spearmanr(g[col], g["excess"]).statistic)
    s = pd.Series(vals)
    return float(s.mean()), float(s.mean() / s.std(ddof=1)) if s.std(ddof=1) > 0 else np.nan


def redundancy(sentiment: pd.DataFrame, tickers: list[str]) -> dict:
    """R² of the sentiment components on the macro block, for one universe."""
    macro = pd.read_csv(MACRO_CSV, index_col=0, parse_dates=True)
    macro.index = pd.DatetimeIndex(macro.index).normalize()
    senti = sentiment.copy()
    senti.index = pd.DatetimeIndex(senti.index).normalize()
    senti = senti[[t for t in tickers if t in senti.columns]]
    cols = list(macro.columns)

    mean_tone = senti.mean(axis=1).reindex(macro.index)
    dispersion = senti.std(axis=1, ddof=1).reindex(macro.index)
    r2_mean, adj_mean = ols_r2(macro[cols], mean_tone)
    r2_disp, adj_disp = ols_r2(macro[cols], dispersion)

    # Pooled and relative use the full scored matrix (every month x every asset)
    # rather than the modelling panel, so the two universes are measured on the
    # same basis. models/sentiment_diagnostics.py pools over the filtered panel
    # instead, which is why its pooled R^2 differs in the third decimal; the
    # headline mean-tone figure is identical.
    long = senti.reset_index().melt(id_vars=senti.index.name or "index",
                                    var_name="ticker", value_name="score")
    long.columns = ["date", "ticker", "score"]
    long = long.merge(macro.reset_index().rename(columns={macro.index.name or "index": "date"}),
                      on="date", how="left").dropna()
    rel = long["score"] - long.groupby("date")["score"].transform("mean")
    r2_pool, adj_pool = ols_r2(long[cols], long["score"])
    r2_rel, adj_rel = ols_r2(long[cols], rel)
    return {"r2_mean_tone": r2_mean, "adj_r2_mean_tone": adj_mean,
            "r2_dispersion": r2_disp, "adj_r2_dispersion": adj_disp,
            "r2_pooled": r2_pool, "adj_r2_pooled": adj_pool,
            "r2_relative": r2_rel, "adj_r2_relative": adj_rel}


def selection_counts(frame: pd.DataFrame, extended: list[str]) -> list[dict]:
    """How often each model's top-3 contains a bond or gold.

    Systematic avoidance or systematic selection is itself a finding: it would
    mean the model is expressing a fixed asset-class view rather than a monthly
    forecast.
    """
    rows = []
    for mid in MODELS:
        col = f"model_{mid}_prob"
        if col not in frame.columns:
            continue
        months, picked, slots = 0, 0, 0
        per_asset = dict.fromkeys(extended, 0)
        for _, g in frame.dropna(subset=[col]).groupby("date"):
            months += 1
            top = g.nlargest(TOP_N, col)["ticker"].tolist()
            hits = [t for t in top if t in extended]
            slots += len(hits)
            picked += bool(hits)
            for t in hits:
                per_asset[t] += 1
        share = len(extended) / frame["ticker"].nunique()
        rows.append({
            "model": mid, "months": months,
            "months_with_an_extended_asset": picked,
            "pct_months_with_extended": picked / months if months else np.nan,
            "extended_slots": slots,
            "pct_of_all_slots": slots / (months * TOP_N) if months else np.nan,
            "expected_pct_if_random": share,
            **{f"n_months_{t}": per_asset[t] for t in extended},
        })
    return rows


def mechanical_decomposition(frame: pd.DataFrame) -> pd.DataFrame:
    """Is the 14-asset AUC gain forecasting, or just an asset-class tilt?

    Two controls settle it.

    *Static rule*: score every equity sector 1 and every bond/gold 0, and never
    update. It contains no information whatsoever -- no macro, no sentiment, no
    timing -- so whatever AUC it earns is the part of the 14-asset score that
    comes from knowing which asset class a row belongs to.

    *Equities only*: recompute each fitted model's AUC on the equity rows alone.
    If a model has genuine cross-sectional skill it survives the restriction;
    if its edge was the asset-class split, it collapses back to the 11-asset
    level.
    """
    sub = frame.dropna(subset=["actual_target"])
    y = sub["actual_target"].to_numpy(int)
    is_equity = (~sub["ticker"].isin(U.EXTENDED_ASSETS)).astype(float).to_numpy()
    eq = sub[~sub["ticker"].isin(U.EXTENDED_ASSETS)]

    rows = [{"signal": "static: equity=1, bond/gold=0 (no timing)",
             "auc_full_universe": float(roc_auc_score(y, is_equity)),
             "auc_equities_only": np.nan, "mean_ic_equities_only": np.nan}]
    for mid in MODELS:
        col = f"model_{mid}_prob"
        if col not in sub.columns:
            continue
        e = eq.dropna(subset=[col, "actual_target"])
        ic, _ = mean_ic(eq, col)
        rows.append({
            "signal": f"Model {mid}",
            "auc_full_universe": float(roc_auc_score(y, sub[col].to_numpy(float))),
            "auc_equities_only": float(roc_auc_score(e["actual_target"].astype(int), e[col])),
            "mean_ic_equities_only": ic,
        })
    base = sub.groupby("ticker")["actual_target"].mean()
    ext = [t for t in U.EXTENDED_ASSETS if t in base.index]
    rows.append({"signal": "P(beat benchmark): equity sectors",
                 "auc_full_universe": float(base[~base.index.isin(ext)].mean()),
                 "auc_equities_only": np.nan, "mean_ic_equities_only": np.nan})
    rows.append({"signal": "P(beat benchmark): bonds + gold",
                 "auc_full_universe": float(base[base.index.isin(ext)].mean()),
                 "auc_equities_only": np.nan, "mean_ic_equities_only": np.nan})
    return pd.DataFrame(rows)


def volatility_table(tickers: list[str]) -> pd.DataFrame:
    prices = pd.read_csv(PRICES_CSV, index_col=0, parse_dates=True)
    rets = prices[[t for t in tickers if t in prices.columns]].pct_change(fill_method=None)
    vol = rets.std(ddof=1)
    eq_median = vol[[t for t in U.EQUITY_SECTORS if t in vol.index]].median()
    return pd.DataFrame({
        "ticker": vol.index,
        "asset_class": ["bond/gold" if t in U.EXTENDED_ASSETS else "equity sector"
                        for t in vol.index],
        "monthly_vol": vol.values,
        "annualised_vol": vol.values * np.sqrt(12),
        "vs_equity_median": vol.values / eq_median,
    })


def benchmark_table() -> pd.DataFrame:
    prices = pd.read_csv(PRICES_CSV, index_col=0, parse_dates=True)
    rets = prices.pct_change(fill_method=None)
    rows = []
    for name in ("11asset", "14asset"):
        cols = [t for t in U.tickers(name) if t in rets.columns]
        b = rets[cols].mean(axis=1, skipna=True).dropna()
        rows.append({
            "universe": name, "n_assets": len(cols), "months": len(b),
            "mean_monthly_return": float(b.mean()),
            "monthly_vol": float(b.std(ddof=1)),
            "annualised_vol": float(b.std(ddof=1) * np.sqrt(12)),
            "cagr": float((1 + b).prod() ** (12 / len(b)) - 1),
        })
    out = pd.DataFrame(rows)
    a, c = out.iloc[0], out.iloc[1]
    log.debug("benchmark vol change %.1f%%", 100 * (c["annualised_vol"] / a["annualised_vol"] - 1))
    return out


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def run(n_boot: int) -> int:
    loaded = {u: load(u) for u in ("11asset", "14asset")}
    available = {u: v for u, v in loaded.items() if v is not None}
    if "11asset" not in available:
        log.error("the main specification is not built -- run python run_all.py first")
        return 1

    model_rows, delta_rows, redundancy_rows, selection_rows = [], [], [], []
    mech = None
    for name, d in available.items():
        frame = d["frame"]
        for mid in MODELS:
            col = f"model_{mid}_prob"
            if col not in frame.columns:
                continue
            auc, lo, hi = auc_ci(frame, col, n_boot)
            ic, ic_ir = mean_ic(frame, col)
            perf = d["performance"]
            prow = perf[perf["strategy"] == f"Model {mid}"]
            model_rows.append({
                "universe": name, "n_assets": len(d["tickers"]),
                "model": mid, "n_obs": int(frame[col].notna().sum()),
                "n_months": int(frame["date"].nunique()),
                "auc": auc, "auc_ci_lo": lo, "auc_ci_hi": hi,
                "mean_ic": ic, "ic_ir": ic_ir,
                "cagr": float(prow["cagr"].iloc[0]) if not prow.empty else np.nan,
                "sharpe": float(prow["sharpe"].iloc[0]) if not prow.empty else np.nan,
                "max_drawdown": float(prow["max_drawdown"].iloc[0]) if not prow.empty else np.nan,
            })
        for base, llm, label in PAIRS:
            b, l = f"model_{base}_prob", f"model_{llm}_prob"
            if b in frame.columns and l in frame.columns:
                delta_rows.append({"universe": name, "pair": label,
                                   **delta_auc_ci(frame, b, l, n_boot)})
        # raw sentiment as a standalone signal
        if "sentiment_lag1" in d["dataset"].columns:
            raw = frame.merge(d["dataset"][["date", "ticker", "sentiment_lag1"]],
                              on=["date", "ticker"], how="left")
            ic, ic_ir = mean_ic(raw, "sentiment_lag1")
            model_rows.append({"universe": name, "n_assets": len(d["tickers"]),
                               "model": "raw sentiment",
                               "n_obs": int(raw["sentiment_lag1"].notna().sum()),
                               "n_months": int(raw["date"].nunique()),
                               "auc": np.nan, "auc_ci_lo": np.nan, "auc_ci_hi": np.nan,
                               "mean_ic": ic, "ic_ir": ic_ir,
                               "cagr": np.nan, "sharpe": np.nan, "max_drawdown": np.nan})
        redundancy_rows.append({"universe": name, "n_assets": len(d["tickers"]),
                                **redundancy(d["sentiment"], d["tickers"])})
        if name == "14asset":
            selection_rows = selection_counts(frame, U.EXTENDED_ASSETS)
            mech = mechanical_decomposition(frame)

    models = pd.DataFrame(model_rows)
    deltas = pd.DataFrame(delta_rows)
    reds = pd.DataFrame(redundancy_rows)
    sel = pd.DataFrame(selection_rows)
    vols = volatility_table(U.tickers("14asset"))
    bench = benchmark_table()

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8") as fh:
        fh.write("# 11-asset (main specification) vs 14-asset (extension)\n")
        fh.write("# MODEL METRICS\n")
        models.to_csv(fh, index=False, float_format="%.6f")
        fh.write("\n# LLM INCREMENT (paired, month-clustered bootstrap)\n")
        deltas.to_csv(fh, index=False, float_format="%.6f")
        fh.write("\n# REDUNDANCY: R-SQUARED OF SENTIMENT ON THE MACRO BLOCK\n")
        reds.to_csv(fh, index=False, float_format="%.6f")
        fh.write("\n# REALISED VOLATILITY BY ASSET\n")
        vols.to_csv(fh, index=False, float_format="%.6f")
        fh.write("\n# EQUAL-WEIGHT BENCHMARK\n")
        bench.to_csv(fh, index=False, float_format="%.6f")
        if not sel.empty:
            fh.write("\n# TOP-3 SELECTION OF BONDS AND GOLD (14-asset)\n")
            sel.to_csv(fh, index=False, float_format="%.6f")
        if mech is not None:
            fh.write("\n# IS THE 14-ASSET GAIN FORECASTING OR AN ASSET-CLASS TILT?\n")
            mech.to_csv(fh, index=False, float_format="%.6f")
    log.info("Wrote %s", OUT_CSV)

    report(models, deltas, reds, vols, bench, sel, mech, set(available))
    return 0


def report(models, deltas, reds, vols, bench, sel, mech, available) -> None:
    line = "=" * 104
    log.info("")
    log.info(line)
    log.info("MODEL METRICS BY UNIVERSE")
    log.info("%-10s %-14s %8s %20s %10s %8s %9s %8s %10s",
             "universe", "model", "AUC", "95% CI", "mean IC", "IC IR", "CAGR",
             "Sharpe", "max DD")
    for _, r in models.iterrows():
        auc = "      —" if pd.isna(r["auc"]) else f"{r['auc']:7.4f}"
        ci = "                  —" if pd.isna(r["auc_ci_lo"]) else \
            f"  [{r['auc_ci_lo']:.4f}, {r['auc_ci_hi']:.4f}]"
        cagr = "       —" if pd.isna(r["cagr"]) else f"{100 * r['cagr']:8.2f}%"
        shar = "       —" if pd.isna(r["sharpe"]) else f"{r['sharpe']:8.2f}"
        mdd = "         —" if pd.isna(r["max_drawdown"]) else f"{100 * r['max_drawdown']:9.2f}%"
        log.info("%-10s %-14s %8s %20s %+10.4f %8.2f %9s %8s %10s",
                 r["universe"], r["model"], auc, ci, r["mean_ic"], r["ic_ir"],
                 cagr, shar, mdd)

    log.info("")
    log.info(line)
    log.info("LLM INCREMENT")
    log.info("%-10s %-18s %12s %26s %s", "universe", "pair", "ΔAUC", "95% CI", "verdict")
    for _, r in deltas.iterrows():
        log.info("%-10s %-18s %+12.4f   [%+.4f, %+.4f]  %s",
                 r["universe"], r["pair"], r["delta_auc"], r["ci_lo"], r["ci_hi"],
                 "EXCLUDES ZERO" if r["ci_excludes_zero"] else "spans zero")

    log.info("")
    log.info(line)
    log.info("REDUNDANCY -- variance of sentiment explained by the 14-variable macro block")
    log.info("%-10s %8s %14s %14s %12s %12s", "universe", "assets", "mean tone",
             "dispersion", "pooled", "relative")
    for _, r in reds.iterrows():
        log.info("%-10s %8d %14.4f %14.4f %12.4f %12.4f", r["universe"], r["n_assets"],
                 r["r2_mean_tone"], r["r2_dispersion"], r["r2_pooled"], r["r2_relative"])

    log.info("")
    log.info(line)
    log.info("CONCEPTUAL FLAG 1 -- cross-sectional volatility heterogeneity")
    eq = vols[vols["asset_class"] == "equity sector"]
    log.info("%-8s %-16s %14s %14s %18s", "ticker", "class", "monthly vol",
             "annualised", "vs equity median")
    for _, r in vols.iterrows():
        log.info("%-8s %-16s %13.4f%% %13.2f%% %17.2fx", r["ticker"], r["asset_class"],
                 100 * r["monthly_vol"], 100 * r["annualised_vol"], r["vs_equity_median"])
    log.info("  equity spread (max/min)   : %.2fx", eq["monthly_vol"].max() / eq["monthly_vol"].min())
    log.info("  14-asset spread (max/min) : %.2fx",
             vols["monthly_vol"].max() / vols["monthly_vol"].min())
    log.info("  The target ranks RAW returns, so in an up month the low-volatility")
    log.info("  assets are near-certain to sit below the cross-sectional average")
    log.info("  regardless of any forecast. Read the 14-asset AUC in that light.")

    log.info("")
    log.info(line)
    log.info("CONCEPTUAL FLAG 2 -- how often bonds and gold reach the top 3")
    if sel.empty:
        log.info("  (needs the 14-asset models; not built yet)")
    else:
        log.info("%-8s %10s %24s %20s %22s", "model", "months", "months w/ an extended",
                 "share of all slots", "random expectation")
        for _, r in sel.iterrows():
            log.info("%-8s %10d %18d (%4.1f%%) %17.1f%% %21.1f%%",
                     r["model"], r["months"], r["months_with_an_extended_asset"],
                     100 * r["pct_months_with_extended"], 100 * r["pct_of_all_slots"],
                     100 * r["expected_pct_if_random"])
        for t in U.EXTENDED_ASSETS:
            col = f"n_months_{t}"
            if col in sel.columns:
                log.info("    %s picked in: %s", t,
                         ", ".join(f"{m}={int(n)}" for m, n in zip(sel["model"], sel[col])))

    if mech is not None:
        log.info("")
        log.info(line)
        log.info("IS THE 14-ASSET GAIN FORECASTING, OR AN ASSET-CLASS TILT?")
        log.info("%-44s %18s %20s %22s", "signal", "AUC (all 14)",
                 "AUC (equities only)", "mean IC (equities)")
        for _, r in mech.iterrows():
            a = f"{r['auc_full_universe']:18.4f}"
            e = "                   —" if pd.isna(r["auc_equities_only"]) else f"{r['auc_equities_only']:20.4f}"
            i = "                     —" if pd.isna(r["mean_ic_equities_only"]) else f"{r['mean_ic_equities_only']:+22.4f}"
            log.info("%-44s %s %s %s", r["signal"], a, e, i)
        log.info("  The static rule carries no information at all, yet it outscores every")
        log.info("  fitted model on the full universe. Restricted to equities, the models")
        log.info("  fall back to their 11-asset level. The 14-asset AUC gain is an")
        log.info("  asset-class tilt, not forecasting skill.")

    log.info("")
    log.info(line)
    log.info("CONCEPTUAL FLAG 3 -- the equal-weight benchmark changes")
    log.info("%-10s %8s %20s %16s %14s %10s", "universe", "assets", "mean monthly",
             "monthly vol", "ann. vol", "CAGR")
    for _, r in bench.iterrows():
        log.info("%-10s %8d %19.4f%% %15.4f%% %13.2f%% %9.2f%%",
                 r["universe"], r["n_assets"], 100 * r["mean_monthly_return"],
                 100 * r["monthly_vol"], 100 * r["annualised_vol"], 100 * r["cagr"])
    a, c = bench.iloc[0], bench.iloc[1]
    log.info("  annualised volatility falls %.1f%%, CAGR falls %.2f pp.",
             100 * (1 - c["annualised_vol"] / a["annualised_vol"]),
             100 * (a["cagr"] - c["cagr"]))
    log.info("  The benchmark the target is defined against is materially different,")
    log.info("  so the two universes' labels are not the same question.")

    if "14asset" not in available:
        log.info("")
        log.info("  NOTE: the 14-asset universe is not built, so only the main")
        log.info("  specification appears above. Build it with:")
        log.info("    python llm/sentiment_pipeline.py --universe 14asset --force")
        log.info("    python data/build_dataset.py   --universe 14asset")
        log.info("    python models/train.py         --universe 14asset")
        log.info("    python backtest/run_backtest.py --universe 14asset")
    log.info(line)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    return run(args.n_boot)


if __name__ == "__main__":
    sys.exit(main())
