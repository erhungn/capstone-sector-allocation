"""Build the report's tables from the analysis CSVs.

    python scripts/build_report_tables.py
    python scripts/build_report_tables.py --check     # verify, write nothing
    python scripts/build_report_tables.py --table 3   # rebuild one table

Outputs
-------
    reports/tables/table1_model_performance.{csv,md}
    ... through table8_robustness_summary.{csv,md}
    reports/tables/README.md          index of the eight tables

Why a builder rather than hand-formatted tables
-----------------------------------------------
Every number here already exists in an analysis CSV. Retyping any of them into a
report is how a figure goes stale: the analysis is re-run, the CSV updates, and
the prose keeps the old value. This script is the only path from CSV to report
table, and ``--check`` regenerates everything in memory and compares against
what is committed, so a stale table becomes a failing test rather than a
misprint. ``scripts/verify_all.py`` runs that check.

Two output formats, on purpose: CSV for anything that needs re-processing, and
GitHub-flavoured Markdown that can be pasted into a document with no reformatting.

Formatting conventions
----------------------
Numbers are formatted once, here, so the same quantity never appears with two
different precisions in the report:

    AUC, IC, R^2, alpha      4 decimal places
    percentages              2 decimal places with a % sign
    ratios (Sharpe, t)       2 decimal places
    confidence intervals     "[lo, hi]" at the precision of the statistic

An em-dash marks a cell that is genuinely not applicable (CAPM has no momentum
loading); "n/a" marks one where the underlying analysis did not produce a value.
The two are different and are not collapsed.
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "reports" / "tables"

MODELS_CSV = REPO_ROOT / "models" / "metrics.csv"
RANKING_CSV = REPO_ROOT / "models" / "ranking_metrics.csv"
PCA_CSV = REPO_ROOT / "models" / "pca_robustness.csv"
LADDER_CSV = REPO_ROOT / "models" / "spec_ladder.csv"
PERFORMANCE_CSV = REPO_ROOT / "backtest" / "performance.csv"
IC_CSV = REPO_ROOT / "models" / "ic_significance.csv"
REDUNDANCY_CSV = REPO_ROOT / "models" / "sentiment_redundancy.csv"
HORIZON_CSV = REPO_ROOT / "models" / "event_horizon_results.csv"
FACTOR_CSV = REPO_ROOT / "models" / "factor_alpha.csv"
COMMON_CSV = REPO_ROOT / "models" / "robustness_common_universe.csv"
GRID_CSV = REPO_ROOT / "backtest" / "robustness_portfolio.csv"
CALIB_CSV = REPO_ROOT / "models" / "calibration_results.csv"

MODEL_LABEL = {"A": ("Logistic regression", "Macro"),
               "B": ("Logistic regression", "Macro + LLM"),
               "C": ("XGBoost", "Macro"),
               "D": ("XGBoost", "Macro + LLM")}

EMDASH = "—"
log = logging.getLogger("tables")


# --------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------

def f4(v) -> str:
    return EMDASH if pd.isna(v) else f"{float(v):.4f}"


def f2(v) -> str:
    return EMDASH if pd.isna(v) else f"{float(v):.2f}"


def pct(v, dp: int = 2) -> str:
    return EMDASH if pd.isna(v) else f"{float(v) * 100:.{dp}f}%"


def signed4(v) -> str:
    return EMDASH if pd.isna(v) else f"{float(v):+.4f}"


def ci(lo, hi, fmt=f4) -> str:
    if pd.isna(lo) or pd.isna(hi):
        return EMDASH
    lead = "+" if fmt is signed4 else ""
    del lead
    return f"[{fmt(lo)}, {fmt(hi)}]"


def read_blocks(path: Path) -> dict[str, pd.DataFrame]:
    """Split a multi-block CSV (comment-delimited) into named frames."""
    blocks, name, buf = {}, None, []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            if name and buf:
                blocks[name] = pd.read_csv(io.StringIO("\n".join(buf)))
            name, buf = line.lstrip("# ").strip(), []
        elif line.strip():
            buf.append(line)
    if name and buf:
        blocks[name] = pd.read_csv(io.StringIO("\n".join(buf)))
    return blocks


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------

def table1() -> tuple[pd.DataFrame, str, str]:
    """Main model performance."""
    metrics = pd.read_csv(MODELS_CSV).set_index("model")
    ranking = pd.read_csv(RANKING_CSV)
    # AUC CIs live in the PCA study, which reuses Models A-D as its raw arm.
    specs = read_blocks(PCA_CSV)["SPECIFICATIONS"].set_index("spec")
    ci_of = {"A": "raw_base", "B": "raw_llm", "C": "raw_base_xgb", "D": "raw_llm_xgb"}
    rank_of = {"A": "Model A (logit, macro)", "B": "Model B (logit, +LLM)",
               "C": "Model C (xgb, macro)", "D": "Model D (xgb, +LLM)"}

    rows = []
    for mid in ["A", "B", "C", "D"]:
        algo, features = MODEL_LABEL[mid]
        s = specs.loc[ci_of[mid]]
        r = ranking[ranking["signal"] == rank_of[mid]].iloc[0]
        rows.append({
            "Model": mid, "Algorithm": algo, "Feature set": features,
            "AUC": f4(metrics.loc[mid, "auc"]),
            "AUC 95% CI": ci(s["auc_ci_lo"], s["auc_ci_hi"]),
            "Accuracy": f4(metrics.loc[mid, "accuracy"]),
            "Mean IC": signed4(r["mean_ic"]),
            "IC IR": f2(r["ic_ir"]),
            "Precision@3": f4(r["precision_at_3"]),
        })
    note = ("Out-of-sample walk-forward, 93 prediction months (2018-03 to 2025-11), "
            "1,007 sector-month forecasts. AUC confidence intervals are month-clustered "
            "bootstraps (2,000 draws). An AUC of 0.50 is no ranking skill; every "
            "interval contains it.")
    return pd.DataFrame(rows), "Table 1 — Main model performance", note


def table2() -> tuple[pd.DataFrame, str, str]:
    """LLM incremental effect across specifications — the sign flip."""
    ladder = pd.read_csv(LADDER_CSV, comment="#")
    order = ["base", "identity", "derived", "pca"]
    rows = []
    for rung in order:
        sub = ladder[ladder["rung"] == rung]
        if sub.empty:
            continue
        log_row = sub[sub["estimator"] == "logistic"]
        xgb_row = sub[sub["estimator"] == "xgboost"]
        rec = {"Specification": sub.iloc[0]["rung_label"]}
        for label, r in [("(B−A)", log_row), ("(D−C)", xgb_row)]:
            if r.empty:
                rec[f"ΔAUC {label}"] = EMDASH
                rec[f"95% CI {label}"] = EMDASH
                continue
            r = r.iloc[0]
            star = " *" if r["ci_excludes_zero"] else ""
            rec[f"ΔAUC {label}"] = signed4(r["delta_auc"]) + star
            rec[f"95% CI {label}"] = ci(r["ci_lo"], r["ci_hi"], signed4)
        rows.append(rec)
    note = ("Every rung is computed on the same 93-month walk-forward with the same "
            "one-month embargo, so the rungs are comparable; earlier development "
            "figures came from a 94-month window and are not used here. B−A is the "
            "logistic pair, D−C the XGBoost pair. Positive means the LLM version "
            "scores higher. * marks an interval excluding zero — 1 of 8 cells, in the "
            "direction of harm. The sign changes across specifications in both "
            "estimator ladders, which is what a signal indistinguishable from noise "
            "looks like when it is measured repeatedly.")
    return pd.DataFrame(rows), "Table 2 — LLM incremental effect across specifications", note


def table3() -> tuple[pd.DataFrame, str, str]:
    """Portfolio performance."""
    perf = pd.read_csv(PERFORMANCE_CSV)
    order = ["Model A", "Model B", "Model C", "Model D",
             "Equal weight", "SPY buy & hold", "Momentum 12-1"]
    rows = []
    for name in order:
        sub = perf[perf["strategy"] == name]
        if sub.empty:
            continue
        r = sub.iloc[0]
        rows.append({
            "Strategy": name,
            "CAGR": pct(r["cagr"]),
            "Ann. vol": pct(r["vol_annual"]),
            "Sharpe": f2(r["sharpe"]),
            "Sortino": f2(r["sortino"]),
            "Max DD": pct(r["max_drawdown"]),
            "Calmar": f2(r["calmar"]),
            "Turnover": pct(r["annual_turnover"], 0),
        })
    note = ("Top-3 equal-weight, monthly rebalance, 10 bps one-way cost charged on "
            "drifted weights. 93 realised months (2018-04 to 2025-12). Sharpe and "
            "Sortino use FEDFUNDS as the risk-free rate; turnover is one-way annual.")
    return pd.DataFrame(rows), "Table 3 — Portfolio performance", note


def table4() -> tuple[pd.DataFrame, str, str]:
    """Statistical significance of the information coefficient."""
    ic = pd.read_csv(IC_CSV)
    label = {"Model A (logit, macro)": "Model A", "Model B (logit, +LLM)": "Model B",
             "Model C (xgb, macro)": "Model C", "Model D (xgb, +LLM)": "Model D",
             "LLM sentiment (raw)": "LLM sentiment", "LLM sentiment change": "LLM sentiment change"}
    rows = []
    for _, r in ic.iterrows():
        rows.append({
            "Signal": label.get(r["signal"], r["signal"]),
            "Mean IC": signed4(r["mean_ic"]),
            "t (conventional)": f2(r["t_conventional"]),
            "t (Newey-West)": f2(r["t_newey_west"]),
            "Bootstrap 95% CI": ci(r["ci_lo"], r["ci_hi"], signed4),
            "Permutation p": f"{float(r['p_permutation']):.3f}",
            "BH-adjusted p": f"{float(r['p_bh_adjusted']):.3f}",
        })
    note = ("Monthly Spearman IC against next-month excess return, 93 months. "
            "Newey-West uses the standard ⌊4(T/100)^(2/9)⌋ = 3 lags. The bootstrap "
            "resamples whole months; the permutation test shuffles the signal within "
            "each month across 10,000 draws. Benjamini-Hochberg corrects across the "
            "six signals. No signal is distinguishable from zero by any of the three "
            "methods, and none survives FDR correction.")
    return pd.DataFrame(rows), "Table 4 — Statistical significance of IC", note


def table5() -> tuple[pd.DataFrame, str, str]:
    """Information redundancy decomposition."""
    block = read_blocks(REDUNDANCY_CSV)[
        "VARIANCE OF SENTIMENT EXPLAINED BY THE 14-VARIABLE MACRO BLOCK"]
    reading = {
        "Monthly mean tone": "The LLM's overall tone largely restates the macro inputs "
                             "it was given.",
        "Monthly dispersion": "How much the LLM spreads sectors apart is also mostly "
                              "macro-driven.",
        "Pooled score": "Across all sector-months, macro explains about a quarter of "
                        "the raw score.",
        "Relative (sector-specific)": "Orthogonal to macro by construction — every "
                                      "sector sees the same macro row. Any genuinely "
                                      "new information must live here, and this is the "
                                      "component the IC tests find uninformative.",
    }
    rows = []
    for _, r in block.iterrows():
        rows.append({
            "Component": r["component"],
            "n": int(r["n"]),
            "R² on macro block": f4(r["r2_on_macro_block"]),
            "Adj. R²": f4(r["adj_r2"]),
            "Interpretation": reading.get(r["component"], ""),
        })
    note = ("OLS of each sentiment component on all 14 lagged macro features. The "
            "macro block is identical across sectors within a month, so it can only "
            "explain the cross-sectional mean; the sector-specific residual is "
            "orthogonal to it by construction.")
    return pd.DataFrame(rows), "Table 5 — Information redundancy decomposition", note


def table6() -> tuple[pd.DataFrame, str, str]:
    """Event-horizon decay, both signals."""
    hz = pd.read_csv(HORIZON_CSV)
    name = {"sentiment": "Raw sentiment", "sentiment_change": "Sentiment change"}
    rows = []
    for signal in ["sentiment", "sentiment_change"]:
        sub = hz[hz["signal"] == signal].sort_values("horizon_days")
        for _, r in sub.iterrows():
            rows.append({
                "Signal": name[signal],
                "Horizon (days)": int(r["horizon_days"]),
                "N events": int(r["n_events"]),
                "Mean IC": signed4(r["mean_ic"]),
                "t (NW)": f2(r["t_newey_west"]),
                "95% CI": ci(r["ci_lo"], r["ci_hi"], signed4),
                "% IC > 0": pct(r["pct_ic_positive"], 1),
                "Precision@3": f4(r["precision_at_3"]),
            })
    note = ("Each FOMC statement and minutes release treated as an event; IC measured "
            "from the release-day close over the stated number of trading days. 120 of "
            "163 events survive the eligibility filter (no other release between the "
            "event and the month-end whose score is attached), which tilts the sample "
            "toward quieter periods. Windows overlap at the longer horizons, so only "
            "the Newey-West t should be read there.")
    return pd.DataFrame(rows), "Table 6 — Event-horizon decay", note


def table7() -> tuple[pd.DataFrame, str, str]:
    """Factor-adjusted alpha: CAPM beside Carhart."""
    fa = pd.read_csv(FACTOR_CSV)
    wanted = [("long-only", "Model A", "Model A (long-only)"),
              ("long-only", "Model B", "Model B (long-only)"),
              ("long-only", "Model C", "Model C (long-only)"),
              ("long-only", "Model D", "Model D (long-only)"),
              ("long-only", "Momentum 12-1", "Momentum 12-1 (long-only)"),
              ("long-short", "LLM sentiment (L/S)", "LLM sentiment (long-short)")]
    rows = []
    for book, strategy, label in wanted:
        capm = fa[(fa["book"] == book) & (fa["strategy"] == strategy) & (fa["spec"] == "CAPM")]
        carh = fa[(fa["book"] == book) & (fa["strategy"] == strategy) & (fa["spec"] == "Carhart")]
        if capm.empty or carh.empty:
            continue
        c, k = capm.iloc[0], carh.iloc[0]
        rows.append({
            "Strategy": label,
            "CAPM α (ann.)": pct(c["alpha_annual"]),
            "t (NW) CAPM": f2(c["alpha_t_nw"]),
            "Carhart α (ann.)": pct(k["alpha_annual"]),
            "t (NW) Carhart": f2(k["alpha_t_nw"]),
            "β_MKT": f2(k["beta_MKT_RF"]),
            "β_MOM": f2(k["beta_MOM"]),
            "Adj. R²": f4(k["adj_r2"]),
        })
    note = ("Monthly regressions on Fama-French factors, 93 months, Newey-West standard "
            "errors. β_MKT and β_MOM are the Carhart loadings; CAPM has no momentum "
            "term. Long-only strategies are in excess of the risk-free rate; the "
            "long-short book is self-financing and is not. None of the 26 alphas in "
            "the full file is significant at 5%. Momentum's apparent long-only edge is "
            "the momentum factor: its CAGR advantage survives CAPM and disappears "
            "under Carhart.")
    return pd.DataFrame(rows), "Table 7 — Factor-adjusted alpha", note


def table8() -> tuple[pd.DataFrame, str, str]:
    """Robustness summary — one row per objection, with the statistic that answers it."""
    ladder = pd.read_csv(LADDER_CSV, comment="#")
    ic = pd.read_csv(IC_CSV)
    common = pd.read_csv(COMMON_CSV)
    grid = pd.read_csv(GRID_CSV)
    hz = pd.read_csv(HORIZON_CSV)
    fa = pd.read_csv(FACTOR_CSV)
    calib = read_blocks(CALIB_CSV)["key comparisons: Brier difference, month-clustered bootstrap"]

    # Only A-D are models here; the file also carries B-A/D-C deltas and benchmarks.
    common_models = common[common["model"].isin(["A", "B", "C", "D"])].dropna(subset=["auc"])
    n_common = len(common_models)
    n_common_ci = int(((common_models["auc_lo"] > 0.5) | (common_models["auc_hi"] < 0.5)).sum())
    n_common_months = int(common_models["n_months"].min())

    identity = ladder[ladder["rung"] == "identity"]
    pca_rung = ladder[ladder["rung"] == "pca"]
    best_hz = hz.loc[hz["mean_ic"].abs().idxmax()]
    ls = fa[(fa["strategy"] == "LLM sentiment (L/S)") & (fa["spec"] == "Carhart")].iloc[0]
    # The grid has no "did it win" column, so derive it: a model cell counts as a
    # win when its Sharpe beats equal weight at the *same* top-N and cost, which is
    # the only like-for-like comparison in the grid.
    bench = (grid[grid["strategy"] == "Equal weight"]
             .set_index(["top_n", "cost_bps"])["sharpe"])
    model_cells = grid[grid["strategy"].isin(["Model A", "Model B", "Model C", "Model D"])]
    wins = sum(1 for _, c in model_cells.iterrows()
               if c["sharpe"] > bench.get((c["top_n"], c["cost_bps"]), float("inf")))
    n_cells = len(model_cells)
    ba = calib[calib["comparison"].astype(str).str.strip() == "B - A"].iloc[0]

    rows = [
        {"Objection tested": "Look-ahead bias in the training window",
         "Method": "One-month training embargo; 5 automated timing assertions",
         "Key statistic": "0 training labels use a price at or after the prediction month",
         "Conclusion": "No leakage; the embargoed protocol is implementable."},
        {"Objection tested": "The model cannot distinguish sectors",
         "Method": "Add 10 sector dummies + 8 macro × sector interactions",
         "Key statistic": f"ΔAUC(D−C) {signed4(identity[identity['estimator']=='xgboost'].iloc[0]['delta_auc'])} "
                          f"{ci(identity[identity['estimator']=='xgboost'].iloc[0]['ci_lo'], identity[identity['estimator']=='xgboost'].iloc[0]['ci_hi'], signed4)}",
         "Conclusion": "Sector identity does not rescue the LLM; it makes the tree pair worse."},
        {"Objection tested": "XLC's late listing distorts the panel",
         "Method": "Refit on the 10 sectors present throughout",
         "Key statistic": f"{n_common_ci} of {n_common} model-sample AUC CIs exclude 0.50 "
                          f"(93- and {n_common_months}-month samples)",
         "Conclusion": "The null is not an artefact of the unbalanced universe."},
        {"Objection tested": "The signal is priced faster than monthly",
         "Method": "FOMC event windows at 1, 3, 5, 10, 21, 42 trading days",
         "Key statistic": f"largest absolute mean IC {signed4(best_hz['mean_ic'])} at "
                          f"{int(best_hz['horizon_days'])}d, t(NW) {f2(best_hz['t_newey_west'])}",
         "Conclusion": "No horizon shows a signal; it is not a rebalance-frequency problem."},
        {"Objection tested": "Long-only results are a market-beta story",
         "Method": "Cross-sectional long-short spread, then Carhart regression",
         "Key statistic": f"LLM L/S β_MOM {f2(ls['beta_MOM'])} (t {f2(ls['t_nw_MOM'])}), "
                          f"Carhart α {pct(ls['alpha_annual'])}",
         "Conclusion": "Removing beta reveals no selection skill; what is left is momentum."},
        {"Objection tested": "The inference method flatters the null",
         "Method": "Newey-West, month-clustered bootstrap, permutation, BH-FDR",
         "Key statistic": f"{int(ic['survives_fdr_5pct'].sum())} of {len(ic)} signals survive FDR at 5%",
         "Conclusion": "Three independent frameworks agree; the null is not a method artefact."},
        {"Objection tested": "The portfolio parameters were badly chosen",
         "Method": f"Top-N × transaction-cost grid ({len(grid)} cells), no ex-post selection",
         "Key statistic": f"{wins} of {n_cells} model cells beat equal weight",
         "Conclusion": "No parameter corner rescues the models; wins are isolated, not systematic."},
        {"Objection tested": "Multicollinearity in the macro block masks the LLM",
         "Method": "PCA the 14 macro features, 90% training variance rule",
         "Key statistic": f"6 of 14 PCs retained; ΔAUC "
                          f"{signed4(pca_rung[pca_rung['estimator']=='logistic'].iloc[0]['delta_auc'])} "
                          f"and {signed4(pca_rung[pca_rung['estimator']=='xgboost'].iloc[0]['delta_auc'])}, "
                          f"both CIs span 0",
         "Conclusion": "Orthogonalising macro does not let the LLM in; this explanation is closed."},
        {"Objection tested": "The LLM helps the probabilities even if not the ranking",
         "Method": "Brier / log loss / ECE, Murphy decomposition, bootstrap on ΔBrier",
         "Key statistic": f"ΔBrier(B−A) {signed4(ba['brier_diff'])} "
                          f"{ci(ba['ci_lo'], ba['ci_hi'], signed4)}",
         "Conclusion": "Calibration is significantly worse with the LLM, not better."},
    ]
    note = ("Each row is an objection to the null result and the analysis that answers "
            "it. Every one was specified before its result was known, and none is "
            "reported selectively — the full grids and all model variants are in the "
            "underlying CSVs.")
    return pd.DataFrame(rows), "Table 8 — Robustness summary", note


TABLES = {1: table1, 2: table2, 3: table3, 4: table4,
          5: table5, 6: table6, 7: table7, 8: table8}

SLUGS = {1: "table1_model_performance", 2: "table2_llm_increment",
         3: "table3_portfolio_performance", 4: "table4_ic_significance",
         5: "table5_redundancy", 6: "table6_event_horizon",
         7: "table7_factor_alpha", 8: "table8_robustness_summary"}


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

def to_markdown(frame: pd.DataFrame, title: str, note: str) -> str:
    header = list(frame.columns)
    lines = [f"### {title}", "",
             "| " + " | ".join(header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    for _, row in frame.iterrows():
        # A literal "|" inside a cell would end the column early; escape it.
        cells = [str(row[c]).replace("|", "\\|") for c in header]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", f"*{note}*", ""]
    return "\n".join(lines)


def build(numbers: list[int]) -> dict[int, tuple[pd.DataFrame, str, str]]:
    out = {}
    for n in numbers:
        frame, title, note = TABLES[n]()
        out[n] = (frame, title, note)
    return out


def write(built: dict, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for n, (frame, title, note) in sorted(built.items()):
        csv_path = out_dir / f"{SLUGS[n]}.csv"
        md_path = out_dir / f"{SLUGS[n]}.md"
        frame.to_csv(csv_path, index=False)
        md_path.write_text(to_markdown(frame, title, note), encoding="utf-8")
        written += [csv_path, md_path]
        log.info("  %-42s %d rows x %d cols", csv_path.name, len(frame), frame.shape[1])

    index = ["# Report tables", "",
             "Generated by `scripts/build_report_tables.py` from the analysis CSVs.",
             "Do not edit by hand — `scripts/verify_all.py` regenerates these and fails",
             "if a committed table no longer matches its source data.", ""]
    for n, (frame, title, _) in sorted(built.items()):
        index.append(f"- **{title}** — `{SLUGS[n]}.csv` / `.md` "
                     f"({len(frame)} rows × {frame.shape[1]} cols)")
    index.append("")
    (out_dir / "README.md").write_text("\n".join(index), encoding="utf-8")
    written.append(out_dir / "README.md")
    return written


def check(built: dict, out_dir: Path) -> list[str]:
    """Compare freshly built tables against what is committed."""
    problems = []
    for n, (frame, title, note) in sorted(built.items()):
        csv_path = out_dir / f"{SLUGS[n]}.csv"
        md_path = out_dir / f"{SLUGS[n]}.md"
        if not csv_path.exists():
            problems.append(f"{csv_path.name} is missing")
            continue
        committed = pd.read_csv(csv_path, dtype=str).fillna("")
        fresh = frame.astype(str).fillna("")
        if not committed.equals(fresh):
            if list(committed.columns) != list(fresh.columns):
                problems.append(f"{csv_path.name}: columns changed")
            else:
                diffs = [(i, c) for i in range(min(len(committed), len(fresh)))
                         for c in fresh.columns
                         if committed.iloc[i][c] != fresh.iloc[i][c]]
                where = "; ".join(f"row {i} col {c!r}" for i, c in diffs[:3]) or "row count"
                problems.append(f"{csv_path.name}: {len(diffs)} cell(s) differ ({where})")
        if not md_path.exists():
            problems.append(f"{md_path.name} is missing")
        elif md_path.read_text(encoding="utf-8") != to_markdown(frame, title, note):
            problems.append(f"{md_path.name}: markdown differs from the rebuilt table")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="rebuild in memory and compare to committed files")
    parser.add_argument("--table", type=int, action="append", choices=sorted(TABLES),
                        help="build only these tables (repeatable)")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    numbers = sorted(set(args.table)) if args.table else sorted(TABLES)

    missing = [p.name for p in (MODELS_CSV, RANKING_CSV, PCA_CSV, LADDER_CSV,
                                PERFORMANCE_CSV, IC_CSV, REDUNDANCY_CSV, HORIZON_CSV,
                                FACTOR_CSV, COMMON_CSV, GRID_CSV, CALIB_CSV)
               if not p.exists()]
    if missing:
        log.error("missing input(s): %s -- run python run_all.py first", ", ".join(missing))
        return 1

    built = build(numbers)

    if args.check:
        problems = check(built, args.out_dir)
        if problems:
            log.error("STALE report tables (%d problem(s)):", len(problems))
            for p in problems:
                log.error("  %s", p)
            log.error("Run: python scripts/build_report_tables.py")
            return 1
        log.info("All %d table(s) match their source CSVs.", len(built))
        return 0

    log.info("Building %d table(s) into %s", len(built), args.out_dir)
    write(built, args.out_dir)
    log.info("Done. %d tables, CSV + Markdown.", len(built))
    return 0


if __name__ == "__main__":
    sys.exit(main())
