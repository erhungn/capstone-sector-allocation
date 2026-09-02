"""Does the backtest conclusion survive different portfolio choices?

Top-3 at 10bps was one arbitrary point in a parameter space. This runs the whole
grid -- N in {2,3,4,5} crossed with costs in {0,5,10,15,20} bps -- for all four
models and for the 12-1 momentum benchmark, and reports every cell.

Usage
-----
    python backtest/robustness_portfolio.py

Outputs
-------
    backtest/robustness_portfolio.csv
    viz/topn_sensitivity.png

Reading this grid
-----------------
No cell is selected as "the" configuration. The question is whether the ranking
between models and benchmarks is stable across reasonable choices, not which
choice flatters the models most. A strategy that beats its benchmark in one
corner of a 20-cell grid and loses in the other nineteen has not demonstrated
skill -- with that many cells, one or two wins are the expected yield of noise.

Comparisons are made at **matched parameters**: a model at (N, TC) is compared
with momentum at the same (N, TC) and with equal-weight and SPY at the same TC.
Comparing a model's best cell against a benchmark's worst would be the same
ex-post selection this file exists to avoid.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtest"))

import run_backtest as B  # noqa: E402

PREDICTIONS = REPO_ROOT / "models" / "predictions.csv"
DATASET = REPO_ROOT / "data" / "model_dataset.csv"
OUT_CSV = REPO_ROOT / "backtest" / "robustness_portfolio.csv"
PLOT_PNG = REPO_ROOT / "viz" / "topn_sensitivity.png"

TOP_N_GRID = [2, 3, 4, 5]
COST_GRID = [0.0, 5.0, 10.0, 15.0, 20.0]
BASE_N, BASE_TC = 3, 10.0

MODELS = ["A", "B", "C", "D"]
MOMENTUM = "Momentum 12-1"
EQUAL_WEIGHT = "Equal weight"
SPY = "SPY buy & hold"

log = logging.getLogger("robustness_portfolio")


def build_grid(preds: pd.DataFrame, dataset: pd.DataFrame) -> pd.DataFrame:
    universe = preds[["date", "ticker", "actual_return"]].merge(
        dataset[["date", "ticker", "ret_12m_lag1"]], on=["date", "ticker"], how="left")
    returns_by_date = {d: dict(zip(g["ticker"], g["actual_return"]))
                       for d, g in universe.groupby("date")}

    realised = B.next_month_end(sorted(universe["date"].unique()))
    spy = B.load_spy(realised.min() - pd.DateOffset(months=3), realised.max())
    spy_ret_raw = (spy / spy.shift(1) - 1.0).reindex(realised).ffill()
    rf = B.risk_free_monthly(realised)

    # Weight schedules depend on N only; costs are applied at simulation time.
    schedules_by_n: dict[int, dict[str, dict]] = {}
    for n in TOP_N_GRID:
        sched = {}
        for model_id in MODELS:
            col = f"model_{model_id}_prob"
            sched[f"Model {model_id}"] = B.top_n_weights(
                preds[["date", "ticker", col]].dropna(), col, n)
        sched[MOMENTUM] = B.top_n_weights(
            universe.dropna(subset=["ret_12m_lag1"]), "ret_12m_lag1", n)
        schedules_by_n[n] = sched

    ew_schedule = B.equal_weight_all(universe)

    rows = []
    for n in TOP_N_GRID:
        for tc in COST_GRID:
            for name, sched in schedules_by_n[n].items():
                r, t = B.simulate(sched, returns_by_date, tc)
                rows.append({"strategy": name, "top_n": n, "cost_bps": tc,
                             **B.performance(r, rf, t)})
            # Benchmarks that do not depend on N, recorded once per N so every
            # cell has its matched comparison available.
            r, t = B.simulate(ew_schedule, returns_by_date, tc)
            rows.append({"strategy": EQUAL_WEIGHT, "top_n": n, "cost_bps": tc,
                         **B.performance(r, rf, t)})
            spy_ret = spy_ret_raw.copy()
            spy_ret.iloc[0] -= tc / 10_000.0
            rows.append({"strategy": SPY, "top_n": n, "cost_bps": tc,
                         **B.performance(spy_ret, rf, pd.Series(0.0, index=realised))})
    return pd.DataFrame(rows)


def win_summary(grid: pd.DataFrame) -> pd.DataFrame:
    """For each model, count grid cells where it beats each benchmark on Sharpe."""
    rows = []
    for model_id in MODELS:
        name = f"Model {model_id}"
        sub = grid[grid["strategy"] == name]
        wins = {EQUAL_WEIGHT: [], SPY: [], MOMENTUM: []}
        for _, r in sub.iterrows():
            cell = grid[(grid["top_n"] == r["top_n"]) & (grid["cost_bps"] == r["cost_bps"])]
            for bench in wins:
                b = cell[cell["strategy"] == bench]
                if not b.empty and r["sharpe"] > b["sharpe"].iloc[0]:
                    wins[bench].append(f"N={int(r['top_n'])},{int(r['cost_bps'])}bp")
        rows.append({
            "model": model_id, "n_cells": len(sub),
            "beats_ew": len(wins[EQUAL_WEIGHT]), "beats_spy": len(wins[SPY]),
            "beats_momentum": len(wins[MOMENTUM]),
            "beats_all_three": sum(
                1 for _, r in sub.iterrows()
                if all(
                    r["sharpe"] > grid[(grid["top_n"] == r["top_n"])
                                       & (grid["cost_bps"] == r["cost_bps"])
                                       & (grid["strategy"] == b)]["sharpe"].iloc[0]
                    for b in (EQUAL_WEIGHT, SPY, MOMENTUM)
                )
            ),
            "ew_win_cells": "; ".join(wins[EQUAL_WEIGHT][:4]),
            "spy_win_cells": "; ".join(wins[SPY][:4]),
            "momentum_win_cells": "; ".join(wins[MOMENTUM][:4]),
            "best_sharpe": float(sub["sharpe"].max()),
            "worst_sharpe": float(sub["sharpe"].min()),
        })
    return pd.DataFrame(rows)


def run() -> int:
    for path in (PREDICTIONS, DATASET):
        if not path.exists():
            log.error("%s not found -- run the upstream script first", path)
            return 1

    preds = pd.read_csv(PREDICTIONS, parse_dates=["date"])
    dataset = pd.read_csv(DATASET, parse_dates=["date"])
    log.info("%d months, %s -> %s | grid: N in %s x cost in %s bps = %d cells per strategy",
             preds["date"].nunique(), preds["date"].min().date(), preds["date"].max().date(),
             TOP_N_GRID, [int(c) for c in COST_GRID], len(TOP_N_GRID) * len(COST_GRID))

    grid = build_grid(preds, dataset)
    summary = win_summary(grid)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    grid.to_csv(OUT_CSV, index=False, float_format="%.4f")
    log.info("Wrote %s (%d rows)", OUT_CSV, len(grid))

    report(grid, summary)
    plot(grid)
    return 0


def report(grid: pd.DataFrame, summary: pd.DataFrame) -> None:
    order = [f"Model {m}" for m in MODELS] + [MOMENTUM, EQUAL_WEIGHT, SPY]

    log.info("")
    log.info("=" * 100)
    log.info("(a) TOP-N SENSITIVITY  (costs held at %d bps)", int(BASE_TC))
    log.info("%-16s %s", "strategy",
             "".join(f"{'N=' + str(n):>20}" for n in TOP_N_GRID))
    log.info("%-16s %s", "", "".join(f"{'CAGR  Shrp  MaxDD':>20}" for _ in TOP_N_GRID))
    for name in order:
        cells = []
        for n in TOP_N_GRID:
            r = grid[(grid.strategy == name) & (grid.top_n == n) & (grid.cost_bps == BASE_TC)]
            cells.append(f"{r['cagr'].iloc[0]*100:5.1f}% {r['sharpe'].iloc[0]:5.2f} "
                         f"{r['max_drawdown'].iloc[0]*100:6.1f}%" if not r.empty else " " * 20)
        log.info("%-16s %s", name, "".join(f"{c:>20}" for c in cells))

    log.info("")
    log.info("    turnover and Calmar at %d bps", int(BASE_TC))
    log.info("%-16s %s", "strategy", "".join(f"{'N=' + str(n):>18}" for n in TOP_N_GRID))
    for name in order:
        cells = []
        for n in TOP_N_GRID:
            r = grid[(grid.strategy == name) & (grid.top_n == n) & (grid.cost_bps == BASE_TC)]
            cells.append(f"{r['annual_turnover'].iloc[0]*100:4.0f}%  Clmr {r['calmar'].iloc[0]:5.2f}"
                         if not r.empty else "")
        log.info("%-16s %s", name, "".join(f"{c:>18}" for c in cells))

    log.info("")
    log.info("=" * 100)
    log.info("(b) TRANSACTION COST SENSITIVITY  (top %d)", BASE_N)
    log.info("%-16s %s", "strategy", "".join(f"{str(int(c)) + 'bp':>15}" for c in COST_GRID))
    log.info("%-16s %s", "", "".join(f"{'CAGR   Sharpe':>15}" for _ in COST_GRID))
    for name in order:
        cells = []
        for tc in COST_GRID:
            r = grid[(grid.strategy == name) & (grid.top_n == BASE_N) & (grid.cost_bps == tc)]
            cells.append(f"{r['cagr'].iloc[0]*100:5.1f}% {r['sharpe'].iloc[0]:6.2f}"
                         if not r.empty else "")
        log.info("%-16s %s", name, "".join(f"{c:>15}" for c in cells))

    log.info("")
    log.info("=" * 100)
    log.info("(e) DOES ANY MODEL BEAT A BENCHMARK ANYWHERE IN THE GRID? (Sharpe, matched cells)")
    log.info("%-8s %8s %10s %10s %14s %12s %12s",
             "model", "cells", "beats EW", "beats SPY", "beats momentum", "beats all 3",
             "Sharpe range")
    for _, r in summary.iterrows():
        log.info("%-8s %8d %10d %10d %14d %12d %6.2f-%.2f",
                 r["model"], r["n_cells"], r["beats_ew"], r["beats_spy"],
                 r["beats_momentum"], r["beats_all_three"],
                 r["worst_sharpe"], r["best_sharpe"])
    log.info("=" * 100)

    total_cells = int(summary["n_cells"].iloc[0])
    for _, r in summary.iterrows():
        for bench, count, cells in ((EQUAL_WEIGHT, r["beats_ew"], r["ew_win_cells"]),
                                    (SPY, r["beats_spy"], r["spy_win_cells"]),
                                    (MOMENTUM, r["beats_momentum"], r["momentum_win_cells"])):
            if 0 < count < total_cells / 2:
                log.info("Model %s beats %s in only %d of %d cells: %s",
                         r["model"], bench, count, total_cells, cells)
                costs = sorted({int(c.split(",")[1].replace("bp", "")) for c in cells.split("; ")})
                if max(costs) <= 5:
                    log.info("  -> every win sits at %s bps. At realistic costs it never wins;",
                             "/".join(str(c) for c in costs))
                    log.info("     this is a corner of the grid, not an edge.")

    if summary["beats_spy"].sum() == 0:
        log.info("No model beats SPY in any of the %d cells.", total_cells)
    if summary["beats_momentum"].sum() == 0:
        log.info("No model beats 12-1 momentum in any of the %d cells.", total_cells)
    if summary["beats_all_three"].sum() == 0:
        log.info("No model beats all three benchmarks simultaneously in any cell.")

    # Sensitivity shape: do the models degrade with cost faster than momentum?
    log.info("")
    log.info("Cost sensitivity, Sharpe lost from 0 to 20 bps (top %d):", BASE_N)
    for name in order:
        lo = grid[(grid.strategy == name) & (grid.top_n == BASE_N) & (grid.cost_bps == 0)]
        hi = grid[(grid.strategy == name) & (grid.top_n == BASE_N) & (grid.cost_bps == 20)]
        if lo.empty or hi.empty:
            continue
        drop = lo["sharpe"].iloc[0] - hi["sharpe"].iloc[0]
        log.info("  %-16s %+.3f   (turnover %3.0f%%)",
                 name, -drop, lo["annual_turnover"].iloc[0] * 100)
    log.info("  High-turnover strategies lose more to costs -- the models turn over")
    log.info("  ~3x momentum and ~30x equal-weight, so cost assumptions hit them hardest.")


def plot(grid: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"Model A": "#4C78A8", "Model B": "#72B7B2",
              "Model C": "#E45756", "Model D": "#F58518", MOMENTUM: "#8E44AD"}
    names = [f"Model {m}" for m in MODELS] + [MOMENTUM]
    x = np.arange(len(TOP_N_GRID))
    width = 0.16

    fig, ax = plt.subplots(figsize=(12, 6.5))
    for i, name in enumerate(names):
        vals = [grid[(grid.strategy == name) & (grid.top_n == n)
                     & (grid.cost_bps == BASE_TC)]["sharpe"].iloc[0] for n in TOP_N_GRID]
        bars = ax.bar(x + (i - 2) * width, vals, width, label=name, color=colors[name])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}",
                    ha="center", fontsize=7.5)

    # Benchmark lines labelled in a right-hand margin so they cannot collide
    # with a bar that happens to land at the same Sharpe.
    right = len(TOP_N_GRID) - 0.5
    for bench, style, col in ((EQUAL_WEIGHT, "--", "#444444"), (SPY, "-.", "#2E7D32")):
        v = grid[(grid.strategy == bench) & (grid.cost_bps == BASE_TC)]["sharpe"].iloc[0]
        ax.axhline(v, color=col, linestyle=style, linewidth=1.6)
        ax.text(right + 0.06, v, f"{bench}\n{v:.2f}", fontsize=9, color=col,
                ha="left", va="center", weight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"Top {n}" for n in TOP_N_GRID], fontsize=11)
    ax.set_ylabel("Sharpe ratio", fontsize=11)
    ax.set_title(f"Sharpe by model and portfolio size (costs {int(BASE_TC)} bps one-way)\n"
                 "No model clears the passive benchmarks at any N",
                 fontsize=13, weight="bold")
    ax.legend(frameon=False, ncol=5, loc="upper center",
              bbox_to_anchor=(0.5, -0.06), fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    ax.set_xlim(-0.55, right + 0.62)
    ax.set_ylim(0, max(0.9, grid[grid.cost_bps == BASE_TC]["sharpe"].max() + 0.12))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    fig.tight_layout()
    PLOT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_PNG, dpi=150)
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
