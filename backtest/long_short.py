"""Long-short diagnostic: is there cross-sectional skill once beta is removed?

**This is a diagnostic, not a proposed strategy.** The top-3 long-only backtest
mixes two things: whether the model picks the right sectors, and whether sectors
went up. A dollar-neutral long-short book cancels the common market move, so what
is left is selection skill alone. If that spread is also flat, the long-only
shortfall was never a beta story.

Usage
-----
    python backtest/long_short.py
    python backtest/long_short.py --cost-bps 5

Outputs
-------
    backtest/long_short_results.csv
    viz/long_short_equity.png

Construction
------------
Each month: rank the available sectors, hold the top 3 at 1/3 each, short the
bottom 3 at 1/3 each. R_LS = R_top3 - R_bottom3. Dollar-neutral, gross exposure
200%, no leverage beyond that.

What this does NOT model
------------------------
Shorting sector ETFs costs money and is not always possible: borrow fees (small
for large SPDRs, but not zero and spiking exactly when a sector is stressed),
margin and financing on the short proceeds, recall risk, and the uptick-style
constraints that bite in fast markets. Short rebates would offset part of that
and are also ignored. Treat the spread as an upper bound on what the selection
signal is worth, not as a tradeable return.

Costs
-----
The same 10bps one-way as the main backtest, charged on **both legs**: a
long-short book turns over roughly twice the notional of a long-only one, so the
cost drag is about double. Gross and net are both reported.

Risk-free rate
--------------
A long-short book is self-financing, so the headline Sharpe uses the raw spread
with no risk-free subtraction. The rf-adjusted figure is reported alongside for
readers who prefer that convention; on this sample it is the more pessimistic of
the two.
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
sys.path.insert(0, str(REPO_ROOT / "common"))

from plotting import DPI  # noqa: E402
from stats import newey_west_t  # noqa: E402

import run_backtest as B  # noqa: E402

PREDICTIONS = REPO_ROOT / "models" / "predictions.csv"
DATASET = REPO_ROOT / "data" / "model_dataset.csv"
OUT_CSV = REPO_ROOT / "backtest" / "long_short_results.csv"
PLOT_PNG = REPO_ROOT / "viz" / "long_short_equity.png"

TOP_N = 3
COST_BPS = 10.0
MONTHS = 12
N_BOOT = 2000
RANDOM_STATE = 0

SIGNALS = {
    "Model A": "model_A_prob",
    "Model B": "model_B_prob",
    "Model C": "model_C_prob",
    "Model D": "model_D_prob",
    "LLM sentiment": "sentiment_lag1",
    "Momentum 12-1": "ret_12m_lag1",
}

log = logging.getLogger("long_short")


# --------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------

def leg_weights(frame: pd.DataFrame, col: str, n: int) -> tuple[dict, dict]:
    """Equal-weight top-n and bottom-n baskets per month."""
    longs, shorts = {}, {}
    for date, g in frame.groupby("date"):
        g = g.dropna(subset=[col])
        if len(g) < 2 * n:
            continue
        ordered = g.sort_values(col, ascending=False)["ticker"].tolist()
        top, bottom = ordered[:n], ordered[-n:]
        longs[date] = {t: 1.0 / n for t in top}
        shorts[date] = {t: 1.0 / n for t in bottom}
    return longs, shorts


def drift(weights: dict, rets: dict) -> dict:
    grown = {t: w * (1.0 + rets.get(t, 0.0)) for t, w in weights.items()}
    total = sum(grown.values())
    return {t: v / total for t, v in grown.items()} if total > 0 else {}


def simulate_long_short(longs: dict, shorts: dict, returns_by_date: dict,
                        cost_bps: float) -> pd.DataFrame:
    """Run both legs forward. Returns a monthly frame of gross, net and turnover.

    Turnover is measured against drifted weights on each leg separately, then
    summed -- that is what makes the long-short cost roughly double the
    long-only cost for the same signal.
    """
    rate = cost_bps / 10_000.0
    prev_l: dict[str, float] = {}
    prev_s: dict[str, float] = {}
    rows = []

    for date in sorted(longs):
        wl, ws = longs[date], shorts[date]
        rets = returns_by_date.get(date, {})

        turn_l = sum(abs(wl.get(t, 0.0) - prev_l.get(t, 0.0)) for t in set(wl) | set(prev_l))
        turn_s = sum(abs(ws.get(t, 0.0) - prev_s.get(t, 0.0)) for t in set(ws) | set(prev_s))
        cost = (turn_l + turn_s) * rate

        r_long = sum(w * rets.get(t, 0.0) for t, w in wl.items())
        r_short = sum(w * rets.get(t, 0.0) for t, w in ws.items())
        gross = r_long - r_short

        rows.append({"formation": date, "r_long": r_long, "r_short": r_short,
                     "gross": gross, "net": gross - cost, "cost": cost,
                     "turnover": turn_l + turn_s})

        prev_l, prev_s = drift(wl, rets), drift(ws, rets)

    out = pd.DataFrame(rows)
    out.index = B.next_month_end(pd.DatetimeIndex(out["formation"]))
    return out.drop(columns=["formation"])


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------



def summarise(name: str, sim: pd.DataFrame, rf: pd.Series, cost_bps: float) -> dict:
    gross, net = sim["gross"], sim["net"]
    n = len(net)
    years = n / MONTHS
    equity = (1.0 + net).cumprod()
    equity_gross = (1.0 + gross).cumprod()

    sd = net.std(ddof=1)
    downside = net.clip(upper=0.0)
    dsd = np.sqrt((downside ** 2).sum() / max(n - 1, 1))
    rf_aligned = rf.reindex(net.index).fillna(0.0)

    t_naive = float(net.mean() / (sd / np.sqrt(n))) if sd > 0 else np.nan
    t_nw, lags = newey_west_t(net.to_numpy())
    rng = np.random.default_rng(RANDOM_STATE)
    boot = [net.to_numpy()[rng.integers(0, n, n)].mean() for _ in range(N_BOOT)]

    return {
        "strategy": name, "n_months": n, "cost_bps": cost_bps,
        "cum_return_gross": float(equity_gross.iloc[-1] - 1.0),
        "cum_return_net": float(equity.iloc[-1] - 1.0),
        "ann_return_gross": float(equity_gross.iloc[-1] ** (1 / years) - 1.0),
        "ann_return_net": float(equity.iloc[-1] ** (1 / years) - 1.0),
        "ann_vol": float(sd * np.sqrt(MONTHS)),
        # Self-financing: the headline Sharpe does not subtract rf.
        "sharpe_no_rf": float(net.mean() / sd * np.sqrt(MONTHS)) if sd > 0 else np.nan,
        "sharpe_rf_adj": float((net - rf_aligned).mean() / sd * np.sqrt(MONTHS)) if sd > 0 else np.nan,
        "sortino": float(net.mean() / dsd * np.sqrt(MONTHS)) if dsd > 0 else np.nan,
        "max_drawdown": float((equity / equity.cummax() - 1.0).min()),
        "hit_rate": float((net > 0).mean()),
        "mean_monthly_spread": float(net.mean()),
        "mean_monthly_gross": float(gross.mean()),
        "t_stat": t_naive,
        "t_newey_west": t_nw, "nw_lags": lags,
        "ci_lo": float(np.percentile(boot, 2.5)), "ci_hi": float(np.percentile(boot, 97.5)),
        "annual_turnover": float(sim["turnover"].mean() * MONTHS / 2.0),
        "annual_cost_drag": float(sim["cost"].mean() * MONTHS),
    }


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def run(cost_bps: float) -> int:
    for path in (PREDICTIONS, DATASET):
        if not path.exists():
            log.error("%s not found -- run the upstream script first", path)
            return 1

    preds = pd.read_csv(PREDICTIONS, parse_dates=["date"])
    data = pd.read_csv(DATASET, parse_dates=["date"])
    frame = preds.merge(data[["date", "ticker", "sentiment_lag1", "ret_12m_lag1"]],
                        on=["date", "ticker"], how="left", validate="one_to_one")
    returns_by_date = {d: dict(zip(g["ticker"], g["actual_return"]))
                       for d, g in frame.groupby("date")}
    rf = B.risk_free_monthly(B.next_month_end(sorted(frame["date"].unique())))

    log.info("%d months, %s -> %s | long top %d, short bottom %d, %.0f bps per leg",
             frame["date"].nunique(), frame["date"].min().date(),
             frame["date"].max().date(), TOP_N, TOP_N, cost_bps)
    log.info("DIAGNOSTIC ONLY -- shorting costs and constraints are not modelled")

    sims, rows = {}, []
    for name, col in SIGNALS.items():
        if col not in frame.columns:
            log.warning("%s: column %s missing, skipped", name, col)
            continue
        longs, shorts = leg_weights(frame[["date", "ticker", col]], col, TOP_N)
        sim = simulate_long_short(longs, shorts, returns_by_date, cost_bps)
        sims[name] = sim
        rows.append(summarise(name, sim, rf, cost_bps))

    results = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT_CSV, index=False, float_format="%.5f")
    log.info("Wrote %s", OUT_CSV)

    report(results)
    plot(sims, results)
    return 0


def report(results: pd.DataFrame) -> None:
    log.info("")
    log.info("=" * 112)
    log.info("LONG-SHORT SPREAD (top %d minus bottom %d, dollar-neutral)", TOP_N, TOP_N)
    log.info("%-16s %8s %8s %8s %8s %9s %8s %8s %9s",
             "strategy", "ann net", "ann gr", "vol", "Sharpe", "Sortino", "maxDD",
             "hit", "turnover")
    for _, r in results.iterrows():
        log.info("%-16s %7.2f%% %7.2f%% %7.2f%% %8.2f %9.2f %7.1f%% %7.1f%% %8.0f%%",
                 r["strategy"], r["ann_return_net"] * 100, r["ann_return_gross"] * 100,
                 r["ann_vol"] * 100, r["sharpe_no_rf"], r["sortino"],
                 r["max_drawdown"] * 100, r["hit_rate"] * 100,
                 r["annual_turnover"] * 100)
    log.info("=" * 112)
    log.info("Sharpe shown without a risk-free subtraction (the book is self-financing).")
    log.info("%-16s %s", "rf-adjusted:",
             "  ".join(f"{r['strategy']} {r['sharpe_rf_adj']:+.2f}" for _, r in results.iterrows()))

    log.info("")
    log.info("IS THE SPREAD DISTINGUISHABLE FROM ZERO?")
    log.info("%-16s %14s %8s %9s %24s %s",
             "strategy", "mean monthly", "t", "t (NW)", "95% CI", "verdict")
    for _, r in results.iterrows():
        sig = "SIGNIFICANT" if (r["ci_lo"] > 0 or r["ci_hi"] < 0) else "no"
        log.info("%-16s %13.4f%% %8.2f %9.2f  [%+.4f%%, %+.4f%%]  %s",
                 r["strategy"], r["mean_monthly_spread"] * 100, r["t_stat"],
                 r["t_newey_west"], r["ci_lo"] * 100, r["ci_hi"] * 100, sig)
    log.info("Newey-West lag %d (rule of thumb for this sample length).",
             int(results["nw_lags"].iloc[0]))

    log.info("")
    log.info("COST OF THE SECOND LEG")
    for _, r in results.iterrows():
        log.info("  %-16s gross %+.2f%%/yr -> net %+.2f%%/yr  (drag %.2f%% on %0.f%% turnover)",
                 r["strategy"], r["ann_return_gross"] * 100, r["ann_return_net"] * 100,
                 r["annual_cost_drag"] * 100, r["annual_turnover"] * 100)

    log.info("")
    log.info("INTERPRETATION")
    models = results[results["strategy"].str.startswith("Model")]
    any_sig = ((results["ci_lo"] > 0) | (results["ci_hi"] < 0)).any()
    if not any_sig:
        log.info("  No strategy's spread is distinguishable from zero. Removing market")
        log.info("  direction does not reveal hidden selection skill -- the long-only")
        log.info("  shortfall was not a beta story, there is simply no cross-sectional edge.")
    else:
        for _, r in results[(results["ci_lo"] > 0) | (results["ci_hi"] < 0)].iterrows():
            log.info("  %s spread is significant: %+.4f%%/month [%+.4f%%, %+.4f%%]",
                     r["strategy"], r["mean_monthly_spread"] * 100, r["ci_lo"] * 100,
                     r["ci_hi"] * 100)
    log.info("  Best model spread: %s at %+.2f%%/yr net (Sharpe %.2f).",
             models.loc[models["ann_return_net"].idxmax(), "strategy"],
             models["ann_return_net"].max() * 100,
             models.loc[models["ann_return_net"].idxmax(), "sharpe_no_rf"])
    log.info("")
    log.info("  Reminder: borrow fees, financing, recall risk and short-sale constraints")
    log.info("  are all excluded. These spreads are an upper bound on the signal's worth.")


def plot(sims: dict[str, pd.DataFrame], results: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"Model A": "#4C78A8", "Model B": "#72B7B2", "Model C": "#E45756",
              "Model D": "#F58518", "LLM sentiment": "#9467BD",
              "Momentum 12-1": "#8E44AD"}
    styles = {"LLM sentiment": "--", "Momentum 12-1": ":"}

    fig, ax = plt.subplots(figsize=(13, 7))
    for name, sim in sims.items():
        eq = (1.0 + sim["net"]).cumprod()
        r = results[results["strategy"] == name].iloc[0]
        ax.plot(eq.index, eq, color=colors.get(name, "grey"),
                linestyle=styles.get(name, "-"),
                linewidth=2.2 if name.startswith("Model") else 1.8,
                label=f"{name:<15} {eq.iloc[-1]:.2f}x   {r['ann_return_net']*100:+5.1f}%/yr   "
                      f"t={r['t_newey_west']:+.2f}")

    ax.axhline(1.0, color="black", linewidth=1.2)
    ax.set_ylabel("Growth of $1 on the long-short spread (net)", fontsize=11)
    ax.set_title("Long-short diagnostic: top 3 minus bottom 3, dollar-neutral\n"
                 "Isolates cross-sectional selection from market direction "
                 "— not a tradeable strategy",
                 fontsize=13, weight="bold")
    ax.legend(loc="upper left", frameon=True, framealpha=0.92,
              prop={"family": "monospace", "size": 8.5}, edgecolor="#CCCCCC")
    ax.grid(alpha=0.25)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.text(0.5, 0.008,
             "Excludes borrow fees, financing, recall risk and short-sale constraints. "
             "Upper bound on the signal, not an achievable return.",
             ha="center", fontsize=9, color="#555555")

    fig.tight_layout(rect=[0, 0.025, 1, 1])
    PLOT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_PNG, dpi=DPI)
    plt.close(fig)
    log.info("Wrote %s", PLOT_PNG)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cost-bps", type=float, default=COST_BPS,
                        help="one-way cost per leg in bps")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s", datefmt="%H:%M:%S",
    )
    return run(args.cost_bps)


if __name__ == "__main__":
    sys.exit(main())
