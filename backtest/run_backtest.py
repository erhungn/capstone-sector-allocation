"""Backtest the four models against three benchmarks.

Each month the strategy ranks the available sectors by predicted probability,
buys the top 3 equal-weighted, and holds for one month. Benchmarks are
equal-weight-all-sectors, SPY buy-and-hold, and 12-1 sector momentum.

Usage
-----
    python backtest/run_backtest.py
    python backtest/run_backtest.py --top-n 4 --cost-bps 5

Outputs
-------
    backtest/equity_curves.csv    cumulative growth of $1 per strategy
    backtest/performance.csv      CAGR, Sharpe, Sortino, MaxDD, Calmar, turnover
    backtest/equity_curves.png

Timing
------
A row in predictions.csv dated *f* holds the probability formed at the close of
*f* and ``actual_return``, the return earned over the *following* month. So a
portfolio formed on *f* earns its return in *f+1*, and the equity curve is
indexed by that realisation month -- not by the formation date. Getting this
backwards would shift every strategy a month early and flatter it during
drawdowns.

Costs
-----
Weights drift with returns between rebalances, so turnover is measured against
the *drifted* weights rather than the target weights. Charging against target
weights would invent trades that never happened. Cost is
``sum |w_new - w_drifted| x 10bps``: that sum counts the sells and the buys
separately, and each side pays the one-way rate.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "models"))

from regime_analysis import VIX_HIGH  # noqa: E402

DATA_DIR = REPO_ROOT / "data"
OUT_DIR = REPO_ROOT / "backtest"

PREDICTIONS = REPO_ROOT / "models" / "predictions.csv"
DATASET = DATA_DIR / "model_dataset.csv"
MACRO = DATA_DIR / "macro_features.csv"
SPY_CACHE = DATA_DIR / "raw" / "spy_prices.csv"

EQUITY_CSV = OUT_DIR / "equity_curves.csv"
PERFORMANCE_CSV = OUT_DIR / "performance.csv"
EQUITY_PNG = OUT_DIR / "equity_curves.png"
REPORT_PNG = REPO_ROOT / "viz" / "equity_curves_report.png"

MODELS = ["A", "B", "C", "D"]
MONTHS_PER_YEAR = 12
# Shading cutoff for the equity plot. Imported from regime_analysis rather than
# repeated, so the shaded bands and the regime tables can never drift apart.
HIGH_VOL_VIX = VIX_HIGH

log = logging.getLogger("backtest")


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------

def next_month_end(dates: pd.Series | pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Formation month-end -> the month-end in which the return is realised."""
    return (pd.DatetimeIndex(dates) + pd.offsets.MonthEnd(1)).normalize()


def load_spy(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Month-end SPY closes, cached after the first download.

    SPY is not part of the sector universe, so it is fetched here rather than in
    collect.py, reusing that module's download path so the adjustment and
    month-end conventions match the sector prices exactly.
    """
    if SPY_CACHE.exists():
        s = pd.read_csv(SPY_CACHE, index_col=0, parse_dates=True)["SPY"]
        s.index = pd.DatetimeIndex(s.index).normalize()
        if s.index.min() <= start and s.index.max() >= end:
            log.info("Loaded cached SPY prices (%d months)", len(s))
            return s

    sys.path.insert(0, str(DATA_DIR))
    import collect

    log.info("Downloading SPY (not cached)")
    daily = collect.fetch_daily_closes(["SPY"], start - pd.DateOffset(months=2), end)
    monthly = collect.to_month_end_prices(daily)["SPY"]
    SPY_CACHE.parent.mkdir(parents=True, exist_ok=True)
    monthly.to_frame("SPY").to_csv(SPY_CACHE, float_format="%.6f")
    return monthly


def risk_free_monthly(realised: pd.DatetimeIndex) -> pd.Series:
    """Monthly risk-free rate from FEDFUNDS (annual %, already lagged)."""
    macro = pd.read_csv(MACRO, index_col=0, parse_dates=True)
    macro.index = pd.DatetimeIndex(macro.index).normalize()
    rf = macro["FEDFUNDS_lag1"] / 100.0 / MONTHS_PER_YEAR
    return rf.reindex(realised).ffill()


# --------------------------------------------------------------------------
# Portfolio simulation
# --------------------------------------------------------------------------

def simulate(
    weights_by_date: dict[pd.Timestamp, dict[str, float]],
    returns_by_date: dict[pd.Timestamp, dict[str, float]],
    cost_bps: float,
) -> tuple[pd.Series, pd.Series]:
    """Run a weight schedule forward. Returns (net monthly return, turnover)."""
    cost_rate = cost_bps / 10_000.0
    prev: dict[str, float] = {}
    net_returns, turnovers, dates = [], [], []

    for formation in sorted(weights_by_date):
        target = weights_by_date[formation]
        rets = returns_by_date[formation]

        traded = set(target) | set(prev)
        gross_turnover = sum(abs(target.get(t, 0.0) - prev.get(t, 0.0)) for t in traded)
        cost = gross_turnover * cost_rate

        gross = sum(w * rets.get(t, 0.0) for t, w in target.items())
        net_returns.append(gross - cost)
        turnovers.append(gross_turnover)
        dates.append(formation)

        # Weights drift with realised returns before the next rebalance.
        grown = {t: w * (1.0 + rets.get(t, 0.0)) for t, w in target.items()}
        total = sum(grown.values())
        prev = {t: v / total for t, v in grown.items()} if total > 0 else {}

    idx = next_month_end(pd.DatetimeIndex(dates))
    return pd.Series(net_returns, index=idx), pd.Series(turnovers, index=idx)


def top_n_weights(scores: pd.DataFrame, score_col: str, top_n: int) -> dict:
    """Equal weight across the top-N sectors by ``score_col`` each month."""
    out = {}
    for date, group in scores.groupby("date"):
        chosen = group.nlargest(top_n, score_col)["ticker"].tolist()
        if chosen:
            out[date] = {t: 1.0 / len(chosen) for t in chosen}
    return out


def equal_weight_all(scores: pd.DataFrame) -> dict:
    out = {}
    for date, group in scores.groupby("date"):
        tickers = group["ticker"].tolist()
        out[date] = {t: 1.0 / len(tickers) for t in tickers}
    return out


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def max_drawdown(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1.0).min())


def performance(returns: pd.Series, rf: pd.Series, turnover: pd.Series | None) -> dict:
    r = returns.dropna()
    equity = (1.0 + r).cumprod()
    n = len(r)
    years = n / MONTHS_PER_YEAR
    cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else np.nan

    excess = r - rf.reindex(r.index).fillna(0.0)
    sd = excess.std(ddof=1)
    sharpe = float(excess.mean() / sd * np.sqrt(MONTHS_PER_YEAR)) if sd > 0 else np.nan

    downside = excess.clip(upper=0.0)
    dsd = np.sqrt((downside ** 2).sum() / max(len(excess) - 1, 1))
    sortino = float(excess.mean() / dsd * np.sqrt(MONTHS_PER_YEAR)) if dsd > 0 else np.nan

    mdd = max_drawdown(equity)
    calmar = float(cagr / abs(mdd)) if mdd < 0 else np.nan
    # One-way annual turnover: the simulator's figure counts both sides.
    ann_turnover = float(turnover.mean() * MONTHS_PER_YEAR / 2.0) if turnover is not None else 0.0

    return {
        "months": n,
        "total_return": float(equity.iloc[-1] - 1.0),
        "cagr": cagr,
        "vol_annual": float(r.std(ddof=1) * np.sqrt(MONTHS_PER_YEAR)),
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": mdd,
        "calmar": calmar,
        "annual_turnover": ann_turnover,
        "hit_rate": float((r > 0).mean()),
    }


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def run(top_n: int, cost_bps: float) -> int:
    for path in (PREDICTIONS, DATASET, MACRO):
        if not path.exists():
            log.error("%s not found -- run the upstream script first", path)
            return 1

    preds = pd.read_csv(PREDICTIONS, parse_dates=["date"])
    data = pd.read_csv(DATASET, parse_dates=["date"])

    # Every strategy picks from the same universe, so differences reflect
    # selection alone rather than which sectors were investable.
    universe = preds[["date", "ticker", "actual_return"]].copy()
    momentum = data[["date", "ticker", "ret_12m_lag1"]]
    universe = universe.merge(momentum, on=["date", "ticker"], how="left")

    returns_by_date = {
        date: dict(zip(g["ticker"], g["actual_return"]))
        for date, g in universe.groupby("date")
    }

    log.info("universe: %d months, %s -> %s, %d-%d sectors per month",
             universe["date"].nunique(), universe["date"].min().date(),
             universe["date"].max().date(),
             universe.groupby("date").size().min(), universe.groupby("date").size().max())
    log.info("rule: top %d by score, equal weight, %.0f bps one-way cost", top_n, cost_bps)
    log.info("")

    schedules: dict[str, dict] = {}
    for model_id in MODELS:
        col = f"model_{model_id}_prob"
        scores = preds[["date", "ticker", col]].dropna()
        schedules[f"Model {model_id}"] = top_n_weights(scores, col, top_n)

    schedules["Equal weight"] = equal_weight_all(universe)
    schedules["Momentum 12-1"] = top_n_weights(
        universe.dropna(subset=["ret_12m_lag1"]), "ret_12m_lag1", top_n
    )

    monthly, turnovers = {}, {}
    for name, sched in schedules.items():
        r, t = simulate(sched, returns_by_date, cost_bps)
        monthly[name] = r
        turnovers[name] = t

    # SPY buy-and-hold: one purchase, then no trading.
    realised = next_month_end(sorted(universe["date"].unique()))
    spy = load_spy(realised.min() - pd.DateOffset(months=3), realised.max())
    spy_ret = (spy / spy.shift(1) - 1.0).reindex(realised)
    if spy_ret.isna().any():
        log.warning("SPY missing %d months; forward-filling", int(spy_ret.isna().sum()))
        spy_ret = spy_ret.ffill()
    spy_ret.iloc[0] -= cost_bps / 10_000.0  # initial purchase only
    monthly["SPY buy & hold"] = spy_ret
    turnovers["SPY buy & hold"] = pd.Series(0.0, index=realised)

    returns_df = pd.DataFrame(monthly).sort_index()
    equity = (1.0 + returns_df).cumprod()
    rf = risk_free_monthly(equity.index)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    equity.round(6).to_csv(EQUITY_CSV, index_label="date")
    log.info("Wrote %s (%d months x %d strategies)", EQUITY_CSV, *equity.shape)

    perf = pd.DataFrame({
        name: performance(returns_df[name], rf, turnovers.get(name))
        for name in returns_df.columns
    }).T
    perf.index.name = "strategy"
    perf.to_csv(PERFORMANCE_CSV, float_format="%.4f")
    log.info("Wrote %s", PERFORMANCE_CSV)

    plot(equity, perf)
    report(perf, rf)
    return 0


def high_vol_spans(equity_index: pd.DatetimeIndex,
                   threshold: float = HIGH_VOL_VIX) -> list[tuple]:
    """Contiguous runs of high-volatility months, on the equity curve's axis.

    The regime is defined on ``VIX_lag1`` at the *formation* month, matching
    models/regime_analysis.py, and then mapped forward to the month whose return
    the equity curve plots. Shading the formation month instead would put the
    band a month to the left of the returns it explains.
    """
    data = pd.read_csv(DATASET, parse_dates=["date"])
    vix = data.groupby("date")["VIX_lag1"].first()
    realised = pd.Series(vix.values, index=next_month_end(vix.index))
    flags = realised.reindex(equity_index) >= threshold

    spans, start = [], None
    for date, hot in flags.items():
        if hot and start is None:
            start = date
        elif not hot and start is not None:
            spans.append((start, date))
            start = None
    if start is not None:
        spans.append((start, flags.index[-1]))
    return spans


def plot(equity: pd.DataFrame, perf: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    styles = {
        "Model A": ("#4C78A8", "-", 2.1), "Model B": ("#72B7B2", "-", 2.1),
        "Model C": ("#E45756", "-", 2.1), "Model D": ("#F58518", "-", 2.1),
        "Equal weight": ("#444444", "--", 1.7), "SPY buy & hold": ("#2E7D32", "--", 1.9),
        "Momentum 12-1": ("#8E44AD", ":", 1.9),
    }
    order = sorted(equity.columns, key=lambda c: -equity[c].iloc[-1])
    spans = high_vol_spans(equity.index)

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(14, 9.5), sharex=True,
        gridspec_kw={"height_ratios": [2.6, 1], "hspace": 0.08},
    )

    for i, (lo, hi) in enumerate(spans):
        for axis in (ax, ax2):
            axis.axvspan(lo, hi, color="#C62828", alpha=0.075, zorder=0,
                         label="High-vol regime (VIX ≥ 25)" if i == 0 and axis is ax else None)

    for name in order:
        color, ls, lw = styles.get(name, ("grey", "-", 1.5))
        stats = perf.loc[name]
        ax.plot(equity.index, equity[name], color=color, linestyle=ls, linewidth=lw,
                label=f"{name:<15} {equity[name].iloc[-1]:>5.2f}x   "
                      f"CAGR {stats['cagr']*100:>5.1f}%   Sharpe {stats['sharpe']:>5.2f}",
                zorder=3 if name.startswith("Model") else 2)

    # End-of-curve labels, nudged apart when strategies finish at the same level
    # (Models C and D both land on 1.42x and would otherwise overprint).
    finals = sorted(((equity[n].iloc[-1], n) for n in order), reverse=True)
    offsets, last_val, last_name = {}, None, None
    for value, name in finals:
        shift = 0.0
        if last_val is not None and abs(np.log(last_val / value)) < 0.035:
            shift = offsets[last_name] - 9.5
        offsets[name] = shift
        last_val, last_name = value, name
    for value, name in finals:
        color, _, _ = styles.get(name, ("grey", "-", 1.5))
        ax.annotate(f"{value:.2f}x", xy=(equity.index[-1], value),
                    xytext=(7, offsets[name]), textcoords="offset points",
                    fontsize=8.5, color=color, va="center", weight="bold")

    ax.set_yscale("log")
    ax.set_ylabel("Growth of $1 (log scale)", fontsize=11)
    ax.set_title(
        f"Out-of-Sample Equity Curves ({equity.index[0]:%Y-%m} to {equity.index[-1]:%Y-%m})",
        fontsize=15, weight="bold", pad=14,
    )
    ax.text(0.0, 1.012,
            "Top 3 sectors by model probability, equal weight, monthly rebalance, 10bps one-way costs",
            transform=ax.transAxes, fontsize=9.5, color="#555555")
    ax.legend(loc="upper left", frameon=True, framealpha=0.92, fontsize=9,
              prop={"family": "monospace", "size": 8.5}, edgecolor="#CCCCCC")
    ax.grid(alpha=0.22, which="major")
    ax.grid(alpha=0.10, which="minor")
    ax.set_yticks([0.8, 1.0, 1.5, 2.0, 2.5, 3.0])
    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}x"))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    for name in order:
        color, ls, lw = styles.get(name, ("grey", "-", 1.2))
        dd = (equity[name] / equity[name].cummax() - 1.0) * 100
        ax2.plot(equity.index, dd, color=color, linestyle=ls,
                 linewidth=1.5 if name.startswith("Model") else 1.2)
    ax2.set_ylabel("Drawdown (%)", fontsize=11)
    ax2.grid(alpha=0.22)
    ax2.axhline(0, color="black", linewidth=0.8)
    for side in ("top", "right"):
        ax2.spines[side].set_visible(False)
    ax2.set_xlim(equity.index[0], equity.index[-1] + pd.Timedelta(days=120))

    fig.align_ylabels([ax, ax2])
    fig.savefig(EQUITY_PNG, dpi=150, bbox_inches="tight")
    REPORT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(REPORT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", EQUITY_PNG)
    log.info("Wrote %s (300 dpi, report quality)", REPORT_PNG)
    log.info("Shaded %d high-volatility spans", len(spans))


def report(perf: pd.DataFrame, rf: pd.Series) -> None:
    log.info("")
    log.info("=" * 104)
    log.info("%-16s %8s %8s %8s %8s %9s %8s %8s %9s",
             "strategy", "total", "CAGR", "vol", "Sharpe", "Sortino", "maxDD", "Calmar", "turnover")
    for name, r in perf.sort_values("cagr", ascending=False).iterrows():
        log.info("%-16s %7.2fx %7.2f%% %7.2f%% %8.2f %9.2f %7.1f%% %8.2f %8.0f%%",
                 name, r["total_return"] + 1, r["cagr"] * 100, r["vol_annual"] * 100,
                 r["sharpe"], r["sortino"], r["max_drawdown"] * 100, r["calmar"],
                 r["annual_turnover"] * 100)
    log.info("=" * 104)
    log.info("Risk-free (FEDFUNDS) averaged %.2f%% annualised over the period",
             rf.mean() * MONTHS_PER_YEAR * 100)

    for a, b, label in (("Model A", "Model B", "Logistic"), ("Model C", "Model D", "XGBoost")):
        if a in perf.index and b in perf.index:
            log.info("%-9s adding sentiment (%s -> %s): CAGR %+.2f%%, Sharpe %+.2f",
                     label, a[-1], b[-1],
                     (perf.loc[b, "cagr"] - perf.loc[a, "cagr"]) * 100,
                     perf.loc[b, "sharpe"] - perf.loc[a, "sharpe"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--top-n", type=int, default=3, help="sectors held each month")
    parser.add_argument("--cost-bps", type=float, default=10.0, help="one-way cost in bps")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    return run(args.top_n, args.cost_bps)


if __name__ == "__main__":
    sys.exit(main())
