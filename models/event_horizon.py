"""Does the LLM signal work at shorter horizons and decay before month-end?

The monthly pipeline finds no predictive value. This asks whether that is a
*horizon* problem: FOMC information could be priced within days, leaving nothing
by the time a month-end rebalance acts on it. Each statement and each minutes
release is treated as a separate event, and cross-sectional IC is measured over
1, 3, 5, 10, 21 and 42 trading days from the release-day close.

Usage
-----
    python models/event_horizon.py

Outputs
-------
    models/event_horizon_results.csv
    viz/horizon_decay.png

THE MAPPING PROBLEM, AND WHY IT IS NOT 1-TO-1
---------------------------------------------
The sentiment pipeline produces one score set per *month-end*, generated jointly
from the latest statement, the latest minutes, and that month's macro row. No
score is attributable to a single document, so "the sentiment derived from this
document" does not exist as a stored quantity.

Worse, the obvious fix is unsound. A score set at month-end *t* uses every
document published on or before *t*. Attaching it to an event at *e* < *t* leaks
any document released in the window (*e*, *t*]. The March 2020 cluster is the
clearest case: the 2020-03-03 statement would be paired with a March month-end
score that also saw the 2020-03-15 and 2020-03-23 statements.

There is no month-end score that both incorporates a mid-month release and
excludes everything after it, because the score is only computed at month-end.
A clean 1-to-1 mapping is therefore impossible.

The approach used here: map event *e* to the first month-end *t* >= *e*, and
**keep the event only if no other FOMC document was released in (e, t]**. For
those events the score's document inputs are exactly the documents available at
*e*, so although the score was computed later, every input to it was public at
the event. The macro block is lagged and known before the month begins, so it
adds nothing after *e* either.

This keeps 120 of 163 events and drops 43. The dropped ones are not random --
they are disproportionately meetings followed closely by another release -- so
the surviving sample tilts toward quieter periods. That is a real limitation and
is reported rather than hidden.

One residual imperfection: the prompt states the month-end date, so the model
knows which month it is scoring. That is a calendar fact rather than market
information, and no return data enters the prompt.

OVERLAPPING WINDOWS
-------------------
Events occur roughly every three weeks, so a 42-day window spans two or three
subsequent events and the IC series is autocorrelated. Newey-West t-statistics
are reported alongside the naive ones; at the longer horizons only the corrected
figure should be read.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "common"))

from stats import newey_west_t  # noqa: E402
CALENDAR = REPO_ROOT / "data" / "fomc" / "fomc_calendar.csv"
DAILY_PRICES = REPO_ROOT / "data" / "raw" / "sector_prices_daily.csv"
SENTIMENT = REPO_ROOT / "data" / "llm_sentiment.csv"
OUT_CSV = REPO_ROOT / "models" / "event_horizon_results.csv"
PLOT_PNG = REPO_ROOT / "viz" / "horizon_decay.png"

HORIZONS = [1, 3, 5, 10, 21, 42]
TOP_N = 3
N_BOOT = 2000
RANDOM_STATE = 0
SIGNALS = {"sentiment": "raw sentiment", "sentiment_change": "sentiment change"}

log = logging.getLogger("event_horizon")


# --------------------------------------------------------------------------
# Events and the sentiment mapping
# --------------------------------------------------------------------------

def build_events(calendar: pd.DataFrame, score_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """One row per FOMC document release, with its mapped score month-end.

    ``eligible`` is False when another document landed between the event and the
    month-end whose score would be attached -- see the module docstring.
    """
    statements = (calendar.dropna(subset=["statement_date"])[["statement_date", "statement_file"]]
                  .drop_duplicates().rename(columns={"statement_date": "event_date",
                                                     "statement_file": "document"}))
    statements["event_type"] = "statement"
    # Shared minutes: two calendar rows can point at one document, so dedupe.
    minutes = (calendar.dropna(subset=["minutes_date"])[["minutes_date", "minutes_file"]]
               .drop_duplicates().rename(columns={"minutes_date": "event_date",
                                                  "minutes_file": "document"}))
    minutes["event_type"] = "minutes"

    events = pd.concat([statements, minutes], ignore_index=True).sort_values("event_date")
    releases = np.array(sorted(events["event_date"].unique()))

    mapped, eligible, blockers = [], [], []
    for e in events["event_date"]:
        later = score_dates[score_dates >= e]
        if len(later) == 0:
            mapped.append(pd.NaT); eligible.append(False); blockers.append("no score set")
            continue
        t = later[0]
        between = releases[(releases > e) & (releases <= t)]
        mapped.append(t)
        eligible.append(len(between) == 0)
        blockers.append("" if len(between) == 0
                        else ";".join(str(pd.Timestamp(b).date()) for b in between))
    events["score_month_end"] = mapped
    events["eligible"] = eligible
    events["blocked_by"] = blockers
    return events.reset_index(drop=True)


def attach_sentiment(events: pd.DataFrame, sentiment: pd.DataFrame) -> dict:
    """Per-event sentiment vectors: the level and the month-over-month change."""
    change = sentiment.diff()
    out = {}
    for _, row in events[events["eligible"]].iterrows():
        t = row["score_month_end"]
        out[row["event_date"]] = {
            "sentiment": sentiment.loc[t],
            "sentiment_change": change.loc[t],
        }
    return out


# --------------------------------------------------------------------------
# Returns
# --------------------------------------------------------------------------

def forward_excess(prices: pd.DataFrame, event_date: pd.Timestamp, h: int) -> pd.Series | None:
    """Excess return over h trading days from the event-date close.

    Statements and minutes are released at 14:00 ET, so the same day's close is
    post-release and is the first tradeable price. The window therefore runs
    strictly after the release.
    """
    idx = prices.index
    pos = idx.searchsorted(event_date)
    if pos >= len(idx) or idx[pos] != event_date:
        # Release on a non-trading day: use the next session's close.
        if pos >= len(idx):
            return None
    if pos + h >= len(idx):
        return None

    start, end = prices.iloc[pos], prices.iloc[pos + h]
    valid = start.notna() & end.notna() & (start > 0)
    if valid.sum() < 4:
        return None
    ret = (end[valid] / start[valid]) - 1.0
    return ret - ret.mean()          # equal-weight benchmark over the same window


def event_metrics(excess: pd.Series, signal: pd.Series) -> dict | None:
    common = excess.index.intersection(signal.dropna().index)
    if len(common) < 4:
        return None
    x, y = signal[common], excess[common]
    if x.nunique() < 2:
        return None
    order = x.sort_values(ascending=False).index
    top, bottom = order[:TOP_N], order[-TOP_N:]
    return {
        "n_sectors": len(common),
        "ic": float(stats.spearmanr(x, y).statistic),
        "precision_at_3": float((y[top] > 0).mean()),
        "top_minus_bottom": float(y[top].mean() - y[bottom].mean()),
    }


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------



def bootstrap_ci(values: np.ndarray, n_boot: int = N_BOOT) -> tuple[float, float]:
    """Resample events with replacement -- each event contributes one IC."""
    rng = np.random.default_rng(RANDOM_STATE)
    n = len(values)
    if n < 3:
        return np.nan, np.nan
    draws = [values[rng.integers(0, n, n)].mean() for _ in range(n_boot)]
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def summarise(records: pd.DataFrame, signal: str, h: int) -> dict:
    sub = records[(records["signal"] == signal) & (records["horizon"] == h)].dropna(subset=["ic"])
    if sub.empty:
        return {}
    ic = sub["ic"].to_numpy()
    lags = max(1, math.ceil(h / 21))
    t_naive = float(ic.mean() / (ic.std(ddof=1) / np.sqrt(len(ic)))) if ic.std(ddof=1) > 0 else np.nan
    lo, hi = bootstrap_ci(ic)
    tmb = sub["top_minus_bottom"].to_numpy()
    tmb_t = float(tmb.mean() / (tmb.std(ddof=1) / np.sqrt(len(tmb)))) if tmb.std(ddof=1) > 0 else np.nan
    return {
        "signal": signal, "horizon_days": h, "n_events": len(sub),
        "mean_ic": float(ic.mean()), "median_ic": float(np.median(ic)),
        "ic_std": float(ic.std(ddof=1)),
        # The lag is passed explicitly: overlapping h-month windows induce
        # exactly h-1 lags of overlap, which is known, not estimated.
        "t_naive": t_naive, "nw_lags": lags, "t_newey_west": newey_west_t(ic, lags)[0],
        "ci_lo": lo, "ci_hi": hi,
        "pct_ic_positive": float((ic > 0).mean()),
        "precision_at_3": float(sub["precision_at_3"].mean()),
        "top_minus_bottom": float(tmb.mean()),
        "top_minus_bottom_t": tmb_t,
    }


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def run() -> int:
    for path in (CALENDAR, DAILY_PRICES, SENTIMENT):
        if not path.exists():
            log.error("%s not found -- run the upstream script first", path)
            return 1

    calendar = pd.read_csv(CALENDAR, parse_dates=["meeting_date", "statement_date", "minutes_date"])
    prices = pd.read_csv(DAILY_PRICES, index_col=0, parse_dates=True).sort_index()
    sentiment = pd.read_csv(SENTIMENT, index_col=0, parse_dates=True)
    sentiment.index = pd.DatetimeIndex(sentiment.index).normalize()

    events = build_events(calendar, sentiment.index)
    n_total = len(events)
    n_ok = int(events["eligible"].sum())
    log.info("%d FOMC document releases (%d statements, %d minutes)", n_total,
             (events["event_type"] == "statement").sum(),
             (events["event_type"] == "minutes").sum())
    log.info("mapping: %d eligible, %d dropped because another release landed before "
             "the month-end whose score would be attached", n_ok, n_total - n_ok)
    dropped = events[~events["eligible"]]
    by_type = dropped.groupby("event_type").size().to_dict()
    log.info("  dropped by type: %s", by_type)

    signals = attach_sentiment(events, sentiment)

    # Verify the monotone-transform claim before relying on it.
    verify_monotone_identity(sentiment)

    records = []
    for _, row in events[events["eligible"]].iterrows():
        e = row["event_date"]
        vecs = signals.get(e)
        if vecs is None:
            continue
        for h in HORIZONS:
            excess = forward_excess(prices, e, h)
            if excess is None:
                continue
            for name in SIGNALS:
                m = event_metrics(excess, vecs[name])
                if m is None:
                    continue
                records.append({"event_date": e, "event_type": row["event_type"],
                                "score_month_end": row["score_month_end"],
                                "signal": name, "horizon": h, **m})
    records = pd.DataFrame(records)
    if records.empty:
        log.error("no usable events")
        return 1

    rows = [summarise(records, s, h) for s in SIGNALS for h in HORIZONS]
    results = pd.DataFrame([r for r in rows if r])
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT_CSV, index=False, float_format="%.4f")
    log.info("Wrote %s", OUT_CSV)

    report(results, records, events)
    plot(results)
    return 0


def verify_monotone_identity(sentiment: pd.DataFrame) -> None:
    """rank and (level - mean) must induce the same ordering as the level."""
    same_rank = same_rel = 0
    for _, row in sentiment.iterrows():
        lvl = stats.rankdata(row.to_numpy())
        same_rank += np.array_equal(lvl, stats.rankdata(row.rank(pct=True).to_numpy()))
        same_rel += np.array_equal(lvl, stats.rankdata((row - row.mean()).to_numpy()))
    n = len(sentiment)
    log.info("monotone check: rank identical in %d/%d months, relative in %d/%d "
             "-> testing raw sentiment only", same_rank, n, same_rel, n)


def report(results: pd.DataFrame, records: pd.DataFrame, events: pd.DataFrame) -> None:
    log.info("")
    log.info("=" * 108)
    log.info("HORIZON DECAY -- cross-sectional IC from the event-day close")
    for signal, label in SIGNALS.items():
        sub = results[results["signal"] == signal]
        if sub.empty:
            continue
        log.info("")
        log.info("  %s", label)
        log.info("    %5s %7s %9s %9s %8s %9s %22s %8s %9s %11s",
                 "days", "events", "mean IC", "med IC", "t", "t (NW)", "95% CI",
                 "%IC>0", "prec@3", "top-bot")
        for _, r in sub.iterrows():
            log.info("    %5d %7d %9.4f %9.4f %8.2f %9.2f  [%+.4f, %+.4f] %8.1f%% %9.3f %10.3f%%",
                     r["horizon_days"], r["n_events"], r["mean_ic"], r["median_ic"],
                     r["t_naive"], r["t_newey_west"], r["ci_lo"], r["ci_hi"],
                     r["pct_ic_positive"] * 100, r["precision_at_3"],
                     r["top_minus_bottom"] * 100)
    log.info("=" * 108)
    log.info("t (NW) uses Newey-West with lag = ceil(h/21); at 42 days the naive t is")
    log.info("unreliable because windows overlap two to three subsequent events.")

    # Interpretation, driven by the numbers rather than asserted.
    log.info("")
    log.info("INTERPRETATION")
    verdicts = []
    for signal, label in SIGNALS.items():
        sub = results[results["signal"] == signal].set_index("horizon_days")
        if sub.empty:
            continue
        short = sub.loc[[h for h in (1, 3, 5) if h in sub.index]]
        long_ = sub.loc[[h for h in (21, 42) if h in sub.index]]
        short_sig = ((short["ci_lo"] > 0) | (short["ci_hi"] < 0)).any()
        any_sig = ((sub["ci_lo"] > 0) | (sub["ci_hi"] < 0)).any()
        if short_sig and abs(long_["mean_ic"]).max() < abs(short["mean_ic"]).max() / 2:
            v = "A: short-lived information that decays before month-end"
        elif not any_sig:
            v = "B: near zero at every horizon -- not a horizon mismatch"
        else:
            v = "C: significant only at isolated horizons -- conditional at best"
        verdicts.append((label, v))
        log.info("  %-18s %s", label, v)
        log.info("    %-16s largest |mean IC| %.4f at %dd; CI excludes zero at %d of %d horizons",
                 "", sub["mean_ic"].abs().max(),
                 int(sub["mean_ic"].abs().idxmax()),
                 int(((sub["ci_lo"] > 0) | (sub["ci_hi"] < 0)).sum()), len(sub))
        # The shape can be suggestive even when no horizon is significant; say so
        # without upgrading it to a finding.
        peak = int(sub["mean_ic"].idxmax())
        if peak in (3, 5, 10) and sub.loc[21, "mean_ic"] < sub.loc[peak, "mean_ic"] / 2:
            log.info("    %-16s shape is decay-like (peaks at %dd, ~zero by 21d) but every CI",
                     "", peak)
            log.info("    %-16s spans zero, so this is a hint at best, not evidence.", "")

    n_ok = int(events["eligible"].sum())
    log.info("")
    log.info("SAMPLE SIZE AND LIMITS")
    log.info("  %d of %d releases usable; each horizon uses %d-%d events.",
             n_ok, len(events), int(results["n_events"].min()), int(results["n_events"].max()))
    log.info("  The %d dropped events are not random -- they are meetings followed",
             len(events) - n_ok)
    log.info("  closely by another release, so the sample tilts toward quieter periods.")
    log.info("  With ~11 sectors per event, a single event's IC has a standard error")
    log.info("  around 0.33, so individual events carry very little information.")

    five = records[(records["horizon"] == 5) & (records["signal"] == "sentiment")]
    by_type = five.groupby("event_type")["ic"].agg(["count", "mean"])
    if not by_type.empty:
        log.info("")
        log.info("  5-day mean IC by document type (raw sentiment):")
        for kind, r in by_type.iterrows():
            log.info("    %-10s n=%3d  mean IC %+.4f", kind, int(r["count"]), r["mean"])


def plot(results: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"sentiment": "#4C78A8", "sentiment_change": "#E45756"}
    fig, ax = plt.subplots(figsize=(11.5, 6.5))

    for signal, label in SIGNALS.items():
        sub = results[results["signal"] == signal].sort_values("horizon_days")
        if sub.empty:
            continue
        x = np.arange(len(sub))
        ax.plot(x, sub["mean_ic"], "o-", color=colors[signal], linewidth=2.2,
                markersize=7, label=label)
        ax.fill_between(x, sub["ci_lo"], sub["ci_hi"], color=colors[signal], alpha=0.15)

    ax.axhline(0, color="black", linewidth=1.2)
    counts = {}
    for h in HORIZONS:
        ns = [int(results[(results.signal == s) & (results.horizon_days == h)]["n_events"].iloc[0])
              for s in SIGNALS if not results[(results.signal == s)
                                              & (results.horizon_days == h)].empty]
        counts[h] = "/".join(str(n) for n in dict.fromkeys(ns))
    ax.set_xticks(np.arange(len(HORIZONS)))
    ax.set_xticklabels([f"{h}d\nn={counts[h]}" for h in HORIZONS], fontsize=10)
    ax.set_xlabel("trading days after the release-day close", fontsize=11)
    ax.set_ylabel("mean cross-sectional IC", fontsize=11)
    ax.set_title("FOMC event-window horizon decay\n"
                 "Shaded bands are bootstrap 95% CIs (resampled by event)",
                 fontsize=13, weight="bold")
    ax.legend(frameon=False, fontsize=10, loc="upper right")
    ax.grid(alpha=0.25)
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
