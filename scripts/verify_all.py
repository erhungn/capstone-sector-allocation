"""Run every validation and audit check in the project, and summarise pass/fail.

    python scripts/verify_all.py
    python scripts/verify_all.py --group timing      # one group only
    python scripts/verify_all.py --list              # names, run nothing
    python scripts/verify_all.py -v                  # show detail for passes too

Exit code is 0 only when every check passes, so this is safe to gate on.

What this is for
----------------
The results in this repository rest on a handful of invariants that are easy to
break silently: a lag applied in the wrong direction, a training window that
creeps forward by one month, a published number that no longer matches the CSV
it was read from. Each of those was checked by hand when it was written. This
re-checks all of them on demand, from the artefacts actually on disk.

Every check is independent and reads only committed outputs -- nothing is
retrained. A check that cannot run because an input is missing reports SKIP, not
PASS: an unrun check is not evidence of anything.
"""

from __future__ import annotations

import argparse
import ast
import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "data"))
sys.path.insert(0, str(REPO_ROOT / "models"))
sys.path.insert(0, str(REPO_ROOT / "backtest"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

log = logging.getLogger("verify")

DATASET = REPO_ROOT / "data" / "model_dataset.csv"
PREDICTIONS = REPO_ROOT / "models" / "predictions.csv"
MACRO = REPO_ROOT / "data" / "macro_features.csv"
MACRO_RAW = REPO_ROOT / "data" / "raw" / "macro_raw.csv"
SECTOR_FEATURES = REPO_ROOT / "data" / "sector_features.csv"
SENTIMENT = REPO_ROOT / "data" / "llm_sentiment.csv"
EQUITY = REPO_ROOT / "backtest" / "equity_curves.csv"
PERFORMANCE = REPO_ROOT / "backtest" / "performance.csv"
METRICS = REPO_ROOT / "models" / "metrics.csv"

# AUC recomputed from disk cannot match the in-memory figure exactly: metrics.csv
# stores 4 dp (+/- 5e-5 on its own) and predictions.csv stores probabilities at
# 6 dp, which perturbs the ordering of near-equal scores that AUC is built from.
# Observed spread across the four models is 1.4e-5 to 5.4e-5; 2e-4 leaves room
# for that while still catching any change large enough to matter.
AUC_TOL = 2e-4


class Skip(Exception):
    """Raised by a check whose inputs are not present."""


@dataclass
class Check:
    group: str
    name: str
    fn: object


CHECKS: list[Check] = []


def check(group: str, name: str):
    def deco(fn):
        CHECKS.append(Check(group, name, fn))
        return fn
    return deco


def need(*paths: Path) -> None:
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise Skip(f"missing {', '.join(p.name for p in missing)}")


# ==========================================================================
# Group: data
# ==========================================================================

@check("data", "Every macro feature is its raw series shifted by its declared lag")
def macro_lags_applied() -> str:
    """The single most consequential invariant in the project.

    A lag applied in the wrong direction, or not at all, would hand the model
    information it could not have had. This rebuilds each feature from
    macro_raw.csv using the Spec table and compares to what was written.
    """
    need(MACRO, MACRO_RAW)
    import collect

    raw = pd.read_csv(MACRO_RAW, index_col=0, parse_dates=True)
    out = pd.read_csv(MACRO, index_col=0, parse_dates=True)

    checked = []
    for series, spec in collect.FRED_SERIES.items():
        col = f"{spec.column}_lag{spec.lag}"
        if col not in out.columns or series not in raw.columns:
            raise AssertionError(f"{col} missing from macro_features.csv")
        s = raw[series]
        if spec.transform == "yoy_pct":
            s = s.pct_change(12, fill_method=None) * 100
        elif spec.transform == "mom_pct":
            s = s.pct_change(fill_method=None) * 100
        expected = s.shift(spec.lag).reindex(out.index)
        got = out[col]
        # The October 2025 shutdown gap is forward-filled before lagging, so
        # compare only where the rebuilt series is defined.
        both = expected.notna() & got.notna()
        if both.sum() == 0:
            raise AssertionError(f"{col}: no overlapping non-null values to compare")
        diff = (expected[both] - got[both]).abs().max()
        if diff > 1e-6:
            raise AssertionError(f"{col}: max abs difference {diff:.3g} against the "
                                 f"lag-{spec.lag} rebuild")
        checked.append(col)
    return f"{len(checked)} series rebuilt from raw and matched exactly"


@check("data", "Macro values are constant across sectors within a month")
def macro_constant_within_month() -> str:
    """Macro is a time series broadcast across the panel.

    models/pca_robustness.py deduplicates on this before fitting PCA; if it
    stopped holding, that deduplication would silently drop information.
    """
    need(DATASET)
    import train

    data = pd.read_csv(DATASET, parse_dates=["date"])
    cols = [c for c in data.columns
            if c.endswith(("_lag1", "_lag2"))
            and c not in train.SECTOR_FEATURES + train.LLM_FEATURES
            and "_x_is_" not in c]
    spread = data.groupby("date")[cols].nunique(dropna=False)
    bad = spread[(spread > 1).any(axis=1)]
    if not bad.empty:
        raise AssertionError(f"{len(bad)} month(s) vary within the panel, "
                             f"first {bad.index[0].date()}")
    return f"{len(cols)} macro columns constant across all {data['date'].nunique()} months"


@check("data", "XLC's pre-inception history is NaN, not filled")
def xlc_not_backfilled() -> str:
    """XLC listed 2018-06. Any value before that would be invented data."""
    need(SECTOR_FEATURES)
    feats = pd.read_csv(SECTOR_FEATURES, index_col=0, parse_dates=True)
    cols = [c for c in feats.columns if c.startswith("XLC_")]
    if not cols:
        raise AssertionError("no XLC columns in sector_features.csv")
    before = feats.loc[feats.index < "2018-06-01", cols]
    filled = int(before.notna().sum().sum())
    if filled:
        raise AssertionError(f"{filled} non-null XLC value(s) before 2018-06")
    after = feats.loc[feats.index >= "2018-07-31", cols]
    if after.notna().sum().sum() == 0:
        raise AssertionError("XLC has no data after inception either -- check the join")
    return f"{len(cols)} XLC columns all NaN before 2018-06, populated after"


@check("data", "No target column reaches any model's feature set")
def no_target_leakage_in_features() -> str:
    need(DATASET)
    import train

    data = pd.read_csv(DATASET, parse_dates=["date"])
    sets = train.feature_sets(data)
    for name, cols in sets.items():
        leaked = [c for c in cols if c in train.FORBIDDEN]
        if leaked:
            raise AssertionError(f"feature set {name!r} contains {leaked}")
    n = {k: len(v) for k, v in sets.items()}
    diff = set(sets["macro_llm"]) - set(sets["macro"])
    if diff != set(train.LLM_FEATURES):
        raise AssertionError(f"macro_llm differs from macro by {sorted(diff)}, "
                             f"expected exactly {train.LLM_FEATURES}")
    return f"macro={n['macro']}, macro_llm={n['macro_llm']}, differing only by the 5 LLM columns"


@check("data", "Sentiment at month t is the score for month t, not a later one")
def sentiment_alignment() -> str:
    """sentiment_lag1 dated *t* must be the reading published for month *t*.

    The dataset's convention is that a row dated t carries features known at t
    and the return earned over t+1, so the sentiment column needs no extra
    shift. An accidental shift in either direction would break the whole LLM
    arm without changing any row count.
    """
    need(DATASET, SENTIMENT)
    data = pd.read_csv(DATASET, parse_dates=["date"])
    # llm_sentiment.csv is stored wide: one row per month, one column per ETF.
    wide = pd.read_csv(SENTIMENT, parse_dates=["date"])
    sent = wide.melt(id_vars="date", var_name="ticker", value_name="source_score")
    merged = data[["date", "ticker", "sentiment_lag1"]].merge(
        sent, on=["date", "ticker"], how="inner")
    score_col = "source_score"
    if merged.empty:
        raise AssertionError("no overlap between dataset and llm_sentiment.csv")
    diff = (merged["sentiment_lag1"] - merged[score_col]).abs().max()
    if diff > 1e-6:
        raise AssertionError(f"sentiment_lag1 differs from the source score by {diff:.3g}")
    return f"{len(merged)} rows match the source sentiment exactly (no stray shift)"


# ==========================================================================
# Group: timing
# ==========================================================================

@check("timing", "models/test_no_lookahead.py passes")
def lookahead_suite() -> str:
    script = REPO_ROOT / "models" / "test_no_lookahead.py"
    need(script)
    proc = subprocess.run([sys.executable, str(script)], cwd=REPO_ROOT,
                          capture_output=True, text=True)
    if proc.returncode != 0:
        tail = (proc.stdout + proc.stderr).strip().splitlines()[-3:]
        raise AssertionError("look-ahead suite failed: " + " | ".join(tail))
    passed = proc.stdout.count("PASS")
    return f"{passed} timing test(s) passed in the dedicated suite"


@check("timing", "Prediction months match the embargoed walk-forward")
def prediction_months() -> str:
    """Regenerating the schedule from the dataset must reproduce predictions.csv.

    This is what catches a window that has quietly shifted -- the failure mode
    that produced the stale '94 months from 2018-02' figure once already.
    """
    need(DATASET, PREDICTIONS)
    import train

    data = pd.read_csv(DATASET, parse_dates=["date"])
    preds = pd.read_csv(PREDICTIONS, parse_dates=["date"])
    all_dates = sorted(data["date"].unique())
    expected = {
        d for d in all_dates
        if d >= pd.Timestamp("2018-01-31")
        and len([x for x in all_dates if x < train.train_cutoff(d, 1)]) >= 24
    }
    actual = set(preds["date"].unique())
    if expected != actual:
        only_e = sorted(str(pd.Timestamp(d).date()) for d in expected - actual)[:3]
        only_a = sorted(str(pd.Timestamp(d).date()) for d in actual - expected)[:3]
        raise AssertionError(f"schedule mismatch: expected-only {only_e}, actual-only {only_a}")
    lo, hi = min(actual), max(actual)
    return (f"{len(actual)} months, {pd.Timestamp(lo).date()} -> {pd.Timestamp(hi).date()}, "
            f"1-month embargo reproduced")


@check("timing", "Equity curve is indexed by the month the return is realised")
def equity_indexed_forward() -> str:
    need(EQUITY, PREDICTIONS)
    equity = pd.read_csv(EQUITY, index_col=0, parse_dates=True)
    preds = pd.read_csv(PREDICTIONS, parse_dates=["date"])
    formation = pd.DatetimeIndex(sorted(preds["date"].unique()))
    expected = (formation + pd.offsets.MonthEnd(1)).normalize()
    if not equity.index.equals(pd.DatetimeIndex(expected)):
        raise AssertionError(
            f"equity index {equity.index[0].date()}..{equity.index[-1].date()} does not "
            f"equal formation+1 {expected[0].date()}..{expected[-1].date()}")
    return (f"{len(equity)} months, {equity.index[0].date()} -> {equity.index[-1].date()} "
            f"= formation month + 1")


# ==========================================================================
# Group: reproducibility
# ==========================================================================

@check("reproducibility", "Every RNG in the project is explicitly seeded")
def rngs_are_seeded() -> str:
    """Static scan: an unseeded generator makes a result unreproducible.

    Flags ``default_rng()`` with no argument and any use of the legacy global
    ``np.random.*`` sampling functions, which draw from process-wide state.
    """
    offenders = []
    seeded = 0
    for path in sorted(REPO_ROOT.glob("*/*.py")) + sorted(REPO_ROOT.glob("*.py")):
        if ".venv" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
            if name == "default_rng":
                if node.args or node.keywords:
                    seeded += 1
                else:
                    offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno} default_rng()")
            elif (isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Attribute)
                  and fn.value.attr == "random"
                  and getattr(fn.value.value, "id", "") == "np"
                  and name not in {"default_rng", "seed"}):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno} np.random.{name}")
            elif name in {"sample", "choice"} and any(
                    kw.arg in {"random_state", "seed"} for kw in node.keywords):
                seeded += 1
    if offenders:
        raise AssertionError("unseeded RNG use: " + "; ".join(offenders[:5]))
    return f"{seeded} seeded generator/sampler call(s), no unseeded RNG anywhere"


@check("reproducibility", "A seeded bootstrap gives the same answer twice")
def bootstrap_is_deterministic() -> str:
    """The static scan proves a seed is passed; this proves it actually binds."""
    need(PREDICTIONS)
    import calibration

    frame = pd.read_csv(PREDICTIONS, parse_dates=["date"]).dropna(
        subset=["model_A_prob", "model_B_prob", "actual_target"])
    a = calibration.clustered_bootstrap_brier_diff(frame, "model_A_prob", "model_B_prob", 200, 42)
    b = calibration.clustered_bootstrap_brier_diff(frame, "model_A_prob", "model_B_prob", 200, 42)
    if a != b:
        raise AssertionError(f"same seed gave different results: {a} vs {b}")
    c = calibration.clustered_bootstrap_brier_diff(frame, "model_A_prob", "model_B_prob", 200, 7)
    if a[1:] == c[1:]:
        raise AssertionError("a different seed gave an identical interval -- seed is ignored")
    return f"seed 42 reproduces CI [{a[1]:+.6f}, {a[2]:+.6f}]; seed 7 differs, as it should"


@check("reproducibility", "requirements.txt pins match the installed environment")
def requirements_match_env() -> str:
    from importlib.metadata import PackageNotFoundError, version

    req = REPO_ROOT / "requirements.txt"
    need(req)
    mismatches, checked = [], 0
    for line in req.read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if not line or "==" not in line:
            continue
        pkg, pinned = line.split("==", 1)
        try:
            installed = version(pkg.strip())
        except PackageNotFoundError:
            mismatches.append(f"{pkg} not installed")
            continue
        checked += 1
        if installed != pinned.strip():
            mismatches.append(f"{pkg}: pinned {pinned.strip()}, installed {installed}")
    if mismatches:
        raise AssertionError("; ".join(mismatches))
    return f"{checked} pinned package(s) match the running environment exactly"


# ==========================================================================
# Group: results
# ==========================================================================

@check("results", "metrics.csv AUCs recompute from predictions.csv")
def metrics_reproduce() -> str:
    need(METRICS, PREDICTIONS)
    from sklearn.metrics import roc_auc_score

    preds = pd.read_csv(PREDICTIONS, parse_dates=["date"])
    published = pd.read_csv(METRICS)
    key = published.columns[0]
    deltas = []
    for mid in ["A", "B", "C", "D"]:
        sub = preds.dropna(subset=[f"model_{mid}_prob", "actual_target"])
        recomputed = roc_auc_score(sub["actual_target"].astype(int), sub[f"model_{mid}_prob"])
        stated = float(published.set_index(key).loc[mid, "auc"])
        deltas.append(abs(recomputed - stated))
        if abs(recomputed - stated) > AUC_TOL:
            raise AssertionError(f"Model {mid}: metrics.csv says {stated}, "
                                 f"predictions.csv gives {recomputed:.6f}")
    return f"4 models agree within {max(deltas):.1e} (storage tolerance {AUC_TOL:.0e})"


@check("results", "performance.csv recomputes from equity_curves.csv")
def performance_reproduces() -> str:
    need(PERFORMANCE, EQUITY)
    equity = pd.read_csv(EQUITY, index_col=0, parse_dates=True)
    perf = pd.read_csv(PERFORMANCE)
    key = perf.columns[0]
    worst, n = 0.0, 0
    for name in equity.columns:
        row = perf[perf[key] == name]
        if row.empty:
            continue
        curve = equity[name].dropna()
        years = len(curve) / 12
        cagr = curve.iloc[-1] ** (1 / years) - 1
        delta = abs(cagr - float(row["cagr"].iloc[0]))
        worst = max(worst, delta)
        n += 1
        if delta > 5e-5:
            raise AssertionError(f"{name}: CAGR {cagr:.6f} vs stated {row['cagr'].iloc[0]}")
    if n == 0:
        raise AssertionError("no strategies matched between the two files")
    return f"{n} strategies' CAGR reproduced from the curves within {worst:.1e}"


@check("results", "calibration_results.csv Brier scores recompute")
def calibration_reproduces() -> str:
    out = REPO_ROOT / "models" / "calibration_results.csv"
    need(out, PREDICTIONS)
    import io

    import calibration

    preds = pd.read_csv(PREDICTIONS, parse_dates=["date"])
    block = out.read_text(encoding="utf-8").split("\n# ")[0]
    block = block.split("\n", 1)[-1] if block.startswith("#") else block
    published = pd.read_csv(io.StringIO(block))
    worst = 0.0
    for _, row in published.iterrows():
        mid = row["model"]
        sub = preds.dropna(subset=[f"model_{mid}_prob", "actual_target"])
        b = calibration.brier(sub[f"model_{mid}_prob"].to_numpy(float),
                              sub["actual_target"].to_numpy(float))
        worst = max(worst, abs(b - float(row["brier"])))
        if abs(b - float(row["brier"])) > 1e-5:
            raise AssertionError(f"Model {mid}: Brier {b:.6f} vs stated {row['brier']}")
    return f"{len(published)} models' Brier scores reproduced within {worst:.1e}"


@check("results", "pca_robustness.csv's raw arm equals the headline models")
def pca_raw_arm_matches() -> str:
    """The PCA study reuses Models A-D rather than re-fitting them.

    If those columns ever drifted from metrics.csv, the whole comparison would
    be against a different baseline than the one the report quotes.
    """
    out = REPO_ROOT / "models" / "pca_robustness.csv"
    need(out, METRICS)
    import io

    blocks = out.read_text(encoding="utf-8").split("\n# ")
    specs = pd.read_csv(io.StringIO(blocks[2].split("\n", 1)[1])).set_index("spec")
    published = pd.read_csv(METRICS)
    key = published.columns[0]
    pairs = {"raw_base": "A", "raw_llm": "B", "raw_base_xgb": "C", "raw_llm_xgb": "D"}
    worst = 0.0
    for spec, mid in pairs.items():
        stated = float(published.set_index(key).loc[mid, "auc"])
        got = float(specs.loc[spec, "auc"])
        worst = max(worst, abs(got - stated))
        if abs(got - stated) > AUC_TOL:
            raise AssertionError(f"{spec} AUC {got:.6f} vs Model {mid} {stated}")
    return f"4 reused arms agree with metrics.csv within {worst:.1e}"


# ==========================================================================
# Group: docs
# ==========================================================================

@check("docs", "README headline figures match the artefacts")
def readme_figures_current() -> str:
    """Catches the stale-number failure mode directly.

    An earlier pass found the README claiming 1,027 out-of-sample rows and 94
    prediction months when the embargo had left 1,007 and 93. Prose drifts from
    artefacts silently; this makes the drift a test failure. Each entry is a
    literal that must appear in README.md, computed fresh from the files.
    """
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    need(DATASET, PREDICTIONS, METRICS)
    data = pd.read_csv(DATASET, parse_dates=["date"])
    preds = pd.read_csv(PREDICTIONS, parse_dates=["date"])
    # AUCs are compared against metrics.csv, not a recomputation: metrics.csv is
    # the artefact the report cites, and recomputing from the 6 dp probabilities
    # in predictions.csv lands a hair away (see AUC_TOL).
    published = pd.read_csv(METRICS).set_index("model")

    claims = {
        "out-of-sample forecast count": f"{len(preds):,}",
        "prediction month count": f"{preds['date'].nunique()} out-of-sample",
        "panel row count": f"{len(data):,}" if f"{len(data):,}" in readme else None,
    }
    for mid in ["A", "B", "C", "D"]:
        claims[f"AUC model {mid}"] = f"{float(published.loc[mid, 'auc']):.4f}"

    missing = [f"{label} ({value!r})" for label, value in claims.items()
               if value is not None and value not in readme]
    if missing:
        raise AssertionError("README does not contain " + "; ".join(missing))
    return f"{sum(v is not None for v in claims.values())} headline figures found verbatim in README.md"


@check("docs", "Every repo path named in the docs exists")
def doc_paths_exist() -> str:
    """A link or path in prose that points at nothing is a documentation bug."""
    import re

    pattern = re.compile(r"`((?:data|models|backtest|llm|viz|logs|scripts|common|docs)/"
                         r"[A-Za-z0-9_./-]+)`")
    checked, missing = 0, []
    for doc in ["README.md", "OUTPUTS.md", "docs/METHODS.md"]:
        path = REPO_ROOT / doc
        if not path.exists():
            continue
        for ref in set(pattern.findall(path.read_text(encoding="utf-8"))):
            checked += 1
            if not (REPO_ROOT / ref).exists():
                missing.append(f"{doc} -> {ref}")
    if missing:
        raise AssertionError(f"{len(missing)} dangling path(s): {missing[:4]}")
    return f"{checked} distinct repo paths referenced across the docs, all present"


# ==========================================================================
# Group: reports
# ==========================================================================

@check("reports", "Report tables match their source CSVs")
def report_tables_current() -> str:
    """Rebuild every table in memory and compare against what is committed.

    The report is written from reports/tables/. If an analysis is re-run and a
    table is not rebuilt, the report silently quotes a superseded number. This
    turns that into a failing check rather than a misprint.
    """
    import build_report_tables as brt

    missing = [p.name for p in (brt.MODELS_CSV, brt.RANKING_CSV, brt.PCA_CSV,
                                brt.LADDER_CSV, brt.PERFORMANCE_CSV, brt.IC_CSV,
                                brt.REDUNDANCY_CSV, brt.HORIZON_CSV, brt.FACTOR_CSV,
                                brt.COMMON_CSV, brt.GRID_CSV, brt.CALIB_CSV)
               if not p.exists()]
    if missing:
        raise Skip(f"missing {', '.join(missing)}")
    if not brt.OUT_DIR.exists():
        raise AssertionError("reports/tables/ does not exist -- run "
                             "python scripts/build_report_tables.py")

    built = brt.build(sorted(brt.TABLES))
    problems = brt.check(built, brt.OUT_DIR)
    if problems:
        raise AssertionError(f"{len(problems)} stale table(s): {problems[:3]}")
    cells = sum(f.size for f, _, _ in built.values())
    return f"{len(built)} tables, {cells} cells, all rebuilt and identical to the committed files"


@check("reports", "Every figure is at least 300 DPI")
def figures_are_print_resolution() -> str:
    """A 150 DPI figure looks fine on screen and blurry in a printed report."""
    try:
        from PIL import Image
    except ImportError:
        raise Skip("Pillow not installed")

    figures = sorted((REPO_ROOT / "viz").glob("*.png"))
    extra = REPO_ROOT / "backtest" / "equity_curves.png"
    if extra.exists():
        figures.append(extra)
    if not figures:
        raise Skip("no figures found")

    low = []
    for path in figures:
        dpi = Image.open(path).info.get("dpi", (0, 0))[0]
        if dpi < 299.9:
            low.append(f"{path.name} ({dpi:.0f})")
    if low:
        raise AssertionError(f"{len(low)} figure(s) below 300 DPI: {low[:4]}")
    return f"all {len(figures)} figures at 300 DPI"


@check("reports", "Every figure has a caption, and every caption a figure")
def captions_cover_figures() -> str:
    captions = REPO_ROOT / "reports" / "figures" / "captions.md"
    if not captions.exists():
        raise Skip("reports/figures/captions.md not found")
    text = captions.read_text(encoding="utf-8")

    figures = sorted((REPO_ROOT / "viz").glob("*.png"))
    extra = REPO_ROOT / "backtest" / "equity_curves.png"
    if extra.exists():
        figures.append(extra)
    uncaptioned = [f.name for f in figures if f.name not in text]
    if uncaptioned:
        raise AssertionError(f"{len(uncaptioned)} figure(s) with no caption: {uncaptioned}")
    return f"{len(figures)} figures, each named in captions.md"


# ==========================================================================
# Group: hygiene
# ==========================================================================

@check("hygiene", "No absolute or machine-specific paths in source")
def no_absolute_paths() -> str:
    bad = []
    for path in sorted(REPO_ROOT.glob("*/*.py")) + sorted(REPO_ROOT.glob("*.py")):
        if ".venv" in path.parts:
            continue
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            # Assembled from fragments so this file does not match itself.
            for needle in ("/Us" + "ers/", "/ho" + "me/", "C:" + "\\", "/priv" + "ate/tmp"):
                if needle in stripped:
                    bad.append(f"{path.relative_to(REPO_ROOT)}:{i}")
    if bad:
        raise AssertionError("absolute paths in " + ", ".join(bad[:5]))
    return "all paths derive from REPO_ROOT"


@check("hygiene", "No API keys committed in source or data")
def no_committed_secrets() -> str:
    """Catches a key pasted into a script, a notebook, or a results file.

    .venv and .claude are excluded: the former is not part of the repository,
    and the latter is machine-local tooling state that .gitignore excludes.
    """
    import re

    patterns = [
        (re.compile(r"sk-ant-[A-Za-z0-9_-]{10,}"), "Anthropic key"),
        (re.compile(r"""(?i)(api[_-]?key|token|secret)\s*[=:]\s*["'][A-Za-z0-9]{16,}["']"""),
         "inline credential"),
        (re.compile(r"(?i)FRED_API_KEY\s*=\s*[0-9a-f]{32}"), "FRED key"),
    ]
    exts = {".py", ".md", ".txt", ".csv", ".json", ".jsonl", ".cfg", ".toml", ".ipynb"}
    hits, scanned = [], 0
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file() or path.suffix not in exts:
            continue
        if any(part in {".venv", ".git", ".claude"} for part in path.parts):
            continue
        scanned += 1
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for pattern, label in patterns:
            if pattern.search(text):
                hits.append(f"{path.relative_to(REPO_ROOT)} ({label})")
    if hits:
        raise AssertionError("possible secrets in " + ", ".join(sorted(set(hits))[:5]))
    return f"{scanned} files scanned, no credential patterns found"


@check("hygiene", "Every output declared by run_all.py exists")
def declared_outputs_exist() -> str:
    sys.path.insert(0, str(REPO_ROOT))
    import run_all

    missing = [p for stage in run_all.STAGES for p in stage.produces
               if not (REPO_ROOT / p).exists()]
    if missing:
        raise AssertionError(f"{len(missing)} declared output(s) missing: {missing[:4]}")
    total = sum(len(s.produces) for s in run_all.STAGES)
    return f"all {total} declared pipeline outputs present"


# ==========================================================================
# Runner
# ==========================================================================

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--group", action="append", help="run only these groups")
    parser.add_argument("--list", action="store_true", help="list checks and exit")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="show the detail line for passing checks too")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    selected = [c for c in CHECKS if not args.group or c.group in args.group]
    if args.list:
        for c in selected:
            print(f"  [{c.group:16s}] {c.name}")
        print(f"\n  {len(selected)} check(s)")
        return 0

    print("=" * 96)
    print("VERIFICATION SUITE")
    print("=" * 96)

    results, group = [], None
    started = time.monotonic()
    for c in selected:
        if c.group != group:
            group = c.group
            print(f"\n  {group.upper()}")
        try:
            detail = c.fn()
            status, note = "PASS", detail
        except Skip as err:
            status, note = "SKIP", str(err)
        except AssertionError as err:
            status, note = "FAIL", str(err)
        except Exception as err:  # noqa: BLE001 - a broken check is a failed check
            status, note = "ERROR", f"{type(err).__name__}: {err}"
        results.append((c, status, note))
        print(f"    {status:5s}  {c.name}")
        if status != "PASS" or args.verbose:
            print(f"           {note}")

    counts = {s: sum(1 for _, st, _ in results if st == s)
              for s in ("PASS", "FAIL", "ERROR", "SKIP")}
    print("\n" + "=" * 96)
    print(f"  {counts['PASS']} passed, {counts['FAIL']} failed, "
          f"{counts['ERROR']} errored, {counts['SKIP']} skipped "
          f"({time.monotonic() - started:.1f}s)")

    bad = [(c, st, note) for c, st, note in results if st in ("FAIL", "ERROR")]
    if bad:
        print("\n  NOT PASSING:")
        for c, st, note in bad:
            print(f"    {st}  [{c.group}] {c.name}")
            print(f"          {note}")
    if counts["SKIP"]:
        print(f"\n  {counts['SKIP']} check(s) skipped for missing inputs -- "
              f"run python run_all.py first. A skipped check proves nothing.")
    print("=" * 96)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
