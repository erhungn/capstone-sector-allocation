"""Run the whole capstone pipeline in dependency order.

    python run_all.py                 # build whatever is missing, then analyse
    python run_all.py --skip-llm      # reuse cached sentiment (the paid step)
    python run_all.py --quick         # smoke test: skip the slow stages
    python run_all.py --force         # rebuild everything from scratch
    python run_all.py --list          # show the plan and exit
    python run_all.py --dry-run       # show what would run, without running it
    python run_all.py --from train    # resume from a stage (see --list for keys)

Caching, and why this script decides it rather than the stages
--------------------------------------------------------------
``data/collect.py`` and ``data/build_features.py`` deliberately refuse to
overwrite an existing output without ``--force`` -- a guard against silently
rebuilding a dataset that published results depend on. That guard makes them
exit non-zero on a second run, which is correct for them and useless for an
orchestrator.

So the decision lives here instead: a stage is **skipped** when every file it
produces already exists, and **run** otherwise. ``--force`` runs everything and
passes each stage its own overwrite flag. The effect is that a fresh clone
builds the whole project, and a finished clone re-runs only the analyses.

The LLM stages are resumable on their own (the scraper skips files it already
has; the sentiment pipeline skips months it has already scored), so they are
safe to re-enter. ``--skip-llm`` exists because the sentiment stage spends real
money on any month that is *not* yet cached.

Failure policy
--------------
Loud and immediate. The first non-zero exit stops the run, prints the command,
its exit code, and where the log is. No stage is run on the assumption that an
earlier one "probably worked".
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
LOG_PATH = REPO_ROOT / "logs" / "run_all.log"


@dataclass(frozen=True)
class Stage:
    key: str
    script: str
    title: str
    group: str
    produces: tuple[str, ...] = ()
    force_flag: str | None = None
    needs_env: tuple[str, ...] = field(default=())
    is_llm: bool = False
    # Slow means "retrains a model, resamples >=10k times, or hits the network".
    # --quick drops these so a smoke test exercises the wiring, not the compute.
    is_slow: bool = False
    # Extra command-line arguments, e.g. ("--universe", "14asset").
    argv: tuple[str, ...] = field(default=())
    note: str = ""


# Order is the dependency order. Within "analysis" the only hard constraint is
# that factor_alpha needs backtest/equity_curves.csv, which run_backtest writes.
STAGES: list[Stage] = [
    Stage("collect", "data/collect.py", "Prices + FRED macro", "data",
          ("data/macro_features.csv", "data/sector_prices.csv",
           "data/raw/sector_prices_daily.csv", "data/raw/macro_raw.csv"),
          force_flag="--force", needs_env=("FRED_API_KEY",), is_slow=True),
    Stage("features", "data/build_features.py", "Sector momentum + volatility", "data",
          ("data/sector_features.csv",), force_flag="--force"),
    Stage("fomc", "llm/fomc_scraper.py", "FOMC statements + minutes", "llm",
          ("data/fomc/fomc_calendar.csv",), force_flag="--force", is_llm=True, is_slow=True,
          note="free, but re-scrapes ~163 documents with polite delays"),
    Stage("sentiment", "llm/sentiment_pipeline.py", "Claude sector sentiment", "llm",
          ("data/llm_sentiment.csv",), force_flag="--force",
          needs_env=("ANTHROPIC_API_KEY",), is_llm=True, is_slow=True,
          note="COSTS MONEY -- roughly $2 for a full 120-month rebuild"),
    Stage("dataset", "data/build_dataset.py", "Merged modelling panel", "dataset",
          ("data/model_dataset.csv",)),
    Stage("train", "models/train.py", "Walk-forward 2x2 training", "models",
          ("models/predictions.csv", "models/metrics.csv",
           "models/feature_names.json", "models/final_models.joblib"),
          is_slow=True,
          note="the long one -- expanding-window refit for every month"),
    Stage("lookahead", "models/test_no_lookahead.py", "Look-ahead unit tests", "models",
          note="must pass before any result below is meaningful"),
    Stage("backtest", "backtest/run_backtest.py", "Top-3 portfolio backtest", "backtest",
          ("backtest/equity_curves.csv", "backtest/performance.csv")),

    # --- standalone analyses, all downstream of train + backtest -------------
    Stage("shap", "models/shap_analysis.py", "SHAP attribution", "analysis",
          ("models/shap_values_c.csv", "models/shap_values_d.csv")),
    Stage("regime", "models/regime_analysis.py", "VIX regime split", "analysis",
          ("models/regime_results.csv",)),
    Stage("ranking", "models/ranking_metrics.py", "IC / precision@3 / NDCG@3", "analysis",
          ("models/ranking_metrics.csv",)),
    Stage("sentiment-diag", "models/sentiment_diagnostics.py",
          "Sentiment redundancy vs macro", "analysis",
          ("models/sentiment_redundancy.csv",)),
    Stage("llm-standalone", "models/llm_standalone.py", "LLM signal on its own", "analysis",
          ("models/llm_standalone_results.csv",)),
    Stage("common-universe", "models/robustness_common_universe.py",
          "Common-universe robustness", "analysis",
          ("models/robustness_common_universe.csv",), is_slow=True),
    Stage("portfolio-grid", "backtest/robustness_portfolio.py",
          "Top-N and cost sensitivity", "analysis",
          ("backtest/robustness_portfolio.csv",)),
    Stage("event-horizon", "models/event_horizon.py", "FOMC horizon decay", "analysis",
          ("models/event_horizon_results.csv",)),
    Stage("long-short", "backtest/long_short.py", "Long-short diagnostic", "analysis",
          ("backtest/long_short_results.csv",)),
    Stage("ic-significance", "models/ic_significance.py", "IC significance tests", "analysis",
          ("models/ic_significance.csv",), is_slow=True),
    Stage("factor-alpha", "models/factor_alpha.py", "CAPM / Carhart alpha", "analysis",
          ("models/factor_alpha.csv",), note="reads backtest/equity_curves.csv"),
    Stage("calibration", "models/calibration.py", "Probability calibration", "analysis",
          ("models/calibration_results.csv",), is_slow=True),
    Stage("pca", "models/pca_robustness.py", "PCA macro robustness", "analysis",
          ("models/pca_robustness.csv",), is_slow=True),
    Stage("spec-ladder", "models/spec_ladder.py", "LLM increment across specifications",
          "analysis", ("models/spec_ladder.csv",), is_slow=True,
          note="refits two earlier feature sets so every rung shares one window"),

    # --- 14-asset extension -------------------------------------------------
    # The main specification above is never touched by these: every output
    # carries a _14asset suffix. See common/universe.py.
    Stage("features-14", "data/build_features.py", "Features, 14-asset universe",
          "extension", ("data/sector_features_14asset.csv",),
          force_flag="--force", argv=("--universe", "14asset")),
    Stage("sentiment-14", "llm/sentiment_pipeline.py", "Claude sentiment, 14-asset universe",
          "extension", ("data/llm_sentiment_14asset.csv",), force_flag="--force",
          needs_env=("ANTHROPIC_API_KEY",), is_llm=True, is_slow=True,
          argv=("--universe", "14asset"),
          note="COSTS MONEY -- about $2.30 for a full 120-month rescore"),
    Stage("dataset-14", "data/build_dataset.py", "Panel, 14-asset universe",
          "extension", ("data/model_dataset_14asset.csv",),
          argv=("--universe", "14asset")),
    Stage("train-14", "models/train.py", "Walk-forward training, 14-asset universe",
          "extension", ("models/predictions_14asset.csv", "models/metrics_14asset.csv"),
          is_slow=True, argv=("--universe", "14asset")),
    Stage("backtest-14", "backtest/run_backtest.py", "Backtest, 14-asset universe",
          "extension", ("backtest/equity_curves_14asset.csv",
                        "backtest/performance_14asset.csv"),
          argv=("--universe", "14asset")),
    Stage("universe-compare", "models/universe_comparison.py", "11 vs 14 asset comparison",
          "extension", ("models/universe_comparison.csv",)),
    Stage("per-asset", "models/per_asset_analysis.py", "Per-asset breakdown, both universes",
          "extension", ("models/per_asset_results.csv",),
          note="exploratory; BH-corrected across assets"),

    # --- report deliverables, downstream of every analysis above -------------
    Stage("tables", "scripts/build_report_tables.py", "Report tables (CSV + Markdown)",
          "report", ("reports/tables/table1_model_performance.csv",
                     "reports/tables/table8_robustness_summary.csv",
                     "reports/tables/README.md")),
]

GROUP_TITLES = {
    "data": "DATA COLLECTION AND FEATURES",
    "llm": "LLM LAYER",
    "dataset": "MERGED DATASET",
    "models": "MODELLING",
    "backtest": "BACKTEST",
    "analysis": "STANDALONE ANALYSES",
    "extension": "14-ASSET EXTENSION (bonds + gold)",
    "report": "REPORT DELIVERABLES",
}


# --------------------------------------------------------------------------
# Planning
# --------------------------------------------------------------------------

def outputs_present(stage: Stage) -> bool:
    """True when the stage has nothing left to produce.

    A stage with no declared outputs (the look-ahead tests) never counts as
    satisfied -- it is a check, and a check that is skipped is not a check.
    """
    return bool(stage.produces) and all((REPO_ROOT / p).exists() for p in stage.produces)


def plan(skip_llm: bool, quick: bool, force: bool, start_from: str | None,
         only: list[str] | None) -> list[tuple[Stage, str]]:
    """Decide run/skip for every stage. Returns (stage, reason) pairs."""
    keys = [s.key for s in STAGES]
    if start_from and start_from not in keys:
        raise SystemExit(f"--from: unknown stage {start_from!r}. Known: {', '.join(keys)}")
    for k in only or []:
        if k not in keys:
            raise SystemExit(f"--only: unknown stage {k!r}. Known: {', '.join(keys)}")

    began = start_from is None
    out = []
    for stage in STAGES:
        if stage.key == start_from:
            began = True
        if only:
            out.append((stage, "run" if stage.key in only else "skip: not in --only"))
            continue
        if not began:
            out.append((stage, "skip: before --from"))
        elif skip_llm and stage.is_llm:
            out.append((stage, "skip: --skip-llm"))
        elif quick and stage.is_slow:
            out.append((stage, "skip: --quick"))
        elif force:
            out.append((stage, "run"))
        elif outputs_present(stage):
            out.append((stage, "skip: outputs already present"))
        else:
            out.append((stage, "run"))
    return out


def missing_env(to_run: list[Stage]) -> list[tuple[str, str]]:
    """Environment variables needed by the stages that will actually run."""
    return [(var, s.key) for s in to_run for var in s.needs_env if not os.environ.get(var)]


# --------------------------------------------------------------------------
# Running
# --------------------------------------------------------------------------

class Tee:
    """Write child output to the console and the run log at the same time."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = path.open("a", encoding="utf-8")

    def write(self, text: str) -> None:
        sys.stdout.write(text)
        sys.stdout.flush()
        self.fh.write(text)
        self.fh.flush()

    def close(self) -> None:
        self.fh.close()


def banner(tee: Tee, text: str, char: str = "=") -> None:
    tee.write(f"\n{char * 78}\n{text}\n{char * 78}\n")


def run_stage(stage: Stage, tee: Tee) -> float:
    """Run one stage, streaming its output. Raises SystemExit on failure.

    If we have decided to run a stage whose outputs already exist, the stage's
    own overwrite guard would reject it -- so pass its force flag. This is not
    conditional on ``--force``: ``--only collect`` and ``--from features`` reach
    the same situation and would otherwise fail on a guard we deliberately
    triggered ourselves.
    """
    cmd = [sys.executable, str(REPO_ROOT / stage.script), *stage.argv]
    if stage.force_flag and outputs_present(stage):
        cmd.append(stage.force_flag)

    tee.write(f"$ {' '.join(Path(c).name if i else c for i, c in enumerate(cmd))}\n")
    started = time.monotonic()
    proc = subprocess.Popen(
        cmd, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        tee.write("  " + line)
    code = proc.wait()
    elapsed = time.monotonic() - started

    if code != 0:
        banner(tee, f"FAILED: {stage.script} (stage {stage.key!r}) exited {code}", "!")
        tee.write(f"  command : {' '.join(cmd)}\n")
        tee.write(f"  after   : {elapsed:.1f}s\n")
        tee.write(f"  full log: {LOG_PATH}\n")
        tee.write("  Nothing downstream was run. Fix this stage and re-run with\n")
        tee.write(f"  `python run_all.py --from {stage.key}`.\n\n")
        tee.close()
        raise SystemExit(code)
    return elapsed


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def print_plan(steps: list[tuple[Stage, str]], stream=sys.stdout) -> None:
    group = None
    for stage, reason in steps:
        if stage.group != group:
            group = stage.group
            print(f"\n  {GROUP_TITLES[group]}", file=stream)
        mark = "RUN " if reason == "run" else "skip"
        print(f"    [{mark}] {stage.key:<16} {stage.script:<38} {stage.title}", file=stream)
        if reason != "run":
            print(f"           {'':<16} ({reason})", file=stream)
        elif stage.note:
            print(f"           {'':<16} note: {stage.note}", file=stream)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--skip-llm", action="store_true",
                        help="skip the FOMC scrape and the paid sentiment call, "
                             "reusing data/llm_sentiment.csv as cached")
    parser.add_argument("--quick", action="store_true",
                        help="smoke test: skip stages that retrain a model, resample "
                             ">=10k times, or hit the network")
    parser.add_argument("--force", action="store_true",
                        help="re-run every stage, overwriting existing outputs")
    parser.add_argument("--from", dest="start_from", metavar="STAGE",
                        help="resume from this stage key (see --list)")
    parser.add_argument("--only", nargs="+", metavar="STAGE",
                        help="run just these stage keys, in pipeline order")
    parser.add_argument("--list", action="store_true", help="print the plan and exit")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan and the preflight checks, run nothing")
    args = parser.parse_args(argv)

    steps = plan(args.skip_llm, args.quick, args.force, args.start_from, args.only)
    to_run = [s for s, r in steps if r == "run"]

    if args.list:
        print("Pipeline stages (dependency order):")
        print_plan(steps)
        print(f"\n  {len(to_run)} of {len(steps)} stage(s) would run.\n")
        return 0

    tee = Tee(LOG_PATH)
    banner(tee, "CAPSTONE PIPELINE")
    tee.write(f"  repo    : {REPO_ROOT}\n")
    tee.write(f"  python  : {sys.version.split()[0]} ({sys.executable})\n")
    tee.write(f"  started : {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    tee.write(f"  log     : {LOG_PATH}\n")

    from io import StringIO
    buf = StringIO()
    print_plan(steps, stream=buf)
    tee.write("\n  PLAN\n" + buf.getvalue())
    tee.write(f"\n  {len(to_run)} of {len(steps)} stage(s) will run.\n")

    missing = missing_env(to_run)
    if missing:
        banner(tee, "MISSING ENVIRONMENT VARIABLES", "!")
        for var, key in missing:
            tee.write(f"  {var} is required by stage {key!r}\n")
        tee.write("\n  Export them, or pass --skip-llm if the cached artefacts are enough.\n")
        tee.write("  Nothing was run.\n\n")
        tee.close()
        return 2

    if args.dry_run:
        tee.write("\n  --dry-run: preflight passed, nothing executed.\n\n")
        tee.close()
        return 0

    if not to_run:
        tee.write("\n  Nothing to do -- every stage already has its outputs.\n")
        tee.write("  Use --force to rebuild, or --only STAGE to re-run one.\n\n")
        tee.close()
        return 0

    timings: list[tuple[str, float]] = []
    overall = time.monotonic()
    for i, (stage, reason) in enumerate(steps, start=1):
        if reason != "run":
            continue
        banner(tee, f"[{len(timings) + 1}/{len(to_run)}]  {stage.key}  --  {stage.title}")
        if stage.note:
            tee.write(f"  note: {stage.note}\n")
        timings.append((stage.key, run_stage(stage, tee)))

    banner(tee, "PIPELINE COMPLETE")
    for key, secs in timings:
        tee.write(f"  {key:<18} {secs:8.1f}s\n")
    tee.write(f"  {'TOTAL':<18} {time.monotonic() - overall:8.1f}s\n")
    tee.write("\n  Next: python scripts/verify_all.py\n\n")
    tee.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
