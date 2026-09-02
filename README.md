# LLM-Enhanced Tactical Sector Allocation

Master's capstone project.

**Research question.** Does a large language model reading Federal Reserve
communications add information that a conventional macro/ML model does not
already have, for the purpose of tactically allocating across the 11 SPDR sector
ETFs?

**Answer: no.** Across five model specifications, three statistical frameworks,
and a factor-adjusted portfolio evaluation, the LLM sentiment feature produces no
detectable incremental value. That null is the finding, and most of this
repository exists to establish it rigorously enough to be believed rather than
to be explained away as a modelling accident.

## Design

The LLM is a **feature generator**, not a decision layer. It never sees the
model's forecasts and never proposes an allocation.

```
FOMC statements + minutes  ──►  Claude Haiku 4.5  ──►  per-sector sentiment scores
                                                                    │
FRED macro (14 series, lagged)  ──┐                                 │
sector momentum + volatility    ──┼──►  monthly panel  ◄────────────┘
cross-sectional ranks           ──┘         │
                                            ▼
                              2 × 2 walk-forward comparison
                     ┌──────────────────────┬──────────────────────┐
                     │  macro only          │  macro + LLM         │
        Logistic     │  Model A             │  Model B             │
        XGBoost      │  Model C             │  Model D             │
                     └──────────────────────┴──────────────────────┘
                                            ▼
                       top-3 equal-weight backtest, 10 bps, monthly
```

Because A/B and C/D differ by *exactly one block of five sentiment columns*, any
difference between them is attributable to the LLM and nothing else. That is the
whole experimental design.

**Target.** For each sector-month, a binary label: did this sector beat the
equal-weight average of the eleven over the following month?

**Sample.** 2016-01 → 2025-11 features, 93 out-of-sample prediction months
(2018-03 → 2025-11), 1,007 sector-month forecasts.

## Repository structure

| Path | Contents |
|---|---|
| `data/` | Collection and feature-building scripts; the generated panel, and `raw/` + `fomc/` source data |
| `llm/` | FOMC scraper and the Anthropic sentiment pipeline |
| `models/` | Walk-forward training, the look-ahead test suite, and every model-level analysis |
| `backtest/` | Portfolio construction, the backtest engine, and portfolio-level analyses |
| `common/` | Routines shared by more than one script (currently the Newey-West estimator) |
| `scripts/` | `verify_all.py`, the repository-wide validation suite |
| `viz/` | Generated figures |
| `logs/` | LLM API call log and the pipeline run log |
| `docs/` | `METHODS.md` — the full methodological record |
| `notebooks/` | Empty; provided for exploratory work |

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.13 is what the results were produced with; `requirements.txt` is pinned
to exact versions. On macOS, XGBoost needs OpenMP:

```bash
brew install libomp
```

Two API keys are read from the environment. Neither is ever written to a file by
this code.

| Variable | Needed by | Where to get one | Cost |
|---|---|---|---|
| `FRED_API_KEY` | `data/collect.py` | https://fredaccount.stlouisfed.org/apikeys | Free |
| `ANTHROPIC_API_KEY` | `llm/sentiment_pipeline.py` | https://console.anthropic.com | ~$2 for a full 120-month rebuild |

```bash
export FRED_API_KEY=...
export ANTHROPIC_API_KEY=...
```

## Running the pipeline

```bash
python run_all.py
```

Stages run in dependency order. A stage is skipped when every file it produces
already exists, so a fresh clone builds everything and a finished clone re-runs
only what is missing.

```bash
python run_all.py --list          # show the plan, run nothing
python run_all.py --dry-run       # plan + preflight checks, run nothing
python run_all.py --skip-llm      # reuse cached sentiment (the step that costs money)
python run_all.py --quick         # smoke test: skip retraining, big resampling, network
python run_all.py --force         # rebuild everything from scratch
python run_all.py --from train    # resume from a stage
python run_all.py --only regime ranking
```

Any stage exiting non-zero stops the run immediately and prints the failing
command, its exit code, and the log path. Nothing downstream runs on the
assumption that an earlier stage "probably worked". Full output is teed to
`logs/run_all.log`.

Then verify:

```bash
python scripts/verify_all.py
```

20 checks across data integrity, timing, reproducibility, result consistency,
documentation and repository hygiene. Exit code is 0 only if all pass, so it is
safe to gate on. A check whose inputs are missing reports SKIP, never PASS.

## Data sources and terms

| Source | Used for | Terms |
|---|---|---|
| **Yahoo Finance** via `yfinance` | 11 SPDR sector ETF and SPY prices | Personal/research use only. Unofficial API, no redistribution rights, no SLA. Prices are adjusted closes and can be revised. |
| **FRED** (St. Louis Fed) via `fredapi` | 14 macro series | Free with an API key. Most series are US federal government works and public domain; a few are third-party (e.g. Michigan consumer sentiment) and carry the provider's own restrictions. Check each series' page before redistributing. |
| **federalreserve.gov** | FOMC statements and minutes | US federal government work, public domain. Scraped with 1–2s delays. Not redistributed here — `.gitignore` excludes `data/fomc/`, regenerate with `llm/fomc_scraper.py`. |
| **Kenneth French Data Library** | Fama-French + momentum factors | Free for research use, credit Kenneth R. French. |
| **Anthropic API** | Sector sentiment from FOMC text | Commercial API, billed per token. Subject to Anthropic's usage policies. |

No source is redistributed in this repository. `.gitignore` excludes every
generated dataset; `run_all.py` rebuilds them.

## Key findings

The detail is in [`docs/METHODS.md`](docs/METHODS.md); the report has the
narrative. In brief:

- **No model ranks sectors better than chance.** Out-of-sample AUCs are 0.4827
  (A), 0.4932 (B), 0.5081 (C), 0.4857 (D). No information coefficient is
  distinguishable from zero under Newey-West, a month-clustered bootstrap, or a
  10,000-draw permutation test; none survives Benjamini-Hochberg correction.
- **The LLM feature adds nothing, in every specification tried.** Raw macro,
  PCA-compressed macro, with and without sector interactions, linear and tree —
  five arms, five ΔAUC confidence intervals spanning zero.
- **Nor is it a representation problem.** Orthogonalising the macro block does
  not let the LLM in. It takes 6 of 14 principal components to reach 90% of
  training variance, so there was never a single dominant macro factor crowding
  it out.
- **The reason is redundancy, established three ways.** The macro block already
  explains 78.5% of the LLM's month-to-month tone; the sector-specific residual
  correlates −0.002 with excess returns; and the LLM long-short portfolio loads
  0.45 on the momentum factor.
- **No alpha anywhere.** 0 of 26 CAPM and Carhart alphas are significant.
  Momentum's apparent long-only edge is the momentum *factor*: CAPM alpha
  +1.76%/yr collapses to a Carhart alpha of +0.07%/yr.
- **Calibration is worse, not better.** Every model scores below a constant 0.5
  forecast. Adding the LLM significantly *worsens* Brier score in the linear pair
  (ΔBrier +0.0045, CI [+0.0006, +0.0092]) — the only statistically significant
  LLM effect found anywhere, and it points the wrong way.

## Known limitations

- **Short sample.** 93 out-of-sample months is small for monthly cross-sectional
  asset pricing. Confidence intervals are wide, and the study is better powered
  to reject large effects than to detect small ones. A true edge of a few basis
  points per month would not be visible here.
- **One regime era.** 2018–2025 covers one hiking cycle, one pandemic crash, and
  one inflation shock. It is not a sample from which regime-conditional claims
  generalise.
- **Revised data, not point-in-time.** FRED series are collected as currently
  revised. Publication lags are modelled (1 month for market data, 2 for
  government and survey releases), but *revisions* are not: the model sees a
  final value where a practitioner would have seen a first print. ALFRED
  vintages would fix this and were out of scope.
- **One LLM, one prompt, one temperature.** Claude Haiku 4.5 at `temperature=0`
  with a frozen prompt. The null result is about this configuration, not about
  language models in general. A larger model, a different prompt, or richer
  source text could behave differently.
- **Sentiment is FOMC-only.** Earnings calls, analyst commentary, and news flow
  are all excluded. The LLM is being asked to infer sector implications from
  monetary-policy text alone, which is a genuinely hard mapping.
- **XLC's short history.** Communication Services listed 2018-06, so it is
  absent from the first part of the sample. Left as NaN, never backfilled; a
  common-universe robustness check covers it.
- **Costs are a simple model.** 10 bps one-way on drifted weights. No market
  impact, no bid-ask spread modelling, no capacity constraints. The long-short
  diagnostic additionally ignores borrow fees and is not a proposed strategy.
- **The October 2025 shutdown** delayed several releases; the gap is
  forward-filled once, before lagging, and logged. See `docs/METHODS.md`.

## Reproducibility

**Fully deterministic — byte-identical on re-run.** Everything reported. Verified
by re-running the 13 fast stages under `run_all.py --quick --force` and diffing
every regenerated artefact: all identical.

- Model fitting: XGBoost and scikit-learn both take `random_state=42`; XGBoost
  runs `n_jobs=1` with `tree_method="hist"`, and grid-search parallelism is
  across folds only, which does not affect results.
- Bootstraps, permutation tests, and shuffles: every one draws from a
  `numpy.random.default_rng(seed)` created with an explicit seed. `verify_all.py`
  scans the AST of every source file to confirm that no unseeded generator and no
  legacy global `np.random.*` call exists anywhere, then proves the seed actually
  binds by re-running a bootstrap and comparing.
- Seed values are **not** uniform across scripts (42 in some, 0 in others). They
  are fixed and recorded, which is what matters; renumbering them would change
  published figures for no benefit.

**Controlled randomness — stable given the seed, would move if reseeded.** The
width of every bootstrap confidence interval, the permutation p-values, and the
shuffle baselines in `ranking_metrics.py`. Point estimates do not depend on any
seed; interval endpoints do, in the last decimal places. None of the conclusions
turns on an interval endpoint — every reported interval either comfortably spans
zero or, in the one significant case, excludes it by a clear margin.

**Not reproducible from scratch.** Re-running `data/collect.py` at a later date
will not reproduce the exact dataset: Yahoo Finance adjusted closes change with
corporate actions, and FRED series are revised. The committed CSVs are the record
of what the results were computed from. `--force` is deliberately opt-in for this
reason, and `collect.py` refuses to overwrite without it.

## Output inventory

Every generated file, what produces it, and its shape: [`OUTPUTS.md`](OUTPUTS.md).

## Repository notes

- Not currently a git repository. Run `git init` before submission; `.gitignore`
  is already written and excludes generated data, logs, model artefacts, and
  `.claude/`.
- `.claude/settings.local.json` is machine-local tooling state. It is gitignored,
  but **it contains a FRED API key in plaintext** from an earlier session — rotate
  that key at https://fredaccount.stlouisfed.org/apikeys before publishing.
