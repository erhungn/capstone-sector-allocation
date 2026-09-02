# Methods and Results

The full methodological record for **LLM-Enhanced Tactical Sector Allocation**.
Every section documents what was built, the decisions taken and why, the results,
and how each was validated. `../README.md` is the short version.

This file is the appendix the capstone report draws on: it keeps the reasoning
that a report has no room for, including the design choices that turned out not
to matter and the checks that caught mistakes along the way.

---

## Data collection

```bash
python data/collect.py
```

Writes two files (`--force` to overwrite, `--start`/`--end` to change the window):

**`data/sector_prices.csv`** — month-end adjusted close, 2016-01 to 2025-12, one
column per ETF: XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, XLB, XLRE, XLC.
Daily bars are pulled from yfinance with `auto_adjust=True` and resampled to the
last trading day of each month; the monthly yfinance endpoint is avoided because
it returns a partial bar for the in-progress month.

**`data/macro_features.csv`** — 14 FRED indicators on the same month-end index.
The applied lag is part of every column name.

Real-time market data (daily series → month-end value), **1-month lag**:

| Column | FRED series | Transform |
| --- | --- | --- |
| `VIX_lag1` | VIXCLS | month-end level |
| `T10Y2Y_lag1` | T10Y2Y | month-end level (10y − 2y, pct) |
| `BAA10Y_lag1` | BAA10Y | month-end level — BBB credit spread, pct |
| `T10YIE_lag1` | T10YIE | month-end level — 10y breakeven inflation, pct |
| `DTWEXBGS_lag1` | DTWEXBGS | month-end level — USD trade-weighted index |
| `WTI_lag1` | DCOILWTICO | month-end level — WTI crude, USD/bbl |
| `FEDFUNDS_lag1` | FEDFUNDS | monthly average level, pct |

Government / survey releases, **2-month lag**:

| Column | FRED series | Transform |
| --- | --- | --- |
| `UNRATE_lag2` | UNRATE | level, pct |
| `CPI_YOY_lag2` | CPIAUCSL | 12-month % change |
| `INDPRO_MOM_lag2` | INDPRO | 1-month % change |
| `UMCSENT_lag2` | UMCSENT | level — Michigan consumer sentiment |
| `M2_YOY_lag2` | M2SL | 12-month % change |
| `PERMIT_lag2` | PERMIT | level — building permits, thousands SAAR |
| `MANEMP_MOM_lag2` | MANEMP | 1-month % change — mfg employment (ISM PMI proxy) |

Untransformed downloads are kept in `data/raw/` so transforms can be re-derived
without re-hitting the APIs — including `sector_prices_daily.csv`, the daily bar
frame `collect.py` already downloads on its way to the month-end panel.

## Sector features

```bash
python data/build_features.py
```

Reads the daily cache above (downloading it only if absent) and writes
**`data/sector_features.csv`** — 120 × 44, same month-end index as
`macro_features.csv`. Four features per ETF, named `{TICKER}_{feature}_lag1`:

| Feature | Definition |
| --- | --- |
| `{T}_ret_1m_lag1` | simple return over the prior 1 month |
| `{T}_ret_3m_lag1` | simple return over the prior 3 months |
| `{T}_ret_12m_lag1` | simple return over the prior 12 months |
| `{T}_rvol_21d_lag1` | std-dev of daily simple returns over 21 trading days |

Simple rather than log returns, so features and the backtest share units — a
portfolio's return is the weighted sum of its holdings' simple returns.

`rvol` is in **daily** units (~0.003–0.09); multiply by `sqrt(252)` to annualize
(4.6%–139%, the top end being XLE in March 2020). Realized vol requires a full
21-day window (`min_periods=21`), so a ticker with a shorter history yields NaN
rather than a noisy estimate off a handful of sessions.

Features are computed on the month-end panel and then shifted one month, so row
*t* is built only from prices through the close of *t−1*. `build_features.py`
runs the same `audit_lags` check as the macro side and aborts if a shift did not
take.

### Missing history is left missing

XLC (inception 2018-06) and XLRE (2015-10) are younger than the 2016-01 start,
so their features begin late — **148 NaNs total, all leading, none interior**:

| Column | First valid |
| --- | --- |
| `XLC_ret_1m_lag1`, `XLC_rvol_21d_lag1` | 2018-08-31 |
| `XLC_ret_3m_lag1` | 2018-10-31 |
| `XLC_ret_12m_lag1` | 2019-07-31 |
| `XLRE_ret_12m_lag1` | 2016-11-30 |

Nothing is forward-filled or interpolated. Note the consequence for the
backtest: a full 11-sector strategy using 12-month momentum cannot start before
**2019-07**. Either start there, or hold sectors without a complete feature row
at zero weight and renormalise — dropping NaN rows silently would change the
sector universe over time and bias the results.

`pct_change(fill_method=None)` is explicit throughout: pandas < 3.0 defaulted to
`'pad'`, which would forward-fill XLC's pre-inception NaNs into fabricated
returns. `requirements.txt` is unpinned, so this matters.

## FOMC corpus

```bash
python llm/fomc_scraper.py
```

Scrapes every FOMC policy statement and set of minutes from federalreserve.gov
(public domain; no `robots.txt` is published). ~170 requests at a 1.5s delay,
about six minutes. Existing files are reused, so re-runs are cheap; `--force`
re-downloads.

```
data/fomc/statements/<statement_date>.txt    83 files
data/fomc/minutes/<meeting_date>.txt         80 files
data/fomc/fomc_calendar.csv                  83 rows
```

Each text file carries a five-line provenance header (document type, dates,
meeting type, source URL) followed by `-` * 70 and then the body as plain text —
split on that separator to get the raw document.

`fomc_calendar.csv` has the five requested columns plus two additions:
`meeting_type` (`scheduled` / `unscheduled` / `notation_vote`) and `notes`.
`meeting_type` is not optional in practice — without it you cannot tell a
regular meeting from an emergency one, and March 2020 has both.

### Publication dates gate usage

`statement_date` and `minutes_date` — not `meeting_date` — determine which
rebalance month may condition on a document. Both are scraped rather than
assumed: statement dates come from the `article__time` stamp on the release
itself, minutes dates from the "(Released ...)" text on the calendar page.
Observed minutes lag is 19–47 days after the meeting, median 21.

Two meetings released their statement **after** the meeting date, so assuming
they coincide would be wrong:

| Meeting | Statement released | |
| --- | --- | --- |
| 2019-10-04 | 2019-10-11 | 7 days later (repo-market response) |
| 2020-03-02 | 2020-03-03 | next day (emergency 50bp cut) |

### Edge cases handled

- **Unscheduled meetings** — 2019-10-04, 2020-03-02, 2020-03-15.
- **Cancelled meetings** — the scheduled 2020-03-17/18 meeting never happened
  and is excluded, which is why 2020 shows 7 scheduled meetings, not 8.
- **Notation votes** — 2020-03-23 (unlimited QE) produced a statement with no
  meeting and no minutes.
- **Shared minutes** — the 2019-10-04 and 2020-03-02 minutes are appended to the
  end of the 2019-10-30 and 2020-03-15 documents rather than published
  separately. Those rows point at the containing file, so **two minutes files
  are each referenced by two calendar rows** — deduplicate on `minutes_file`
  before counting documents.
- **Look-alike releases** — the policy statement is identified by link text, not
  by URL. `monetary20200331a.htm` holds the bare `a` suffix but is a press
  release about the FIMA repo facility, and `monetary20250822a.htm` is the
  Statement on Longer-Run Goals. Filtering on the URL pattern alone would pull
  in three documents that are not policy statements.

Files are gitignored — the corpus is ~8 MB and fully reproducible.

## LLM sentiment scores

```bash
export ANTHROPIC_API_KEY=...
python llm/sentiment_pipeline.py --dry-run    # 3 months, ~$0.07
python llm/sentiment_pipeline.py              # full run, ~$2.70
```

Writes **`data/llm_sentiment.csv`** (120 × 11, same month-end index as the other
feature files) and **`logs/llm_calls.jsonl`** (one record per call: timestamp,
month, prompt length, raw response, parsed scores, token counts, cost, latency,
request id, and the documents used).

Model `claude-haiku-4-5-20251001`, `temperature=0`, `max_tokens=2000`. The run
prints a cost estimate before spending anything, and is **resumable** — months
already in the CSV are skipped unless `--force`, so an interrupted run doesn't
re-pay for completed work.

### Which documents each month sees

For month *t*, the pipeline takes the most recent statement and the most recent
minutes whose **publication date** is on or before the last day of *t* —
`statement_date` and `minutes_date` from `fomc_calendar.csv`, never
`meeting_date`. Verified across all 120 months: no selected document was
published after its month-end. The provenance header is stripped from each text
file before it is sent, so document metadata never enters the prompt.

### Prompt standardization

`SYSTEM_PROMPT` and `USER_TEMPLATE` are frozen. Only three things vary between
calls: statement text, minutes text, and the macro block (fixed column order,
fixed `%.2f` formatting). The sector list, the −1.0/+1.0 scale, the JSON schema,
and the constraints are byte-identical in every period, so scores are comparable
across the sample.

The template is also held fixed **across retries**. Appending a repair
instruction after a parse failure would make that month's prompt differ from
every other month's, which is exactly the comparability the design is protecting.
The tradeoff: at `temperature=0` a retry is close to deterministic, so if a month
fails validation three times the run aborts rather than silently degrading —
partial results are already saved, so re-running resumes.

### Structured outputs

The response shape is enforced server-side via `output_config.format` with
`OUTPUT_SCHEMA`, so generation is constrained to valid JSON matching the schema.
This makes four failure modes impossible at the API level rather than something
to detect afterwards: malformed JSON, a missing sector, an unexpected sector, and
a wrong-typed score (`additionalProperties: false` at all 13 object levels, all
11 tickers `required`, `sentiment` typed `number`).

This is a **request parameter, not prompt text** — `USER_TEMPLATE` is unchanged,
so prompt standardization across periods is unaffected. The first call with a new
schema pays a one-time compilation cost; after that it is cached for 24 hours.

**The schema cannot express everything.** The API rejects numeric bounds
(`minimum`/`maximum`) and string length (`maxLength`), so two checks must stay
outside the schema:

| Constraint | Enforced by |
| --- | --- |
| valid JSON, 11 sectors, no extras, numeric score | schema (server-side) |
| score within [−1.0, 1.0] | `parse_and_validate()` |
| rationale ≤ 2 sentences | prompt only (unenforced) |

A truncated (`stop_reason: max_tokens`) or refused (`stop_reason: refusal`)
response can still violate the schema, so both are checked before parsing and
count as a failed attempt.

### Why this replaced JSON repair

The first live run failed on 2016-01 after three attempts. The log showed it was
**not** truncation: `stop_reason: "end_turn"`, ~715 of 2000 output tokens used,
response ending in a complete `}}`. Haiku 4.5 was emitting 13 opening braces and
12 closing — omitting the single root-level `}`, identically on all three
attempts, because `temperature=0` makes the defect reproducible. Retries could
never have recovered it.

`repair_json()` was written to close unterminated delimiters and did fix all
three captured responses. It is **retained as a canary, not as the mechanism**:
with the schema enforced, a `"json_repaired": true` record in
`logs/llm_calls.jsonl` now means the API returned something the schema should
have prevented, which is worth knowing. It never fabricates data — it runs
before validation, so a genuinely truncated response is still rejected for the
missing sector rather than being scored on ten.

### Cost, and why there are 120 calls not ~80

Spec point 1 says to carry forward when no new FOMC document has appeared since
the last rebalance. **On real data that never happens.** With eight meetings a
year, statements change in 79 of 119 month-transitions and minutes in 80 — and
they alternate, so at least one is new in *every* month:

| Month-end | Latest statement | Latest minutes |
| --- | --- | --- |
| 2016-04-30 | 2016-04-27 | 2016-03-16 |
| 2016-05-31 | 2016-04-27 | 2016-04-27 |
| 2016-06-30 | 2016-06-15 | 2016-04-27 |

The carry-forward branch is implemented and tested (six months pinned to one
document pair produce one API call and five copies), it just never fires here.
Budget for **120 calls, ~$2.70** — not the ~80 the rule implies. It would start
firing if the meeting calendar thinned or the window changed.

## Modeling dataset

```bash
python data/build_dataset.py
```

Merges all three feature files into **`data/model_dataset.csv`** — a
`(date, ticker)` panel of **1,257 rows × 27 columns**: 2 identifiers, 21
features, 4 target columns. 2016-01-31 → 2025-11-30.

| Group | Columns |
| --- | --- |
| identifiers | `date`, `ticker` |
| macro (shared within a month) | the 14 from `macro_features.csv` |
| sector (per ticker) | `ret_1m_lag1`, `ret_3m_lag1`, `ret_12m_lag1`, `rvol_21d_lag1` |
| LLM | `sentiment_lag1` |
| cross-sectional | `ret_1m_rank`, `rvol_21d_rank` |
| target — **never use as features** | `y_fwd_return`, `y_benchmark_return`, `y_excess_return`, `target` |

The four outcome columns carry a `y_` prefix precisely so a stray
`df.drop(columns=["target"])` cannot leave the raw forward return in the feature
matrix.

**Cross-sectional ranks are percentiles, not 1–11.** The universe is 10 sectors
before XLC's 2018-06 inception, so an integer rank would not mean the same thing
in 2016 as in 2020. Both ranks are computed from already-lagged features.

### Target

`target = 1` if a sector's forward return beats the equal-weight average of all
sectors that month, else 0. The benchmark averages every sector with a forward
return that month, so XLC joins from 2018-06 when its prices begin — independent
of when its *features* become complete.

**Class balance: 49.4%** (621 of 1,257), and between 45.5% and 52.5% in every
individual year. Slightly under half is expected: the benchmark is a *mean*, and
a few large winners pull it above the median.

### Alignment — no extra shift is applied

Each source file is already stamped so row *t* holds only what an investor knows
at the close of *t*, so none is shifted again during the merge:

| File | Row *t* contains |
| --- | --- |
| `macro_features.csv` | observations from *t−1* (market) or *t−2* (survey) |
| `sector_features.csv` | prices through the close of *t−1* |
| `llm_sentiment.csv` | FOMC text published on or before *t*, scored against the macro row for *t* |

`sentiment_lag1` keeps the `_lag1` suffix of its siblings to signal "safe to use
at *t*", but it differs in kind: the macro and sector features are built from
observations dated *before* t, whereas the sentiment score is formed *at* t from
information published by then. All three are known at *t*; only the target is not.

Worked example, XLK at 2016-01-31 — `sentiment_lag1` is 0.35, exactly
`llm_sentiment.csv[2016-01-31, XLK]`, and `y_fwd_return` is −0.00654727, exactly
the **February 2016** return. That is the requirement: the January score predicts
the February outcome.

### Dropped rows

1,320 candidate rows (120 × 11) → 1,257:

- **−40** no forward return: 2025-12 for all 11 tickers (no January 2026 price
  in the downloaded window), plus XLC's 29 pre-inception months.
- **−23** incomplete features: XLC 13 rows through 2019-06 and XLRE 10 rows
  through 2016-10, both waiting on a 12-month return.

XLC therefore contributes 77 rows and XLRE 109, against 119 for every other
sector — worth accounting for in any per-sector analysis.

### Verification

`build_dataset.py` re-derives its own claims from source on every run and exits
non-zero if any fail: the target recomputed from the price panel, macro and
sentiment matched to their own month, sentiment checked across all 1,257 rows
against both its own and the prior month (1257/1257 vs 414/1257 — so it is not
shifted), the last row confirmed to be the second-to-last price month, the label
checked against its own ingredients, and a tripwire for any feature correlating
above 0.99 with the forward return. The strongest feature correlation is 0.124
(`WTI_lag1`), which is what an honest monthly-return panel should look like.

## Models

```bash
python models/train.py                    # ~70 seconds
python models/train.py --retune-every 6   # faster; tune less often
```

The 2×2 design crosses estimator against feature set, so the only difference
between A/B and between C/D is the single `sentiment_lag1` column:

| | macro + sector + ranks (20) | + `sentiment_lag1` (21) |
| --- | --- | --- |
| **Logistic (L2, CV-tuned C)** | A | B |
| **XGBoost (CV-tuned grid)** | C | D |

Outputs: `models/predictions.csv` (1,007 out-of-sample rows), `models/metrics.csv`,
`models/feature_names.json`, and `models/final_models.joblib` for SHAP.

### Protocol

Expanding-window walk-forward with a **one-month training embargo**: for
prediction month *P*, train on rows dated through *P−2*, predict the rows dated
*P*. 93 prediction months, 2018-03 → 2025-11.

The embargo is not cosmetic. A row dated *d* carries the return from *d* to
*d+1*, so the *P−1* row's label does not settle until the close of *P*. Training
on it is not a strict look-ahead violation — the label is known at the close of
*P*, which is when the forecast is made — but it is **not implementable**: you
would have to observe P(*P*), finalise that label, retrain, predict, and trade at
P(*P*) in the same instant. Withholding *P−1* gives:

```
close of P-1   observe prices; every training label through P-2 has settled
(between)      retrain, no deadline pressure
close of P     read month-P features, score, take positions
```

`models/test_no_lookahead.py` asserts the invariant — for every prediction row
dated *t*, no training label consumes a price at or after *t* — and includes a
guard test confirming the check fails when the embargo is removed, so it cannot
pass vacuously. Run `--embargo-months 0` to reproduce the original protocol.

**Inner CV splits on dates, not rows.** Hyperparameter tuning uses a custom
`PanelDateSplit` rather than sklearn's `TimeSeriesSplit`. On a panel with ~11
rows per month, splitting by row index cuts through the middle of a month and
puts some tickers in train while their same-month peers sit in validation.
Because the target is *cross-sectional* (beat the equal-weight average that
month), that leaks the answer. On this panel, whose months hold 9–11 rows, plain
`TimeSeriesSplit` bleeds a month across **3 of 3 folds**; `PanelDateSplit` bleeds
none.

### Derived sentiment features, and the redundancy problem

`build_dataset.py` also emits four derived views of the LLM score, all built from
the full 120 × 11 sentiment matrix (so XLC gets a valid change at its first panel
month) and all using information from month *t* or earlier:

| Feature | Definition |
| --- | --- |
| `sentiment_change` | month-over-month change for that sector |
| `sentiment_rank` | percentile rank across sectors within the month |
| `relative_sentiment` | score minus that month's cross-sectional mean |
| `sentiment_dispersion` | std across sectors, constant within a month |

`macro_llm` is now 43 features against `macro`'s 38. Cost: the first month
(2016-01) has no prior sentiment, so 9 training rows drop and the walk-forward
starts one month later (93 prediction months).

**They add nothing detectable, and the controls prove you cannot say more.**
Models A and C have *unchanged* feature sets, so they should be identical across
the two runs. They are not:

| Model | Feature set | Raw sentiment only | + 4 derived | Δ |
| --- | --- | --- | --- | --- |
| A | macro (control) | 0.4869 | 0.4827 | −0.004 |
| C | macro (control) | 0.4945 | 0.5081 | **+0.014** |
| B | logistic + LLM | 0.4809 | 0.4933 | +0.012 |
| D | xgboost + LLM | 0.5010 | 0.4857 | −0.015 |

The controls moved by −0.004 and +0.014 purely from 9 dropped rows and
hyperparameter re-selection. The treated models moved +0.012 and −0.015 — inside
the noise floor the controls establish. **No conclusion about the derived
features is supportable at this sample size.**

The LLM contrast has now taken the values −0.031, −0.010, +0.007, −0.022 across
four specifications. A quantity whose sign flips with every change to the feature
set is noise.

#### The macro block explains 78% of the LLM's tone

`models/sentiment_diagnostics.py` tests the redundancy directly. The prompt hands
Claude the same 14 macro variables the model receives as raw features, so the
score can restate them rather than add to them.

The decomposition is clean, because the macro block is *identical across sectors*
within a month. It can therefore only explain the cross-sectional mean:

```
sentiment(i,t) = monthly_mean(t)  +  relative(i,t)
                 macro can explain    macro cannot, by construction
```

| Component | R² on the 14 macro features | adj R² |
| --- | --- | --- |
| **Monthly mean tone** (n=120) | **0.785** | **0.756** |
| Monthly dispersion (n=120) | 0.648 | 0.601 |
| Pooled score (n=1248) | 0.266 | 0.257 |
| Relative, sector-specific (n=1248) | 0.001 | −0.010 |

Strongest single relationships with the monthly tone: `T10Y2Y_lag1` +0.67 (a
steeper curve reads as bullish), `M2_YOY_lag2` +0.55, `DTWEXBGS_lag1` −0.55 (a
stronger dollar as bearish), `CPI_YOY_lag2` −0.48. Economically sensible readings
— the model is not confused — but readings of data the model already has.

**This is the cleanest explanation of the whole null result.** The LLM's output
splits into two parts and neither can help:

1. **The time-varying tone is ~78% a linear restatement of the macro block.**
   Whatever it adds there, the model already had directly.
2. **The sector-specific view is the only part that could be new** — orthogonal
   to macro by construction (R² 0.001) — **and it is the part every AUC test
   found uninformative** (correlation with realised excess return −0.002).

The design gives the LLM no channel through which to contribute. Feeding it the
same macro variables the model uses guarantees the overlap; the only room left
for new information is sector differentiation, and on this evidence the FOMC text
does not carry it at a one-month horizon.

That points to a concrete redesign rather than abandonment: **remove the macro
block from the prompt** so the score is a pure read of the FOMC text, then
re-test. Signal that survives would be additive by construction.
`viz/sentiment_redundancy.png` plots both panels.

### Sector identity and macro × sector interactions

Both feature sets carry sector identity: ten one-hot dummies (`is_XLK` … `is_XLRE`,
with **XLC as the reference**) and eight economically motivated interactions —
`VIX × {XLE, XLU, XLF}`, `T10Y2Y × {XLF, XLRE, XLU}`, `WTI × XLE`,
`CPI_YOY × XLP`. Without them every sector sees an identical macro row each
month, so a linear model must use one coefficient per macro variable across all
eleven sectors and cannot represent energy and utilities reacting to the same VIX
print in opposite directions.

Feature counts go 20 → **38** (macro) and 21 → **39** (macro+LLM); the sets still
differ by exactly `sentiment_lag1`. Interactions are given to the trees too —
redundant, since XGBoost can build them from the dummies, but it keeps one
feature matrix so an A-vs-C difference is the estimator and not the inputs.

**It did not help.** Measured on the same 94 months, identical rows and targets:

| Model | AUC without sector identity | with | Δ | 95% CI |
| --- | --- | --- | --- | --- |
| A | 0.4728 | 0.4817 | +0.009 | [−0.027, +0.044] |
| B | 0.4724 | 0.4744 | +0.002 | [−0.032, +0.035] |
| C | 0.5112 | 0.4954 | −0.016 | [−0.044, +0.013] |
| D | 0.5012 | 0.5025 | +0.001 | [−0.026, +0.028] |

Every change is indistinguishable from zero. The interactions do get used —
`CPI_YOY_lag2_x_is_XLP` ranks 10th and `VIX_lag1_x_is_XLE` 12th of 39 by SHAP in
Model C — they just carry no predictive value, the same pattern as the LLM
feature.

#### The dummies turned the linear models into a static bet

This is the result worth reporting. Sector dummies have no time variation, so
they can only express a sector's *unconditional* tendency to beat the
cross-sectional average — and the linear models latched onto exactly that:

| | Holdings changed per month | Turnover | Top-3 sectors' share of all slots |
| --- | --- | --- | --- |
| Model A, before | ~2.0 of 3 | 781% | — |
| **Model A, after** | **0.78 of 3** | **326%** | **71%** |
| Model C, after | 2.08 of 3 | 832% | 35% |

Model A now holds XLI in 72 of 94 months, XLK in 71, XLB in 57. Its backtest
CAGR jumped 5.79% → 10.45%, but that is a **factor tilt, not forecasting skill**:
a fixed XLI+XLK+XLB portfolio with no timing whatsoever returned **13.72% CAGR at
0.79 Sharpe** over the same window — better than the model that approximates it.
AUC meanwhile stayed at 0.482, below a coin flip.

So the CAGR improvement is the model discovering that tech and industrials led
2018–2025 and betting on them persistently. That is an in-sample regularity of
this particular window, and it is the main risk of adding sector identity to a
cross-sectional target. The tree models kept their conditional behaviour
(2.08 of 3 holdings changing monthly, top-3 share 35%).

### Results — still a null result

| Model | Estimator | Features | Accuracy | Precision | Recall | AUC |
| --- | --- | --- | --- | --- | --- | --- |
| A | Logistic | macro (38) | 0.4866 | 0.4767 | 0.4970 | 0.4827 |
| B | Logistic | + LLM (43) | 0.4945 | 0.4845 | 0.5071 | 0.4932 |
| C | XGBoost | macro (38) | 0.5005 | 0.4890 | 0.4503 | 0.5081 |
| D | XGBoost | + LLM (43) | 0.4777 | 0.4651 | 0.4462 | 0.4857 |
No model shows meaningful skill: AUCs sit between 0.47 and 0.51 against a 0.50
coin flip. **Adding the LLM feature does not help** — B − A is −0.007 and D − C
is +0.007, both indistinguishable from zero, and neither sign is stable across
specifications (D − C was −0.031 with no embargo, −0.010 with the embargo and no
sector identity, +0.007 here). A coefficient whose sign flips with the feature
set is noise, not signal.

### Effect of the embargo

Measured on the 94 months common to both runs, with identical rows, targets, and
realised returns:

| Model | AUC without embargo | AUC with embargo | Δ | 95% CI (month-clustered) |
| --- | --- | --- | --- | --- |
| A | 0.4729 | 0.4728 | −0.000 | [−0.016, +0.015] |
| B | 0.4672 | 0.4724 | +0.005 | [−0.012, +0.022] |
| C | 0.4852 | 0.5112 | +0.026 | [−0.007, +0.058] |
| D | 0.4547 | 0.5012 | **+0.047** | **[+0.015, +0.078]** |

Removing the non-implementable month *improved* the tree models. The mechanism
is worth stating carefully, because two things are mixed together:

1. **A real effect.** The withheld month is the most recent cross-section, and
   monthly sector returns show short-horizon reversal — last month's relative
   winners tend to be next month's laggards. Training on it fed the model an
   anti-signal. Model C supports this reading: 0.485 → 0.511 → 0.512 across
   embargoes of 0, 1, 2 — it improves once and then stays put.
2. **Model instability.** Model D goes 0.455 → 0.501 → 0.489: up, then partly
   back. And the linear models barely move (A: −0.0002) while the tree models
   swing by 0.03–0.05. One month out of ~119 should not move predictions this
   much — old-vs-new prediction correlation is only 0.60 for XGBoost — so
   monthly hyperparameter retuning is re-selecting quite different trees on
   nearly identical data.

Neither reading changes the conclusion: every AUC stays within ~0.01–0.05 of
0.50, and the LLM contrast stays negative. The embargo makes the models look
*less bad*, not good.

## Backtest

```bash
python backtest/run_backtest.py
python backtest/run_backtest.py --top-n 4 --cost-bps 5
```

Top 3 sectors by predicted probability each month, equal weight, long-only,
monthly rebalance, 10bps one-way costs. Outputs `equity_curves.csv`,
`performance.csv`, and `equity_curves.png` (growth plus a drawdown panel).

| Strategy | Total | CAGR | Vol | Sharpe | Sortino | MaxDD | Calmar | Turnover |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SPY buy & hold | 2.92× | 14.84% | 16.5% | 0.77 | 1.18 | −23.9% | 0.62 | 0% |
| Momentum 12-1 | 2.71× | 13.72% | 16.0% | 0.73 | 1.11 | −15.7% | 0.87 | 265% |
| Equal weight | 2.43× | 12.12% | 16.1% | 0.63 | 0.96 | −22.6% | 0.54 | 23% |
| Model B | 2.41× | 12.00% | 17.7% | 0.59 | 0.88 | −26.2% | 0.46 | 428% |
| Model C | 2.35× | 11.67% | 16.1% | 0.61 | 0.98 | −20.8% | 0.56 | 776% |
| Model A | 2.32× | 11.47% | 17.5% | 0.56 | 0.82 | −26.7% | 0.43 | 355% |
| Model D | 2.05× | 9.71% | 16.3% | 0.49 | 0.75 | −22.5% | 0.43 | 827% |

**Every model still loses to every benchmark.** The best (Model B, 12.00%)
trails SPY by nearly 3 points of CAGR and 0.18 of Sharpe, with the worst drawdown
of any strategy tested (−26.2%).

Costs are a minor part of it: at ~790% turnover the 10bps fee costs ~1.6%/yr
against a ~4%/yr gap to SPY. The rest is selection.

Two conventions that materially affect the numbers:

- **The equity curve is indexed by the month the return is *realised*,** not the
  formation month. A row dated *f* carries the return earned over *f+1*, so the
  curve starts 2018-04-30 from a 2018-03-31 formation. Indexing by formation
  date would shift every strategy a month early.
- **Turnover is measured against drifted weights.** Weights move with returns
  between rebalances; charging against target weights would invent trades that
  never happened.

All strategies pick from the same universe (the sectors present in
`predictions.csv`), so differences reflect selection alone, not which sectors
were investable. Verified independently: Model C gross 1.613× vs net 1.4225×
implies 1.60%/yr cost drag, matching 797% turnover × 20bps round-trip.

## SHAP attribution

```bash
python models/shap_analysis.py
```

Explains the XGBoost pair. Outputs `viz/shap_model_c.png`,
`viz/shap_model_d.png`, and per-row values in `models/shap_values_{c,d}.csv`.

**`sentiment_lag1` ranks 6th of 39 features in Model D** (5.6% of total
attribution) — behind only two price-based features and ahead of every macro
variable including VIX. The model leans on it heavily.

That makes the out-of-sample result sharper, not weaker. Despite being the 3rd
most-used feature, it does not improve out-of-sample skill (ΔAUC −0.010, CI
[−0.033, +0.013]). The direct test of the raw signal explains why:

| | value |
| --- | --- |
| corr(`sentiment_lag1`, realised excess return) | **−0.002** |
| corr(SHAP of sentiment, realised excess return) | +0.24 |

Mean excess return by sentiment quintile is flat and non-monotonic (Q1 +0.04%,
Q3 −0.43%, Q5 +0.18%). The raw score carries essentially **zero** information
about next-month relative performance; the high SHAP correlation is the
in-sample refit memorising noise. That is the mechanism by which a feature can
be prominent in attribution and still destroy out-of-sample performance — a
useful thing to be able to demonstrate rather than assert.

SHAP values here are in-sample by construction (they explain the full-sample
refits), so they describe how the model *uses* a feature, not whether it
predicts.

## Regime analysis

```bash
python models/regime_analysis.py
```

Splits the out-of-sample window by `VIX_lag1` (< 20 low, ≥ 25 high, else
medium) to test the guideline's hypothesis that Fed communication matters most
under stress. Outputs `models/regime_results.csv` and
`viz/regime_auc_comparison.png`. The regime label uses the *lagged* VIX, so it
is known when the forecast is made — regimes are not assigned with hindsight.

| Regime | Months | Obs | Mean VIX | AUC A | AUC B | AUC C | AUC D |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Low (<20) | 61 | 656 | 15.9 | 0.465 | 0.467 | 0.502 | 0.491 |
| Medium (20–25) | 14 | 153 | 22.3 | 0.521 | 0.519 | 0.549 | 0.542 |
| High (≥25) | 19 | 208 | 31.2 | 0.460 | 0.453 | 0.509 | 0.494 |
| All | 94 | 1017 | 19.9 | 0.473 | 0.472 | 0.511 | 0.501 |

**The hypothesis is not supported.** The LLM feature does not help under stress —
the point estimate is negative in every regime, and most negative in high-vol:

| Regime | Pair | ΔAUC | 95% CI | Verdict |
| --- | --- | --- | --- | --- |
| Low | D − C | −0.010 | [−0.042, +0.023] | no detectable effect |
| Medium | D − C | −0.007 | [−0.054, +0.040] | no detectable effect |
| High | D − C | −0.014 | [−0.054, +0.025] | no detectable effect |
| Low | B − A | +0.002 | [−0.007, +0.012] | no detectable effect |
| High | B − A | −0.008 | [−0.023, +0.007] | no detectable effect |

Under the pre-embargo protocol the high-vol XGBoost contrast was significantly
negative (−0.060, CI [−0.106, −0.018]). With the embargo it is smaller and no
longer distinguishable from zero — one of several results that softened once the
non-implementable month was removed. The direction is unchanged; the strength of
the claim is not.

Correlation between `sentiment_lag1` and realised excess return is ≈ 0 in every
regime: +0.006 high-vol, −0.002 low-vol, −0.013 medium.

The most interesting row is sentiment *dispersion* — the average within-month
spread across the 11 sectors:

| Regime | Mean | Std | Dispersion | corr(excess) |
| --- | --- | --- | --- | --- |
| Low (<20) | +0.054 | 0.362 | 0.306 | −0.0020 |
| Medium | −0.056 | 0.453 | 0.373 | −0.0133 |
| High (≥25) | −0.016 | 0.510 | **0.428** | +0.0055 |

**The LLM becomes markedly more opinionated as volatility rises** (dispersion
0.305 → 0.428, +40%) while the informational content stays at zero. It
differentiates sectors most confidently exactly when it is least reliable, which
is why the XGBoost model — which leans on the feature heavily — degrades most in
the high-vol bucket.

**Bootstrap is clustered by month, not by row.** Sectors within a month share a
macro state and their targets are defined against each other, so rows are
strongly dependent. Resampling rows independently would treat ~11 correlated
observations as 11 independent ones and produce intervals far too narrow.

**Sample caveat.** The 19 high-vol months are not 19 independent events — 8 fall
in 2020 and 7 in 2022. This is roughly four episodes (2019-01, COVID, early
2021, the 2022 bear), so the high-vol column should be read as directional, not
conclusive.

## Ranking metrics — the portfolio-appropriate view

```bash
python models/ranking_metrics.py
```

AUC and accuracy treat every (date, ticker) row as an independent classification.
A sector strategy ranks sectors *against each other* within a month and buys the
top few. These metrics score that directly, and they evaluate the raw LLM signal
standalone rather than only as a feature buried in a model. Outputs
`models/ranking_metrics.csv` and `viz/ic_timeseries.png`.

93 OOS months, 2018-03 → 2025-11. Base rate: 48.8% of sector-months beat the
equal-weight mean.

| Signal | Mean IC | IC std | IC IR | % IC>0 | t-stat | Prec@3 | Top-3 excess/yr | NDCG@3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Model A (logit, macro) | +0.011 | 0.352 | 0.030 | 45.2% | 0.29 | 0.477 | +0.33% | 0.543 |
| Model B (logit, +LLM) | +0.007 | 0.332 | 0.020 | 48.4% | 0.19 | 0.477 | +0.97% | 0.537 |
| Model C (xgb, macro) | **+0.046** | 0.373 | 0.124 | 53.8% | 1.19 | 0.498 | +1.10% | 0.557 |
| Model D (xgb, +LLM) | −0.024 | 0.373 | −0.063 | 46.2% | −0.61 | 0.484 | −0.55% | 0.531 |
| LLM sentiment (raw) | −0.002 | 0.411 | −0.006 | 48.4% | −0.05 | 0.484 | +0.17% | 0.544 |
| **LLM sentiment change** | **+0.043** | 0.343 | 0.127 | 56.5% | 1.22 | 0.518 | +1.43% | 0.553 |
| LLM sentiment rank | −0.002 | 0.411 | −0.006 | 48.4% | −0.05 | 0.484 | +0.17% | 0.544 |
| LLM relative sentiment | −0.002 | 0.411 | −0.006 | 48.4% | −0.05 | 0.484 | +0.17% | 0.544 |
| *Random baseline* | *0* | — | — | — | — | *0.486* | *−0.19%* | *0.540* |

**No signal has a mean IC distinguishable from zero** (best t-stat 1.22, p≈0.23).
Every precision@3 sits within a point or two of the 0.486 random baseline, and
every NDCG@3 is inside the shuffle distribution — the 95th percentile of random
is 0.568, above all eight signals. The random baseline column matters: NDCG does
not start at zero, and 0.54 is what a coin flip scores.

For scale, a monthly IC of 0.02–0.05 is a weak-but-usable equity signal and
below ~0.01 is noise. Model C at +0.046 and sentiment-change at +0.043 are in
usable territory *as point estimates*, but with 93 months neither clears
significance — IC std of ~0.35 against a mean of 0.04 needs far more months.

### Three of the four derived sentiment features are the same signal

`sentiment_lag1`, `sentiment_rank`, and `relative_sentiment` produce **byte-identical
metrics**, verified as identical rankings in 93 of 93 months. This is arithmetic,
not coincidence: within a month, `rank()` is a monotone transform of the level
and `relative` subtracts a constant. Neither changes the ordering, so for any
cross-sectional ranking task they are one signal wearing three hats.

Only `sentiment_change` re-orders the sectors (0 of 93 months matching), and it
is the best-performing LLM signal in the table — highest precision@3 (0.518),
highest top-3 excess (+1.43%/yr), 56.5% of months with positive IC. Still not
significant, but it is the one derived feature that added information rather
than a relabelling, and the one worth carrying into future work.

This also reframes the earlier feature-engineering result: adding four derived
features to `macro_llm` really added **one** new degree of freedom plus three
near-collinear copies — which is a good reason the models could not exploit them.

### What the IC plot shows

`viz/ic_timeseries.png` — 6-month rolling IC for Models C and D and the raw LLM
signal, with the raw signal's monthly IC as bars underneath. The rolling lines
swing between −0.45 and +0.30 with no persistence; the raw LLM signal is positive
in 48% of months. That is the visual form of the same conclusion: sign is a coin
flip, and any full-sample mean is an average over noise rather than a stable edge.

## LLM signal standalone tests

```bash
python models/llm_standalone.py
python models/llm_standalone.py --window full   # 118 months instead of 93
```

Tests the LLM signal directly, with no model wrapped around it. Since
`sentiment_rank` and `relative_sentiment` are identical to the level
cross-sectionally, only two distinct signals exist: **level** and **change**.
Outputs `models/llm_standalone_results.csv` and `viz/quintile_returns.png`.

### (a) Quintile sort — no monotonic relationship

| Bin | LLM level excess/yr | hit rate | LLM change excess/yr | hit rate |
| --- | --- | --- | --- | --- |
| Q1 (bearish) | −1.08% | 47.1% | −2.97% | 42.2% |
| Q2 | +1.26% | 49.5% | +0.19% | 53.2% |
| Q3 | +3.92% | 54.8% | +1.16% | 48.9% |
| Q4 | **−5.61%** | 43.0% | +2.23% | 52.2% |
| Q5 (bullish) | +2.19% | 50.0% | +0.82% | 50.0% |
| **Q5−Q1** | **+3.27%** (t=0.60, p=0.55) | | **+3.79%** (t=0.78, p=0.44) | |

Neither is monotonic. The level's Q4 is the *worst* bin at −5.61%/yr, sitting
between two positive ones — the shape of noise, not a signal. Both long-short
spreads are positive but nowhere near significance.

**A resolution problem undercuts the change sort specifically.** 33% of
`sentiment_change` values are exactly zero and **6.8 of 10.8 sectors per month
sit in a tie group (63%)** — the LLM emits scores on a 0.05 grid and frequently
repeats a sector's score verbatim. So most bin assignments for that signal are
arbitrary tie-breaks rather than signal. The level is clean by comparison (5%
tied). This also explains why sentiment-change "won" the earlier ranking table:
it is a low-resolution signal whose apparent edge rests on very few genuine
distinctions.

### (b) Regime-conditional IC — no detectable difference

| Signal | Low VIX | Medium | High VIX | High − Low (95% CI) |
| --- | --- | --- | --- | --- |
| LLM level | −0.034 | +0.006 | **+0.092** | +0.126 [−0.058, +0.314] |
| LLM change | **+0.068** | −0.047 | +0.034 | −0.033 [−0.195, +0.138] |

The level does score its best IC in high-vol (+0.092, 52.6% positive) — the
direction the guideline predicted — but the confidence interval spans zero, so
it is not distinguishable from the low-vol figure. The change signal points the
opposite way, working best in *low* vol. Two signals disagreeing on the sign of
the regime effect is what noise looks like.

### (c) IC persistence — none

| Signal | lag 1 | lag 2 | lag 3 | white-noise band |
| --- | --- | --- | --- | --- |
| LLM level | −0.043 | −0.266 | −0.015 | ±0.203 |
| LLM change | +0.044 | −0.131 | −0.081 | ±0.204 |

Only the level's lag-2 coefficient escapes the band, and it is *negative* — mean
reversion, not persistence. With three lags tested at 5%, one marginal exceedance
is ~14% likely by chance. A good IC month tells you nothing about the next.

### (d) Big opinion shifts predict nothing

| Bucket | n | share | excess/yr | hit rate | t |
| --- | --- | --- | --- | --- | --- |
| Large positive (> +0.3) | 56 | 5.6% | +0.77% | 51.8% | 0.12 |
| Large negative (< −0.3) | 61 | 6.1% | −0.32% | 47.5% | −0.07 |
| Small (\|change\| ≤ 0.3) | 890 | 88.4% | −0.03% | 48.7% | −0.02 |

The signs go the right way — big upgrades slightly positive, big downgrades
slightly negative — but the magnitudes are trivial and both t-stats are under
0.15. A dramatic change of view carries no more information than a small one.

### More data makes it weaker, not stronger

Re-run over the full 118-month panel (legitimate here — the signal is never
fitted, so there is no leakage in using every month):

| Signal | OOS (93 months) | Full (118 months) |
| --- | --- | --- |
| LLM level | −0.002 | −0.007 |
| LLM change | **+0.043** (t=1.22) | **+0.029** (t=0.90) |

The change signal's IC *shrinks* by a third when 25 more months are added. A real
effect would firm up with more data; an artefact dilutes. That is the strongest
single piece of evidence that the earlier +0.043 was noise.

## Robustness: common-universe sample

```bash
python models/robustness_common_universe.py
```

XLC's prices start 2018-06 and its 12-month return feature completes 2019-07, so
before that the panel holds ten sectors. Both the cross-sectional benchmark and
the ranking task change composition mid-sample. This re-runs everything on the
window where all eleven sectors are present throughout — **from 2019-07-31**,
77 months, contiguous — and reports it beside the main result. Saved to
`models/robustness_common_universe.csv`.

The walk-forward, tuning grid, CV splitter, embargo, and portfolio simulation are
imported from `train.py` and `run_backtest.py` rather than reimplemented, so any
difference between runs is the sample and nothing else.

Three columns separate two effects that would otherwise be confounded: the
shorter **window**, and the restricted **training data**.

| | main (93 mo) | main, common months (52) | retrained (52 mo) |
| --- | --- | --- | --- |
| **AUC A** | 0.4827 [0.447, 0.518] | 0.4721 [0.421, 0.526] | 0.4872 [0.444, 0.531] |
| **AUC B** | 0.4933 [0.459, 0.527] | 0.4881 [0.440, 0.535] | 0.4914 [0.452, 0.532] |
| **AUC C** | 0.5081 [0.476, 0.543] | 0.5209 [0.477, 0.568] | 0.5118 [0.466, 0.564] |
| **AUC D** | 0.4857 [0.449, 0.520] | 0.4913 [0.447, 0.534] | 0.5045 [0.458, 0.556] |
| **B − A** | +0.011 [−0.008, +0.027] | +0.016 [−0.011, +0.039] | +0.004 [−0.024, +0.030] |
| **D − C** | −0.022 [−0.056, +0.009] | −0.030 [−0.074, +0.012] | −0.007 [−0.043, +0.027] |

**The conclusion does not change.** All four retrained AUCs fall inside the
main-sample confidence intervals — the largest shift is Model D at 0.53 CI
half-widths, the rest under 0.15. Both LLM contrasts still straddle zero in every
column, and both shrink *toward* zero when retrained. Every AUC remains within
~0.02 of a coin flip.

### Backtest on the restricted window

| Strategy | main (93 mo) | common months (52) | retrained (52 mo) |
| --- | --- | --- | --- |
| SPY buy & hold | 14.8% / 0.77 | 11.5% / 0.54 | 11.5% / 0.54 |
| Momentum 12-1 | 13.7% / 0.73 | 12.1% / 0.57 | 12.1% / 0.57 |
| Equal weight | 12.1% / 0.63 | 9.2% / 0.42 | 9.2% / 0.42 |
| Model A | 11.5% / 0.56 | 8.6% / 0.36 | 5.4% / 0.18 |
| Model B | 12.0% / 0.59 | 8.1% / 0.33 | 7.0% / 0.28 |
| Model C | 11.7% / 0.61 | 8.7% / 0.38 | 8.4% / 0.36 |
| Model D | 9.7% / 0.49 | 7.0% / 0.29 | 9.0% / 0.43 |

*CAGR / Sharpe. The three benchmarks are numerically identical across the two
52-month columns, as they must be — they do not depend on the models. That
identity is a useful check that the window is applied consistently.*

Every model still loses to every benchmark. Note that the benchmarks themselves
weaken in the shorter window (SPY 14.8% → 11.5%), so the models' lower CAGRs
mostly reflect a less favourable market period rather than degradation — which is
exactly why the benchmark row matters for reading this table.

### Power, stated plainly

52 OOS months against 93. Mean AUC interval width rises from 0.069 to 0.091 —
**31% wider**. Several point estimates move in interesting-looking directions
(Model C's mean IC goes +0.046 → +0.088, Model D's −0.024 → +0.048), but with
intervals this wide none of it is evidence. The correct reading of this table is
"consistent with the main result", not "different from it" — and the restricted
sample is not capable of resolving differences of the size being discussed.

## Robustness: portfolio construction

```bash
python backtest/robustness_portfolio.py
```

Top-3 at 10bps was one arbitrary point. This runs the full grid — N ∈ {2,3,4,5}
crossed with costs ∈ {0,5,10,15,20} bps, **20 cells per strategy, 140 rows** —
for all four models and the 12-1 momentum benchmark. No cell is selected as "the"
configuration; the question is whether the ranking is stable, not which choice
flatters the models. Saved to `backtest/robustness_portfolio.csv`, plotted in
`viz/topn_sensitivity.png`.

### (a) Top-N sensitivity (10 bps)

| Strategy | N=2 | N=3 | N=4 | N=5 |
| --- | --- | --- | --- | --- |
| Model A | 9.5% / 0.46 | 11.5% / 0.56 | 11.9% / 0.60 | 9.9% / 0.49 |
| Model B | 7.2% / 0.33 | 12.0% / 0.59 | 11.1% / 0.55 | 10.8% / 0.54 |
| Model C | 10.1% / 0.49 | 11.7% / 0.61 | 10.5% / 0.54 | 11.7% / 0.63 |
| Model D | 5.5% / 0.25 | 9.7% / 0.49 | 8.8% / 0.44 | 8.9% / 0.47 |
| **Momentum 12-1** | **15.7% / 0.79** | 13.7% / 0.73 | 13.3% / 0.73 | 13.0% / 0.73 |
| Equal weight | 12.1% / 0.63 | — | — | — |
| SPY buy & hold | 14.8% / 0.77 | — | — | — |

*CAGR / Sharpe.* Momentum is stable across N (0.73–0.79) while the models swing
by 0.2–0.3 of Sharpe with no consistent pattern — Model A peaks at N=4, C at N=5,
B and D at N=3. That instability is itself informative: a real signal should not
care much whether you hold three names or four.

### (b) Transaction-cost sensitivity (top 3)

| Strategy | 0bp | 5bp | 10bp | 15bp | 20bp | Sharpe lost | Turnover |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Model A | 0.60 | 0.58 | 0.56 | 0.54 | 0.52 | −0.081 | 355% |
| Model B | 0.64 | 0.61 | 0.59 | 0.56 | 0.54 | −0.098 | 428% |
| Model C | 0.70 | 0.65 | 0.61 | 0.56 | 0.51 | −0.192 | 776% |
| Model D | 0.59 | 0.54 | 0.49 | 0.44 | 0.39 | −0.202 | 827% |
| Momentum 12-1 | 0.76 | 0.74 | 0.73 | 0.71 | 0.69 | −0.066 | 265% |
| Equal weight | 0.64 | 0.64 | 0.63 | 0.63 | 0.63 | −0.006 | 23% |
| SPY buy & hold | 0.77 | 0.77 | 0.77 | 0.77 | 0.77 | −0.002 | 0% |

Cost damage tracks turnover exactly. The XGBoost models turn over ~800%/yr — 3×
momentum, 30× equal-weight — so they surrender 0.19–0.20 of Sharpe across the
cost range while the benchmarks lose almost nothing. Any conclusion favouring
these models would be highly sensitive to the cost assumption.

### (e) Does any model beat a benchmark anywhere in the grid?

| Model | Cells | Beats EW | Beats SPY | Beats momentum | Beats all three | Sharpe range |
| --- | --- | --- | --- | --- | --- | --- |
| A | 20 | 0 | 0 | 0 | 0 | 0.40–0.63 |
| B | 20 | 0 | 0 | 0 | 0 | 0.27–0.64 |
| C | 20 | **4** | 0 | 0 | 0 | 0.39–0.70 |
| D | 20 | 0 | 0 | 0 | 0 | 0.14–0.59 |

- **No model beats SPY in any of the 80 cells.**
- **No model beats 12-1 momentum in any cell.**
- **No model beats all three benchmarks simultaneously anywhere.**

Model C's four wins over equal-weight are the only positives in the entire grid,
and they are exactly the fragility pattern this test was built to expose: they
sit at **N=3 and N=5 with costs of 0 and 5 bps**. At 10 bps or above it loses in
every cell. A win that requires assuming away transaction costs is a corner of
the grid, not an edge.

Comparisons are made at **matched parameters** — a model at (N, TC) against
momentum at the same (N, TC) and against EW/SPY at the same TC. Comparing a
model's best cell to a benchmark's worst would be the ex-post selection this
file exists to avoid.

## FOMC event-window / horizon decay

```bash
python models/event_horizon.py
```

Diagnostic separate from the monthly pipeline. If FOMC information is priced
within days, a month-end rebalance would act on it too late — so this measures
cross-sectional IC over 1, 3, 5, 10, 21 and 42 trading days from each release-day
close. Statements and minutes are treated as separate events. Outputs
`models/event_horizon_results.csv` and `viz/horizon_decay.png`.

### The mapping is not 1-to-1, and here is why

The pipeline produces one score set per **month-end**, generated jointly from the
latest statement, the latest minutes, and that month's macro row. No score is
attributable to a single document.

The obvious fix — attach the containing month's score to the event — is unsound.
A score at month-end *t* uses every document published on or before *t*, so
pairing it with an event at *e* < *t* leaks anything released in (*e*, *t*]. The
March 2020 cluster is the clearest case: the 2020-03-03 statement would be paired
with a March score that also saw the 03-15 and 03-23 statements.

There is no month-end score that both incorporates a mid-month release and
excludes everything after it, because the score is only computed at month-end.
**A clean 1-to-1 mapping is impossible.**

Approach used: map event *e* to the first month-end *t* ≥ *e*, and keep the event
**only if no other FOMC document was released in (e, t]**. For those events the
score's document inputs are exactly what was public at *e* — computed later, but
using nothing newer. Verified: **0 of 120 eligible events have a mapped score
that saw a document published after the event.**

This keeps **120 of 163 releases** (61 statements, 59 minutes) and drops 43. The
dropped events are *not random* — they are meetings followed closely by another
release — so the surviving sample tilts toward quieter periods. Residual
imperfection: the prompt states the month-end date, a calendar fact rather than
market information.

### Horizon decay

**Raw sentiment**

| Days | Events | Mean IC | Median IC | t | t (NW) | 95% CI | %IC>0 | Prec@3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 120 | +0.012 | −0.041 | 0.29 | 0.29 | [−0.067, +0.088] | 49.2% | 0.481 |
| 3 | 119 | +0.022 | +0.055 | 0.56 | 0.54 | [−0.055, +0.103] | 57.1% | 0.515 |
| 5 | 119 | **+0.030** | +0.064 | 0.75 | 0.80 | [−0.051, +0.103] | 54.6% | 0.482 |
| 10 | 119 | +0.028 | +0.055 | 0.73 | 0.79 | [−0.046, +0.100] | 57.1% | 0.524 |
| 21 | 119 | −0.004 | 0.000 | −0.09 | −0.09 | [−0.080, +0.070] | 49.6% | 0.496 |
| 42 | 118 | +0.009 | −0.021 | 0.23 | 0.21 | [−0.065, +0.080] | 48.3% | 0.492 |

**Sentiment change**

| Days | Events | Mean IC | t (NW) | 95% CI | %IC>0 |
| --- | --- | --- | --- | --- | --- |
| 1 | 118 | +0.036 | 1.15 | [−0.030, +0.102] | 51.7% |
| 3 | 117 | +0.004 | 0.13 | [−0.065, +0.065] | 49.6% |
| 5 | 117 | −0.013 | −0.37 | [−0.076, +0.051] | 46.2% |
| 10 | 117 | +0.013 | 0.38 | [−0.051, +0.078] | 49.6% |
| 21 | 117 | +0.022 | 0.70 | [−0.043, +0.087] | 50.4% |
| 42 | 116 | +0.034 | 1.11 | [−0.033, +0.101] | 54.3% |

Newey-West uses lag = ⌈h/21⌉; at 42 days the naive t is unreliable because
windows span two to three subsequent events.

### Verdict: **B — near zero at every horizon**

**Not a horizon mismatch.** No CI excludes zero at any horizon for either signal
(0 of 6, both signals). The largest |mean IC| anywhere is 0.036, and the largest
NW t-statistic is 1.15.

One honest nuance: raw sentiment's *shape* is decay-like — it rises from +0.012
at 1 day to +0.030 at 5 days, then collapses to −0.004 by 21 days, which is what
hypothesis A predicts. But every interval spans zero, and the change signal shows
no such pattern (it is lowest at 5 days and highest at 42). A shape visible only
in one of two signals with all CIs covering zero is a hint at best, not evidence.

### Sample limits

120 usable events, 116–120 per horizon. With ~11 sectors per event a single
event's IC has a standard error near 0.33, so individual events carry almost no
information and the mean is doing all the work. Splitting by document type at 5
days: statements +0.048 (n=61), minutes +0.011 (n=59) — suggestive that
statements matter more than minutes, but on 61 events with these error bars it is
not a claim worth making.

## Long-short diagnostic

```bash
python backtest/long_short.py
```

**A diagnostic, not a proposed strategy.** The top-3 long-only backtest conflates
two things: whether the model picks the right sectors, and whether sectors went
up. A dollar-neutral book (long top 3, short bottom 3, 1/3 each) cancels the
common market move, leaving selection skill alone. Outputs
`backtest/long_short_results.csv` and `viz/long_short_equity.png`.

| Strategy | Ann. net | Ann. gross | Vol | Sharpe | MaxDD | Hit | Turnover |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Model A | −3.60% | −2.29% | 11.7% | −0.26 | −44.3% | 48.4% | 674% |
| Model B | −1.07% | +0.51% | 11.3% | −0.04 | −46.3% | 46.2% | 794% |
| Model C | **+0.40%** | +3.42% | 12.5% | 0.09 | −33.1% | 49.5% | 1486% |
| Model D | −6.39% | −3.42% | 13.1% | −0.44 | −46.2% | 45.2% | 1560% |
| LLM sentiment | −0.91% | +0.43% | 13.9% | 0.00 | −34.6% | 47.3% | 669% |
| Momentum 12-1 | −2.69% | −1.70% | 14.9% | −0.11 | −33.0% | 53.8% | 507% |

Sharpe is shown without a risk-free subtraction, since the book is
self-financing; the rf-adjusted figures are all worse (−0.11 to −0.63).

### The spread confirms the null

| Strategy | Mean monthly | t | t (NW) | 95% CI | Significant? |
| --- | --- | --- | --- | --- | --- |
| Model A | −0.249% | −0.71 | −0.65 | [−0.925%, +0.411%] | no |
| Model B | −0.036% | −0.11 | −0.09 | [−0.727%, +0.629%] | no |
| Model C | +0.098% | 0.26 | 0.28 | [−0.613%, +0.831%] | no |
| Model D | −0.478% | −1.22 | −1.48 | [−1.231%, +0.292%] | no |
| LLM sentiment | +0.003% | 0.01 | 0.01 | [−0.790%, +0.792%] | no |
| Momentum 12-1 | −0.135% | −0.30 | −0.33 | [−0.986%, +0.724%] | no |

**No spread is distinguishable from zero.** Removing market direction does not
reveal hidden selection skill — so the long-only shortfall was *not* a beta
story. There is simply no cross-sectional edge to extract. The raw LLM signal is
the starkest case: +0.003% per month, a mean of essentially exactly zero.

Costs matter more here because the second leg roughly doubles traded notional
(Model C: 776% long-only → 1486% long-short). Model C's only positive net figure,
+0.40%/yr, is what remains of +3.42% gross after a 2.97%/yr cost drag — and its
CI spans zero either way.

### An incidental result worth reporting

**Momentum's long-short spread is negative (−2.69%/yr net) despite its long-only
success (13.7% CAGR).** Its hit rate is the *highest* of any strategy at 53.8%
and its median month is +0.47%, but the mean is −0.05% — a handful of violent
reversals do the damage:

| Month | Spread |
| --- | --- |
| 2022-12 | −13.1% |
| 2018-09 | −10.6% |
| 2021-01 | −9.3% |
| 2020-03 | −8.0% |

This is the textbook momentum-crash profile: right most months, catastrophically
wrong in sharp reversals. It also means momentum's long-only advantage came from
holding the *level* of strong sectors rather than from top-versus-bottom
discrimination — worth stating, since momentum is the benchmark the models most
need to beat.

### Not modelled

Borrow fees (small for large SPDRs but not zero, and spiking exactly when a
sector is stressed), financing on short proceeds, recall risk, and short-sale
constraints in fast markets. Short rebates would offset part of that and are also
ignored. These spreads are an **upper bound** on what the selection signal is
worth, not an achievable return.

## IC statistical significance

```bash
python models/ic_significance.py
python models/ic_significance.py --n-perm 50000
```

The project's headline claim is that no signal has an IC distinguishable from
zero. Everything before this leaned on a conventional t-statistic and a naive
bootstrap. This tests it properly. Outputs `models/ic_significance.csv` and
`viz/permutation_null.png`.

| Signal | Months | Mean IC | IC IR | %>0 | t | t (NW) | Bootstrap 95% CI | Perm. p | BH p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Model A | 93 | +0.011 | +0.030 | 45.2% | 0.29 | 0.24 | [−0.059, +0.083] | 0.737 | 0.943 |
| Model B | 93 | +0.007 | +0.020 | 48.4% | 0.19 | 0.16 | [−0.061, +0.073] | 0.848 | 0.943 |
| Model C | 93 | **+0.046** | +0.124 | 53.8% | 1.19 | 1.13 | [−0.028, +0.121] | **0.165** | 0.583 |
| Model D | 93 | −0.024 | −0.063 | 46.2% | −0.61 | −0.66 | [−0.098, +0.052] | 0.472 | 0.943 |
| LLM sentiment | 93 | −0.002 | −0.006 | 48.4% | −0.05 | −0.07 | [−0.086, +0.082] | 0.943 | 0.943 |
| LLM sentiment change | 92 | **+0.043** | +0.127 | 56.5% | 1.22 | 1.29 | [−0.026, +0.113] | **0.194** | 0.583 |

**0 of 6 signals reach raw p < 0.05. 0 of 6 survive Benjamini-Hochberg at FDR 5%.**
The claim is confirmed under every test.

### Method notes

**Newey-West** uses the standard ⌊4·(T/100)^(2/9)⌋ = **3 lags** at T = 93. It
barely moves any t-statistic (largest change 0.07), which is expected: monthly
ICs are non-overlapping and only weakly autocorrelated.

**Bootstrap scheme chosen from the data, not assumed.** Lag-1 IC autocorrelation
is compared against the white-noise band (±0.203); all six fall inside, so the
iid bootstrap is used. Models A and B are borderline at +0.187, so I re-ran both
with a moving-block bootstrap (block 4): CIs widen to [−0.073, +0.098] and
[−0.071, +0.093] and still contain zero. The borderline call changes nothing.

**The permutation test is the primary evidence.** Within each month it shuffles
the *sector labels* of the signal, leaving returns untouched — so the null is
exactly "no cross-sectional information", and the number of sectors, the
distribution of signal values, the return cross-section, and the tie structure
are all preserved by construction. No distributional assumption enters. 10,000
permutations, two-tailed, with the +1 correction so a p-value is never exactly
zero.

I validated the machinery before trusting it: the fast rank-based IC matches
`scipy.spearmanr` on 500 random months **including ties** (0 mismatches), a
signal set equal to the returns returns IC = 1.0 at minimum p, and pure-noise
inputs produce approximately uniform p-values.

### The three methods agree, which is itself the point

The permutation null's standard deviation (0.033) sits within 10–20% of the
conventional standard error (0.035–0.043) for every signal. The t-test,
bootstrap, and permutation test would only diverge if a distributional assumption
were doing the work — they don't, so none is.

Model C is the strongest signal in the study and lands at the **92nd percentile
of its own null**: shuffling the sector labels produces a mean IC that large or
larger about 8% of the time in one direction, 16.5% two-tailed. `sentiment_change`
sits at the 90th. Both are the kind of result a null model generates routinely,
and neither survives correcting for having looked at six signals.

## Factor-adjusted alpha

```bash
python models/factor_alpha.py
python models/factor_alpha.py --refresh-factors
```

Losing to SPY is not the same as having no alpha — a low-beta strategy can trail
the market and still add value, and a high-beta one can beat it while adding
none. This prices out market, size, value and momentum exposure and asks what is
left. Factors come from Ken French's library, cached to `data/raw/ff_factors.csv`;
results in `models/factor_alpha.csv`.

### Long-only (excess of RF)

| Strategy | CAPM α/yr | t | t_NW | Carhart α/yr | t | t_NW | β_MKT | β_MOM | adj R² |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Model A | −2.24% | −1.17 | −1.29 | −3.21% | −1.73 | −1.96 | 1.02 | 0.05 | 0.917 |
| Model B | −1.88% | −1.00 | −1.09 | −2.58% | −1.39 | −1.64 | 1.02 | 0.03 | 0.919 |
| Model C | −0.48% | −0.17 | −0.16 | −1.19% | −0.45 | −0.46 | 0.87 | 0.05 | 0.806 |
| Model D | −2.66% | −1.04 | −1.13 | −2.74% | −1.07 | −1.28 | 0.86 | −0.04 | 0.819 |
| Equal weight | −0.99% | −0.60 | −0.53 | −1.17% | −0.95 | −1.12 | 0.91 | −0.04 | 0.958 |
| Momentum 12-1 | +1.76% | 0.56 | 0.57 | +0.07% | 0.03 | 0.03 | 0.90 | **0.29** | 0.812 |
| SPY buy & hold | +0.79% | 1.28 | 1.31 | +0.07% | 0.17 | 0.18 | 0.98 | −0.01 | 0.996 |

### Long-short (raw spread — self-financing, so no RF subtraction)

| Strategy | CAPM α/yr | t_NW | Carhart α/yr | t_NW | β_MKT | β_MOM | adj R² |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Model A (L/S) | −4.16% | −0.96 | −6.02% | −1.94 | 0.17 | 0.17 | 0.287 |
| Model B (L/S) | −1.53% | −0.32 | −3.37% | −1.08 | 0.15 | 0.11 | 0.307 |
| Model C (L/S) | +4.56% | 1.16 | +2.49% | 0.63 | −0.16 | 0.23 | 0.175 |
| Model D (L/S) | −3.38% | −0.91 | −4.33% | −1.25 | −0.14 | 0.13 | 0.130 |
| LLM sentiment (L/S) | +4.19% | 0.92 | +0.69% | 0.17 | −0.15 | **0.45** | 0.334 |
| Momentum 12-1 (L/S) | +1.52% | 0.30 | −2.08% | −0.59 | −0.02 | **0.80** | 0.486 |

### Verdict: no alpha anywhere

**Not one of the 26 regressions produces an alpha distinguishable from zero** at
5% on Newey-West standard errors. Every strategy's return is accounted for by
market, size, value and momentum exposure.

The two closest calls are worth naming precisely, because both cut the wrong way
for the models: Model A's Carhart alpha is **−3.21%/yr (t_NW −1.955, p 0.054)**
and its long-short version **−6.02%/yr (t_NW −1.942, p 0.055)**. The nearest
thing to a significant result in the whole table is *negative* alpha.

### Two things the factor loadings reveal

**Momentum's edge is the momentum factor.** Its long-only CAPM alpha of +1.76%
collapses to **+0.07%** under Carhart, with β_MOM = 0.29 (t = 5.62). The
benchmark that beat every model was not skill — it was a well-known style
premium, available at low cost through any momentum fund.

**The LLM long-short is a partial momentum clone.** β_MOM = **0.45 (t = 4.82)**,
and its CAPM alpha of +4.19% shrinks to **+0.69%** once momentum is priced in —
84% of the raw spread was factor exposure. This is the redundancy story appearing
in a third guise: having already shown the LLM's tone restates the macro block
(R² 0.78), its sector rankings turn out to partly restate price momentum too.
Whatever the model reads out of FOMC text, it is largely information the market
already encodes.

### Validation

The setup is verified against a known answer: SPY's Carhart regression returns
β_MKT = 0.979 with alpha +0.07%/yr (t = 0.18) and adj R² = 0.996 — textbook. The
alignment check reports corr(SPY monthly return, FF market return) = **0.9949**;
a one-month offset would collapse that toward zero. FF's RF averages 2.52%/yr
over the window against FEDFUNDS' 2.55%, so the two risk-free sources agree.

The OLS and HAC code was checked against `scipy.linregress` (coefficients, SEs,
R² identical), against `lstsq` for multiple regression, and the Bartlett kernel
at lag 0 reproduces White standard errors exactly.

**Risk-free choice:** these regressions use FF's own RF column, not FEDFUNDS.
MKT is constructed as (market return − FF's RF), so using a different rate on the
left-hand side leaves a wedge that lands in the intercept and gets read as alpha.
Sharpe ratios elsewhere in this project use FEDFUNDS; that is a separate
calculation and does not enter here.

## PCA macro robustness

```bash
python models/pca_robustness.py
python models/pca_robustness.py --skip-xgboost      # logistic arms only
python models/pca_robustness.py --variance 0.95     # different pre-set rule
```

Outputs `models/pca_robustness.csv` and `viz/pca_variance.png`.

The null result has a plausible boring explanation. The 14 macro features are
correlated with each other — VIX with credit spreads, the term spread with the
funds rate, industrial production with payrolls. A model handed a collinear,
noisy block can spend its capacity unstably and drown out a small independent
signal sitting beside it. If that were happening, replacing the raw macro block
with orthogonal principal components should give the LLM room to contribute.

This tests that and nothing else. **The raw specification is not modified** —
Models A–D are read back from `models/predictions.csv` exactly as `train.py`
produced them, so the raw arm is the real headline specification rather than a
re-fit of it.

### What is compressed

PCA is applied to the **14 lagged FRED macro features only**. Left out by design:
the LLM sentiment block (compressing it with macro would destroy the separation
the experiment is about), sector momentum/volatility (sector-level, not macro),
cross-sectional ranks (already normalised within month), sector dummies
(indicators, not continuous macro), and interaction terms (see below).

Scaler and rotation are fitted on training months only, at every step, then
applied unchanged to the prediction month — under the same one-month embargo
used everywhere else. Because macro is a time series broadcast across the 11
sectors, both are fitted on the **deduplicated monthly** matrix; fitting on panel
rows would count every month eleven times and quietly down-weight the pre-2018
months that carry only ten sectors.

**The interaction problem.** The raw spec carries eight `macro × sector`
interaction terms, which are products of the *raw* macro columns. Carrying them
into a "PCA" spec would smuggle the raw block back in and defeat the test;
dropping them confounds the comparison the other way, since the PCA arm would
lose the ability to let sectors respond differently to macro. Both were
therefore specified in advance and both are run: `pca` (PC scores + PC1 × each
sector dummy, for structural parity) and `pca_noint` (PC scores only). Neither
was chosen on results.

### Component selection — pre-specified, never on out-of-sample performance

Smallest *k* reaching 90% cumulative explained variance **on the training
window**, re-applied as the window expands:

| | value |
|---|---|
| components retained | mean 5.62, min 5, max 7 (of 14) |
| distribution | k=5 in 37 months, k=6 in 54, k=7 in 2 |
| variance captured at k | mean 0.9166 (min 0.9000, max 0.9382) |
| PC1 alone | 41.7% |
| PC1 + PC2 | 61.0% |

It takes **6 of 14 components to reach 90%**, and that is itself informative:
the macro block is correlated, but nowhere near rank-1. There was never a single
"macro factor" for the LLM to be competing against.

### Results

| spec | AUC | 95% CI | mean IC | IC IR | P@3 | NDCG@3 | Sharpe | CAGR |
|---|---|---|---|---|---|---|---|---|
| Raw macro (= Model A) | 0.4827 | [0.4486, 0.5199] | 0.0106 | 0.030 | 0.477 | 0.543 | 0.56 | 11.47% |
| Raw macro + LLM (= Model B) | 0.4933 | [0.4599, 0.5273] | 0.0066 | 0.020 | 0.480 | 0.537 | 0.59 | 12.00% |
| PCA macro | 0.4916 | [0.4553, 0.5249] | −0.0181 | −0.052 | 0.444 | 0.522 | 0.41 | 8.56% |
| PCA macro + LLM | 0.4882 | [0.4526, 0.5218] | −0.0261 | −0.074 | 0.473 | 0.528 | 0.38 | 8.07% |
| PCA, no interactions | 0.4951 | [0.4595, 0.5319] | 0.0233 | 0.065 | 0.487 | 0.541 | 0.61 | 12.72% |
| PCA + LLM, no interactions | 0.4918 | [0.4565, 0.5255] | −0.0068 | −0.020 | 0.477 | 0.531 | 0.52 | 10.54% |
| Raw macro (= Model C, xgb) | 0.5081 | [0.4749, 0.5414] | 0.0462 | 0.124 | 0.498 | 0.557 | 0.61 | 11.67% |
| Raw macro + LLM (= Model D, xgb) | 0.4857 | [0.4499, 0.5200] | −0.0236 | −0.063 | 0.487 | 0.531 | 0.49 | 9.71% |
| PCA macro (xgb) | 0.4578 | [0.4187, 0.4975] | −0.0091 | −0.026 | 0.470 | 0.540 | 0.56 | 11.00% |
| PCA macro + LLM (xgb) | 0.4803 | [0.4394, 0.5182] | −0.0051 | −0.014 | 0.473 | 0.544 | 0.51 | 9.55% |

Every AUC interval contains 0.50. PCA does not rescue the base model either —
compression neither helps nor meaningfully hurts, which is what you expect when
there is no signal to represent more cleanly.

### Does the LLM add value once macro is orthogonal?

| arm | ΔAUC | 95% CI | ΔIC | 95% CI | ΔSharpe | ΔCAGR |
|---|---|---|---|---|---|---|
| Logistic / raw macro | +0.0106 | [−0.0077, +0.0283] | −0.0039 | [−0.0354, +0.0284] | +0.02 | +0.53% |
| Logistic / PCA macro | −0.0034 | [−0.0161, +0.0097] | −0.0080 | [−0.0394, +0.0228] | −0.03 | −0.50% |
| Logistic / PCA, no interactions | −0.0033 | [−0.0205, +0.0132] | −0.0301 | [−0.0688, +0.0061] | −0.09 | −2.18% |
| XGBoost / raw macro | −0.0224 | [−0.0556, +0.0103] | −0.0698 | [−0.1466, +0.0062] | −0.12 | −1.96% |
| XGBoost / PCA macro | +0.0225 | [−0.0134, +0.0579] | +0.0040 | [−0.0645, +0.0671] | −0.06 | −1.45% |

Positive means the LLM version is better. CIs are paired, month-clustered
bootstraps: months are resampled whole, because sectors within a month share a
macro state and their targets are defined against each other.

**Every interval contains zero.** The largest LLM increment anywhere under PCA
is +0.0225 AUC (XGBoost), with a CI running from −0.013 to +0.058 — the same
kind of nothing seen in the raw specification. The two PCA variants agree, so
the answer does not hinge on the interaction design choice.

### Verdict: this explanation is closed off

Multicollinearity in the raw macro block would only explain the null if
orthogonalising the block changed the answer. It does not. The LLM contributes
nothing incremental whether macro enters as 14 correlated raw features or as
~6 orthogonal components, under a linear model or a tree ensemble, with or
without sector interactions — five arms, five intervals spanning zero.

Taken with the earlier diagnostics — the macro block already explains 78.5% of
the LLM's monthly tone, the sector-specific residual correlates −0.002 with
excess returns, and the LLM long-short loads 0.45 on the momentum factor — the
representation of the macro features was never the binding constraint.

### One caveat on interpretation

The logistic pipeline standardises its inputs, so the PC scores are re-scaled to
unit variance before the fit. The specification is therefore *whitened* macro,
not merely rotated macro: PC6 is penalised on the same footing as PC1 despite
carrying a fraction of the variance. That is the identical treatment the raw
columns receive, which is why it was kept, but it means low-variance components
are not automatically down-weighted. Component signs are stabilised (each
component flipped so its largest-magnitude loading is positive) so the PC1
interaction terms stay coherent across monthly refits.

### Validation

Explained-variance ratios match a direct `sklearn` PCA exactly and sum to 1.0.
The retained components are orthogonal on the training window (max off-diagonal
correlation 1.1e-15). The selection rule is checked to be tight — cumulative
variance at *k−1* is below the threshold and at *k* is above it. A training-only
fit is confirmed to differ from a full-sample fit in both means and loadings,
and the prediction month's projection is reproduced by hand from the
training-fitted scaler and rotation. The reused raw arm reproduces the published
Model A–D numbers exactly: AUC 0.4827 / 0.4933 / 0.5081 / 0.4857, Sharpe 0.5635 /
0.5871 / 0.6069 / 0.4914, and Model A's mean IC 0.0106, IC IR 0.030, P@3 0.4767,
NDCG@3 0.5430.

## Probability calibration

`models/calibration.py` → `models/calibration_results.csv`, `viz/calibration_curves.png`

AUC and IC measure *ranking*. Calibration is a separate quality: when a model
says 0.70, does the sector beat the benchmark 70% of the time? A model can rank
at chance and still be well calibrated, or rank well and be badly scaled. This
asks whether the LLM feature helps on the dimension the other tests do not see.

Every headline number below is the **raw** model output. No Platt scaling, no
isotonic regression — fitting a calibrator on the evaluation data would improve
every metric by construction. 1,007 predictions over 93 months, base rate 0.490.

### Raw calibration

| model | spec | Brier | BSS vs 0.5 | log loss | ECE | bins used |
|---|---|---|---|---|---|---|
| A | Logistic, macro | 0.26145 | **−0.0458** | 0.72195 | 0.0705 | 10/10 |
| B | Logistic, +LLM | 0.26596 | **−0.0638** | 0.73604 | 0.0757 | 10/10 |
| C | XGBoost, macro | 0.28884 | **−0.1554** | 0.80556 | 0.1616 | 10/10 |
| D | XGBoost, +LLM | 0.29816 | **−0.1926** | 0.82709 | 0.1932 | 10/10 |

Baselines: a constant 0.5 scores Brier 0.25000; the base rate scores 0.24989 with
log loss 0.69293. **Every Brier skill score is negative** — all four models are
worse than answering "0.5" for every sector-month, and all four lose to the base
rate on log loss too. The ordering is consistent: adding the LLM feature hurts in
both pairs, and XGBoost is much worse than logistic regression, which is what
unregularised probability output on a near-random target looks like.

### Murphy decomposition — Brier = reliability − resolution + uncertainty

| model | reliability ↓ | resolution ↑ | uncertainty | sum | residual |
|---|---|---|---|---|---|
| A | 0.01376 | 0.00265 | 0.24989 | 0.26100 | 0.00045 |
| B | 0.01631 | 0.00133 | 0.24989 | 0.26486 | 0.00109 |
| C | 0.04016 | 0.00170 | 0.24989 | 0.28835 | 0.00049 |
| D | 0.05151 | 0.00339 | 0.24989 | 0.29801 | 0.00015 |

This separates the two failures. **Resolution is near zero for every model** —
at most 0.00339 against an uncertainty of 0.24989, meaning the bins barely
separate outcomes at all. That is the AUC null restated in calibration language.
The rest of the damage is **reliability**, which is a scale problem: the models
are confident in the wrong places, worst for D. The residual is the binning loss
— the identity is exact only for the binned Brier score, so it is shown rather
than absorbed into the components.

### Does the LLM feature improve the probabilities?

| pair | ΔBrier | 95% CI (month-clustered bootstrap) | verdict |
|---|---|---|---|
| B − A | +0.004502 | [+0.000626, +0.009201] | **significantly worse** |
| D − C | +0.009323 | [−0.004100, +0.022826] | no detectable difference |

Negative would mean the LLM version is better calibrated. Months are resampled
whole, because sectors within a month share a macro state and their targets are
defined against each other.

**B − A is the first statistically significant LLM effect anywhere in this
project, and it points the wrong way**: in the linear pair, adding sentiment
makes the probabilities measurably worse. Two comparisons were run here, so this
is one significant result out of two, and it should be read against the null AUC,
IC, and alpha results rather than as a standalone discovery. The honest reading
is not "the LLM feature is harmful" but "there is no evidence it helps, and the
one significant estimate is a degradation."

### Probability spread — how much these metrics mean

| model | min | p25 | median | p75 | max | std | within ±0.05 of 0.5 |
|---|---|---|---|---|---|---|---|
| A | 0.057 | 0.470 | 0.501 | 0.531 | 0.935 | 0.0926 | **66.8%** |
| B | 0.056 | 0.466 | 0.501 | 0.531 | 0.995 | 0.1158 | **64.0%** |
| C | 0.010 | 0.346 | 0.482 | 0.628 | 0.993 | 0.2009 | 19.6% |
| D | 0.021 | 0.336 | 0.485 | 0.630 | 0.990 | 0.2110 | 21.5% |

This cuts differently for the two model families, and it matters:

- **A and B are narrow-band.** Two-thirds of their predictions sit within ±0.05
  of 0.5. All ten bins are technically populated, but the outer ones hold 1, 2,
  7, and 12 observations — the wild excursions at the ends of Model A's curve in
  the plot are small-*n* artifacts, not evidence about behaviour at extreme
  probabilities. ECE for A and B is dominated by the two central bins holding
  ~85% of the mass. Read their calibration numbers as "well-behaved because they
  barely commit," not as a strength.
- **C and D genuinely spread out.** Bin counts run 21–224 across the full range,
  so their much larger ECE and reliability terms are real: XGBoost states
  confident probabilities on both tails and is wrong at the stated rate.

That asymmetry is itself informative. The logistic models look better calibrated
mostly by declining to say anything, and even so they cannot beat a constant.

### Secondary — what recalibration could repair

Clearly labelled as secondary, and fitted prequentially: the calibrator for month
*t* uses only walk-forward predictions from months **before** *t*, so no month
informs its own calibration. The first 24 months are consumed as the initial
fitting window, leaving n = 759.

| model | Brier raw | Brier calibrated | ECE raw | ECE calibrated |
|---|---|---|---|---|
| A | 0.25248 | 0.25018 | 0.0388 | 0.0103 |
| B | 0.25203 | 0.25070 | 0.0433 | 0.0083 |
| C | 0.27935 | **0.25039** | 0.1386 | 0.0276 |
| D | 0.29369 | **0.25036** | 0.1800 | 0.0100 |

Recalibration works: D's ECE falls from 0.180 to 0.010. But look where every
model lands — Brier ≈ 0.2504 for all four, against the 0.24989 base rate. Platt
scaling removes the miscalibration penalty by shrinking everything toward the
base rate, which is the correct response to forecasts with no resolution. It
repairs the scale; it cannot manufacture ranking skill, and the fact that all
four converge to the same point confirms there was none to preserve.

### Validation

Brier and log loss match `sklearn.metrics` exactly. `bin_table` matches
`sklearn.calibration.calibration_curve` (uniform strategy) exactly on both mean
predicted probability and observed frequency. The Murphy identity holds exactly
on the binned Brier score. A perfect forecaster scores 0; a climatology forecast
scores p(1−p).

## Look-ahead bias

Each macro feature is shifted forward by its own lag, set in the `FRED_SERIES`
table at the top of `collect.py`:

- **1 month** for series observable in real time — daily market prices and rates.
- **2 months** for government and survey releases, which are not published until
  part-way through the following month.

`audit_lags()` re-derives every shift on each run and aborts the collection if
one did not take, so the no-look-ahead claim is checked rather than assumed. The
warm-up window is computed from the config (12 months for the YoY transforms
plus the largest lag), so adding a longer-lagged series cannot silently truncate
it.

One caveat worth addressing in the writeup:

**Revisions.** `fredapi.get_series` returns the *current* vintage, not the number
as first printed. INDPRO, UNRATE, PERMIT, and MANEMP are all revised materially,
so the lag protects against look-ahead in *time* but not against seeing a
revised figure the market had not yet been given. For a fully point-in-time
dataset use ALFRED vintages (`fred.get_series_first_release` /
`get_series_as_of_date`). This is the main remaining gap in the point-in-time
treatment.

## Known data gap: October 2025

October 2025 unemployment and CPI were never published — the government shutdown
that autumn cancelled the BLS household survey and the October CPI release
outright. It is a permanent hole in the source, not a late print. `PERMIT` and
`MANEMP` were delayed but did publish, so they are unaffected.

`fill_release_gaps()` carries the September value forward across the missing
month, which matches what an allocator actually faced: no fresh data, so act on
the last print. Under the 2-month lag this feeds the **2025-12-31** row. The fill
is capped at one month (`limit=1`) so a longer outage cannot be absorbed
silently, and every filled cell is logged at WARNING level on each run:

```
WARNING Gap fill: UNRATE at 2025-10-31 was unpublished; carried forward 4.4000
WARNING Gap fill: CPI_YOY at 2025-10-31 was unpublished; carried forward 3.0226
```

The fill runs on the transformed frame *before* lagging, so each feature remains
a clean shift of its own source and the lag audit still holds.

Two consequences to keep in mind. The 2025-11-30 and 2025-12-31 rows now carry
the *same* `UNRATE`/`CPI_YOY` values, so those two months are not independent
observations. And because the raw October CPI *level* is still missing, a window
extended past October 2026 will produce a fresh `CPI_YOY` gap — the YoY
calculation needs that missing base. Filling `CPIAUCSL` in `macro_raw` rather
than `CPI_YOY` in the transformed frame would close that too, at the cost of
assuming a flat month of prices.

## Data coverage

XLRE (inception 2015-10) and XLC (inception 2018-06) are younger than the
2016-01 start, so `XLC` is empty until 2018-06. Handle this explicitly in the
backtest — either start the full 11-sector strategy in 2018-07 or hold the
missing sectors at zero weight and renormalise. Silently dropping the NaNs
would bias the sector universe.
