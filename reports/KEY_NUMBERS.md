# Key numbers reference card

Every figure that appears in the report text, with the file it comes from.

**Look it up here rather than re-deriving it.** A number retyped from memory is
how a report ends up quoting a value the analysis no longer produces. If a
number you need is not on this page, add it here first, then cite it.

Regenerate the tables that back this page with `python scripts/build_report_tables.py`;
`python scripts/verify_all.py` fails if any of them has drifted from its source.

---

## Dataset scale

| Quantity | Value | Source |
|---|---|---|
| Sector ETFs | 11 (SPDR sector series) | `data/collect.py` |
| Feature window | 2016-01 → 2025-11 (120 months) | `data/macro_features.csv` |
| Macro indicators | 14 FRED series | `data/macro_features.csv` |
| Publication lags | 1 month (market data), 2 months (government/survey) | `data/collect.py` |
| Sector feature columns | 44 | `data/sector_features.csv` |
| Modelling panel | 1,248 rows × 49 columns | `data/model_dataset.csv` |
| Feature count, macro only | 38 | `models/feature_names.json` |
| Feature count, macro + LLM | 43 | `models/feature_names.json` |
| FOMC meetings scraped | 83 | `data/fomc/fomc_calendar.csv` |
| FOMC statements / minutes | 83 / 80 | `data/fomc/` |
| Sentiment months scored | 120 | `data/llm_sentiment.csv` |
| Out-of-sample forecasts | 1,007 | `models/predictions.csv` |
| Prediction months | 93 (2018-03 → 2025-11) | `models/predictions.csv` |
| Realised return months | 93 (2018-04 → 2025-12) | `backtest/equity_curves.csv` |
| Target base rate | 0.4896 | `models/calibration_results.csv` |

## LLM layer

| Quantity | Value | Source |
|---|---|---|
| Model | Claude Haiku 4.5, `temperature=0` | `llm/sentiment_pipeline.py` |
| Output enforcement | Structured outputs (JSON schema) | `llm/sentiment_pipeline.py` |
| Successful calls | 120 of 120 months | `logs/llm_calls.jsonl` |
| Total API cost | $2.15 | `logs/llm_calls.jsonl` |
| Input / output tokens | 1,699,575 / 92,854 | `logs/llm_calls.jsonl` |
| Mean latency | 10.0 s per call | `logs/llm_calls.jsonl` |
| Calls needing a retry | 6 | `logs/llm_calls.jsonl` |
| JSON repairs required | 0 | `logs/llm_calls.jsonl` |

## Model performance — Table 1

| Model | AUC | 95% CI | Accuracy | Mean IC | Precision@3 |
|---|---|---|---|---|---|
| A — Logistic, macro | 0.4827 | [0.4486, 0.5199] | 0.4866 | +0.0106 | 0.4767 |
| B — Logistic, +LLM | 0.4932 | [0.4599, 0.5273] | 0.4945 | +0.0066 | 0.4767 |
| C — XGBoost, macro | 0.5081 | [0.4749, 0.5414] | 0.5005 | +0.0462 | 0.4982 |
| D — XGBoost, +LLM | 0.4857 | [0.4499, 0.5200] | 0.4777 | −0.0236 | 0.4839 |

Source: `models/metrics.csv`, `models/ranking_metrics.csv`, `models/pca_robustness.csv`.
**Every interval contains 0.50.**

## LLM effect — Table 2

| Specification | ΔAUC (B−A) | ΔAUC (D−C) | Source |
|---|---|---|---|
| Base (no sector identity) | +0.0035 | +0.0050 | `models/spec_ladder.csv` |
| + Sector identity | +0.0010 | **−0.0255** (CI excludes 0) | `models/spec_ladder.csv` |
| + Derived sentiment | +0.0106 | −0.0224 | `models/spec_ladder.csv` |
| PCA macro | −0.0034 | +0.0225 | `models/spec_ladder.csv` |

Sign sequence: logistic `+ + + −`, XGBoost `+ − − +`. **Both ladders change sign.**
One of eight cells has an interval excluding zero, in the direction of harm.

| Other LLM-effect figures | Value | Source |
|---|---|---|
| ΔBrier (B−A) | +0.004502, CI [+0.000626, +0.009201] — **worse, significant** | `models/calibration_results.csv` |
| ΔBrier (D−C) | +0.009323, CI [−0.004100, +0.022826] | `models/calibration_results.csv` |
| LLM sentiment IC | −0.0023, permutation p = 0.943 | `models/ic_significance.csv` |
| LLM sentiment change IC | +0.0434, permutation p = 0.194 | `models/ic_significance.csv` |
| Q5−Q1 spread, LLM level | +3.27%/yr (t = 0.60, p = 0.55) | `models/llm_standalone_results.csv` |

## Portfolio — Table 3

| Strategy | CAGR | Sharpe | Max DD | Turnover |
|---|---|---|---|---|
| Model A | 11.47% | 0.56 | −26.68% | 355% |
| Model B | 12.00% | 0.59 | −26.19% | 428% |
| Model C | 11.67% | 0.61 | −20.78% | 776% |
| Model D | 9.71% | 0.49 | −22.53% | 827% |
| Equal weight | 12.12% | 0.63 | −22.65% | 23% |
| SPY buy & hold | **14.84%** | **0.77** | −23.93% | 0% |
| Momentum 12-1 | 13.72% | 0.73 | −15.70% | 265% |

Source: `backtest/performance.csv`. Top-3 equal-weight, 10 bps, monthly.
**No model beats either benchmark.**

## Redundancy — Table 5

| Component | R² on macro block | Adj. R² | Source |
|---|---|---|---|
| Monthly mean tone | **0.7849** | 0.7562 | `models/sentiment_redundancy.csv` |
| Monthly dispersion | 0.6479 | 0.6010 | `models/sentiment_redundancy.csv` |
| Pooled score | 0.2656 | 0.2573 | `models/sentiment_redundancy.csv` |
| Relative (sector-specific) | **0.0009** | −0.0104 | `models/sentiment_redundancy.csv` |

The headline sentence: *the macro block already explains 78.5% of the LLM's
month-to-month tone, and the sector-specific residual is orthogonal to macro by
construction.*

## Significance — Table 4

| Quantity | Value | Source |
|---|---|---|
| Signals tested | 6 | `models/ic_significance.csv` |
| Signals with raw p < 0.05 | **0** | `models/ic_significance.csv` |
| Signals surviving BH-FDR at 5% | **0** | `models/ic_significance.csv` |
| Newey-West lags at T = 93 | 3 | `models/ic_significance.csv` |
| Strongest signal | Model C, IC +0.0462, permutation p = 0.165 | `models/ic_significance.csv` |
| Permutation draws | 10,000 | `models/ic_significance.py` |

## Factor adjustment — Table 7

| Quantity | Value | Source |
|---|---|---|
| Alphas tested | 26 | `models/factor_alpha.csv` |
| Significant at 5% (Newey-West) | **0** | `models/factor_alpha.csv` |
| Momentum long-only, CAPM α | +1.76%/yr (t = 0.57) | `models/factor_alpha.csv` |
| Momentum long-only, Carhart α | **+0.07%/yr** (t = 0.03) | `models/factor_alpha.csv` |
| LLM long-short, CAPM α | +4.19%/yr (t = 0.92) | `models/factor_alpha.csv` |
| LLM long-short, Carhart α | +0.69%/yr (t = 0.17) | `models/factor_alpha.csv` |
| LLM long-short β_MOM | **0.45** (t = 4.82) | `models/factor_alpha.csv` |
| Closest to significance | Model A Carhart −3.21%/yr (t = −1.96) | `models/factor_alpha.csv` |
| SPY validation: β_MKT, adj R² | 0.979, 0.996 | `models/factor_alpha.csv` |

## Robustness — Table 8

| Quantity | Value | Source |
|---|---|---|
| Timing assertions passing | 5 of 5 | `models/test_no_lookahead.py` |
| Common-universe AUC CIs excluding 0.50 | 0 of 12 | `models/robustness_common_universe.csv` |
| Event horizons tested | 6 (1–42 trading days) × 2 signals | `models/event_horizon_results.csv` |
| Eligible events | 120 of 163 | `models/event_horizon_results.csv` |
| Best horizon IC | +0.0364 at 1 day, t(NW) = 1.15 | `models/event_horizon_results.csv` |
| Portfolio grid cells | 140 (4 top-N × 5 costs × 7 strategies) | `backtest/robustness_portfolio.csv` |
| Model cells beating equal weight | **4 of 80**, all Model C at 0–5 bps | `backtest/robustness_portfolio.csv` |
| PCs retained (90% variance rule) | mean 5.62, min 5, max 7, of 14 | `models/pca_robustness.csv` |
| PC1 / PC1+PC2 variance | 41.7% / 61.0% | `models/pca_robustness.csv` |

## Calibration

| Model | Brier | Skill vs constant 0.5 | ECE | Source |
|---|---|---|---|---|
| A | 0.2615 | −0.0458 | 0.0705 | `models/calibration_results.csv` |
| B | 0.2660 | −0.0638 | 0.0757 | `models/calibration_results.csv` |
| C | 0.2888 | −0.1554 | 0.1616 | `models/calibration_results.csv` |
| D | 0.2982 | −0.1926 | 0.1932 | `models/calibration_results.csv` |

Baselines: constant 0.5 → Brier 0.2500; base rate → 0.2499.
**All four models are worse than answering 0.5 every time.**

---

## Numbers that are easy to get wrong

- **93 prediction months, not 94.** The one-month embargo drops the first
  candidate month. Development-era text used 94; that window is superseded.
- **1,007 forecasts, not 1,027.** Same cause.
- **The equity curve starts 2018-04-30**, one month after the first formation
  month (2018-03-31), because it is indexed by the month the return is realised.
- **Sharpe uses FEDFUNDS; the factor regressions use Fama-French's own RF.**
  These are deliberately different and should not be reconciled.
- **Turnover is one-way and annualised.** The simulator's internal figure counts
  both sides and is halved before reporting.
- **Table 2's rungs are recomputed**, not lifted from development logs. Earlier
  sector-identity figures were measured on a 94-month window and are not
  comparable to anything else in this report.
