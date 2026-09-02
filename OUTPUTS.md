# Output inventory

Every file the pipeline generates: what makes it, what is in it, and how big it
is. Shapes are as currently on disk.

**Report** marks a file the capstone report draws numbers or figures from.
**Intermediate** marks a file that exists so a later stage can read it; correct,
but not something the report quotes.

Nothing here is committed — `.gitignore` excludes all of it. `python run_all.py`
rebuilds everything; see [README](README.md#running-the-pipeline).

---

## Data

| Path | Produced by | Contents | Shape | Role |
|---|---|---|---|---|
| `data/raw/macro_raw.csv` | `data/collect.py` | 14 FRED series at month-end, **untransformed and unlagged**. Kept so the lag audit can rebuild the features from source. | 136 × 15 | Intermediate |
| `data/raw/sector_prices_raw.csv` | `data/collect.py` | Month-end adjusted closes, 11 sector ETFs, before the analysis window is trimmed | 136 × 12 | Intermediate |
| `data/raw/sector_prices_daily.csv` | `data/collect.py` | Daily adjusted closes. Feeds realised volatility and the FOMC event-window study. | 2,809 × 12 | Intermediate |
| `data/raw/spy_prices.csv` | `backtest/run_backtest.py` | Cached SPY monthly series for the buy-and-hold benchmark | 101 × 2 | Intermediate |
| `data/raw/ff_factors.csv` | `models/factor_alpha.py` | Fama-French 3 factors + momentum + RF, monthly, from the French library | 1,194 × 6 | Intermediate |
| `data/macro_features.csv` | `data/collect.py` | The 14 macro features, transformed and **lagged by 1 or 2 months** per series. Column names carry the lag (`VIX_lag1`, `CPI_YOY_lag2`). | 120 × 15 | Report |
| `data/sector_prices.csv` | `data/collect.py` | Month-end prices over the analysis window | 120 × 12 | Intermediate |
| `data/sector_features.csv` | `data/build_features.py` | Per-ETF 1/3/12-month returns and 21-day realised volatility, all lagged 1 month. XLC is NaN before 2018-06. | 120 × 45 | Report |
| `data/fomc/fomc_calendar.csv` | `llm/fomc_scraper.py` | One row per FOMC meeting: dates, statement and minutes publication dates, file paths | 83 × 7 | Report |
| `data/fomc/statements/*.txt` | `llm/fomc_scraper.py` | 83 statement texts, plain text with a provenance header | 83 files | Intermediate |
| `data/fomc/minutes/*.txt` | `llm/fomc_scraper.py` | 80 minutes texts, same format | 80 files | Intermediate |
| `data/llm_sentiment.csv` | `llm/sentiment_pipeline.py` | Monthly per-sector sentiment in [−1, 1] from Claude Haiku 4.5. Wide: one row per month, one column per ETF. | 120 × 12 | Report |
| `data/model_dataset.csv` | `data/build_dataset.py` | The modelling panel: one row per sector-month with all features, cross-sectional ranks, sector dummies, macro × sector interactions, and the binary target. **The single input to every model.** | 1,248 × 49 | Report |

## Models

| Path | Produced by | Contents | Shape | Role |
|---|---|---|---|---|
| `models/predictions.csv` | `models/train.py` | Out-of-sample probability per (date, ticker) for all four models, plus the realised target and return. **The input to almost every analysis below.** | 1,007 × 8 | Report |
| `models/metrics.csv` | `models/train.py` | Accuracy, precision, recall and AUC per model | 4 × 8 | Report |
| `models/feature_names.json` | `models/train.py` | The two feature sets and each model's spec, so SHAP explains the right columns | 3 keys | Intermediate |
| `models/final_models.joblib` | `models/train.py` | Full-sample refits, for SHAP only. Never used to produce a forecast. | 495 KB | Intermediate |
| `models/shap_values_c.csv` | `models/shap_analysis.py` | Per-row SHAP values, Model C (macro only) | 1,248 × 40 | Report |
| `models/shap_values_d.csv` | `models/shap_analysis.py` | Per-row SHAP values, Model D (macro + LLM) | 1,248 × 45 | Report |
| `models/regime_results.csv` | `models/regime_analysis.py` | AUC per model within VIX regimes, with bootstrap CIs on the A→B and C→D differences | 4 × 23 | Report |
| `models/ranking_metrics.csv` | `models/ranking_metrics.py` | IC, IC IR, precision@3, NDCG@3 per signal, plus shuffle baselines | 9 × 13 | Report |
| `models/sentiment_redundancy.csv` | `models/sentiment_diagnostics.py` | How much of the LLM signal the macro block already explains | 14 × 5 | Report |
| `models/llm_standalone_results.csv` | `models/llm_standalone.py` | The LLM signal evaluated alone: quintiles, regime-conditional IC, IC persistence, big-move conditioning | 31 × 36 | Report |
| `models/robustness_common_universe.csv` | `models/robustness_common_universe.py` | The 2×2 refit on the 10 sectors present throughout, so XLC's late listing cannot drive anything | 27 × 14 | Report |
| `models/event_horizon_results.csv` | `models/event_horizon.py` | FOMC event-window returns and IC decay by horizon | 12 × 15 | Report |
| `models/ic_significance.csv` | `models/ic_significance.py` | Newey-West t, month-clustered bootstrap CI, 10,000-draw permutation p, and BH-adjusted p per signal | 6 × 20 | Report |
| `models/factor_alpha.csv` | `models/factor_alpha.py` | CAPM and Carhart regressions for every strategy: alpha, factor loadings, HAC t-statistics | 26 × 25 | Report |
| `models/calibration_results.csv` | `models/calibration.py` | Four blocks: raw calibration (4 × 21), Brier-difference comparisons (2 × 6), calibration-curve bins (40 × 7), prequential Platt secondary (4 × 6) | 4 blocks | Report |
| `models/pca_robustness.csv` | `models/pca_robustness.py` | Three blocks: specifications (10 × 17), LLM increment with CIs (5 × 14), component count per month (93 × 6) | 3 blocks | Report |

## Backtest

| Path | Produced by | Contents | Shape | Role |
|---|---|---|---|---|
| `backtest/equity_curves.csv` | `backtest/run_backtest.py` | Growth of $1 for 4 models + 3 benchmarks, **indexed by the month the return is realised** (formation + 1) | 93 × 8 | Report |
| `backtest/performance.csv` | `backtest/run_backtest.py` | CAGR, volatility, Sharpe, Sortino, max drawdown, Calmar, turnover, hit rate per strategy | 7 × 11 | Report |
| `backtest/long_short_results.csv` | `backtest/long_short.py` | Top-3 minus bottom-3 spread per signal, with Newey-West t-statistics. A diagnostic, not a proposed strategy. | 6 × 22 | Report |
| `backtest/robustness_portfolio.csv` | `backtest/robustness_portfolio.py` | The full top-N × transaction-cost grid, reported whole rather than at a chosen corner | 140 × 13 | Report |

## Figures

All 300 DPI or better. `viz/equity_curves_report.png` is the presentation copy of
`backtest/equity_curves.png`.

| Path | Produced by | Shows | Size |
|---|---|---|---|
| `backtest/equity_curves.png` | `backtest/run_backtest.py` | Equity curves, log scale, VIX ≥ 25 shading, drawdown subplot | 450 KB |
| `viz/equity_curves_report.png` | `backtest/run_backtest.py` | The same figure at presentation resolution | 1.0 MB |
| `viz/shap_model_c.png` | `models/shap_analysis.py` | SHAP feature importance, Model C | 255 KB |
| `viz/shap_model_d.png` | `models/shap_analysis.py` | SHAP feature importance, Model D — where the LLM columns rank | 340 KB |
| `viz/regime_auc_comparison.png` | `models/regime_analysis.py` | AUC by VIX regime with bootstrap intervals | 72 KB |
| `viz/ic_timeseries.png` | `models/ranking_metrics.py` | Monthly IC and its rolling mean | 207 KB |
| `viz/sentiment_redundancy.png` | `models/sentiment_diagnostics.py` | LLM tone against the macro block's fitted values | 83 KB |
| `viz/quintile_returns.png` | `models/llm_standalone.py` | Forward excess return by LLM sentiment quintile | 79 KB |
| `viz/topn_sensitivity.png` | `backtest/robustness_portfolio.py` | Sharpe across the top-N × cost grid | 66 KB |
| `viz/horizon_decay.png` | `models/event_horizon.py` | Signal decay by holding horizon after an FOMC release | 121 KB |
| `viz/long_short_equity.png` | `backtest/long_short.py` | Long-short spread equity curves | 310 KB |
| `viz/permutation_null.png` | `models/ic_significance.py` | Permutation null distributions against observed IC | 83 KB |
| `viz/calibration_curves.png` | `models/calibration.py` | Reliability diagrams with probability histograms, all four models | 266 KB |
| `viz/pca_variance.png` | `models/pca_robustness.py` | Scree plot for the final training window, and component count across the walk-forward | 166 KB |

## Logs

| Path | Produced by | Contents | Shape | Role |
|---|---|---|---|---|
| `logs/llm_calls.jsonl` | `llm/sentiment_pipeline.py` | One record per API call: prompt hash, raw response, stop reason, token counts, cost, latency, retry and repair flags. 129 records for 120 months — the excess is retries and dry-run calls. | 129 records | Report (cost and reliability figures) |
| `logs/run_all.log` | `run_all.py` | Appended transcript of every pipeline run, with per-stage timings | varies | Intermediate |

---

## Reading the multi-block CSVs

`models/calibration_results.csv` and `models/pca_robustness.csv` hold several
tables in one file, separated by `#` comment lines. `pandas.read_csv` will not
parse them whole. Split on the comment markers first:

```python
import io, pathlib, pandas as pd

blocks, name, buf = {}, None, []
for line in pathlib.Path("models/pca_robustness.csv").read_text().splitlines():
    if line.startswith("#"):
        if name and buf:
            blocks[name] = pd.read_csv(io.StringIO("\n".join(buf)))
        name, buf = line.lstrip("# ").strip(), []
    elif line.strip():
        buf.append(line)
blocks[name] = pd.read_csv(io.StringIO("\n".join(buf)))

blocks["SPECIFICATIONS"]        # 10 x 17
blocks["COMPONENT COUNT PER PREDICTION MONTH"]   # 93 x 6
```

`scripts/verify_all.py` reads both files this way and checks their contents
against `models/metrics.csv` and `models/predictions.csv`.
