# Figure captions

One paragraph per figure, written to be pasted directly beneath it. Each states
what is plotted, how to read it, and what it shows — a caption that only names
the axes makes the reader do the work twice.

All figures are 300 DPI. Colours are fixed in `common/plotting.py`: Model A blue,
Model B teal, Model C red, Model D orange, LLM sentiment purple, Momentum violet,
equal weight grey, SPY green. A series keeps its colour in every figure.

---

### Figure 1 — `backtest/equity_curves.png` (report copy: `viz/equity_curves_report.png`)

**Out-of-sample equity curves, April 2018 to December 2025.** Growth of $1 for
the four models and three benchmarks, on a log scale so equal vertical distances
are equal percentage moves. Shaded bands mark months when lagged VIX closed at or
above 25; the lower panel shows drawdown from the running peak. All strategies
select from the same universe each month, so the spread between them reflects
selection alone rather than which sectors were investable. Every model finishes
below both benchmarks: SPY buy-and-hold ends highest at 14.84% CAGR and the
equal-weight sector portfolio at 12.12%, against 9.71% to 12.00% for Models A
through D. The models are not merely failing to add value over a passive
alternative — they are paying 355% to 827% annual turnover to underperform it.

### Figure 2 — `viz/shap_model_c.png`

**SHAP feature attribution, Model C (macro only).** Mean absolute SHAP value per
feature, which measures how much each input moves the model's output regardless
of direction. Features are ranked by that magnitude and the top 20 of 38 are
shown. This is the reference figure for Figure 3: it establishes what the model
attends to when the LLM feature is not available, and shows that attention is
spread thinly across the macro block rather than concentrated in one dominant
driver — consistent with the finding that no single macro variable carries the
signal either.

### Figure 3 — `viz/shap_model_d.png`

**SHAP feature attribution, Model D (macro + LLM), with the sentiment feature
highlighted.** The same construction as Figure 2, over the 43-feature set that
includes the five LLM columns. The comparison that matters is where the
sentiment features land in the ranking: the model does *use* them — they receive
non-trivial attribution — but using a feature and profiting from it are
different things, and Model D's out-of-sample AUC (0.4857) is lower than Model
C's (0.5081). Attribution shows what the model looked at, not whether looking
helped.

### Figure 4 — `viz/regime_auc_comparison.png`

**Model AUC by volatility regime, with bootstrap confidence intervals.** Months
are split on lagged VIX into low (< 20), medium (20–25) and high (≥ 25) regimes,
and AUC is computed within each; intervals are month-clustered bootstraps. The
motivating hypothesis was that FOMC language should matter most when uncertainty
is highest, so the LLM models should separate from the controls in the
high-volatility bucket. They do not. Every interval spans 0.50 in every regime,
and the high-volatility subsample is small enough that its intervals are the
widest on the chart — which is itself the honest reading: this design cannot
resolve a regime-conditional effect even if one existed.

### Figure 5 — `viz/ic_timeseries.png`

**Rolling information coefficient, six-month window.** The upper panel plots the
rolling mean Spearman correlation between each signal and the following month's
excess return, with each series' full-sample mean as a dotted line. The lower
panel shows the raw monthly IC of the LLM sentiment signal as bars. The rolling
lines wander well above and below zero — Model C reaches +0.42 in 2019 and −0.45
in 2020 — but they cross zero continually and none holds a sign. The bar panel
makes the point plainly: 48% of months have positive IC, against the 50% a coin
flip would deliver. Persistent-looking runs in a rolling mean are what noise
looks like when it is smoothed.

### Figure 6 — `viz/sentiment_redundancy.png`

**How much of the LLM's signal the macro block already contains.** The left panel
ranks the macro features by correlation with the LLM's monthly tone; the right
panel gives the R² from regressing each component of the sentiment score on all
14 macro variables jointly. The macro block explains 78.5% of the month-to-month
tone — the LLM is largely restating inputs the model already receives directly.
The sector-specific component tells the other half of the story: its R² is 0.0009,
because the macro row is identical across sectors within a month and cannot
explain cross-sectional differences by construction. Any genuinely new
information the LLM offers must live in that component, and that is precisely
the component the IC tests find uninformative.

### Figure 7 — `viz/quintile_returns.png`

**Forward excess return by LLM signal quintile.** Sector-months are sorted into
quintiles by the LLM score (left) and by its month-over-month change (right),
with mean annualised forward excess return per bucket and ±1 standard error
bars. A working signal produces a monotone staircase from Q1 to Q5. Neither
panel does: the level sort peaks at Q3 and puts its worst bucket at Q4, and the
change sort peaks at Q4. The Q5−Q1 spreads (+3.27%/yr and +3.79%/yr) look
superficially encouraging but carry t-statistics of 0.60 and 0.78 — smaller than
the error bars on individual buckets, and entirely consistent with noise.

### Figure 8 — `viz/topn_sensitivity.png`

**Sharpe ratio across the portfolio-construction grid.** Every combination of
top-N (2 to 5 holdings) and transaction cost (0 to 20 bps) for each model and the
momentum benchmark — 140 cells, reported whole rather than at a chosen corner.
The purpose is to test whether the null result is an artefact of the specific
top-3 / 10 bps parameterisation. It is not: 4 of the 80 model cells beat equal
weight, all of them Model C at 0 or 5 bps, i.e. only where trading is assumed to
be nearly free. A result that survives only in the cheapest corner of the grid is
evidence of fragility rather than skill.

### Figure 9 — `viz/horizon_decay.png`

**Information coefficient by holding horizon after an FOMC release.** Each
statement and minutes release is treated as an event, and cross-sectional IC is
measured from the release-day close over 1, 3, 5, 10, 21 and 42 trading days,
with Newey-West intervals to account for overlapping windows. The hypothesis
under test is that FOMC information is priced within days, so a month-end
rebalance would arrive too late. The data do not support it: no horizon shows an
IC distinguishable from zero, the largest absolute value being +0.0364 at one day
(t = 1.15). The signal is not being missed by acting too slowly — there is no
signal at any speed.

### Figure 10 — `viz/long_short_equity.png`

**Long-short spread portfolios — a diagnostic, not a proposed strategy.** Each
curve is a self-financing top-3 minus bottom-3 portfolio, which removes market
direction and isolates cross-sectional selection. If the models' long-only
shortfall were a beta story, removing beta should reveal skill underneath. It
does not: no spread is distinguishable from zero, and the best model spread is
Model C at +0.40%/yr net (Sharpe 0.09). Borrow fees, financing, recall risk and
short-sale constraints are all excluded, so these are an upper bound on what the
selection signal is worth.

### Figure 11 — `viz/permutation_null.png`

**Permutation null distributions against observed information coefficients.**
For each signal, the LLM score is shuffled within each month 10,000 times,
rebuilding the null distribution of mean IC under the hypothesis of no
cross-sectional information while preserving the panel's structure. The observed
IC is the vertical line. Every observed value sits comfortably inside its own
null — the most extreme is Model C at the 84th percentile (p = 0.165) — and none
survives Benjamini-Hochberg correction across the six signals. This is the
non-parametric confirmation of the parametric result, and it agrees.

### Figure 12 — `viz/calibration_curves.png`

**Reliability diagrams for the four models, with probability histograms.** Each
panel plots observed frequency against predicted probability in ten equal-width
bins, with the diagonal marking perfect calibration and point labels giving bin
counts; the histogram beneath shows how the predictions are distributed. All four
curves sit far flatter than the diagonal — the models' probabilities carry almost
no resolution. Note the asymmetry: the logistic models look better calibrated
largely because they barely commit (two-thirds of their predictions fall within
±0.05 of 0.5, and their outer bins hold as few as one observation), while the
XGBoost models genuinely spread out and are genuinely miscalibrated. Every model
scores worse than answering 0.5 every time.

### Figure 13 — `viz/pca_variance.png`

**Principal component structure of the macro block.** The upper panel is a scree
plot for the final training window, with filled bars marking the components
retained by the pre-specified 90%-variance rule and the red line showing
cumulative variance; the lower panel tracks how many components that rule
retained at each step of the walk-forward. The figure answers whether
multicollinearity in the raw macro block was masking the LLM. It was not: six of
fourteen components are needed to reach 90% (PC1 alone explains only 41.7%), so
there was never a single dominant macro factor crowding the LLM out — and
orthogonalising the block does not change the LLM's incremental value.

### Figure 14 — `viz/equity_curves_report_14asset.png` (also `backtest/equity_curves_14asset.png`)

**Out-of-sample equity curves for the 14-asset extension.** The same
construction as Figure 1, over the universe extended with TLT, IEF and GLD.
Model A finishes ahead of the equal-weight benchmark here (12.42% versus 10.88%
CAGR), which it did not do in the 11-asset universe — but the cause is visible
in the holdings rather than the curve: Model A selects a bond or gold in zero of
93 months, so it is simply an equity portfolio measured against a benchmark that
holds three lower-returning assets. Both Models C and D, which do hold bonds and
gold about a third of the time, finish below equal weight. The extension
therefore reproduces the main finding and adds a cautionary one: in a universe
with heterogeneous volatility, a static asset-class tilt can look like skill on
every headline metric at once.

### Figure 15 — `viz/per_asset_heatmap.png`

**Hit rate by asset and model, with each asset's base rate beside it.** Rows are
assets sorted by mean hit rate, columns are the four models, and colour diverges
around 0.50 — blue beats a coin flip, red loses to one. The narrow panel to the
right of each heatmap gives that asset's base rate (how often it actually beat
the equal-weight benchmark) and, as a diamond, the majority-class baseline: what
a model scores by always predicting the commoner outcome. The pairing is the
point. TLT has the highest hit rate anywhere on the chart (0.68 for Model A) and
is simultaneously the asset where every model *loses* to a constant prediction,
because it beats the benchmark in only 30% of months and so rewards a model for
never selecting it. Read alone, the heatmap suggests the models work best on
bonds; read against the base-rate panel, it shows the opposite. Across both
universes exactly one asset of twenty-five beats its own majority baseline on
average, and that margin is +0.011.
