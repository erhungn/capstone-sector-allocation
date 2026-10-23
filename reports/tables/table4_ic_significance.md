### Table 4 — Statistical significance of IC

| Signal | Mean IC | t (conventional) | t (Newey-West) | Bootstrap 95% CI | Permutation p | BH-adjusted p |
|---|---|---|---|---|---|---|
| Model A | +0.0106 | 0.29 | 0.24 | [-0.0594, +0.0831] | 0.737 | 0.943 |
| Model B | +0.0066 | 0.19 | 0.16 | [-0.0605, +0.0730] | 0.848 | 0.943 |
| Model C | +0.0462 | 1.19 | 1.13 | [-0.0283, +0.1208] | 0.165 | 0.583 |
| Model D | -0.0236 | -0.61 | -0.66 | [-0.0983, +0.0517] | 0.472 | 0.943 |
| LLM sentiment | -0.0023 | -0.05 | -0.07 | [-0.0861, +0.0824] | 0.943 | 0.943 |
| LLM sentiment change | +0.0434 | 1.22 | 1.29 | [-0.0258, +0.1131] | 0.194 | 0.583 |

*Monthly Spearman IC against next-month excess return, 93 months. Newey-West uses the standard ⌊4(T/100)^(2/9)⌋ = 3 lags. The bootstrap resamples whole months; the permutation test shuffles the signal within each month across 10,000 draws. Benjamini-Hochberg corrects across the six signals. No signal is distinguishable from zero by any of the three methods, and none survives FDR correction.*
