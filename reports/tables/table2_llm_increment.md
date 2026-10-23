### Table 2 — LLM incremental effect across specifications

| Specification | ΔAUC (B−A) | 95% CI (B−A) | ΔAUC (D−C) | 95% CI (D−C) |
|---|---|---|---|---|
| Base (no sector identity) | +0.0035 | [-0.0031, +0.0104] | +0.0050 | [-0.0232, +0.0329] |
| + Sector identity | +0.0010 | [-0.0074, +0.0099] | -0.0255 * | [-0.0511, -0.0025] |
| + Derived sentiment | +0.0106 | [-0.0077, +0.0283] | -0.0224 | [-0.0556, +0.0103] |
| PCA macro | -0.0034 | [-0.0161, +0.0097] | +0.0225 | [-0.0134, +0.0579] |

*Every rung is computed on the same 93-month walk-forward with the same one-month embargo, so the rungs are comparable; earlier development figures came from a 94-month window and are not used here. B−A is the logistic pair, D−C the XGBoost pair. Positive means the LLM version scores higher. * marks an interval excluding zero — 1 of 8 cells, in the direction of harm. The sign changes across specifications in both estimator ladders, which is what a signal indistinguishable from noise looks like when it is measured repeatedly.*
