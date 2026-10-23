### Table 1 — Main model performance

| Model | Algorithm | Feature set | AUC | AUC 95% CI | Accuracy | Mean IC | IC IR | Precision@3 |
|---|---|---|---|---|---|---|---|---|
| A | Logistic regression | Macro | 0.4827 | [0.4486, 0.5199] | 0.4866 | +0.0106 | 0.03 | 0.4767 |
| B | Logistic regression | Macro + LLM | 0.4932 | [0.4599, 0.5273] | 0.4945 | +0.0066 | 0.02 | 0.4767 |
| C | XGBoost | Macro | 0.5081 | [0.4749, 0.5414] | 0.5005 | +0.0462 | 0.12 | 0.4982 |
| D | XGBoost | Macro + LLM | 0.4857 | [0.4499, 0.5200] | 0.4777 | -0.0236 | -0.06 | 0.4839 |

*Out-of-sample walk-forward, 93 prediction months (2018-03 to 2025-11), 1,007 sector-month forecasts. AUC confidence intervals are month-clustered bootstraps (2,000 draws). An AUC of 0.50 is no ranking skill; every interval contains it.*
