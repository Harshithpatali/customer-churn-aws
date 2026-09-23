# Statistical Modeling

The project separates statistical inference from predictive machine learning.

## Methods

- Chi-square and Cramer's V for categorical variables versus churn.
- Mann-Whitney U tests for numeric distributions between churn classes.
- Logistic regression for coefficients, odds ratios, p-values and confidence intervals.

## Interpretation

Statistical significance is treated as evidence of association, not causal evidence. Predictive model quality is evaluated separately using ROC-AUC, PR-AUC, recall, precision, F1 and Brier score.

Run:

```bash
python scripts/statistical_analysis.py
```

Outputs are written to `reports/`.
