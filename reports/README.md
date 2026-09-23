# Generated Model and Data Reports

The analytical pipeline produces:

- data_quality.json
- categorical_significance.csv
- numeric_significance.csv
- statistical_logit_odds_ratios.csv
- statistical_summary.json
- model_comparison.csv
- hyperparameter_search.csv
- test_predictions.csv
- shap_global_importance.csv

Regenerate them with:

python scripts/run_pipeline.py
python scripts/explainability.py

These are experiment artifacts and should be regenerated whenever data, features or model configuration changes.
