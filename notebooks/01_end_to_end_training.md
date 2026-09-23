# End-to-end training notebook plan

1. Load IBM Telco CSV
2. Validate schema, duplicates and missing values
3. Explore churn distribution
4. Engineer service count, tenure groups and value ratios
5. Split stratified train/test
6. Compare Logistic Regression, Random Forest and XGBoost
7. Run randomized CV on XGBoost using PR-AUC
8. Optimize threshold for retention recall/precision trade-off
9. Save pipeline, feature contract and metrics
10. Inspect global/local SHAP explanations
11. Export scored customer file for dashboard use
