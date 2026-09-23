# Explainability

The serving API exposes local SHAP explanations through `POST /explain`.

Generate global SHAP importance with:

```bash
python scripts/explainability.py
```

The project uses TreeSHAP for the XGBoost model. SHAP values explain model behavior; they are not causal effects.
