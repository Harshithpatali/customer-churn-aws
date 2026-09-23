# Data Pipeline

The pipeline validates the IBM Telco schema, converts numeric fields, removes duplicate customer IDs, maps the churn target to binary values, and creates reusable customer-level features.

Run:

```bash
python -m services.data_pipeline.run
```

The processed dataset is written to `data/processed/model_ready.csv` and the data-quality profile to `reports/data_quality.json`.
