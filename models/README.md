# Model artifacts

`churn_pipeline.joblib` is a bootstrap smoke-test artifact generated from `data/raw/bootstrap_telco.csv` so the API can be tested immediately.

**It is not an IBM-trained model.**

For the real project run:

```bash
python scripts/train.py --download
```

or put the official IBM CSV at `data/raw/Telco-Customer-Churn.csv` and run:

```bash
python scripts/train.py
```

That command overwrites the artifact and writes fresh metrics and the feature contract.
