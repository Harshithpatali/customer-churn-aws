# Customer Churn Intelligence Platform — AWS Ready

Production-style customer churn prediction system built around the **IBM Telco Customer Churn** sample dataset.

> Dataset: IBM sample, 7,043 customers, 21 raw columns; `Churn` is the target. The official IBM repository is archived but remains publicly accessible.

## Architecture

```text
                         Internet
                            |
                +-----------+-----------+
                |                       |
        AWS ECS Express Mode     AWS ECS Express Mode
          FastAPI API              Streamlit UI
             :8000                    :8501
                |                       |
                +-----------+-----------+
                            |
                     ML inference layer
                            |
              +-------------+-------------+
              | XGBoost + preprocessing  |
              | SHAP explainability      |
              | risk + retention engine  |
              +---------------------------+
```

The frontend calls the backend through `BACKEND_URL`. Both components are independently containerized and can be deployed as separate ECS Express Mode services.

## Features

- IBM Telco Customer Churn data pipeline
- Leakage-safe preprocessing with `ColumnTransformer`
- Logistic Regression, Random Forest and XGBoost benchmarking
- Cross-validation and hyperparameter search
- Business threshold optimization
- SHAP local/global explainability
- Risk bands and revenue-at-risk estimate
- Retention recommendation engine
- Single-customer prediction API
- CSV batch prediction API
- Streamlit executive dashboard
- Dockerfiles for API and frontend
- Docker Compose for local integration
- AWS ECR + ECS Express Mode deployment scripts
- Health endpoint and API documentation
- Unit tests
- CI workflow

## Production Data Science Architecture

The project now separates the data/training plane from the online serving plane:

```text
Raw Telco CSV
    |
    +--> Data quality + deterministic feature pipeline
    |           |
    |           +--> processed/model_ready.csv
    |
    +--> Statistical inference
    |           +--> chi-square
    |           +--> Mann-Whitney U
    |           +--> logistic odds ratios / CI / AIC / BIC
    |
    +--> ML experimentation
                +--> Logistic Regression
                +--> Random Forest
                +--> XGBoost
                +--> 5-fold CV + randomized search
                +--> business threshold optimization
                            |
                            v
                    versioned model artifact
                            |
                            v
                    FastAPI model service
                       /predict /explain
                       /batch-predict
                            |
                            v
                     Streamlit decision UI
```

### Reproducible commands

```bash
python scripts/run_pipeline.py
python scripts/statistical_analysis.py
python scripts/explainability.py
pytest -q
```

The existing AWS ECS GitHub Actions workflows and Dockerfiles are intentionally kept unchanged by this upgrade.

## Production upgrade roadmap

See `docs/PRODUCTION_UPGRADE_PLAN.md` for the staged merge plan that preserves the existing AWS ECS CI/CD setup while adding data engineering, statistical modeling, ML experimentation, SHAP explainability, testing and modern UI layers.

## Dataset

Official source:
https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv

The training script can download the source automatically when network access is available:

```bash
python scripts/train.py --download
```

Or place the file at:

```text
data/raw/Telco-Customer-Churn.csv
```

## Local setup

```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# Linux / WSL
source .venv/bin/activate

pip install -r requirements.txt
python scripts/train.py
uvicorn api.main:app --reload --port 8000
```

Frontend:

```bash
streamlit run frontend/app.py
```

Set `BACKEND_URL=http://localhost:8000`.

## Docker

```bash
docker compose up --build
```

- API: http://localhost:8000/docs
- Health: http://localhost:8000/health
- Dashboard: http://localhost:8501

## AWS deployment

AWS ECS Express Mode accepts a container image and manages the Fargate service, HTTPS load balancer, networking, monitoring and autoscaling defaults. See `docs/AWS_DEPLOYMENT.md`.

High-level flow:

```text
GitHub -> Docker -> Amazon ECR -> ECS Express Mode -> public HTTPS URL
```

Deploy the API first, then deploy the frontend with:

```text
BACKEND_URL=https://<api-service>.ecs.<region>.on.aws
```

## Important training note

The repository contains the complete reproducible training pipeline. Because this build environment cannot transfer the full remote CSV into the local container runtime, the included artifact under `models/` is a **bootstrap smoke-test artifact**, not a claim of a fresh full-IBM training run. Run `python scripts/train.py --download` (or place the official CSV in `data/raw/`) to generate the production artifact and metrics from all 7,043 IBM records.

This distinction is intentional so the project never presents an unverified metric as a fresh experiment.
