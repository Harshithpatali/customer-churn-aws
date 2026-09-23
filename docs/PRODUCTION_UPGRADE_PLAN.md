# Production Upgrade & Merge Plan

## Objective

Evolve the current Customer Churn Intelligence application into an industry-style, reproducible Data Science and ML platform without breaking the existing AWS ECS deployment.

## Current production baseline

- FastAPI backend on AWS ECS Express Mode
- Streamlit frontend on AWS ECS Express Mode
- Amazon ECR images
- GitHub Actions CI
- GitHub Actions API and frontend ECS auto-deployment
- IBM Telco Customer Churn dataset
- XGBoost churn model
- SHAP explainability
- Batch prediction
- Dockerized services

## Merge rules

The existing Git repository and production deployment remain the source of truth.

### Never overwrite

- .git/
- .github/workflows/deploy-api.yml
- .github/workflows/deploy-frontend.yml
- .github/workflows/ci.yml
- Dockerfile.api
- Dockerfile.frontend
- AWS service configuration

These files are already connected to the live ECS deployment.

### Add incrementally

1. Data engineering layer
   - services/data_pipeline/
   - schema validation
   - data-quality checks
   - reproducible feature generation
   - processed-data artifacts

2. Statistical analysis layer
   - src/statistics.py
   - scripts/statistical_analysis.py
   - categorical association tests
   - numeric distribution tests
   - logistic-regression inference
   - odds ratios, confidence intervals, p-values, AIC/BIC

3. ML experimentation layer
   - model benchmarking
   - cross-validation
   - hyperparameter search
   - threshold optimization
   - calibration metrics
   - reproducible training reports

4. Explainability layer
   - scripts/explainability.py
   - global SHAP importance
   - local SHAP explanations
   - retention-driver reporting

5. Serving layer
   - preserve FastAPI compatibility
   - health/readiness endpoints
   - prediction endpoint
   - explanation endpoint
   - batch scoring endpoint
   - model metadata

6. Product/UI layer
   - executive overview
   - customer 360 scoring
   - SHAP explanation view
   - batch scoring
   - model governance
   - modern responsive styling

7. Testing and quality
   - unit tests
   - API tests
   - data-contract tests
   - training smoke test
   - CI enforcement

8. Documentation
   - architecture
   - data pipeline
   - statistical methodology
   - model methodology
   - explainability
   - deployment
   - model governance

## Safe rollout sequence

Do not replace the complete application in one commit.

### Phase 1 — Additive structure
Add new services, analysis scripts, reports, notebooks and documentation without changing the live API/frontend.

### Phase 2 — Local validation
Run:

    python scripts/run_pipeline.py
    python scripts/statistical_analysis.py
    python scripts/train.py
    python scripts/explainability.py
    pytest -q

Verify artifacts and metrics before changing serving code.

### Phase 3 — Serving integration
Integrate the validated model service and new explanation/reporting endpoints while preserving existing API contracts.

### Phase 4 — UI integration
Replace dashboard components only after the API is verified locally.

### Phase 5 — Production deployment
Push to main and let the existing CI/CD workflows build ECR images and force ECS deployments.

### Phase 6 — Production smoke test
Verify:

- /health
- /ready
- /metrics
- /predict
- /explain
- batch prediction
- Streamlit dashboard
- API-to-frontend communication

## Target architecture

    Raw CSV
       |
       v
    Data Quality
       |
       v
    Feature Pipeline
       |
       +-------------------+
       |                   |
       v                   v
    Statistical        ML Training
    Analysis               |
       |                   v
       |              Model Registry
       |                   |
       +---------+---------+
                 |
                 v
          FastAPI Model Service
                 |
          +------+------+
          |             |
          v             v
      Prediction      SHAP
          |
          v
      Streamlit UI
          |
          v
      AWS ECS

## Important principle

The project should demonstrate depth rather than simply adding more files. Every new component must have a clear purpose, reproducible output, tests where appropriate, and documentation.
