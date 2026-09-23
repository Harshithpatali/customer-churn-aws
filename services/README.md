# Service Boundaries

The project separates the data and training plane from online serving.

- data_pipeline: ingestion, schema validation, deterministic cleaning and feature engineering.
- model_service: serving boundary around the serialized model and explanation engine.
- api: HTTP boundary implemented by FastAPI.

The existing Dockerfiles and ECS deployment workflows are infrastructure and remain unchanged.
