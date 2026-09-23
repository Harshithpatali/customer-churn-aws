# AWS deployment — two ECS Express Mode services

## 1. Build and push API image

Set these first:

```bash
export AWS_REGION=ap-south-1
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export ECR_API=customer-churn-api
export ECR_UI=customer-churn-ui
```

Create repositories:

```bash
aws ecr create-repository --repository-name $ECR_API --region $AWS_REGION
aws ecr create-repository --repository-name $ECR_UI --region $AWS_REGION
```

Authenticate:

```bash
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com
```

Build/push:

```bash
docker build -f Dockerfile.api -t $ECR_API:latest .
docker tag $ECR_API:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_API:latest
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_API:latest

docker build -f Dockerfile.frontend -t $ECR_UI:latest .
docker tag $ECR_UI:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_UI:latest
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_UI:latest
```

## 2. Deploy API

Use the AWS console or CLI. Express Mode requires an ECR/private-registry image plus the required IAM roles. The AWS console can create the roles during the guided setup.

CLI shape:

```bash
aws ecs create-express-gateway-service \
  --service-name customer-churn-api \
  --primary-container "image=$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_API:latest,containerPort=8000" \
  --execution-role-arn <TASK_EXECUTION_ROLE_ARN> \
  --infrastructure-role-arn <EXPRESS_INFRA_ROLE_ARN> \
  --health-check-path /health \
  --monitor-resources
```

Copy the generated HTTPS URL.

## 3. Deploy Streamlit UI

```bash
aws ecs create-express-gateway-service \
  --service-name customer-churn-ui \
  --primary-container "image=$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_UI:latest,containerPort=8501,environment=[{name=BACKEND_URL,value=https://<API_URL>}]" \
  --execution-role-arn <TASK_EXECUTION_ROLE_ARN> \
  --infrastructure-role-arn <EXPRESS_INFRA_ROLE_ARN> \
  --health-check-path /_stcore/health \
  --monitor-resources
```

Replace `<API_URL>` with the API service hostname without a trailing slash.

## 4. Updating the application

Build and push a new image tag, then update the Express Mode service to that image. Do not put secrets in Git. Use ECS environment variables or Secrets Manager for credentials.

## Cost warning

ECS Express Mode itself has no separate Express Mode fee, but AWS bills the underlying resources such as Fargate, Application Load Balancer, CloudWatch and data transfer. Free-tier/credit eligibility varies by account and resource, so check the current AWS pricing page before leaving services running.
