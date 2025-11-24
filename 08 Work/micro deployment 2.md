# 🚀 Guide: Deploying the Insights Analyzer as an AWS Microservice

  

This guide will walk you through converting the `server/microservices/insights_analyzer` component into a standalone, scheduled AWS microservice.

  

Since your main application is already deployed using **AWS App Runner** (based on `.github/workflows/dev.yml`), we will deploy this background worker using **AWS ECS Fargate**. This is the standard, cost-effective way to run scheduled tasks (cron jobs) in AWS that need to share the same database and network configuration.

  

---

  

## 📋 Architecture Overview

  

* **What:** A Python script that runs every 3 days.

* **Where:** AWS ECS (Elastic Container Service) using Fargate (Serverless Compute).

* **Trigger:** AWS EventBridge (Scheduler).

* **Database:** Connects to your existing RDS PostgreSQL (same as your main app).

  

---

  

## ✅ Step 1: The Dockerfile

  

We need to package just this microservice into a Docker image. I have already created the `Dockerfile` for you at:

`server/microservices/insights_analyzer/Dockerfile`

  

**Key details:**

* It uses `python:3.12-slim` (matching your main app).

* It copies the entire `server/` directory so it can access shared code (like database models).

* It sets the entrypoint to run `server.microservices.insights_analyzer.main`.

  

---

  

## 🛠️ Step 2: AWS Infrastructure Setup (One-Time)

  

You need to create a few resources in AWS. You can do this via the AWS Console or CLI.

  

### 1. Create an ECR Repository

This is where we store the Docker image.

* **Name:** `lokam/insights-analyzer` (or similar)

* **Command:**

```bash

aws ecr create-repository --repository-name lokam/insights-analyzer --region us-east-1

```

  

### 2. Create a CloudWatch Log Group

To see the logs from your script.

* **Name:** `/ecs/insights-analyzer`

* **Command:**

```bash

aws logs create-log-group --log-group-name /ecs/insights-analyzer --region us-east-1

```

  

### 3. Create an ECS Cluster (if you don't have one)

App Runner manages its own infrastructure, so you likely need a standard ECS cluster for this.

* **Name:** `lokam-microservices`

* **Command:**

```bash

aws ecs create-cluster --cluster-name lokam-microservices --region us-east-1

```

  

---

  

## 🤖 Step 3: Automate Deployment (GitHub Actions)

  

We will create a **new** GitHub Actions workflow file to build and deploy this microservice automatically. This keeps it separate from your main `dev.yml` but uses the same secrets.

  

**Create a file: `.github/workflows/deploy-insights.yml`**

  

```yaml

name: Deploy Insights Microservice

  

on:

push:

branches: [ "playground" ] # Or your main branch

paths:

- 'server/microservices/insights_analyzer/**' # Only run when this folder changes

- 'server/app/models/**' # Or when shared models change

  

jobs:

deploy:

runs-on: ubuntu-latest

steps:

- name: Checkout code

uses: actions/checkout@v4

  

- name: Configure AWS credentials

uses: aws-actions/configure-aws-credentials@v4

with:

aws-access-key-id: ${{ secrets.DEV_AWS_ACCESS_KEY_ID }}

aws-secret-access-key: ${{ secrets.DEV_AWS_SECRET_ACCESS_KEY }}

aws-region: ${{ secrets.DEV_AWS_REGION }}

  

- name: Login to Amazon ECR

id: login-ecr

uses: aws-actions/amazon-ecr-login@v2

  

- name: Build, tag, and push image to Amazon ECR

env:

ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}

ECR_REPOSITORY: lokam/insights-analyzer

IMAGE_TAG: ${{ github.sha }}

run: |

# Build from the root context using the specific Dockerfile

docker build -f server/microservices/insights_analyzer/Dockerfile -t $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG .

docker push $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG

# Also push 'latest' tag

docker tag $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG $ECR_REGISTRY/$ECR_REPOSITORY:latest

docker push $ECR_REGISTRY/$ECR_REPOSITORY:latest

  

- name: Update ECS Task Definition

id: task-def

uses: aws-actions/amazon-ecs-render-task-definition@v1

with:

task-definition: .aws/insights-task-definition.json # We will create this next

container-name: insights-analyzer

image: ${{ steps.login-ecr.outputs.registry }}/lokam/insights-analyzer:${{ github.sha }}

  

- name: Deploy to ECS

uses: aws-actions/amazon-ecs-deploy-task-definition@v1

with:

task-definition: ${{ steps.task-def.outputs.task-definition }}

cluster: lokam-microservices

# We don't specify 'service' because this is a scheduled task, not a continuous service.

# However, updating the task definition is enough for the EventBridge scheduler to pick up the new 'latest' revision if configured correctly,

# OR we can update the EventBridge rule here.

# For simplicity, we just update the Task Definition.

```

  

---

  

## 📝 Step 4: The Task Definition

  

You need a JSON file that tells AWS how to run your container (CPU, Memory, Environment Variables).

  

**Create a file: `.aws/insights-task-definition.json`** (in your repo)

  

```json

{

"family": "insights-analyzer",

"networkMode": "awsvpc",

"requiresCompatibilities": ["FARGATE"],

"cpu": "512",

"memory": "1024",

"executionRoleArn": "arn:aws:iam::YOUR_AWS_ACCOUNT_ID:role/ecsTaskExecutionRole",

"taskRoleArn": "arn:aws:iam::YOUR_AWS_ACCOUNT_ID:role/ecsTaskExecutionRole",

"containerDefinitions": [

{

"name": "insights-analyzer",

"image": "lokam/insights-analyzer:latest",

"essential": true,

"logConfiguration": {

"logDriver": "awslogs",

"options": {

"awslogs-group": "/ecs/insights-analyzer",

"awslogs-region": "us-east-1",

"awslogs-stream-prefix": "ecs"

}

},

"environment": [

{ "name": "DATABASE_URL", "value": "YOUR_PRODUCTION_DB_URL_HERE" },

{ "name": "OPENAI_API_KEY", "value": "YOUR_OPENAI_KEY_HERE" }

]

}

]

}

```

*Note: For better security, use AWS Secrets Manager for the environment variables, similar to how your `dev.yml` injects them.*

  

---

  

## ⏰ Step 5: Scheduling (EventBridge)

  

Finally, tell AWS to run this Task every 3 days.

  

1. Go to **Amazon EventBridge** > **Schedules** > **Create Schedule**.

2. **Schedule pattern**: Recurring schedule > Cron-based schedule.

* Cron expression: `0 0 */3 * *` (Every 3 days at midnight).

3. **Target**: AWS ECS > Run Task.

* **Cluster**: `lokam-microservices`

* **Task Definition**: `insights-analyzer` (latest)

* **Launch Type**: FARGATE

* **Subnets**: Select your VPC subnets (use Private subnets if you have NAT Gateway, or Public subnets with "Auto-assign Public IP" enabled if you don't).

* **Security Groups**: Select a security group that allows outbound access (to internet for OpenAI and to your RDS).

  

---

  

## 🎯 Summary

  

1. **Code**: You have the code in `server/microservices/insights_analyzer`.

2. **Docker**: We created a `Dockerfile` to package it.

3. **CI/CD**: The new GitHub Action builds and pushes this image to ECR whenever you change the code.

4. **Run**: AWS EventBridge wakes up every 3 days, tells ECS to "Run Task" using the latest image, and your script executes!

  

This setup is robust, scalable, and completely separate from your main App Runner deployment, so a heavy analysis job won't slow down your website.