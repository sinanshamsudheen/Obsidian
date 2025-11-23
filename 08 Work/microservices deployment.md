# Insights Microservice AWS Deployment Guide

  

## 📚 Prerequisites - What You Need to Learn First

  

Before deploying this microservice to AWS, you should understand these core concepts:

  

### 1. **Docker Fundamentals** (2-3 hours)

- **Why:** Containerization is essential for deploying microservices

- **What to Learn:**

- Docker images and containers

- Dockerfile creation

- Docker commands (build, run, push)

- Docker registries

  

**Recommended Tutorial:**

- [Docker Tutorial for Beginners - FreeCodeCamp](https://www.youtube.com/watch?v=fqMOX6JJhGo) (2.5 hours)

- [Docker in 100 Seconds](https://www.youtube.com/watch?v=Gjnup-PuquQ) (Quick overview)

  

### 2. **AWS Basics** (3-4 hours)

- **Why:** You need to understand AWS core services

- **What to Learn:**

- AWS account setup

- IAM roles and permissions

- AWS CLI basics

- Security groups and networking basics

  

**Recommended Tutorial:**

- [AWS Tutorial For Beginners](https://www.youtube.com/watch?v=k1RI5locZE4) (4 hours)

- [AWS IAM Tutorial](https://www.youtube.com/watch?v=ExjW3HCFHGY) (30 min)

  

### 3. **AWS ECS/Fargate** (2-3 hours)

- **Why:** Best option for running containerized microservices

- **What to Learn:**

- ECS clusters and services

- Task definitions

- Fargate vs EC2 launch types

- ECR (Elastic Container Registry)

  

**Recommended Tutorial:**

- [AWS ECS Tutorial](https://www.youtube.com/watch?v=esISkPlnxL0) (1.5 hours)

- [Deploy Docker to AWS ECS](https://www.youtube.com/watch?v=zs3tyVgiBQQ) (30 min)

  

### 4. **AWS EventBridge/CloudWatch Events** (1-2 hours)

- **Why:** For scheduling the microservice to run every 3 days

- **What to Learn:**

- EventBridge rules

- Cron expressions

- Triggering ECS tasks

  

**Recommended Tutorial:**

- [AWS EventBridge Tutorial](https://www.youtube.com/watch?v=28B4L1fnnGM) (20 min)

  

### 5. **PostgreSQL on AWS RDS** (1-2 hours)

- **Why:** Your database needs to be accessible from AWS

- **What to Learn:**

- RDS setup

- Security groups for database access

- Connection strings

  

**Recommended Tutorial:**

- [AWS RDS PostgreSQL Setup](https://www.youtube.com/watch?v=_Ul50H98VXo) (25 min)

  

**Total Learning Time:** ~12-15 hours (can be spread over 2-3 days)

  

---

  

## 🎯 Deployment Options

  

### Option 1: AWS Lambda (Simplest, but with limitations)

**Pros:**

- Serverless (no infrastructure management)

- Pay per execution

- Easy to set up

  

**Cons:**

- ⚠️ **15-minute timeout limit** (your microservice might exceed this)

- Memory limited to 10GB

- Cold starts

  

**Best for:** Quick tests, but **NOT RECOMMENDED** for this use case due to timeout.

  

### Option 2: AWS ECS Fargate (RECOMMENDED)

**Pros:**

- ✅ No timeout limits

- ✅ Scalable

- ✅ No server management

- ✅ Pay only when running

  

**Cons:**

- Slightly more complex setup

- Costs ~$0.04/hour when running

  

**Best for:** Production deployments of long-running tasks

  

### Option 3: AWS ECS with EC2

**Pros:**

- More control

- Can be cheaper for continuous workloads

  

**Cons:**

- Need to manage EC2 instances

- More complex

  

**Best for:** Heavy continuous workloads

  

### Option 4: AWS Batch

**Pros:**

- Designed for batch jobs

- Automatic scaling

  

**Cons:**

- Overkill for simple tasks

- More complex

  

---

  

## 🚀 Recommended Deployment: AWS ECS Fargate + EventBridge

  

This guide will use **ECS Fargate** + **EventBridge** for scheduling.

  

---

  

## Step-by-Step Deployment Guide

  

### Phase 1: Prepare Your Application

  

#### 1.1 Create a Dockerfile

  

Create `server/microservices/insights_analyzer/Dockerfile`:

  

```dockerfile

FROM python:3.11-slim

  

# Set working directory

WORKDIR /app

  

# Install system dependencies

RUN apt-get update && apt-get install -y \

gcc \

postgresql-client \

&& rm -rf /var/lib/apt/lists/*

  

# Copy requirements

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

  

# Copy the entire server directory

COPY . /app/server/

  

# Set Python path

ENV PYTHONPATH=/app/server

  

# Run the microservice

CMD ["python", "-m", "server.microservices.insights_analyzer.main"]

```

  

#### 1.2 Create requirements.txt

  

Create `server/microservices/insights_analyzer/requirements.txt`:

  

```text

fastapi==0.104.1

sqlalchemy==2.0.23

asyncpg==0.29.0

httpx==0.25.2

pydantic==2.5.2

pydantic-settings==2.1.0

python-dotenv==1.0.0

alembic==1.13.0

```

  

#### 1.3 Prepare Environment Variables

  

You'll need these environment variables in AWS:

  

```bash

DATABASE_URL=postgresql+asyncpg://user:password@your-rds-endpoint:5432/lokamspace

OPENAI_API_KEY=sk-your-openai-key

OPENAI_MODEL=gpt-4o

OPENAI_BASE_URL=https://api.openai.com/v1

```

  

---

  

### Phase 2: AWS Setup

  

#### 2.1 Install AWS CLI

  

```bash

# macOS

brew install awscli

  

# Linux

curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"

unzip awscliv2.zip

sudo ./aws/install

  

# Windows

# Download from: https://awscli.amazonaws.com/AWSCLIV2.msi

```

  

#### 2.2 Configure AWS CLI

  

```bash

aws configure

```

  

Enter:

- AWS Access Key ID

- AWS Secret Access Key

- Default region (e.g., `us-east-1`)

- Default output format: `json`

  

#### 2.3 Create ECR Repository

  

```bash

# Create repository for Docker images

aws ecr create-repository \

--repository-name lokam/insights-analyzer \

--region us-east-1

```

  

**Output will include:**

```json

{

"repository": {

"repositoryUri": "123456789012.dkr.ecr.us-east-1.amazonaws.com/lokam/insights-analyzer"

}

}

```

  

Save this `repositoryUri` - you'll need it!

  

---

  

### Phase 3: Build and Push Docker Image

  

#### 3.1 Build Docker Image

  

```bash

cd /home/zero/Lokam/lokamspace/server/microservices/insights_analyzer

  

# Build the image

docker build -t insights-analyzer:latest .

```

  

#### 3.2 Authenticate Docker with ECR

  

```bash

# Get login password

aws ecr get-login-password --region us-east-1 | \

docker login --username AWS --password-stdin \

123456789012.dkr.ecr.us-east-1.amazonaws.com

```

  

#### 3.3 Tag and Push Image

  

```bash

# Tag the image

docker tag insights-analyzer:latest \

123456789012.dkr.ecr.us-east-1.amazonaws.com/lokam/insights-analyzer:latest

  

# Push to ECR

docker push \

123456789012.dkr.ecr.us-east-1.amazonaws.com/lokam/insights-analyzer:latest

```

  

---

  

### Phase 4: Create ECS Resources

  

#### 4.1 Create ECS Cluster

  

```bash

aws ecs create-cluster \

--cluster-name lokam-microservices \

--region us-east-1

```

  

#### 4.2 Create IAM Execution Role

  

Create `ecs-task-execution-role-policy.json`:

  

```json

{

"Version": "2012-10-17",

"Statement": [

{

"Effect": "Allow",

"Principal": {

"Service": "ecs-tasks.amazonaws.com"

},

"Action": "sts:AssumeRole"

}

]

}

```

  

```bash

# Create the role

aws iam create-role \

--role-name ecsTaskExecutionRole \

--assume-role-policy-document file://ecs-task-execution-role-policy.json

  

# Attach policy

aws iam attach-role-policy \

--role-name ecsTaskExecutionRole \

--policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy

```

  

#### 4.3 Create Task Definition

  

Create `insights-task-definition.json`:

  

```json

{

"family": "insights-analyzer",

"networkMode": "awsvpc",

"requiresCompatibilities": ["FARGATE"],

"cpu": "512",

"memory": "1024",

"executionRoleArn": "arn:aws:iam::123456789012:role/ecsTaskExecutionRole",

"containerDefinitions": [

{

"name": "insights-analyzer",

"image": "123456789012.dkr.ecr.us-east-1.amazonaws.com/lokam/insights-analyzer:latest",

"essential": true,

"environment": [

{

"name": "DATABASE_URL",

"value": "postgresql+asyncpg://user:password@rds-endpoint:5432/lokamspace"

},

{

"name": "OPENAI_API_KEY",

"value": "sk-your-api-key"

},

{

"name": "OPENAI_MODEL",

"value": "gpt-4o"

},

{

"name": "OPENAI_BASE_URL",

"value": "https://api.openai.com/v1"

}

],

"logConfiguration": {

"logDriver": "awslogs",

"options": {

"awslogs-group": "/ecs/insights-analyzer",

"awslogs-region": "us-east-1",

"awslogs-stream-prefix": "ecs"

}

}

}

]

}

```

  

**⚠️ Security Note:** For production, use **AWS Secrets Manager** instead of plain text for sensitive values!

  

Register the task:

  

```bash

aws ecs register-task-definition \

--cli-input-json file://insights-task-definition.json

```

  

#### 4.4 Create CloudWatch Log Group

  

```bash

aws logs create-log-group \

--log-group-name /ecs/insights-analyzer \

--region us-east-1

```

  

---

  

### Phase 5: Schedule with EventBridge

  

#### 5.1 Create EventBridge IAM Role

  

Create `eventbridge-ecs-role-policy.json`:

  

```json

{

"Version": "2012-10-17",

"Statement": [

{

"Effect": "Allow",

"Principal": {

"Service": "events.amazonaws.com"

},

"Action": "sts:AssumeRole"

}

]

}

```

  

```bash

# Create role

aws iam create-role \

--role-name EventBridgeECSRole \

--assume-role-policy-document file://eventbridge-ecs-role-policy.json

  

# Attach permissions

aws iam attach-role-policy \

--role-name EventBridgeECSRole \

--policy-arn arn:aws:iam::aws:policy/service-role/AmazonEC2ContainerServiceEventsRole

```

  

#### 5.2 Create EventBridge Rule

  

**Run every 3 days at 2 AM UTC:**

  

```bash

aws events put-rule \

--name insights-analyzer-schedule \

--description "Run insights analyzer every 3 days" \

--schedule-expression "cron(0 2 */3 * ? *)" \

--region us-east-1

```

  

**Cron Expression Breakdown:**

- `0` - Minute: 0

- `2` - Hour: 2 AM

- `*/3` - Day: Every 3 days

- `*` - Month: Every month

- `?` - Day of week: No specific day

- `*` - Year: Every year

  

#### 5.3 Add ECS Task as Target

  

Create `eventbridge-ecs-target.json`:

  

```json

{

"Targets": [

{

"Id": "1",

"Arn": "arn:aws:ecs:us-east-1:123456789012:cluster/lokam-microservices",

"RoleArn": "arn:aws:iam::123456789012:role/EventBridgeECSRole",

"EcsParameters": {

"TaskDefinitionArn": "arn:aws:ecs:us-east-1:123456789012:task-definition/insights-analyzer:1",

"TaskCount": 1,

"LaunchType": "FARGATE",

"NetworkConfiguration": {

"awsvpcConfiguration": {

"Subnets": ["subnet-12345678"],

"SecurityGroups": ["sg-12345678"],

"AssignPublicIp": "ENABLED"

}

}

}

}

]

}

```

  

**⚠️ Replace:**

- `subnet-12345678` with your VPC subnet ID

- `sg-12345678` with your security group ID (allow outbound to OpenAI and RDS)

  

```bash

aws events put-targets \

--rule insights-analyzer-schedule \

--cli-input-json file://eventbridge-ecs-target.json

```

  

---

  

### Phase 6: Test the Deployment

  

#### 6.1 Manual Test Run

  

```bash

aws ecs run-task \

--cluster lokam-microservices \

--task-definition insights-analyzer:1 \

--launch-type FARGATE \

--network-configuration "awsvpcConfiguration={subnets=[subnet-12345678],securityGroups=[sg-12345678],assignPublicIp=ENABLED}"

```

  

#### 6.2 Check Logs

  

```bash

# Get task ID from output above, then:

aws logs tail /ecs/insights-analyzer --follow

```

  

#### 6.3 Verify in Database

  

```bash

psql $DATABASE_URL -c "SELECT COUNT(*) FROM insights WHERE updated_at > NOW() - INTERVAL '1 hour';"

```

  

---

  

## 🔒 Security Best Practices

  

### 1. Use AWS Secrets Manager for Sensitive Data

  

```bash

# Create secret

aws secretsmanager create-secret \

--name lokam/insights/env \

--secret-string '{

"DATABASE_URL":"postgresql+asyncpg://...",

"OPENAI_API_KEY":"sk-..."

}'

  

# Update task definition to reference secrets

# Add to containerDefinitions:

"secrets": [

{

"name": "DATABASE_URL",

"valueFrom": "arn:aws:secretsmanager:us-east-1:123456789012:secret:lokam/insights/env:DATABASE_URL::"

},

{

"name": "OPENAI_API_KEY",

"valueFrom": "arn:aws:secretsmanager:us-east-1:123456789012:secret:lokam/insights/env:OPENAI_API_KEY::"

}

]

```

  

### 2. Network Security

  

- Use **VPC Private Subnets** for tasks

- Use **NAT Gateway** for internet access (OpenAI API)

- Restrict **Security Groups** to only necessary ports

- Use **RDS Security Groups** to allow only ECS tasks

  

### 3. IAM Least Privilege

  

- Create specific IAM roles with minimum permissions

- Use task-level roles, not instance-level roles

  

---

  

## 💰 Cost Estimation

  

### ECS Fargate Costs

- **CPU:** 512 vCPU = $0.04048/hour

- **Memory:** 1GB = $0.004445/hour

- **Total:** ~$0.045/hour

  

**Monthly Cost:**

- Run every 3 days = ~10 runs/month

- Average runtime: ~5 minutes/run

- **Cost:** 10 × (5/60) × $0.045 = **$0.04/month**

  

### Data Transfer

- Minimal (API calls to OpenAI)

- **~ $0.01/month**

  

### CloudWatch Logs

- **~ $0.50/month** (500MB/month)

  

**Total Estimated Monthly Cost: ~$0.55/month** 🎉

  

---

  

## 📊 Monitoring & Alerts

  

### Set Up CloudWatch Alarms

  

```bash

# Alarm for failed tasks

aws cloudwatch put-metric-alarm \

--alarm-name insights-task-failure \

--alarm-description "Alert when insights task fails" \

--metric-name TaskFailedCount \

--namespace AWS/ECS \

--statistic Sum \

--period 300 \

--evaluation-periods 1 \

--threshold 1 \

--comparison-operator GreaterThanThreshold \

--dimensions Name=ClusterName,Value=lokam-microservices

```

  

### Dashboard

  

Create a CloudWatch dashboard to monitor:

- Task run count

- Task failures

- Log errors

- Execution duration

  

---

  

## 🔧 Troubleshooting

  

### Issue: Task Fails to Start

  

**Check:**

```bash

# Describe the task

aws ecs describe-tasks \

--cluster lokam-microservices \

--tasks <task-id>

```

  

**Common causes:**

- Image not found in ECR

- IAM role permissions

- Security group blocking outbound

  

### Issue: Database Connection Failed

  

**Check:**

- RDS security group allows ECS subnet

- Connection string is correct

- Database is accessible

  

**Test connection from ECS:**

```bash

# Run a test container

aws ecs run-task \

--cluster lokam-microservices \

--task-definition insights-analyzer:1 \

--overrides '{"containerOverrides":[{"name":"insights-analyzer","command":["python","-c","import psycopg2; print(\"OK\")"]}]}'

```

  

### Issue: OpenAI API Timeout

  

**Check:**

- Security group allows outbound HTTPS

- NAT Gateway is configured (if using private subnet)

- OpenAI API key is valid

  

---

  

## 🔄 CI/CD Integration (Optional)

  

### GitHub Actions Workflow

  

Create `.github/workflows/deploy-insights.yml`:

  

```yaml

name: Deploy Insights Analyzer

  

on:

push:

branches: [main]

paths:

- 'server/microservices/insights_analyzer/**'

  

jobs:

deploy:

runs-on: ubuntu-latest

steps:

- uses: actions/checkout@v3

- name: Configure AWS Credentials

uses: aws-actions/configure-aws-credentials@v2

with:

aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}

aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}

aws-region: us-east-1

- name: Login to ECR

id: login-ecr

uses: aws-actions/amazon-ecr-login@v1

- name: Build and Push

env:

ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}

ECR_REPOSITORY: lokam/insights-analyzer

IMAGE_TAG: ${{ github.sha }}

run: |

cd server/microservices/insights_analyzer

docker build -t $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG .

docker push $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG

docker tag $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG $ECR_REGISTRY/$ECR_REPOSITORY:latest

docker push $ECR_REGISTRY/$ECR_REPOSITORY:latest

- name: Update Task Definition

run: |

aws ecs update-service \

--cluster lokam-microservices \

--service insights-analyzer \

--force-new-deployment

```

  

---

  

## 📝 Quick Reference Commands

  

### Update Docker Image

```bash

# Rebuild and push

docker build -t insights-analyzer:latest .

docker tag insights-analyzer:latest 123456789012.dkr.ecr.us-east-1.amazonaws.com/lokam/insights-analyzer:latest

docker push 123456789012.dkr.ecr.us-east-1.amazonaws.com/lokam/insights-analyzer:latest

  

# Force ECS to pull new image

aws ecs update-service --cluster lokam-microservices --service insights-analyzer --force-new-deployment

```

  

### Manual Trigger

```bash

aws ecs run-task --cluster lokam-microservices --task-definition insights-analyzer:1 --launch-type FARGATE --network-configuration "awsvpcConfiguration={subnets=[subnet-12345678],securityGroups=[sg-12345678],assignPublicIp=ENABLED}"

```

  

### View Logs

```bash

aws logs tail /ecs/insights-analyzer --follow

```

  

### Update Schedule

```bash

# Change to run daily at 3 AM

aws events put-rule --name insights-analyzer-schedule --schedule-expression "cron(0 3 * * ? *)"

```

  

---

  

## ✅ Deployment Checklist

  

- [ ] Learn Docker basics (2-3 hours)

- [ ] Learn AWS ECS fundamentals (2-3 hours)

- [ ] Set up AWS account and IAM user

- [ ] Install AWS CLI

- [ ] Create ECR repository

- [ ] Build and push Docker image

- [ ] Create ECS cluster

- [ ] Create task definition

- [ ] Set up EventBridge schedule

- [ ] Test manual run

- [ ] Verify logs in CloudWatch

- [ ] Check database for new insights

- [ ] Set up monitoring/alerts

- [ ] Document credentials in password manager

- [ ] **Switch to AWS Secrets Manager for production**

  

---

  

## 🎓 Additional Resources

  

### Documentation

- [AWS ECS Documentation](https://docs.aws.amazon.com/ecs/)

- [AWS EventBridge Documentation](https://docs.aws.amazon.com/eventbridge/)

- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)

  

### Videos

- [Complete AWS Tutorial](https://www.youtube.com/watch?v=3hLmDS179YE) (10 hours)

- [AWS ECS Deep Dive](https://www.youtube.com/watch?v=I9VAMGEjW-Q) (50 min)

- [Docker Crash Course](https://www.youtube.com/watch?v=pg19Z8LL06w) (1 hour)

  

### Communities

- [r/aws](https://reddit.com/r/aws)

- [AWS Community Discord](https://discord.gg/aws)

- [Docker Community Forums](https://forums.docker.com/)

  

---

  

## 🚨 Important Notes

  

1. **Test in Development First**: Set up a dev environment before production

2. **Use Secrets Manager**: Never hardcode credentials

3. **Monitor Costs**: Set up billing alerts

4. **Backup Strategy**: Ensure your RDS has automated backups

5. **Version Control**: Tag Docker images with versions

6. **Rollback Plan**: Keep previous task definitions for quick rollback

  

---

  

**Good luck with your deployment! 🚀**

  

For questions or issues, refer to AWS documentation or community forums.