# FedGAN SaaS Products - Deployment Guide

Complete guide for deploying all 6 SaaS products to production.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Local Development](#local-development)
4. [Docker Deployment](#docker-deployment)
5. [Cloud Deployment](#cloud-deployment)
6. [Production Checklist](#production-checklist)
7. [Monitoring & Maintenance](#monitoring--maintenance)

---

## Overview

This guide covers deploying 6 distinct SaaS products:

| Product | Port | Database | Use Case |
|---------|------|----------|----------|
| FedMed Platform | 8000 | PostgreSQL | Enterprise federated learning |
| SyntheticHealth | 8001 | SQLite | Image generation API |
| PrivacyGuard | 8002 | PostgreSQL | Privacy auditing |
| FedTrain Express | 8003 | SQLite | Simplified FL training |
| MedImageQuality | 8004 | N/A | Image quality metrics |
| DataSim | 8005 | N/A | Data distribution simulator |

---

## Prerequisites

### System Requirements

```bash
# Minimum
- CPU: 4 cores
- RAM: 16 GB
- Storage: 100 GB SSD
- OS: Ubuntu 20.04+ / Debian 11+ / CentOS 8+

# Recommended for Production
- CPU: 8+ cores
- RAM: 32 GB
- Storage: 500 GB NVMe SSD
- GPU: NVIDIA Tesla T4 or better (for ML operations)
```

### Software Dependencies

```bash
# Install Python 3.11
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev

# Install PostgreSQL
sudo apt install postgresql-15 postgresql-contrib

# Install Redis
sudo apt install redis-server

# Install Docker (optional but recommended)
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo apt install docker-compose
```

---

## Local Development

### Option 1: Run Individual Products

#### FedMed Platform

```bash
cd saas_products/fedmed_platform

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Setup database
createdb fedmed_db

# Configure environment
cp .env.example .env
nano .env  # Edit configuration

# Run
uvicorn app.main:app --reload --port 8000
```

Access at: http://localhost:8000

#### SyntheticHealth

```bash
cd saas_products/synthetic_health
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --port 8001
```

Access at: http://localhost:8001

#### PrivacyGuard

```bash
cd saas_products/privacy_guard
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
createdb privacyguard_db
uvicorn app.main:app --reload --port 8002
```

Access at: http://localhost:8002

#### FedTrain Express

```bash
cd saas_products/fedtrain_express
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --port 8003
```

Access at: http://localhost:8003

#### MedImageQuality

```bash
cd saas_products/medimage_quality
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --port 8004
```

Access at: http://localhost:8004

#### DataSim

```bash
cd saas_products/datasim
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --port 8005
```

Access at: http://localhost:8005

### Option 2: Run All Products Simultaneously

Create a master script:

```bash
#!/bin/bash
# run_all_saas.sh

# Start FedMed Platform
cd fedmed_platform && uvicorn app.main:app --port 8000 &

# Start SyntheticHealth
cd ../synthetic_health && uvicorn app.main:app --port 8001 &

# Start PrivacyGuard
cd ../privacy_guard && uvicorn app.main:app --port 8002 &

# Start FedTrain Express
cd ../fedtrain_express && uvicorn app.main:app --port 8003 &

# Start MedImageQuality
cd ../medimage_quality && uvicorn app.main:app --port 8004 &

# Start DataSim
cd ../datasim && uvicorn app.main:app --port 8005 &

echo "All SaaS products started!"
echo "FedMed Platform: http://localhost:8000"
echo "SyntheticHealth: http://localhost:8001"
echo "PrivacyGuard: http://localhost:8002"
echo "FedTrain Express: http://localhost:8003"
echo "MedImageQuality: http://localhost:8004"
echo "DataSim: http://localhost:8005"
```

---

## Docker Deployment

### Individual Product Deployment

#### FedMed Platform (Docker)

```bash
cd fedmed_platform
docker-compose up -d
```

This starts:
- Web server (port 8000)
- PostgreSQL database
- Redis cache
- Celery worker

### Deploy All Products with Docker

Create `docker-compose-all.yml`:

```yaml
version: '3.8'

services:
  fedmed:
    build: ./fedmed_platform
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@fedmed_db:5432/fedmed
    depends_on:
      - fedmed_db
      - redis

  fedmed_db:
    image: postgres:15-alpine
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
      - POSTGRES_DB=fedmed
    volumes:
      - fedmed_data:/var/lib/postgresql/data

  synthetic_health:
    build: ./synthetic_health
    ports:
      - "8001:8001"

  privacy_guard:
    build: ./privacy_guard
    ports:
      - "8002:8002"

  fedtrain_express:
    build: ./fedtrain_express
    ports:
      - "8003:8003"

  medimage_quality:
    build: ./medimage_quality
    ports:
      - "8004:8004"

  datasim:
    build: ./datasim
    ports:
      - "8005:8005"

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./certs:/etc/nginx/certs
    depends_on:
      - fedmed
      - synthetic_health
      - privacy_guard
      - fedtrain_express
      - medimage_quality
      - datasim

volumes:
  fedmed_data:
  redis_data:
```

Deploy:
```bash
docker-compose -f docker-compose-all.yml up -d
```

---

## Cloud Deployment

### AWS Deployment

#### Architecture

```
Route 53 (DNS)
    ↓
CloudFront (CDN)
    ↓
Application Load Balancer
    ↓
┌────────────────────────────────────┐
│  ECS Fargate / EC2 Instances       │
│  - FedMed Platform (8000)          │
│  - SyntheticHealth (8001)          │
│  - PrivacyGuard (8002)             │
│  - FedTrain Express (8003)         │
│  - MedImageQuality (8004)          │
│  - DataSim (8005)                  │
└────────────────────────────────────┘
    ↓
┌─────────────────┬──────────────────┐
│  RDS (Postgres) │  ElastiCache     │
│                 │  (Redis)         │
└─────────────────┴──────────────────┘
```

#### Step-by-Step AWS Deployment

1. **Setup RDS PostgreSQL**:
```bash
aws rds create-db-instance \
  --db-instance-identifier fedmed-db \
  --db-instance-class db.t3.medium \
  --engine postgres \
  --master-username admin \
  --master-user-password <secure-password> \
  --allocated-storage 100
```

2. **Setup ElastiCache Redis**:
```bash
aws elasticache create-cache-cluster \
  --cache-cluster-id fedmed-redis \
  --cache-node-type cache.t3.micro \
  --engine redis \
  --num-cache-nodes 1
```

3. **Deploy to ECS**:
```bash
# Build and push Docker images
aws ecr create-repository --repository-name fedmed-platform
docker build -t fedmed-platform ./fedmed_platform
docker tag fedmed-platform:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/fedmed-platform
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/fedmed-platform

# Create ECS task definition
# Create ECS service
# Configure load balancer
```

### Google Cloud Platform

```bash
# Create GKE cluster
gcloud container clusters create fedgan-cluster \
  --num-nodes=3 \
  --machine-type=n1-standard-4

# Deploy with Kubernetes
kubectl apply -f k8s/deployments/
kubectl apply -f k8s/services/
kubectl apply -f k8s/ingress.yml
```

### Azure

```bash
# Create AKS cluster
az aks create \
  --resource-group fedgan-rg \
  --name fedgan-aks \
  --node-count 3 \
  --node-vm-size Standard_D4s_v3

# Deploy
kubectl apply -f azure-deployment.yml
```

---

## Production Checklist

### Security

- [ ] Change all default passwords and secret keys
- [ ] Enable HTTPS with valid SSL certificates (Let's Encrypt)
- [ ] Configure firewall rules (allow only necessary ports)
- [ ] Enable database encryption at rest
- [ ] Setup VPN for database access
- [ ] Implement rate limiting
- [ ] Enable CORS only for trusted domains
- [ ] Setup WAF (Web Application Firewall)
- [ ] Regular security audits

### Configuration

- [ ] Set `DEBUG=False` in all `.env` files
- [ ] Configure production database URLs
- [ ] Setup email/SMS notifications
- [ ] Configure backup schedules
- [ ] Setup monitoring and alerting
- [ ] Configure log rotation
- [ ] Setup CDN for static files
- [ ] Enable Gzip compression

### Performance

- [ ] Setup database connection pooling
- [ ] Enable Redis caching
- [ ] Configure horizontal auto-scaling
- [ ] Optimize database indexes
- [ ] Enable query caching
- [ ] Setup load balancer health checks
- [ ] Configure CDN caching policies

### Monitoring

- [ ] Setup Prometheus metrics collection
- [ ] Configure Grafana dashboards
- [ ] Enable application performance monitoring (APM)
- [ ] Setup error tracking (Sentry)
- [ ] Configure uptime monitoring
- [ ] Setup log aggregation (ELK stack)
- [ ] Enable database query monitoring

---

## Monitoring & Maintenance

### Health Checks

Each product exposes a `/health` endpoint:

```bash
# Check all services
curl http://localhost:8000/health  # FedMed
curl http://localhost:8001/health  # SyntheticHealth
curl http://localhost:8002/health  # PrivacyGuard
curl http://localhost:8003/health  # FedTrain
curl http://localhost:8004/health  # MedImageQuality
curl http://localhost:8005/health  # DataSim
```

### Log Management

```bash
# View logs (Docker)
docker-compose logs -f fedmed
docker-compose logs -f synthetic_health

# View logs (systemd)
journalctl -u fedmed.service -f
```

### Database Backups

```bash
# Automated PostgreSQL backups
#!/bin/bash
pg_dump -U user -d fedmed_db > backup_$(date +%Y%m%d_%H%M%S).sql

# Restore
psql -U user -d fedmed_db < backup_20240101_120000.sql
```

### Performance Monitoring

```bash
# Prometheus metrics
curl http://localhost:8000/metrics
```

### Scaling

```bash
# Docker Swarm
docker service scale fedmed_web=5

# Kubernetes
kubectl scale deployment fedmed --replicas=5

# AWS ECS
aws ecs update-service --service fedmed --desired-count 5
```

---

## Troubleshooting

### Common Issues

**Database Connection Failed**:
```bash
# Check PostgreSQL is running
sudo systemctl status postgresql

# Check connection
psql -U user -d fedmed_db -h localhost
```

**Port Already in Use**:
```bash
# Find process using port
sudo lsof -i :8000

# Kill process
kill -9 <PID>
```

**Out of Memory**:
```bash
# Check memory usage
free -h

# Increase swap
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

---

## Support

For deployment issues:
- Documentation: See individual product READMEs
- GitHub Issues: [Repository Issues]
- Email: support@fedgan.ai

---

**Last Updated**: 2024
