# FedTrain Express

Simplified federated training-as-a-service for small clinics and research labs.

## Overview

FedTrain Express makes federated learning accessible to non-technical users with pre-configured templates and one-click setup. No PhD required!

## Key Features

- **One-Click Setup**: Pre-configured federated training templates
- **Simple UI**: No code required, just upload data and click train
- **Freemium Model**: Free for up to 3 clients
- **Pre-trained Models**: Start from pre-trained weights
- **Auto-tuning**: Automatic hyperparameter optimization
- **Docker Clients**: Easy deployment with Docker

## Quick Start

```bash
cd saas_products/fedtrain_express
docker-compose up -d
```

Access at `http://localhost:8003`

## Use Cases

- **Small Clinics**: 2-3 clinics collaborating on disease detection
- **Research Labs**: Academic researchers without ML expertise
- **Startups**: MVP federated learning for product demos
- **Education**: Teaching federated learning concepts

## Pricing

- **Free Tier**: Up to 3 clients, 10 rounds/month
- **Starter**: $29/month - 5 clients, unlimited rounds
- **Professional**: $99/month - 10 clients, priority support
- **Enterprise**: Custom - 10+ clients, SLA

## Templates

- Diabetic retinopathy detection
- Pneumonia detection (chest X-ray)
- Skin lesion classification
- Brain tumor segmentation

## Tech Stack

- FastAPI + HTMX
- SQLite (simple deployment)
- Docker (client deployment)
- TensorFlow

---

**Federated Learning Made Simple**
