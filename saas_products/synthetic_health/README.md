# SyntheticHealth

On-demand synthetic medical image generation API with pay-per-image pricing.

## Overview

SyntheticHealth provides a simple REST API for generating privacy-preserving synthetic medical images. Perfect for researchers, medical device companies, and AI developers who need training data without privacy concerns.

## Key Features

- **Pay-Per-Image Pricing**: Credit-based system with free tier
- **Multiple Modalities**: Retinal images, CT scans, X-rays (coming soon)
- **REST API**: Simple HTTP API for easy integration
- **Batch Generation**: Generate up to 100 images at once
- **S3 Integration**: Optional cloud storage for generated images
- **Fast Generation**: ~0.5s per image

## Quick Start

```bash
cd saas_products/synthetic_health
cp .env.example .env
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8001
```

Access at `http://localhost:8001`

## API Usage

### Generate Images

```bash
curl -X POST http://localhost:8001/api/v1/generate \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "dcgan",
    "num_images": 5,
    "image_size": 128
  }'
```

### Check Credits

```bash
curl -X GET http://localhost:8001/api/v1/credits \
  -H "X-API-Key: your-api-key"
```

### Download Image

```bash
curl -X GET http://localhost:8001/images/{image_id}/download \
  -H "X-API-Key: your-api-key" \
  -o image.png
```

## Pricing

- **Free Tier**: 10 credits (10 images)
- **Starter**: $10/month - 100 credits
- **Professional**: $50/month - 1000 credits
- **Enterprise**: Custom pricing

## Tech Stack

- FastAPI (async Python web framework)
- SQLite (lightweight database)
- TensorFlow (ML models)
- S3/MinIO (optional image storage)

## Documentation

Full API docs: `http://localhost:8001/api/docs`

---

**Built with Python & FastAPI**
