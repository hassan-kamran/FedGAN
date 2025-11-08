# MedImageQuality

Automated medical image quality evaluation API using FID, IS, and custom metrics.

## Overview

MedImageQuality provides instant quality metrics for synthetic or real medical images using industry-standard metrics like FID (Fréchet Inception Distance) and IS (Inception Score).

## Key Features

- **FID Score Calculation**: Measure image quality against reference dataset
- **Inception Score**: Evaluate image realism
- **Batch Evaluation**: Process 1000+ images at once
- **Custom Metrics**: Domain-specific quality measures
- **API-First**: RESTful API for easy integration
- **Fast**: < 100ms per image

## Quick Start

```bash
cd saas_products/medimage_quality
pip install -r requirements.txt
uvicorn app.main:app --port 8004
```

## API Usage

### Calculate FID Score

```bash
curl -X POST http://localhost:8004/api/v1/fid \
  -H "X-API-Key: your-api-key" \
  -F "real_images=@real_dataset.zip" \
  -F "generated_images=@generated_dataset.zip"
```

Response:
```json
{
  "fid_score": 45.23,
  "quality_rating": "good",
  "num_real_images": 1000,
  "num_generated_images": 500
}
```

### Calculate Inception Score

```bash
curl -X POST http://localhost:8004/api/v1/inception_score \
  -F "images=@images.zip"
```

## Metrics

- **FID Score**: Lower is better (< 50 is excellent)
- **Inception Score**: Higher is better (> 5 is good)
- **SSIM**: Structural similarity (0-1)
- **PSNR**: Peak signal-to-noise ratio

## Pricing

- **Free**: 100 evaluations/month
- **Starter**: $19/month - 1000 evaluations
- **Pro**: $99/month - 10000 evaluations
- **Enterprise**: Custom pricing

## Integration

```python
import requests

response = requests.post(
    "http://localhost:8004/api/v1/fid",
    headers={"X-API-Key": "your-key"},
    files={
        "real_images": open("real.zip", "rb"),
        "generated_images": open("gen.zip", "rb")
    }
)

fid_score = response.json()["fid_score"]
print(f"FID Score: {fid_score}")
```

## Tech Stack

- FastAPI
- TensorFlow/InceptionV3
- NumPy/SciPy
- PIL/Pillow

---

**Measure Image Quality in Seconds**
