# DataSim

Non-IID Data Distribution Simulator for Federated Learning Research

## Overview

DataSim helps researchers create realistic federated learning scenarios by generating non-IID (non-independent-identically-distributed) data splits. Perfect for academic research and benchmarking federated algorithms.

## Key Features

- **Non-IID Splits**: Label skew, feature shift, quantity skew
- **Multiple Strategies**: Dirichlet, pathological, practical
- **Visualization**: Interactive data distribution charts
- **Export Formats**: TFRecord, PyTorch, NumPy, CSV
- **Statistical Analysis**: KL divergence, Jensen-Shannon distance
- **Research-Focused**: Free for academic use

## Quick Start

```bash
cd saas_products/datasim
pip install -r requirements.txt
uvicorn app.main:app --port 8005
```

## API Usage

### Create Non-IID Splits

```bash
curl -X POST http://localhost:8005/api/v1/split \
  -H "Content-Type: application/json" \
  -d '{
    "dataset": "retinopathy",
    "num_clients": 5,
    "strategy": "dirichlet",
    "alpha": 0.5,
    "min_samples_per_client": 100
  }'
```

Response:
```json
{
  "split_id": "abc123",
  "num_clients": 5,
  "total_samples": 5000,
  "distribution": {
    "client_0": {"class_0": 120, "class_1": 80},
    "client_1": {"class_0": 50, "class_1": 200},
    ...
  },
  "heterogeneity_score": 0.67
}
```

### Analyze Distribution

```bash
curl -X GET http://localhost:8005/api/v1/analyze/{split_id}
```

## Distribution Strategies

### 1. Dirichlet Distribution
```python
# Controlled by alpha parameter
# alpha < 1: High heterogeneity
# alpha > 1: Low heterogeneity
{
  "strategy": "dirichlet",
  "alpha": 0.5
}
```

### 2. Pathological Non-IID
```python
# Each client gets only K classes
{
  "strategy": "pathological",
  "classes_per_client": 2
}
```

### 3. Practical Non-IID
```python
# Real-world hospital distributions
{
  "strategy": "practical",
  "institution_type": "hospital"
}
```

## Use Cases

- **Academic Research**: Benchmark FL algorithms
- **Algorithm Testing**: Test robustness to data heterogeneity
- **Education**: Teaching federated learning concepts
- **Simulation**: Pre-deployment testing

## Pricing

- **Academic**: FREE (with .edu email)
- **Commercial Research**: $49/month
- **Enterprise**: $199/month (unlimited splits)

## Supported Datasets

- Diabetic Retinopathy
- Chest X-ray (Pneumonia)
- Skin Lesions (HAM10000)
- Brain Tumors (MRI)
- Custom datasets (upload your own)

## Tech Stack

- FastAPI
- Pandas/NumPy
- Plotly (visualizations)
- TensorFlow (TFRecord export)

## Integration

```python
import requests

response = requests.post(
    "http://localhost:8005/api/v1/split",
    json={
        "dataset": "retinopathy",
        "num_clients": 3,
        "strategy": "dirichlet",
        "alpha": 0.3
    }
)

split_id = response.json()["split_id"]

# Download splits
for client_id in range(3):
    response = requests.get(
        f"http://localhost:8005/api/v1/download/{split_id}/{client_id}"
    )
    with open(f"client_{client_id}_data.tfrecord", "wb") as f:
        f.write(response.content)
```

---

**Simulate Real-World Federated Scenarios**
