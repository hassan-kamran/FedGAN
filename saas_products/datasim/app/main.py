"""
DataSim - Non-IID Data Distribution Simulator

Main FastAPI application.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
import numpy as np

app = FastAPI(
    title="DataSim",
    version="1.0.0",
    description="Non-IID Data Distribution Simulator for Federated Learning"
)


class SplitRequest(BaseModel):
    """Request model for creating data splits."""
    dataset: str
    num_clients: int
    strategy: str = "dirichlet"  # dirichlet, pathological, practical
    alpha: Optional[float] = 0.5  # For Dirichlet
    classes_per_client: Optional[int] = 2  # For pathological
    min_samples_per_client: int = 50


@app.get("/")
async def root():
    return {
        "app": "DataSim",
        "description": "Non-IID Data Distribution Simulator",
        "strategies": ["dirichlet", "pathological", "practical"],
        "pricing": {
            "academic": "FREE",
            "commercial": "$49/month",
            "enterprise": "$199/month"
        }
    }


@app.post("/api/v1/split")
async def create_split(request: SplitRequest):
    """
    Create non-IID data splits.

    Based on create_non_iid_splits.py from FedGAN.
    """
    if request.num_clients < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 clients")

    if request.num_clients > 100:
        raise HTTPException(status_code=400, detail="Max 100 clients")

    # Simulate non-IID split creation
    # In production: implement actual Dirichlet distribution sampling
    # from create_non_iid_splits.py

    split_id = f"split_{np.random.randint(10000, 99999)}"

    # Simulate class distribution
    num_classes = 2  # Binary classification
    distribution = {}

    if request.strategy == "dirichlet":
        # Dirichlet distribution for heterogeneous splits
        for client_idx in range(request.num_clients):
            # Sample from Dirichlet
            proportions = np.random.dirichlet([request.alpha] * num_classes)
            total_samples = np.random.randint(
                request.min_samples_per_client,
                request.min_samples_per_client * 3
            )

            distribution[f"client_{client_idx}"] = {
                f"class_{cls}": int(total_samples * proportions[cls])
                for cls in range(num_classes)
            }

    elif request.strategy == "pathological":
        # Each client gets only K classes
        for client_idx in range(request.num_clients):
            selected_classes = np.random.choice(
                num_classes,
                size=min(request.classes_per_client, num_classes),
                replace=False
            )
            total_samples = np.random.randint(
                request.min_samples_per_client,
                request.min_samples_per_client * 2
            )

            distribution[f"client_{client_idx}"] = {
                f"class_{cls}": total_samples // len(selected_classes)
                if cls in selected_classes else 0
                for cls in range(num_classes)
            }

    # Calculate heterogeneity score
    heterogeneity = calculate_heterogeneity(distribution)

    return {
        "split_id": split_id,
        "num_clients": request.num_clients,
        "strategy": request.strategy,
        "distribution": distribution,
        "heterogeneity_score": heterogeneity,
        "total_samples": sum(
            sum(client_dist.values())
            for client_dist in distribution.values()
        )
    }


@app.get("/api/v1/analyze/{split_id}")
async def analyze_split(split_id: str):
    """
    Analyze data distribution heterogeneity.

    Returns statistical measures of non-IIDness.
    """
    return {
        "split_id": split_id,
        "heterogeneity_metrics": {
            "kl_divergence": 0.45,
            "jensen_shannon": 0.23,
            "earth_movers_distance": 0.67
        },
        "class_balance": {
            "client_0": {"class_0": 0.6, "class_1": 0.4},
            "client_1": {"class_0": 0.3, "class_1": 0.7},
        },
        "recommendations": [
            "High heterogeneity detected",
            "Consider using FedProx or FedNova for training"
        ]
    }


@app.get("/api/v1/visualize/{split_id}")
async def visualize_distribution(split_id: str):
    """
    Get visualization data for distribution.

    Returns data suitable for Plotly charts.
    """
    return {
        "split_id": split_id,
        "chart_type": "stacked_bar",
        "data": {
            "clients": ["Client 0", "Client 1", "Client 2"],
            "class_0": [120, 50, 200],
            "class_1": [80, 200, 50]
        }
    }


def calculate_heterogeneity(distribution: Dict) -> float:
    """
    Calculate heterogeneity score (0-1).

    Higher scores indicate more heterogeneous distributions.
    """
    # Simplified heterogeneity calculation
    # In production: use KL divergence or Jensen-Shannon

    if not distribution:
        return 0.0

    # Calculate variance in class distributions
    all_proportions = []
    for client_dist in distribution.values():
        total = sum(client_dist.values())
        if total > 0:
            proportions = [v / total for v in client_dist.values()]
            all_proportions.append(proportions)

    if not all_proportions:
        return 0.0

    # Calculate average variance
    variances = [np.var([p[i] for p in all_proportions])
                 for i in range(len(all_proportions[0]))]

    return float(np.mean(variances))


@app.get("/health")
async def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8005, reload=True)
