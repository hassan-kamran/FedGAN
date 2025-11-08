"""Federated learning components."""

from fedgan.federated.aggregation import (
    AggregationStrategy,
    FedAvgStrategy,
    WeightedFedAvgStrategy,
    get_aggregation_strategy,
)
from fedgan.federated.trainer import FederatedTrainer

__all__ = [
    # Aggregation
    "AggregationStrategy",
    "FedAvgStrategy",
    "WeightedFedAvgStrategy",
    "get_aggregation_strategy",
    # Trainer
    "FederatedTrainer",
]
