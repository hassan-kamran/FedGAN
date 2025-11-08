"""
Aggregation strategies for federated learning.

This module implements the Strategy pattern for different weight
aggregation methods in federated learning.
"""
from abc import ABC, abstractmethod
from typing import List

import tensorflow as tf


class AggregationStrategy(ABC):
    """Abstract strategy for aggregating model weights.
    
    This implements the Strategy pattern, allowing different
    aggregation algorithms to be used interchangeably.
    """
    
    @abstractmethod
    def aggregate(self, client_weights: List[List[tf.Tensor]]) -> List[tf.Tensor]:
        """Aggregate weights from multiple clients.
        
        Args:
            client_weights: List of weight lists, one per client.
        
        Returns:
            Aggregated weights.
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Get the name of this aggregation strategy."""
        pass


class FedAvgStrategy(AggregationStrategy):
    """Federated Averaging (FedAvg) aggregation strategy.
    
    This is the standard federated learning aggregation method,
    which computes the arithmetic mean of client weights.
    
    Reference:
        McMahan et al. "Communication-Efficient Learning of Deep Networks
        from Decentralized Data" (2017)
    """
    
    @property
    def name(self) -> str:
        """Get strategy name."""
        return "fedavg"
    
    def aggregate(self, client_weights: List[List[tf.Tensor]]) -> List[tf.Tensor]:
        """Compute arithmetic mean of client weights.
        
        Args:
            client_weights: List of weight lists from clients.
        
        Returns:
            Averaged weights.
        """
        if not client_weights:
            raise ValueError("No client weights provided")
        
        num_layers = len(client_weights[0])
        aggregated_weights = []
        
        for layer_idx in range(num_layers):
            # Collect weights for this layer from all clients
            layer_weights = [
                client_w[layer_idx] for client_w in client_weights
            ]
            
            # Stack and compute mean
            stacked = tf.stack(layer_weights, axis=0)
            layer_mean = tf.reduce_mean(stacked, axis=0)
            
            aggregated_weights.append(layer_mean)
        
        return aggregated_weights


class WeightedFedAvgStrategy(AggregationStrategy):
    """Weighted Federated Averaging strategy.
    
    This strategy weights client contributions by their dataset sizes,
    giving more influence to clients with more data.
    
    Args:
        client_dataset_sizes: List of dataset sizes for each client.
    """
    
    def __init__(self, client_dataset_sizes: List[int]):
        if not client_dataset_sizes:
            raise ValueError("Dataset sizes required")
        
        self.dataset_sizes = client_dataset_sizes
        
        # Compute normalized weights
        total_size = sum(client_dataset_sizes)
        self.weights = [size / total_size for size in client_dataset_sizes]
    
    @property
    def name(self) -> str:
        """Get strategy name."""
        return "weighted_fedavg"
    
    def aggregate(self, client_weights: List[List[tf.Tensor]]) -> List[tf.Tensor]:
        """Compute weighted average of client weights.
        
        Args:
            client_weights: List of weight lists from clients.
        
        Returns:
            Weighted averaged weights.
        """
        if len(client_weights) != len(self.weights):
            raise ValueError(
                f"Expected {len(self.weights)} clients, got {len(client_weights)}"
            )
        
        num_layers = len(client_weights[0])
        aggregated_weights = []
        
        for layer_idx in range(num_layers):
            # Initialize weighted sum
            weighted_sum = None
            
            for client_idx, client_w in enumerate(client_weights):
                layer_weight = client_w[layer_idx]
                client_contribution = layer_weight * self.weights[client_idx]
                
                if weighted_sum is None:
                    weighted_sum = client_contribution
                else:
                    weighted_sum += client_contribution
            
            aggregated_weights.append(weighted_sum)
        
        return aggregated_weights


# Registry of aggregation strategies
AGGREGATION_STRATEGIES = {
    'fedavg': FedAvgStrategy,
    'weighted_fedavg': WeightedFedAvgStrategy,
}


def get_aggregation_strategy(
    strategy_name: str,
    **kwargs
) -> AggregationStrategy:
    """Get an aggregation strategy by name.
    
    Args:
        strategy_name: Name of the strategy ('fedavg', 'weighted_fedavg').
        **kwargs: Strategy-specific arguments.
    
    Returns:
        AggregationStrategy instance.
    
    Raises:
        ValueError: If strategy name is unknown.
    """
    strategy_class = AGGREGATION_STRATEGIES.get(strategy_name)
    
    if strategy_class is None:
        raise ValueError(
            f"Unknown aggregation strategy: {strategy_name}. "
            f"Available: {list(AGGREGATION_STRATEGIES.keys())}"
        )
    
    return strategy_class(**kwargs)
