"""
Federated learning trainer.

This module orchestrates federated learning by coordinating
local training on clients and aggregating updates.
"""
from typing import Dict, Optional

import tensorflow as tf

from fedgan.config import ExperimentConfig
from fedgan.federated.aggregation import AggregationStrategy, get_aggregation_strategy
from fedgan.models.factory import ModelFactory
from fedgan.training.base_trainer import GANTrainer


class FederatedTrainer:
    """Orchestrates federated learning across multiple clients.
    
    This class coordinates:
    1. Distribution of global model to clients
    2. Local training on each client
    3. Aggregation of client updates
    4. Update of global model
    
    Args:
        config: Experiment configuration.
        model_factory: Factory for creating models.
        aggregation_strategy: Strategy for aggregating weights (optional).
    
    Example:
        >>> config = ExperimentConfig.from_yaml("config.yaml")
        >>> factory = get_model_factory('dcgan', config.model)
        >>> trainer = FederatedTrainer(config, factory)
        >>> history = trainer.train_federated(client_datasets, rounds=10)
    """
    
    def __init__(
        self,
        config: ExperimentConfig,
        model_factory: ModelFactory,
        aggregation_strategy: Optional[AggregationStrategy] = None
    ):
        self.config = config
        self.model_factory = model_factory
        
        # Set up aggregation strategy
        if aggregation_strategy is None:
            strategy_name = config.federated.aggregation_strategy
            self.aggregation_strategy = get_aggregation_strategy(strategy_name)
        else:
            self.aggregation_strategy = aggregation_strategy
        
        # Create global models
        self.global_generator = model_factory.create_generator()
        self.global_discriminator = model_factory.create_discriminator()
        
        self.round = 0
    
    def train_federated_round(
        self,
        client_datasets: Dict[int, tf.data.Dataset]
    ) -> Dict[str, any]:
        """Execute one round of federated learning.
        
        Args:
            client_datasets: Dictionary mapping client ID to dataset.
        
        Returns:
            Dictionary of metrics from this round.
        """
        client_gen_weights = []
        client_disc_weights = []
        client_metrics = {}
        
        # Local training on each client
        for client_id, dataset in client_datasets.items():
            # Create local trainer with global weights
            local_trainer = self._create_local_trainer(client_id)
            
            # Train locally
            history = local_trainer.train(
                dataset,
                epochs=self.config.federated.local_epochs
            )
            
            # Collect updated weights
            client_gen_weights.append(local_trainer.generator.get_weights())
            client_disc_weights.append(local_trainer.discriminator.get_weights())
            client_metrics[client_id] = history
        
        # Aggregate weights
        aggregated_gen = self.aggregation_strategy.aggregate(client_gen_weights)
        aggregated_disc = self.aggregation_strategy.aggregate(client_disc_weights)
        
        # Update global models
        self.global_generator.set_weights(aggregated_gen)
        self.global_discriminator.set_weights(aggregated_disc)
        
        self.round += 1
        
        return {
            'round': self.round,
            'num_clients': len(client_datasets),
            'client_metrics': client_metrics,
            'aggregation_strategy': self.aggregation_strategy.name
        }
    
    def train_federated(
        self,
        client_datasets: Dict[int, tf.data.Dataset],
        rounds: Optional[int] = None
    ) -> Dict[str, any]:
        """Train for multiple federated rounds.
        
        Args:
            client_datasets: Dictionary mapping client ID to dataset.
            rounds: Number of federated rounds (uses config if None).
        
        Returns:
            Dictionary of training history.
        """
        rounds = rounds or self.config.federated.federated_rounds
        history = {'rounds': []}
        
        for r in range(rounds):
            round_metrics = self.train_federated_round(client_datasets)
            history['rounds'].append(round_metrics)
            
            # Log progress
            print(f"Federated Round {r+1}/{rounds} complete. "
                  f"Clients: {round_metrics['num_clients']}")
        
        return history
    
    def _create_local_trainer(self, client_id: int) -> GANTrainer:
        """Create a local trainer for a client.
        
        Args:
            client_id: ID of the client.
        
        Returns:
            GANTrainer initialized with global model weights.
        """
        # Create fresh models
        local_gen = self.model_factory.create_generator()
        local_disc = self.model_factory.create_discriminator()
        
        # Initialize with global weights
        local_gen.set_weights(self.global_generator.get_weights())
        local_disc.set_weights(self.global_discriminator.get_weights())
        
        # Create trainer
        trainer = GANTrainer(self.config, local_gen, local_disc)
        trainer.steps_per_epoch = 100  # Limit steps per epoch
        
        return trainer
    
    def get_global_models(self) -> Dict[str, tf.keras.Model]:
        """Get the current global models.
        
        Returns:
            Dictionary with 'generator' and 'discriminator'.
        """
        return {
            'generator': self.global_generator,
            'discriminator': self.global_discriminator
        }
    
    def save_global_models(self, save_dir: str) -> None:
        """Save global models to disk.
        
        Args:
            save_dir: Directory to save models.
        """
        from pathlib import Path
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        self.global_generator.save(save_path / "global_generator.keras")
        self.global_discriminator.save(save_path / "global_discriminator.keras")
