"""
High-level data loaders for federated learning scenarios.

This module provides Facade pattern interfaces that simplify
loading federated data splits for training and evaluation.
"""
from pathlib import Path
from typing import Dict, List, Optional

import tensorflow as tf

from fedgan.config import ExperimentConfig
from fedgan.data.factory import DatasetFactory
from fedgan.data.parsers import ParserFactory


class FederatedDataLoader:
    """Facade for loading federated data splits.
    
    This class provides a simple interface for loading client datasets
    in federated learning scenarios, hiding the complexity of dataset
    creation and configuration.
    
    Args:
        config: Experiment configuration.
        parser_type: Type of parser to use ('unlabeled', 'labeled', 'flexible').
    
    Example:
        >>> from fedgan.config import ExperimentConfig
        >>> config = ExperimentConfig.from_yaml("config.yaml")
        >>> loader = FederatedDataLoader(config)
        >>> client_datasets = loader.load_client_datasets(num_clients=5)
        >>> train_data = client_datasets[0]  # Get client 0's data
    """
    
    def __init__(
        self,
        config: ExperimentConfig,
        parser_type: str = 'unlabeled'
    ):
        self.config = config
        self.parser_type = parser_type
        
        # Create parser
        self.parser = ParserFactory.create_parser(
            parser_type,
            image_size=config.data.image_size,
            channels=config.data.channels
        )
        
        # Create factory
        self.factory = DatasetFactory(config.data, self.parser)
    
    def load_client_datasets(
        self,
        num_clients: Optional[int] = None,
        split: str = 'train'
    ) -> Dict[int, tf.data.Dataset]:
        """Load datasets for all federated clients.
        
        Args:
            num_clients: Number of clients (uses config if None).
            split: Data split ('train', 'val', 'test').
        
        Returns:
            Dictionary mapping client_id to tf.data.Dataset.
        
        Raises:
            FileNotFoundError: If client data files don't exist.
        """
        num_clients = num_clients or self.config.federated.num_clients
        
        # Construct base path for non-IID splits
        base_path = (
            self.config.paths.data_dir
            if 'non_iid' not in str(self.config.paths.data_dir)
            else self.config.paths.data_dir.parent / f"non_iid_clusters_{num_clients}"
        )
        
        client_datasets = {}
        
        for client_id in range(num_clients):
            # Construct path to client's TFRecord
            tfrecord_path = base_path / f"client_{client_id}_{split}.tfrecord"
            
            if not tfrecord_path.exists():
                raise FileNotFoundError(
                    f"Client {client_id} {split} data not found: {tfrecord_path}"
                )
            
            # Create dataset for this client
            dataset = self.factory.create_training_dataset([str(tfrecord_path)])
            client_datasets[client_id] = dataset
        
        return client_datasets
    
    def load_client_dataset(
        self,
        client_id: int,
        num_clients: Optional[int] = None,
        split: str = 'train'
    ) -> tf.data.Dataset:
        """Load dataset for a specific client.
        
        Args:
            client_id: ID of the client.
            num_clients: Total number of clients.
            split: Data split ('train', 'val', 'test').
        
        Returns:
            tf.data.Dataset for the specified client.
        """
        all_datasets = self.load_client_datasets(num_clients, split)
        return all_datasets[client_id]
    
    def load_centralized_dataset(
        self,
        split: str = 'train'
    ) -> tf.data.Dataset:
        """Load centralized (non-federated) dataset.
        
        Args:
            split: Data split ('train', 'val', 'test').
        
        Returns:
            tf.data.Dataset with all data combined.
        """
        data_path = self.config.paths.data_dir / f"{split}.tfrecord"
        
        if not data_path.exists():
            # Try alternative naming
            data_path = self.config.paths.data_dir / f"{split}_data.tfrecord"
        
        if not data_path.exists():
            raise FileNotFoundError(f"Centralized {split} data not found")
        
        if split == 'train':
            return self.factory.create_training_dataset([str(data_path)])
        else:
            return self.factory.create_evaluation_dataset([str(data_path)])
    
    def load_validation_dataset(self) -> tf.data.Dataset:
        """Load validation dataset.
        
        Returns:
            tf.data.Dataset for validation.
        """
        return self.load_centralized_dataset(split='val')
    
    def load_test_dataset(self) -> tf.data.Dataset:
        """Load test dataset.
        
        Returns:
            tf.data.Dataset for testing.
        """
        return self.load_centralized_dataset(split='test')
    
    def get_client_data_info(
        self,
        num_clients: Optional[int] = None
    ) -> Dict[int, dict]:
        """Get information about each client's data.
        
        Args:
            num_clients: Number of clients.
        
        Returns:
            Dictionary mapping client_id to data statistics.
        """
        num_clients = num_clients or self.config.federated.num_clients
        client_info = {}
        
        for client_id in range(num_clients):
            base_path = self.config.paths.data_dir.parent / f"non_iid_clusters_{num_clients}"
            tfrecord_path = base_path / f"client_{client_id}_train.tfrecord"
            
            if tfrecord_path.exists():
                info = self.factory.get_dataset_info([str(tfrecord_path)])
                client_info[client_id] = info
        
        return client_info


class DataLoaderBuilder:
    """Builder for constructing data loaders with custom configuration.
    
    This provides a fluent interface for creating data loaders.
    
    Example:
        >>> loader = (DataLoaderBuilder()
        ...     .with_config(config)
        ...     .with_parser('labeled')
        ...     .build())
    """
    
    def __init__(self):
        self._config: Optional[ExperimentConfig] = None
        self._parser_type: str = 'unlabeled'
    
    def with_config(self, config: ExperimentConfig) -> 'DataLoaderBuilder':
        """Set the experiment configuration.
        
        Args:
            config: Experiment configuration.
        
        Returns:
            Self for chaining.
        """
        self._config = config
        return self
    
    def with_parser(self, parser_type: str) -> 'DataLoaderBuilder':
        """Set the parser type.
        
        Args:
            parser_type: Parser type ('unlabeled', 'labeled', 'flexible').
        
        Returns:
            Self for chaining.
        """
        self._parser_type = parser_type
        return self
    
    def build(self) -> FederatedDataLoader:
        """Build the data loader.
        
        Returns:
            Configured FederatedDataLoader.
        
        Raises:
            ValueError: If configuration is not set.
        """
        if self._config is None:
            raise ValueError("Configuration must be set before building")
        
        return FederatedDataLoader(self._config, self._parser_type)


class DatasetSplitter:
    """Utility for splitting datasets for federated learning.
    
    This class helps create non-IID splits from a centralized dataset.
    """
    
    @staticmethod
    def create_non_iid_splits(
        dataset: tf.data.Dataset,
        num_clients: int,
        strategy: str = 'label_based'
    ) -> List[tf.data.Dataset]:
        """Create non-IID data splits for federated learning.
        
        Args:
            dataset: Source dataset to split.
            num_clients: Number of clients to split data among.
            strategy: Splitting strategy ('label_based', 'quantity_based').
        
        Returns:
            List of datasets, one per client.
        
        Note:
            This is a placeholder for future implementation.
            The actual split logic from create_non_iid_splits.py
            will be migrated here.
        """
        # TODO: Implement splitting logic
        raise NotImplementedError(
            "Dataset splitting will be implemented in migration phase"
        )
