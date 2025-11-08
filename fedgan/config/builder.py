"""
Builder pattern implementation for constructing experiment configurations.

This module provides a fluent interface for building complex experiment
configurations programmatically.
"""
from typing import Any, Dict, List, Optional, Tuple

from fedgan.config.config import (
    DataConfig,
    ExperimentConfig,
    FederatedConfig,
    ModelConfig,
    PathConfig,
)


class ConfigBuilder:
    """Fluent interface builder for ExperimentConfig.
    
    This class implements the Builder pattern, allowing for step-by-step
    construction of complex configuration objects with a readable, chainable API.
    
    Example:
        >>> config = (ConfigBuilder()
        ...     .for_experiment("retinopathy_fedgan")
        ...     .with_description("Federated GAN on diabetic retinopathy data")
        ...     .with_model_config(latent_dim=200, learning_rate=0.0002)
        ...     .with_federated_config(num_clients=5, federated_rounds=10)
        ...     .with_seed(42)
        ...     .build())
    """
    
    def __init__(self) -> None:
        """Initialize builder with default configuration."""
        self._name: str = "unnamed_experiment"
        self._description: str = ""
        self._seed: Optional[int] = 42
        self._model_config: Optional[ModelConfig] = None
        self._data_config: Optional[DataConfig] = None
        self._federated_config: Optional[FederatedConfig] = None
        self._path_config: Optional[PathConfig] = None
        self._metadata: Dict[str, Any] = {}
    
    def for_experiment(self, name: str) -> "ConfigBuilder":
        """Set the experiment name.
        
        Args:
            name: Name of the experiment.
        
        Returns:
            Self for method chaining.
        """
        self._name = name
        return self
    
    def with_description(self, description: str) -> "ConfigBuilder":
        """Set the experiment description.
        
        Args:
            description: Description of the experiment.
        
        Returns:
            Self for method chaining.
        """
        self._description = description
        return self
    
    def with_seed(self, seed: int) -> "ConfigBuilder":
        """Set the random seed for reproducibility.
        
        Args:
            seed: Random seed value.
        
        Returns:
            Self for method chaining.
        """
        self._seed = seed
        return self
    
    def with_model_config(self, **kwargs: Any) -> "ConfigBuilder":
        """Configure model architecture parameters.
        
        Args:
            **kwargs: Model configuration parameters (see ModelConfig).
        
        Returns:
            Self for method chaining.
        """
        self._model_config = ModelConfig(**kwargs)
        return self
    
    def with_data_config(self, **kwargs: Any) -> "ConfigBuilder":
        """Configure data processing parameters.
        
        Args:
            **kwargs: Data configuration parameters (see DataConfig).
        
        Returns:
            Self for method chaining.
        """
        self._data_config = DataConfig(**kwargs)
        return self
    
    def with_federated_config(self, **kwargs: Any) -> "ConfigBuilder":
        """Configure federated learning parameters.
        
        Args:
            **kwargs: Federated config parameters (see FederatedConfig).
        
        Returns:
            Self for method chaining.
        """
        self._federated_config = FederatedConfig(**kwargs)
        return self
    
    def with_path_config(self, **kwargs: Any) -> "ConfigBuilder":
        """Configure file paths.
        
        Args:
            **kwargs: Path configuration parameters (see PathConfig).
        
        Returns:
            Self for method chaining.
        """
        self._path_config = PathConfig(**kwargs)
        return self
    
    def with_metadata(self, **kwargs: Any) -> "ConfigBuilder":
        """Add metadata to the configuration.
        
        Args:
            **kwargs: Key-value pairs to add to metadata.
        
        Returns:
            Self for method chaining.
        """
        self._metadata.update(kwargs)
        return self
    
    def with_latent_dim(self, latent_dim: int) -> "ConfigBuilder":
        """Convenience method to set latent dimension.
        
        Args:
            latent_dim: Latent vector dimensionality.
        
        Returns:
            Self for method chaining.
        """
        if self._model_config is None:
            self._model_config = ModelConfig()
        self._model_config.latent_dim = latent_dim
        return self
    
    def with_batch_size(self, batch_size: int) -> "ConfigBuilder":
        """Convenience method to set batch size.
        
        Args:
            batch_size: Training batch size.
        
        Returns:
            Self for method chaining.
        """
        if self._data_config is None:
            self._data_config = DataConfig()
        self._data_config.batch_size = batch_size
        return self
    
    def with_num_clients(self, num_clients: int) -> "ConfigBuilder":
        """Convenience method to set number of clients.
        
        Args:
            num_clients: Number of federated clients.
        
        Returns:
            Self for method chaining.
        """
        if self._federated_config is None:
            self._federated_config = FederatedConfig()
        self._federated_config.num_clients = num_clients
        return self
    
    def with_learning_rate(self, learning_rate: float) -> "ConfigBuilder":
        """Convenience method to set learning rate.
        
        Args:
            learning_rate: Learning rate for optimizer.
        
        Returns:
            Self for method chaining.
        """
        if self._model_config is None:
            self._model_config = ModelConfig()
        self._model_config.learning_rate = learning_rate
        return self
    
    def build(self) -> ExperimentConfig:
        """Build and validate the final configuration.
        
        Returns:
            Validated ExperimentConfig instance.
        
        Raises:
            ValueError: If configuration validation fails.
        """
        # Use provided configs or defaults
        model_config = self._model_config or ModelConfig()
        data_config = self._data_config or DataConfig()
        federated_config = self._federated_config or FederatedConfig()
        path_config = self._path_config or PathConfig()
        
        # Create experiment config
        config = ExperimentConfig(
            name=self._name,
            description=self._description,
            seed=self._seed,
            model=model_config,
            data=data_config,
            federated=federated_config,
            paths=path_config,
            metadata=self._metadata,
        )
        
        # Validate before returning
        config.validate()
        
        return config
    
    def reset(self) -> "ConfigBuilder":
        """Reset builder to initial state.
        
        Returns:
            Self for method chaining.
        """
        self.__init__()
        return self


class PresetConfigBuilder:
    """Factory for creating commonly used configuration presets.
    
    This class provides pre-configured builders for typical use cases,
    reducing boilerplate code.
    """
    
    @staticmethod
    def for_quick_test() -> ConfigBuilder:
        """Create a minimal config for quick testing.
        
        Returns:
            ConfigBuilder with test-friendly defaults.
        """
        return (
            ConfigBuilder()
            .for_experiment("quick_test")
            .with_description("Quick test configuration")
            .with_data_config(image_size=64, batch_size=4)
            .with_model_config(latent_dim=100)
            .with_federated_config(
                num_clients=2, local_epochs=1, federated_rounds=1
            )
        )
    
    @staticmethod
    def for_federated_gan(
        num_clients: int = 5, rounds: int = 10
    ) -> ConfigBuilder:
        """Create a standard federated GAN configuration.
        
        Args:
            num_clients: Number of federated clients.
            rounds: Number of federated rounds.
        
        Returns:
            ConfigBuilder with federated GAN defaults.
        """
        return (
            ConfigBuilder()
            .for_experiment(f"fedgan_{num_clients}clients_{rounds}rounds")
            .with_description(
                f"Federated GAN with {num_clients} clients, {rounds} rounds"
            )
            .with_data_config(image_size=128, batch_size=16)
            .with_model_config(latent_dim=200, learning_rate=0.0002)
            .with_federated_config(
                num_clients=num_clients,
                local_epochs=5,
                federated_rounds=rounds,
                aggregation_strategy="fedavg",
            )
        )
    
    @staticmethod
    def for_medical_imaging() -> ConfigBuilder:
        """Create a configuration for medical imaging experiments.
        
        Returns:
            ConfigBuilder with medical imaging defaults.
        """
        return (
            ConfigBuilder()
            .for_experiment("medical_imaging_fedgan")
            .with_description("Federated GAN for medical image generation")
            .with_data_config(
                image_size=128,
                channels=1,  # Grayscale
                batch_size=16,
                clip_limit=2.0,
                gamma=1.5,
            )
            .with_model_config(
                latent_dim=200,
                learning_rate=0.0002,
                use_batch_norm=True,
            )
            .with_federated_config(
                num_clients=5,
                local_epochs=5,
                federated_rounds=10,
            )
        )
