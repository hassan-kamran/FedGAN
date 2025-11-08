"""
Configuration management for FedGAN experiments.

This module provides dataclasses for managing all configuration aspects
of federated GAN experiments including model architecture, data processing,
federated learning parameters, and experimental setup.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


@dataclass
class ModelConfig:
    """Configuration for GAN model architecture.
    
    Attributes:
        latent_dim: Dimensionality of the latent noise vector.
        learning_rate: Learning rate for Adam optimizer.
        beta_1: Beta1 parameter for Adam optimizer.
        beta_2: Beta2 parameter for Adam optimizer.
        generator_filters: List of filter counts for generator layers.
        discriminator_filters: List of filter counts for discriminator layers.
        use_batch_norm: Whether to use batch normalization.
        batch_norm_momentum: Momentum for batch normalization layers.
        activation: Activation function name (e.g., 'relu', 'leaky_relu').
        leaky_relu_alpha: Alpha parameter for LeakyReLU activation.
    """
    
    latent_dim: int = 200
    learning_rate: float = 0.0002
    beta_1: float = 0.5
    beta_2: float = 0.999
    generator_filters: List[int] = field(
        default_factory=lambda: [1024, 512, 256, 128, 64, 32]
    )
    discriminator_filters: List[int] = field(
        default_factory=lambda: [64, 128, 256, 512, 1024]
    )
    use_batch_norm: bool = True
    batch_norm_momentum: float = 0.1
    activation: str = "leaky_relu"
    leaky_relu_alpha: float = 0.2
    
    def validate(self) -> None:
        """Validate configuration parameters.
        
        Raises:
            ValueError: If any configuration parameter is invalid.
        """
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if not 0 < self.learning_rate < 1:
            raise ValueError("learning_rate must be between 0 and 1")
        if not 0 < self.beta_1 < 1:
            raise ValueError("beta_1 must be between 0 and 1")
        if not 0 < self.beta_2 < 1:
            raise ValueError("beta_2 must be between 0 and 1")
        if len(self.generator_filters) == 0:
            raise ValueError("generator_filters cannot be empty")
        if len(self.discriminator_filters) == 0:
            raise ValueError("discriminator_filters cannot be empty")


@dataclass
class DataConfig:
    """Configuration for data preprocessing and loading.
    
    Attributes:
        image_size: Size of input images (assumed square).
        channels: Number of image channels (1 for grayscale, 3 for RGB).
        batch_size: Batch size for training.
        clip_limit: Clip limit for CLAHE preprocessing.
        tile_grid_size: Tile grid size for CLAHE preprocessing.
        gamma: Gamma value for gamma correction.
        bin_size: Number of bins for pixel binning.
        normalization_range: Tuple of (min, max) for normalization.
        shuffle_buffer_size: Buffer size for dataset shuffling.
        prefetch_size: Number of batches to prefetch.
    """
    
    image_size: int = 128
    channels: int = 1
    batch_size: int = 16
    clip_limit: float = 2.0
    tile_grid_size: Tuple[int, int] = (8, 8)
    gamma: float = 1.5
    bin_size: int = 16
    normalization_range: Tuple[float, float] = (-1.0, 1.0)
    shuffle_buffer_size: int = 1000
    prefetch_size: int = 2
    
    def validate(self) -> None:
        """Validate configuration parameters.
        
        Raises:
            ValueError: If any configuration parameter is invalid.
        """
        if self.image_size % 16 != 0:
            raise ValueError("image_size must be divisible by 16")
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")
        if self.channels not in [1, 3]:
            raise ValueError("channels must be 1 (grayscale) or 3 (RGB)")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.clip_limit <= 0:
            raise ValueError("clip_limit must be positive")
        if self.gamma <= 0:
            raise ValueError("gamma must be positive")
        if self.bin_size <= 0:
            raise ValueError("bin_size must be positive")


@dataclass
class FederatedConfig:
    """Configuration for federated learning.
    
    Attributes:
        num_clients: Number of clients participating in federated learning.
        local_epochs: Number of local training epochs per federated round.
        federated_rounds: Total number of federated rounds.
        aggregation_strategy: Strategy for aggregating client weights.
        client_selection_strategy: Strategy for selecting clients each round.
        client_selection_fraction: Fraction of clients to select per round.
        min_clients_per_round: Minimum number of clients required per round.
        use_differential_privacy: Whether to use differential privacy.
        dp_noise_multiplier: Noise multiplier for differential privacy.
        dp_l2_norm_clip: L2 norm clipping threshold for DP.
    """
    
    num_clients: int = 5
    local_epochs: int = 5
    federated_rounds: int = 2
    aggregation_strategy: str = "fedavg"
    client_selection_strategy: str = "all"
    client_selection_fraction: float = 1.0
    min_clients_per_round: int = 2
    use_differential_privacy: bool = False
    dp_noise_multiplier: float = 0.1
    dp_l2_norm_clip: float = 1.0
    
    def validate(self) -> None:
        """Validate configuration parameters.
        
        Raises:
            ValueError: If any configuration parameter is invalid.
        """
        if self.num_clients <= 0:
            raise ValueError("num_clients must be positive")
        if self.local_epochs <= 0:
            raise ValueError("local_epochs must be positive")
        if self.federated_rounds <= 0:
            raise ValueError("federated_rounds must be positive")
        if self.aggregation_strategy not in ["fedavg", "weighted_fedavg"]:
            raise ValueError(
                f"Unknown aggregation strategy: {self.aggregation_strategy}"
            )
        if not 0 < self.client_selection_fraction <= 1:
            raise ValueError("client_selection_fraction must be between 0 and 1")
        if self.min_clients_per_round <= 0:
            raise ValueError("min_clients_per_round must be positive")


@dataclass
class PathConfig:
    """Configuration for file paths.
    
    Attributes:
        data_dir: Directory containing training data.
        model_dir: Directory for saving models.
        log_dir: Directory for logs and TensorBoard files.
        output_dir: Directory for experiment outputs.
        checkpoint_dir: Directory for model checkpoints.
    """
    
    data_dir: Path = field(default_factory=lambda: Path("data"))
    model_dir: Path = field(default_factory=lambda: Path("models"))
    log_dir: Path = field(default_factory=lambda: Path("logs"))
    output_dir: Path = field(default_factory=lambda: Path("outputs"))
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints"))
    
    def __post_init__(self) -> None:
        """Convert string paths to Path objects."""
        self.data_dir = Path(self.data_dir)
        self.model_dir = Path(self.model_dir)
        self.log_dir = Path(self.log_dir)
        self.output_dir = Path(self.output_dir)
        self.checkpoint_dir = Path(self.checkpoint_dir)
    
    def create_directories(self) -> None:
        """Create all configured directories if they don't exist."""
        for path in [
            self.data_dir,
            self.model_dir,
            self.log_dir,
            self.output_dir,
            self.checkpoint_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)


@dataclass
class ExperimentConfig:
    """Root configuration for a FedGAN experiment.
    
    Attributes:
        name: Name of the experiment.
        description: Description of the experiment.
        seed: Random seed for reproducibility.
        model: Model architecture configuration.
        data: Data processing configuration.
        federated: Federated learning configuration.
        paths: File path configuration.
        metadata: Additional metadata (for extensibility).
    """
    
    name: str
    description: str = ""
    seed: Optional[int] = 42
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    federated: FederatedConfig = field(default_factory=FederatedConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> None:
        """Validate all configuration components.
        
        Raises:
            ValueError: If any configuration is invalid.
        """
        if not self.name:
            raise ValueError("Experiment name cannot be empty")
        
        self.model.validate()
        self.data.validate()
        self.federated.validate()
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ExperimentConfig":
        """Load configuration from a YAML file.
        
        Args:
            yaml_path: Path to YAML configuration file.
        
        Returns:
            ExperimentConfig instance loaded from YAML.
        
        Raises:
            FileNotFoundError: If YAML file doesn't exist.
            ValueError: If YAML is malformed or invalid.
        """
        yaml_file = Path(yaml_path)
        if not yaml_file.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")
        
        with open(yaml_file, "r") as f:
            config_dict = yaml.safe_load(f)
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ExperimentConfig":
        """Create configuration from a dictionary.
        
        Args:
            config_dict: Dictionary containing configuration.
        
        Returns:
            ExperimentConfig instance.
        """
        # Extract nested configs
        model_dict = config_dict.pop("model", {})
        data_dict = config_dict.pop("data", {})
        federated_dict = config_dict.pop("federated", {})
        paths_dict = config_dict.pop("paths", {})
        
        # Create config objects
        config = cls(
            name=config_dict.get("name", "unnamed"),
            description=config_dict.get("description", ""),
            seed=config_dict.get("seed", 42),
            model=ModelConfig(**model_dict),
            data=DataConfig(**data_dict),
            federated=FederatedConfig(**federated_dict),
            paths=PathConfig(**paths_dict),
            metadata=config_dict.get("metadata", {}),
        )
        
        config.validate()
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of configuration.
        """
        return {
            "name": self.name,
            "description": self.description,
            "seed": self.seed,
            "model": {
                "latent_dim": self.model.latent_dim,
                "learning_rate": self.model.learning_rate,
                "beta_1": self.model.beta_1,
                "beta_2": self.model.beta_2,
                "generator_filters": self.model.generator_filters,
                "discriminator_filters": self.model.discriminator_filters,
                "use_batch_norm": self.model.use_batch_norm,
                "batch_norm_momentum": self.model.batch_norm_momentum,
            },
            "data": {
                "image_size": self.data.image_size,
                "channels": self.data.channels,
                "batch_size": self.data.batch_size,
                "clip_limit": self.data.clip_limit,
                "tile_grid_size": list(self.data.tile_grid_size),
                "gamma": self.data.gamma,
                "bin_size": self.data.bin_size,
            },
            "federated": {
                "num_clients": self.federated.num_clients,
                "local_epochs": self.federated.local_epochs,
                "federated_rounds": self.federated.federated_rounds,
                "aggregation_strategy": self.federated.aggregation_strategy,
            },
            "paths": {
                "data_dir": str(self.paths.data_dir),
                "model_dir": str(self.paths.model_dir),
                "log_dir": str(self.paths.log_dir),
                "output_dir": str(self.paths.output_dir),
            },
            "metadata": self.metadata,
        }
    
    def to_yaml(self, yaml_path: str) -> None:
        """Save configuration to YAML file.
        
        Args:
            yaml_path: Path where to save YAML file.
        """
        with open(yaml_path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
