"""Configuration management for FedGAN experiments."""

from fedgan.config.builder import ConfigBuilder, PresetConfigBuilder
from fedgan.config.config import (
    DataConfig,
    ExperimentConfig,
    FederatedConfig,
    ModelConfig,
    PathConfig,
)
from fedgan.config.registry import (
    ConfigRegistry,
    get_config,
    has_config,
    set_config,
)

__all__ = [
    # Core config classes
    "ExperimentConfig",
    "ModelConfig",
    "DataConfig",
    "FederatedConfig",
    "PathConfig",
    # Builder pattern
    "ConfigBuilder",
    "PresetConfigBuilder",
    # Registry (Singleton)
    "ConfigRegistry",
    "get_config",
    "set_config",
    "has_config",
]
