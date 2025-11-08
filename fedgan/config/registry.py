"""
Singleton registry for managing the active experiment configuration.

This module provides a global registry for accessing the current configuration
throughout the application without passing config objects explicitly.
"""
from typing import Optional

from fedgan.config.config import ExperimentConfig


class ConfigRegistry:
    """Singleton registry for the active experiment configuration.
    
    This class implements the Singleton pattern to ensure only one
    configuration is active at any time across the application.
    
    Example:
        >>> from fedgan.config.registry import ConfigRegistry
        >>> from fedgan.config.builder import ConfigBuilder
        >>> 
        >>> # Set configuration
        >>> config = ConfigBuilder().for_experiment("test").build()
        >>> registry = ConfigRegistry()
        >>> registry.set_config(config)
        >>> 
        >>> # Access configuration from anywhere
        >>> registry = ConfigRegistry()
        >>> config = registry.get_config()
        >>> print(config.name)  # "test"
    
    Note:
        The singleton pattern ensures that all instances of ConfigRegistry
        refer to the same underlying configuration object.
    """
    
    _instance: Optional["ConfigRegistry"] = None
    _config: Optional[ExperimentConfig] = None
    
    def __new__(cls) -> "ConfigRegistry":
        """Create or return the singleton instance.
        
        Returns:
            The singleton ConfigRegistry instance.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def set_config(self, config: ExperimentConfig) -> None:
        """Set the active experiment configuration.
        
        Args:
            config: ExperimentConfig to set as active.
        
        Raises:
            ValueError: If config is None.
        """
        if config is None:
            raise ValueError("Cannot set None as configuration")
        
        # Validate config before setting
        config.validate()
        
        self._config = config
    
    def get_config(self) -> ExperimentConfig:
        """Get the active experiment configuration.
        
        Returns:
            The active ExperimentConfig.
        
        Raises:
            RuntimeError: If no configuration has been set.
        """
        if self._config is None:
            raise RuntimeError(
                "No configuration has been initialized. "
                "Call set_config() before accessing configuration."
            )
        return self._config
    
    def has_config(self) -> bool:
        """Check if a configuration has been set.
        
        Returns:
            True if configuration is set, False otherwise.
        """
        return self._config is not None
    
    def clear_config(self) -> None:
        """Clear the active configuration.
        
        This is primarily useful for testing and resetting state.
        """
        self._config = None
    
    def get_or_default(self, default: ExperimentConfig) -> ExperimentConfig:
        """Get the active config or return a default if not set.
        
        Args:
            default: Default configuration to return if none is set.
        
        Returns:
            Active config if set, otherwise the provided default.
        """
        if self._config is None:
            return default
        return self._config
    
    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance.
        
        This is primarily useful for testing to ensure a clean state.
        WARNING: This will affect all references to the singleton.
        """
        cls._instance = None
        cls._config = None


def get_config() -> ExperimentConfig:
    """Convenience function to get the active configuration.
    
    Returns:
        The active ExperimentConfig.
    
    Raises:
        RuntimeError: If no configuration has been set.
    """
    return ConfigRegistry().get_config()


def set_config(config: ExperimentConfig) -> None:
    """Convenience function to set the active configuration.
    
    Args:
        config: ExperimentConfig to set as active.
    """
    ConfigRegistry().set_config(config)


def has_config() -> bool:
    """Convenience function to check if config is set.
    
    Returns:
        True if configuration is set, False otherwise.
    """
    return ConfigRegistry().has_config()
