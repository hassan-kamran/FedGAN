"""
Factory pattern for creating GAN models.

This module provides factories for creating generator and discriminator
models with different architectures and configurations.
"""
from abc import ABC, abstractmethod
from typing import Optional, Tuple

from tensorflow import keras

from fedgan.config import ModelConfig
from fedgan.models.architectures import build_discriminator, build_generator


class ModelFactory(ABC):
    """Abstract factory for creating GAN models.
    
    This implements the Abstract Factory pattern, allowing different
    GAN architectures to be created through a common interface.
    """
    
    @abstractmethod
    def create_generator(self) -> keras.Model:
        """Create a generator model.
        
        Returns:
            Keras Model for the generator.
        """
        pass
    
    @abstractmethod
    def create_discriminator(self) -> keras.Model:
        """Create a discriminator model.
        
        Returns:
            Keras Model for the discriminator.
        """
        pass
    
    def create_models(self) -> Tuple[keras.Model, keras.Model]:
        """Create both generator and discriminator.
        
        Returns:
            Tuple of (generator, discriminator) models.
        """
        return self.create_generator(), self.create_discriminator()


class DCGANFactory(ModelFactory):
    """Factory for creating DCGAN models.
    
    Args:
        config: Model configuration object.
        image_size: Size of generated/input images.
        channels: Number of image channels (1 for grayscale, 3 for RGB).
    
    Example:
        >>> from fedgan.config import ModelConfig
        >>> config = ModelConfig(latent_dim=200)
        >>> factory = DCGANFactory(config, image_size=128, channels=1)
        >>> generator, discriminator = factory.create_models()
    """
    
    def __init__(
        self,
        config: ModelConfig,
        image_size: int = 128,
        channels: int = 1
    ):
        self.config = config
        self.image_size = image_size
        self.channels = channels
    
    def create_generator(self) -> keras.Model:
        """Create DCGAN generator.
        
        Returns:
            Generator model configured according to ModelConfig.
        """
        return build_generator(
            latent_dim=self.config.latent_dim,
            filters=self.config.generator_filters,
            batch_norm_momentum=self.config.batch_norm_momentum,
            output_channels=self.channels
        )
    
    def create_discriminator(self) -> keras.Model:
        """Create DCGAN discriminator.
        
        Returns:
            Discriminator model configured according to ModelConfig.
        """
        return build_discriminator(
            image_shape=(self.image_size, self.image_size, self.channels),
            filters=self.config.discriminator_filters,
            batch_norm_momentum=self.config.batch_norm_momentum,
            leaky_relu_alpha=self.config.leaky_relu_alpha
        )


class SimpleGANFactory(ModelFactory):
    """Factory for creating simple GAN models for testing.
    
    Args:
        latent_dim: Dimensionality of latent vector.
        image_size: Size of images.
    """
    
    def __init__(self, latent_dim: int = 100, image_size: int = 64):
        self.latent_dim = latent_dim
        self.image_size = image_size
    
    def create_generator(self) -> keras.Model:
        """Create simple generator for testing."""
        return build_generator(
            latent_dim=self.latent_dim,
            filters=[256, 128, 64],
            output_channels=1
        )
    
    def create_discriminator(self) -> keras.Model:
        """Create simple discriminator for testing."""
        return build_discriminator(
            image_shape=(self.image_size, self.image_size, 1),
            filters=[32, 64, 128]
        )


# Registry of available factories
MODEL_FACTORIES = {
    'dcgan': DCGANFactory,
    'simple': SimpleGANFactory,
}


def get_model_factory(
    architecture: str,
    config: Optional[ModelConfig] = None,
    **kwargs
) -> ModelFactory:
    """Get a model factory by architecture name.
    
    Args:
        architecture: Name of architecture ('dcgan', 'simple').
        config: Model configuration (required for 'dcgan').
        **kwargs: Additional factory-specific arguments.
    
    Returns:
        ModelFactory instance.
    
    Raises:
        ValueError: If architecture is unknown or config is missing.
    
    Example:
        >>> from fedgan.config import ModelConfig
        >>> config = ModelConfig(latent_dim=200)
        >>> factory = get_model_factory('dcgan', config, image_size=128)
        >>> generator = factory.create_generator()
    """
    factory_class = MODEL_FACTORIES.get(architecture)
    
    if factory_class is None:
        raise ValueError(
            f"Unknown architecture: {architecture}. "
            f"Available: {list(MODEL_FACTORIES.keys())}"
        )
    
    if architecture == 'dcgan':
        if config is None:
            raise ValueError("ModelConfig required for DCGAN factory")
        return factory_class(config, **kwargs)
    else:
        return factory_class(**kwargs)


def register_factory(name: str, factory_class: type) -> None:
    """Register a custom model factory.
    
    Args:
        name: Name for the factory.
        factory_class: Factory class (must inherit from ModelFactory).
    
    Raises:
        TypeError: If factory_class doesn't inherit from ModelFactory.
    """
    if not issubclass(factory_class, ModelFactory):
        raise TypeError("Factory class must inherit from ModelFactory")
    MODEL_FACTORIES[name] = factory_class
