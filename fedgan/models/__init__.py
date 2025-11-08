"""Model architectures and factories for FedGAN."""

from fedgan.models.architectures import (
    build_discriminator,
    build_generator,
    build_simple_discriminator,
    build_simple_generator,
)
from fedgan.models.factory import (
    DCGANFactory,
    ModelFactory,
    SimpleGANFactory,
    get_model_factory,
    register_factory,
)
from fedgan.models.layers import ConvLayer, TransposedConvLayer

__all__ = [
    # Layers
    "ConvLayer",
    "TransposedConvLayer",
    # Architecture builders
    "build_generator",
    "build_discriminator",
    "build_simple_generator",
    "build_simple_discriminator",
    # Factories
    "ModelFactory",
    "DCGANFactory",
    "SimpleGANFactory",
    "get_model_factory",
    "register_factory",
]
