"""
GAN model architectures.

This module provides the generator and discriminator architectures
for DCGAN (Deep Convolutional GAN).
"""
from typing import List, Tuple

from tensorflow import keras
from tensorflow.keras import layers

from fedgan.models.layers import ConvLayer, TransposedConvLayer


def build_generator(
    latent_dim: int = 200,
    initial_size: int = 4,
    filters: List[int] = [1024, 512, 256, 128, 64, 32],
    batch_norm_momentum: float = 0.1,
    output_channels: int = 1,
    output_activation: str = 'tanh'
) -> keras.Model:
    """Build DCGAN generator model.
    
    The generator takes a latent vector and produces an image through
    a series of transposed convolution layers with upsampling.
    
    Args:
        latent_dim: Dimensionality of the latent input vector.
        initial_size: Initial spatial size after dense layer (e.g., 4 for 4x4).
        filters: List of filter counts for each transposed conv layer.
        batch_norm_momentum: Momentum for batch normalization.
        output_channels: Number of output channels (1 for grayscale, 3 for RGB).
        output_activation: Activation function for output layer.
    
    Returns:
        Keras Model for the generator.
    
    Example:
        >>> generator = build_generator(latent_dim=200)
        >>> noise = tf.random.normal([16, 200])
        >>> generated_images = generator(noise)
        >>> generated_images.shape  # (16, 128, 128, 1)
    """
    inputs = layers.Input(shape=(latent_dim,), name='latent_input')
    
    # Initial dense layer and reshape
    initial_filters = filters[0]
    x = layers.Dense(
        initial_size * initial_size * initial_filters,
        use_bias=False,
        name='initial_dense'
    )(inputs)
    x = layers.Reshape((initial_size, initial_size, initial_filters))(x)
    
    # Transposed convolution layers (upsampling)
    for i, num_filters in enumerate(filters[1:], start=1):
        x = TransposedConvLayer(
            filters=num_filters,
            kernel_size=4,
            strides=2,
            padding='same',
            momentum=batch_norm_momentum,
            name=f'upconv_{i}'
        )(x)
    
    # Output layer
    x = layers.Conv2DTranspose(
        output_channels,
        kernel_size=4,
        padding='same',
        use_bias=False,
        name='output_conv'
    )(x)
    outputs = layers.Activation(output_activation, name='output_activation')(x)
    
    model = keras.Model(inputs, outputs, name='generator')
    return model


def build_discriminator(
    image_shape: Tuple[int, int, int] = (128, 128, 1),
    filters: List[int] = [64, 128, 256, 512, 1024],
    batch_norm_momentum: float = 0.1,
    leaky_relu_alpha: float = 0.2
) -> keras.Model:
    """Build DCGAN discriminator model.
    
    The discriminator takes an image and classifies it as real or fake
    through a series of convolutional layers with downsampling.
    
    Args:
        image_shape: Shape of input images (height, width, channels).
        filters: List of filter counts for each conv layer.
        batch_norm_momentum: Momentum for batch normalization.
        leaky_relu_alpha: Negative slope for LeakyReLU activation.
    
    Returns:
        Keras Model for the discriminator.
    
    Example:
        >>> discriminator = build_discriminator(image_shape=(128, 128, 1))
        >>> images = tf.random.normal([16, 128, 128, 1])
        >>> predictions = discriminator(images)
        >>> predictions.shape  # (16, 1)
    """
    inputs = layers.Input(shape=image_shape, name='image_input')
    
    x = inputs
    
    # Convolutional layers (downsampling)
    for i, num_filters in enumerate(filters):
        x = ConvLayer(
            filters=num_filters,
            kernel_size=4,
            strides=2,
            padding='same',
            momentum=batch_norm_momentum,
            alpha=leaky_relu_alpha,
            name=f'downconv_{i}'
        )(x)
    
    # Flatten and output
    x = layers.Flatten(name='flatten')(x)
    outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
    
    model = keras.Model(inputs, outputs, name='discriminator')
    return model


def build_simple_generator(latent_dim: int = 100) -> keras.Model:
    """Build a smaller generator for testing.
    
    Args:
        latent_dim: Dimensionality of latent vector.
    
    Returns:
        Simple generator model.
    """
    return build_generator(
        latent_dim=latent_dim,
        initial_size=4,
        filters=[256, 128, 64],
        output_channels=1
    )


def build_simple_discriminator(image_size: int = 64) -> keras.Model:
    """Build a smaller discriminator for testing.
    
    Args:
        image_size: Size of input images.
    
    Returns:
        Simple discriminator model.
    """
    return build_discriminator(
        image_shape=(image_size, image_size, 1),
        filters=[32, 64, 128]
    )
