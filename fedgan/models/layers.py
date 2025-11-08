"""
Custom Keras layers for GAN architectures.

This module provides reusable layer components that encapsulate
common patterns in GAN architectures.
"""
from typing import Any, Dict

from tensorflow.keras import layers


class TransposedConvLayer(layers.Layer):
    """Transposed convolution layer with batch normalization and ReLU.
    
    This layer encapsulates the common pattern of:
    Conv2DTranspose -> BatchNormalization -> ReLU
    
    Args:
        filters: Number of output filters.
        kernel_size: Size of the convolution kernel.
        strides: Stride of the convolution.
        padding: Padding mode ('same' or 'valid').
        momentum: Momentum for batch normalization.
        **kwargs: Additional layer arguments.
    """
    
    def __init__(
        self,
        filters: int,
        kernel_size: int = 4,
        strides: int = 2,
        padding: str = 'same',
        momentum: float = 0.1,
        **kwargs: Any
    ):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.momentum = momentum
        
        # Sub-layers
        self.transposed_conv = layers.Conv2DTranspose(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            use_bias=False
        )
        self.batch_norm = layers.BatchNormalization(momentum=momentum)
        self.relu = layers.ReLU()
    
    def build(self, input_shape: tuple) -> None:
        """Build layer weights.
        
        Args:
            input_shape: Shape of input tensor.
        """
        self.transposed_conv.build(input_shape)
        output_shape = self.transposed_conv.compute_output_shape(input_shape)
        self.batch_norm.build(output_shape)
        self.built = True
    
    def call(self, input_tensor: layers.Layer) -> layers.Layer:
        """Forward pass.
        
        Args:
            input_tensor: Input tensor.
        
        Returns:
            Output tensor after transposed conv, batch norm, and ReLU.
        """
        x = self.transposed_conv(input_tensor)
        x = self.batch_norm(x)
        return self.relu(x)
    
    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization.
        
        Returns:
            Configuration dictionary.
        """
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'momentum': self.momentum
        })
        return config


class ConvLayer(layers.Layer):
    """Convolution layer with batch normalization and LeakyReLU.
    
    This layer encapsulates the common pattern of:
    Conv2D -> BatchNormalization -> LeakyReLU
    
    Args:
        filters: Number of output filters.
        kernel_size: Size of the convolution kernel.
        strides: Stride of the convolution.
        padding: Padding mode ('same' or 'valid').
        momentum: Momentum for batch normalization.
        alpha: Negative slope coefficient for LeakyReLU.
        **kwargs: Additional layer arguments.
    """
    
    def __init__(
        self,
        filters: int,
        kernel_size: int = 4,
        strides: int = 2,
        padding: str = 'same',
        momentum: float = 0.1,
        alpha: float = 0.2,
        **kwargs: Any
    ):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.momentum = momentum
        self.alpha = alpha
        
        # Sub-layers
        self.conv = layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            use_bias=False
        )
        self.batch_norm = layers.BatchNormalization(momentum=momentum)
        self.leaky_relu = layers.LeakyReLU(negative_slope=alpha)
    
    def build(self, input_shape: tuple) -> None:
        """Build layer weights.
        
        Args:
            input_shape: Shape of input tensor.
        """
        self.conv.build(input_shape)
        output_shape = self.conv.compute_output_shape(input_shape)
        self.batch_norm.build(output_shape)
        self.built = True
    
    def call(self, input_tensor: layers.Layer) -> layers.Layer:
        """Forward pass.
        
        Args:
            input_tensor: Input tensor.
        
        Returns:
            Output tensor after conv, batch norm, and LeakyReLU.
        """
        x = self.conv(input_tensor)
        x = self.batch_norm(x)
        return self.leaky_relu(x)
    
    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization.
        
        Returns:
            Configuration dictionary.
        """
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'momentum': self.momentum,
            'alpha': self.alpha
        })
        return config
