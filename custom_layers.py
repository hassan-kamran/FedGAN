from tensorflow.keras.layers import (
    Layer, Conv2DTranspose, BatchNormalization, ReLU, Conv2D, LeakyReLU
)


class TransposedConvLayer(Layer):
    def __init__(self, filters, kernel_size, strides, padding, momentum, **kwargs):
        super(TransposedConvLayer, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.momentum = momentum
        
        self.transposed_conv = Conv2DTranspose(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            use_bias=False
        )
        self.batch_norm = BatchNormalization(momentum=momentum)
        self.relu = ReLU()

    def build(self, input_shape):
        # This method is called before the first call to this layer
        # It's where we create the layer's weights
        self.transposed_conv.build(input_shape)
        # Calculate output shape for batch normalization
        output_shape = self.transposed_conv.compute_output_shape(input_shape)
        self.batch_norm.build(output_shape)
        self.built = True

    def call(self, input_tensor):
        x = self.transposed_conv(input_tensor)
        x = self.batch_norm(x)
        return self.relu(x)

    def get_config(self):
        config = super(TransposedConvLayer, self).get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'momentum': self.momentum
        })
        return config

class ConvLayer(Layer):
    def __init__(self, filters, kernel_size, strides, padding, momentum, alpha, **kwargs):
        super(ConvLayer, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.momentum = momentum
        self.alpha = alpha
        
        self.conv = Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            use_bias=False
        )
        self.batch_norm = BatchNormalization(momentum=momentum)
        self.leaky_relu = LeakyReLU(negative_slope=alpha)

    def build(self, input_shape):
        self.conv.build(input_shape)
        output_shape = self.conv.compute_output_shape(input_shape)
        self.batch_norm.build(output_shape)
        self.built = True

    def call(self, input_tensor):
        x = self.conv(input_tensor)
        x = self.batch_norm(x)
        return self.leaky_relu(x)

    def get_config(self):
        config = super(ConvLayer, self).get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'momentum': self.momentum,
            'alpha': self.alpha
        })
        return config
