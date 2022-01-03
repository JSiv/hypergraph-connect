import tensorflow as tf
from torch import nn


class GatedConvolution(nn.Module):
    def __init__(self, channels, kernel_size, stride=1, dilation=1, padding='same', activation='ELU'):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.activation = activation

        self.conv = nn.Conv2d(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation
        )

        # tf.keras.initializers.glorot_normal()
        nn.init.xavier_normal_(self.conv.weight)

    def forward(self, conv_input):
        # Apply convolution to the Input features
        x = self.conv(conv_input)

        # If we have final layer then we don't apply any activation
        if self.channels == 3 and self.activation is None:
            return x

        x, y = tf.split(x, 2, 3)

        if self.activation == 'LeakyReLU':
            x = nn.LeakyReLU(x)
        elif self.activation == 'ReLU':
            x = nn.ReLU(x)
        elif self.activation == 'ELU':
            x = nn.ELU(x)
        else:
            print("NO ACTIVATION!!!")

        # Gated Convolutiopn
        y = tf.nn.sigmoid(y)
        x = x * y

        return x


# Gated Deconvolution layer -> Upsampling + Gated Convolution
class GatedDeConvolution(nn.Module):
    def __init__(self, channels, kernel_size, stride=1, dilation=1, padding='same', activation='ELU'):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.activation = activation

        self.up_sample = tf.keras.layers.UpSampling2D(size=2)
        self.conv = GatedConvolution(self.channels, self.kernel_size, self.stride, self.dilation, self.padding)

    def forward(self, input):
        x = self.up_sample(input)
        x = self.conv(x)

        if self.activation == 'LeakyReLU':
            x = nn.LeakyReLU(x)
        elif self.activation == 'ReLU':
            x = nn.ReLU(x)
        elif self.activation == 'ELU':
            x = nn.ELU(x)

        return x