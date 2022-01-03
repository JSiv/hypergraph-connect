from torch import nn

from src.hypergraph.models.gc_layer import GatedConvolution

class GatedConvolution():
    def __init__(self, channel, kernel_size, stride=1, dilation=1, padding='same', activation='ELU'):
        super(GatedConvolution, self).__init__()
        self.channels = channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.activation = activation

        # kernel_initializer = tf.keras.initializers.glorot_normal()

        self.conv_0 = nn.Conv2d(in_channels=self.channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation)


class HyperGraphGenerator(nn.Module):
    def __init__(self, image_height, image_width):
        super(HyperGraphGenerator, self).__init__()

        channels = 64
        self.encoder(nn.Sequential(

            GatedConvolution(channels=channels,kernel_size=7,stride=1,dilation=1,padding='same',activation='ELU')


        ))
        self.deconvolution()
        self.decoder()

