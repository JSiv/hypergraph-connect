import torch
import torch.nn as nn
import tensorflow as tf

from src.hypergraph.models.gc_layer import GatedConvolution, GatedDeConvolution
from src.hypergraph.models.hypergraph_layer import HypergraphConv


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)


class InpaintGenerator(BaseNetwork):
    def __init__(self, residual_blocks=8, init_weights=True):
        super(InpaintGenerator, self).__init__()

        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=4, out_channels=64, kernel_size=7, padding=0),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(True)
        )

        blocks = []
        for _ in range(residual_blocks):
            block = ResnetBlock(256, 2)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, padding=0),
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        x = (torch.tanh(x) + 1) / 2

        return x


class EdgeGenerator(BaseNetwork):
    def __init__(self, residual_blocks=8, use_spectral_norm=True, init_weights=True):
        super(EdgeGenerator, self).__init__()

        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            spectral_norm(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, padding=0), use_spectral_norm),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
                          use_spectral_norm),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
                          use_spectral_norm),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(True)
        )

        blocks = []
        for _ in range(residual_blocks):
            block = ResnetBlock(256, 2, use_spectral_norm=use_spectral_norm)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

        self.decoder = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
                          use_spectral_norm),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            spectral_norm(nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
                          use_spectral_norm),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=7, padding=0),
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        x = torch.sigmoid(x)
        return x


class Discriminator(BaseNetwork):
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True, init_weights=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid

        self.conv1 = self.features = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        outputs = conv5
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)

        return outputs, [conv1, conv2, conv3, conv4, conv5]


class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1, use_spectral_norm=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=dilation,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
        )

    def forward(self, x):
        out = x + self.conv_block(x)

        # Remove ReLU at the end of the residual block
        # http://torch.ch/blog/2016/02/04/resnets.html

        return out


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module


class HyperGraphGenerator(nn.Module):
    def __init__(self, input_height, input_width):
        super(HyperGraphGenerator, self).__init__()
        self.image_width = input_width
        self.image_height = input_height

    def forward(self, input_img, coarse_img, input_mask):
        x = coarse_img * input_mask + input_img * (1 - input_mask)
        x = tf.keras.layers.Concatenate()([x, input_mask])

        channels = 64
        skip_connections = []
        down_samples = 4
        current_image_height = self.image_height
        current_image_width = self.image_width

        x = GatedConvolution(channels=channels, kernel_size=7, stride=1, dilation=1, padding='same', activation='ELU')(
            x)

        for i in range(1, down_samples + 1):
            x = GatedConvolution(channels=channels * (2 ** i), kernel_size=3, stride=2, dilation=1, padding='same',
                                 activation='ELU')(x)

            current_image_height = current_image_height // 2
            current_image_width = current_image_width // 2

            dilation_rate = 1
            count = 2

            if i == down_samples - 1:
                dilation_rate = 2
                count = 3

            for j in range(count):
                x = GatedConvolution(channels=channels * (2 ** i), kernel_size=3, stride=1, dilation=dilation_rate,
                                     padding='same', activation='ELU')(x)

            if i != down_samples:
                skip_connections.append(list([x, channels * (2 ** (i - 1)), current_image_height, current_image_width]))

        # Apply Hypergraph convolution on last skip connections
        for i in range(len(skip_connections) - 1, len(skip_connections) - 3, -1):
            mult = 1
            if i == len(skip_connections) - 1:
                mult = 2
            skip_connections[i][0] = HypergraphConv(
                in_features=skip_connections[i][1],
                out_features=skip_connections[i][1] * mult,
                features_height=skip_connections[i][2],
                features_width=skip_connections[i][3],
                edges=256,
                filters=128,
                apply_bias=True,
                trainable=True
            )(skip_connections[i][0])

            skip_connections[i][0] = nn.ELU()(skip_connections[i][0])

        # Doing the first Deconvolution operation
        x = GatedDeConvolution(channels=channels * (2 ** (down_samples - 1)), kernel_size=3, stride=1, dilation=1,
                               padding='same', activation='ELU')(x)

        x = torch.cat([x, skip_connections[len(skip_connections) - 1][0]])

        # Decoder for Refine Network
        current = len(skip_connections) - 2
        for i in range(down_samples - 1, 0, -1):

            dilation_rate = 1
            count = 2
            if i == down_samples - 1:
                dilation_rate = 2
                count = 3

            for j in range(count):
                x = GatedConvolution(channels=channels * (2 ** i), kernel_size=3, stride=1, dilation=dilation_rate,
                                     padding='same', activation='ELU')(x)

            x = GatedDeConvolution(channels=channels * (2 ** (i - 1)), kernel_size=3, stride=1, dilation=1,
                                   padding='same', activation='ELU')(x)

            if current != -1:
                x = torch.cat([x, skip_connections[current][0]])
                current -= 1

        x = GatedConvolution(channels=channels, kernel_size=3, stride=1, dilation=1, padding='same', activation='ELU')(
            x)

        x = GatedConvolution(channels=channels, kernel_size=3, stride=1, dilation=1, padding='same', activation='ELU')(
            x)

        refine_out = GatedConvolution(channels=3, kernel_size=3, stride=1, dilation=1, padding='same', activation=None)(
            x)

        return refine_out


class HyperGraphDiscriminator(nn.Module):
    def __init__(self, image_height, image_width):
        super(HyperGraphDiscriminator, self).__init__()
        self.image_width = image_width
        self.image_height = image_height

    def forward(self, input_img, input_mask):
        x = torch.cat([input_img, input_mask])
        channels = 64
        x = GatedConvolution(channels=channels, kernel_size=3, stride=1, dilation=1, padding='same',
                             activation='LeakyReLU')(x)
        for i in range(1, 7):
            mult = (2 ** i) if (2 ** i) < 8 else 8
            x = GatedConvolution(channels=channels * mult, kernel_size=3, stride=2, dilation=1, padding='same',
                                 activation='LeakyReLU')(x)

        return x

# class HyperGraphGenerator(nn.Module):
#     def build_generator(self, input_height, input_width):
#         input_img = tf.keras.layers.Input(shape=(input_height, input_width, 3))
#         coarse_img = tf.keras.layers.Input(shape=(input_height, input_width, 3))
#         input_mask = tf.keras.layers.Input(shape=(input_height, input_width, 1))
#
#         channels = 64
#
#         x = coarse_img * input_mask + input_img * (1 - input_mask)
#         x = tf.keras.layers.Concatenate()([x, input_mask])
#         x = GatedConvolution(
#             channels=channels,
#             kernel_size=7,
#             stride=1,
#             dilation=1,
#             padding='same',
#             activation='ELU'
#         )(x)
#
#         # Encoder For Refine Network
#         skip_connections = []
#         downsamples = 4
#         current_image_height = input_height
#         current_image_width = input_width
#
#         for i in range(1, downsamples + 1):
#             x = GatedConvolution(channels=channels * (2 ** i), kernel_size=3, stride=2, dilation=1, padding='same',
#                                  activation='ELU')(x)
#
#             current_image_height = current_image_height // 2
#             current_image_width = current_image_width // 2
#
#             dilation_rate = 1
#             count = 2
#
#             if i == downsamples - 1:
#                 dilation_rate = 2
#                 count = 3
#
#             for j in range(count):
#                 x = GatedConvolution(channels=channels * (2 ** i), kernel_size=3, stride=1, dilation=dilation_rate,
#                                      padding='same', activation='ELU')(x)
#
#             if i != downsamples:
#                 skip_connections.append(list([x, channels * (2 ** (i - 1)), current_image_height, current_image_width]))
#
#         # Apply Hypergraph convolution on last skip connections
#         for i in range(len(skip_connections) - 1, len(skip_connections) - 3, -1):
#             mult = 1
#             if i == len(skip_connections) - 1:
#                 mult = 2
#             skip_connections[i][0] = HypergraphConv(
#                 in_features=skip_connections[i][1],
#                 out_features=skip_connections[i][1] * mult,
#                 features_height=skip_connections[i][2],
#                 features_width=skip_connections[i][3],
#                 edges=256,
#                 filters=128,
#                 apply_bias=True,
#                 trainable=True
#             )(skip_connections[i][0])
#
#             skip_connections[i][0] = tf.keras.layers.ELU()(skip_connections[i][0])
#
#         # Doing the first Deconvolution operation
#         x = GatedDeConvolution(channels=channels * (2 ** (downsamples - 1)), kernel_size=3, stride=1, dilation=1,
#                                padding='same', activation='ELU')(x)
#
#         x = tf.keras.layers.Concatenate()([x, skip_connections[len(skip_connections) - 1][0]])
#
#         # Decoder for Refine Network
#         current = len(skip_connections) - 2
#         for i in range(downsamples - 1, 0, -1):
#
#             dilation_rate = 1
#             count = 2
#             if i == downsamples - 1:
#                 dilation_rate = 2
#                 count = 3
#
#             for j in range(count):
#                 x = GatedConvolution(channels=channels * (2 ** i), kernel_size=3, stride=1, dilation=dilation_rate,
#                                      padding='same', activation='ELU')(x)
#
#             x = GatedDeConvolution(channels=channels * (2 ** (i - 1)), kernel_size=3, stride=1, dilation=1,
#                                    padding='same', activation='ELU')(x)
#
#             if current != -1:
#                 x = tf.keras.layers.Concatenate()([x, skip_connections[current][0]])
#                 current -= 1
#
#         x = GatedConvolution(channels=channels, kernel_size=3, stride=1, dilation=1, padding='same', activation='ELU')(
#             x)
#
#         x = GatedConvolution(channels=channels, kernel_size=3, stride=1, dilation=1, padding='same', activation='ELU')(
#             x)
#
#         refine_out = GatedConvolution(channels=3, kernel_size=3, stride=1, dilation=1, padding='same', activation=None)(
#             x)
#
#         return tf.keras.Model(inputs=[input_img, coarse_img, input_mask], outputs=[coarse_img, refine_out])
#
#     def build_discriminator(self, image_height, image_width):
#         input_img = tf.keras.layers.Input(shape=[image_height, image_width, 3])
#         input_mask = tf.keras.layers.Input(shape=[image_height, image_width, 1])
#
#         x = tf.keras.layers.Concatenate()([input_img, input_mask])
#
#         channels = 64
#         x = GatedConvolution(channels=channels, kernel_size=3, stride=1, dilation=1, padding='same',
#                              activation='LeakyReLU')(x)
#         for i in range(1, 7):
#             mult = (2 ** i) if (2 ** i) < 8 else 8
#             x = GatedConvolution(channels=channels * mult, kernel_size=3, stride=2, dilation=1, padding='same',
#                                  activation='LeakyReLU')(x)
#
#         return tf.keras.Model(inputs=[input_img, input_mask], outputs=x)
