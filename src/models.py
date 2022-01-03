import os

import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim

from src.loss import AdversarialLoss, PerceptualLoss, StyleLoss
from src.networks import InpaintGenerator, EdgeGenerator, Discriminator, HyperGraphGenerator, HyperGraphDiscriminator


class BaseModel(nn.Module):
    def __init__(self, name, config):
        super(BaseModel, self).__init__()

        self.name = name
        self.config = config
        self.iteration = 0

        self.gen_weights_path = os.path.join(config.PATH, name + '_gen.pth')
        self.dis_weights_path = os.path.join(config.PATH, name + '_dis.pth')

    def load(self):
        if os.path.exists(self.gen_weights_path):
            print('Loading %s generator...' % self.name)

            if torch.cuda.is_available():
                data = torch.load(self.gen_weights_path)
            else:
                data = torch.load(self.gen_weights_path, map_location=lambda storage, loc: storage)

            self.generator.load_state_dict(data['generator'])
            self.iteration = data['iteration']

        # load discriminator only when training
        if self.config.MODE == 1 and os.path.exists(self.dis_weights_path):
            print('Loading %s discriminator...' % self.name)

            if torch.cuda.is_available():
                data = torch.load(self.dis_weights_path)
            else:
                data = torch.load(self.dis_weights_path, map_location=lambda storage, loc: storage)

            self.discriminator.load_state_dict(data['discriminator'])

    def save(self):
        print('\nsaving %s...\n' % self.name)
        torch.save({
            'iteration': self.iteration,
            'generator': self.generator.state_dict()
        }, self.gen_weights_path)

        torch.save({
            'discriminator': self.discriminator.state_dict()
        }, self.dis_weights_path)


class EdgeModel(BaseModel):

    def __init__(self, config):
        super(EdgeModel, self).__init__('EdgeModel', config)

        # generator input: [grayscale(1) + edge(1) + mask(1)]
        # discriminator input: (grayscale(1) + edge(1))
        generator = EdgeGenerator(use_spectral_norm=True)
        discriminator = Discriminator(in_channels=2, use_sigmoid=config.GAN_LOSS != 'hinge')
        if len(config.GPU) > 1:
            generator = nn.DataParallel(generator, config.GPU)
            discriminator = nn.DataParallel(discriminator, config.GPU)
        l1_loss = nn.L1Loss()
        adversarial_loss = AdversarialLoss(type=config.GAN_LOSS)

        self.add_module('generator', generator)
        self.add_module('discriminator', discriminator)

        self.add_module('l1_loss', l1_loss)
        self.add_module('adversarial_loss', adversarial_loss)

        self.gen_optimizer = optim.Adam(
            params=generator.parameters(),
            lr=float(config.LR),
            betas=(config.BETA1, config.BETA2)
        )

        self.dis_optimizer = optim.Adam(
            params=discriminator.parameters(),
            lr=float(config.LR) * float(config.D2G_LR),
            betas=(config.BETA1, config.BETA2)
        )

    def process(self, images, edges, masks):
        self.iteration += 1

        # zero optimizers
        self.gen_optimizer.zero_grad()
        self.dis_optimizer.zero_grad()

        # process outputs
        outputs = self(images, edges, masks)
        gen_loss = 0
        dis_loss = 0

        # discriminator loss
        dis_input_real = torch.cat((images, edges), dim=1)
        dis_input_fake = torch.cat((images, outputs.detach()), dim=1)
        dis_real, dis_real_feat = self.discriminator(dis_input_real)  # in: (grayscale(1) + edge(1))
        dis_fake, dis_fake_feat = self.discriminator(dis_input_fake)  # in: (grayscale(1) + edge(1))
        dis_real_loss = self.adversarial_loss(dis_real, True, True)
        dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
        dis_loss += (dis_real_loss + dis_fake_loss) / 2

        # generator adversarial loss
        gen_input_fake = torch.cat((images, outputs), dim=1)
        gen_fake, gen_fake_feat = self.discriminator(gen_input_fake)  # in: (grayscale(1) + edge(1))
        gen_gan_loss = self.adversarial_loss(gen_fake, True, False)
        gen_loss += gen_gan_loss

        # generator feature matching loss
        gen_fm_loss = 0
        for i in range(len(dis_real_feat)):
            gen_fm_loss += self.l1_loss(gen_fake_feat[i], dis_real_feat[i].detach())
        gen_fm_loss = gen_fm_loss * self.config.FM_LOSS_WEIGHT
        gen_loss += gen_fm_loss

        # create logs
        logs = [
            ("l_d1", dis_loss.item()),
            ("l_g1", gen_gan_loss.item()),
            ("l_fm", gen_fm_loss.item()),
        ]

        return outputs, gen_loss, dis_loss, logs

    def forward(self, images, edges, masks):
        edges_masked = (edges * (1 - masks))
        images_masked = (images * (1 - masks)) + masks
        inputs = torch.cat((images_masked, edges_masked, masks), dim=1)
        outputs = self.generator(inputs)  # in: [grayscale(1) + edge(1) + mask(1)]
        return outputs

    def backward(self, gen_loss=None, dis_loss=None):
        if dis_loss is not None:
            dis_loss.backward()
        self.dis_optimizer.step()

        if gen_loss is not None:
            gen_loss.backward()
        self.gen_optimizer.step()


class InpaintingModel(BaseModel):

    def __init__(self, config):
        super(InpaintingModel, self).__init__('InpaintingModel', config)

        # generator input: [rgb(3) + edge(1)]
        # discriminator input: [rgb(3)]
        generator = InpaintGenerator()
        discriminator = Discriminator(in_channels=3, use_sigmoid=config.GAN_LOSS != 'hinge')
        if len(config.GPU) > 1:
            generator = nn.DataParallel(generator, config.GPU)
            discriminator = nn.DataParallel(discriminator, config.GPU)

        l1_loss = nn.L1Loss()
        perceptual_loss = PerceptualLoss()
        style_loss = StyleLoss()
        adversarial_loss = AdversarialLoss(type=config.GAN_LOSS)

        self.add_module('generator', generator)
        self.add_module('discriminator', discriminator)

        self.add_module('l1_loss', l1_loss)
        self.add_module('perceptual_loss', perceptual_loss)
        self.add_module('style_loss', style_loss)
        self.add_module('adversarial_loss', adversarial_loss)

        self.gen_optimizer = optim.Adam(
            params=generator.parameters(),
            lr=float(config.LR),
            betas=(config.BETA1, config.BETA2)
        )

        self.dis_optimizer = optim.Adam(
            params=discriminator.parameters(),
            lr=float(config.LR) * float(config.D2G_LR),
            betas=(config.BETA1, config.BETA2)
        )

    def process(self, images, edges, masks):
        self.iteration += 1

        # zero optimizers
        self.gen_optimizer.zero_grad()
        self.dis_optimizer.zero_grad()

        # process outputs
        outputs = self(images, edges, masks)
        gen_loss = 0
        dis_loss = 0

        # discriminator loss
        dis_input_real = images
        dis_input_fake = outputs.detach()
        dis_real, _ = self.discriminator(dis_input_real)  # in: [rgb(3)]
        dis_fake, _ = self.discriminator(dis_input_fake)  # in: [rgb(3)]
        dis_real_loss = self.adversarial_loss(dis_real, True, True)
        dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
        dis_loss += (dis_real_loss + dis_fake_loss) / 2

        # generator adversarial loss
        gen_input_fake = outputs
        gen_fake, _ = self.discriminator(gen_input_fake)  # in: [rgb(3)]
        gen_gan_loss = self.adversarial_loss(gen_fake, True, False) * self.config.INPAINT_ADV_LOSS_WEIGHT
        gen_loss += gen_gan_loss

        # generator l1 loss
        gen_l1_loss = self.l1_loss(outputs, images) * self.config.L1_LOSS_WEIGHT / torch.mean(masks)
        gen_loss += gen_l1_loss

        # generator perceptual loss
        gen_content_loss = self.perceptual_loss(outputs, images)
        gen_content_loss = gen_content_loss * self.config.CONTENT_LOSS_WEIGHT
        gen_loss += gen_content_loss

        # generator style loss
        gen_style_loss = self.style_loss(outputs * masks, images * masks)
        gen_style_loss = gen_style_loss * self.config.STYLE_LOSS_WEIGHT
        gen_loss += gen_style_loss

        # create logs
        logs = [
            ("l_d2", dis_loss.item()),
            ("l_g2", gen_gan_loss.item()),
            ("l_l1", gen_l1_loss.item()),
            ("l_per", gen_content_loss.item()),
            ("l_sty", gen_style_loss.item()),
        ]

        return outputs, gen_loss, dis_loss, logs

    def forward(self, images, edges, masks):
        images_masked = (images * (1 - masks).float()) + masks
        inputs = torch.cat((images_masked, edges), dim=1)
        outputs = self.generator(inputs)  # in: [rgb(3) + edge(1)]
        return outputs

    def backward(self, gen_loss=None, dis_loss=None):
        dis_loss.backward()
        self.dis_optimizer.step()

        gen_loss.backward()
        self.gen_optimizer.step()


class RefineModel(BaseModel):

    def __init__(self, config):
        super(RefineModel, self).__init__('RefineModel', config)

        generator = HyperGraphGenerator(config.INPUT_SIZE, config.INPUT_SIZE)
        discriminator = HyperGraphDiscriminator(config.INPUT_SIZE, config.INPUT_SIZE)
        # generator = model.build_generator()
        # model.build_discriminator(config.INPUT_SIZE, config.INPUT_SIZE)

        if len(config.GPU) > 1:
            generator = nn.DataParallel(generator, config.GPU)
            discriminator = nn.DataParallel(discriminator, config.GPU)

        l1_loss = nn.L1Loss()
        perceptual_loss = PerceptualLoss()
        style_loss = StyleLoss()
        adversarial_loss = AdversarialLoss(type=config.GAN_LOSS)

        self.add_module('generator', generator)
        self.add_module('discriminator', discriminator)
        # self.generator = generator
        # self.discriminator = discriminator

        self.add_module('l1_loss', l1_loss)
        self.add_module('perceptual_loss', perceptual_loss)
        self.add_module('style_loss', style_loss)
        self.add_module('adversarial_loss', adversarial_loss)

        learning_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=config.learning_rate,
            decay_steps=config.decay_steps,
            decay_rate=config.decay_rate,
            staircase=True
        )

        self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_schedule)
        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_schedule)

        self.HOLE_LOSS_WEIGHT = config.HOLE_LOSS_WEIGHT
        self.VALID_LOSS_WEIGHT = config.VALID_LOSS_WEIGHT
        self.EDGE_LOSS_WEIGHT = config.EDGE_LOSS_WEIGHT
        self.GAN_LOSS_WEIGHT = config.GAN_LOSS_WEIGHT
        self.PERCEPTUAL_LOSS_COMP_WEIGHT = config.PERCEPTUAL_LOSS_COMP_WEIGHT
        self.PERCEPTUAL_LOSS_OUT_WEIGHT = config.PERCEPTUAL_LOSS_OUT_WEIGHT

    def process(self, original_image, mask, prediction_coarse, vgg_model, selected_layers):

        self.iteration += 1

        # zero optimizers
        self.gen_optimizer.zero_grad()
        self.dis_optimizer.zero_grad()

        # process outputs
        prediction_refine = self(original_image, mask, prediction_coarse)
        gen_loss = 0
        dis_loss = 0

        disc_original_output = self.discriminator([original_image, mask], training=True)
        disc_generated_output = self.discriminator([prediction_refine, mask], training=True)

        # generator loss
        loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
        hole_l1_loss = tf.reduce_mean(tf.math.abs(mask * (original_image - prediction_coarse))) * 0.5
        hole_l1_loss += tf.reduce_mean(tf.math.abs(mask * (original_image - prediction_refine)))
        valid_l1_loss = tf.reduce_mean(tf.math.abs((1 - mask) * (original_image - prediction_coarse))) * 0.5
        valid_l1_loss += tf.reduce_mean(tf.math.abs((1 - mask) * (original_image - prediction_refine)))
        vgg_gen_output = vgg_model(tf.keras.applications.vgg19.preprocess_input(prediction_refine * 255.0))
        vgg_comp = vgg_model(tf.keras.applications.vgg19.preprocess_input((prediction_refine * mask + original_image * (1 - mask)) * 255.0))
        vgg_target = vgg_model(tf.keras.applications.vgg19.preprocess_input(original_image * 255.0))
        perceptual_loss_out = 0
        perceptual_loss_comp = 0

        for i in range(len(selected_layers)):
            perceptual_loss_out += tf.reduce_mean(tf.math.abs(vgg_gen_output[i] - vgg_target[i]))
            perceptual_loss_comp += tf.reduce_mean(tf.math.abs(vgg_comp[i] - vgg_target[i]))

        edge_loss = tf.reduce_mean(
            tf.math.abs(tf.image.sobel_edges(prediction_refine) - tf.image.sobel_edges(original_image)))

        # valid_l1_loss, hole_l1_loss, edge_loss, gan_loss, perceptual_loss_out, perceptual_loss_comp

        gen_loss = self.VALID_LOSS_WEIGHT * valid_l1_loss + self.HOLE_LOSS_WEIGHT * hole_l1_loss + self.EDGE_LOSS_WEIGHT * edge_loss + self.GAN_LOSS_WEIGHT * gan_loss + self.PERCEPTUAL_LOSS_OUT_WEIGHT * perceptual_loss_out + self.PERCEPTUAL_LOSS_COMP_WEIGHT * perceptual_loss_comp

        # discriminator loss
        original_loss = loss_object(tf.ones_like(disc_original_output), disc_original_output)
        generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

        dis_loss = original_loss + generated_loss

        # create logs
        logs = [
            ("dis_loss", dis_loss.item()),
            ("gen_loss", gen_loss.item()),
            ("valid_l1_loss, hole_l1_loss, edge_loss, gan_loss, perceptual_loss_out, perceptual_loss_comp",
             (valid_l1_loss, hole_l1_loss, edge_loss, gan_loss, perceptual_loss_out, perceptual_loss_comp)),
        ]

        return prediction_refine, gen_loss, dis_loss, logs

    def forward(self, original_image, mask, prediction_coarse):
        # images_masked = (original_image * (1 - mask).float()) + mask
        # inputs = torch.cat((images_masked, prediction_coarse),dim=1)
        outputs = self.generator(original_image, prediction_coarse, mask) # images_masked, prediction_coarse, mask
        return outputs

    def backward(self, gen_loss=None, dis_loss=None):
        dis_loss.backward()
        self.discriminator_optimizer.step()

        gen_loss.backward()
        self.generator_optimizer.step()