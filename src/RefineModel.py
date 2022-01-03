import tensorflow as tf
import torch
from torch import nn, optim

from src.loss import PerceptualLoss, StyleLoss, AdversarialLoss
from src.models import BaseModel
from src.networks import HyperGraphGenerator


class RefineModel(BaseModel):

    def __init__(self, config):
        super(RefineModel, self).__init__('RefineModel', config)

        model = HyperGraphGenerator()
        generator = model.build_generator(config.INPUT_SIZE, config.INPUT_SIZE)
        discriminator = model.build_discriminator(config.INPUT_SIZE, config.INPUT_SIZE)

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

        disc_learning_schedule = optim.lr_scheduler.StepLR(
            optimizer=optim.Adam(
                params=discriminator.parameters(),
                lr=float(config.LR),
                betas=(config.BETA1, config.BETA2)),
            gamma=config.DECAY_RATE,
            step_size=config.DECAY_STEPS
        )

        gen_learning_schedule = optim.lr_scheduler.StepLR(
            optimizer=optim.Adam(
                params=generator.parameters(),
                lr=float(config.LR),
                betas=(config.BETA1, config.BETA2)),
            step_size=config.DECAY_STEPS,
            gamma=config.DECAY_RATE
        )

        self.discriminator_optimizer = disc_learning_schedule
        self.generator_optimizer = gen_learning_schedule

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
        images_masked = (original_image * (1 - mask).float()) + mask
        inputs = torch.cat((images_masked, prediction_coarse),dim=1)
        outputs = self.generator(inputs) #images_masked, prediction_coarse, mask
        return outputs

    def backward(self, gen_loss=None, dis_loss=None):
        dis_loss.backward()
        self.discriminator_optimizer.step()

        gen_loss.backward()
        self.generator_optimizer.step()
