#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 14:01:52 2017

@author: leminen
"""
import sys
import os
import tensorflow as tf
import numpy as np
import itertools
import functools
import matplotlib.pyplot as plt
import datetime

sys.path.append('/home/leminen/Documents/RoboWeedMaps/GAN/ACGAN_collections')
import src.data.process_dataset as process_dataset
import src.utils as utils

tfgan = tf.contrib.gan
layers = tf.contrib.layers
framework = tf.contrib.framework
ds = tf.contrib.distributions

leaky_relu = lambda net: tf.nn.leaky_relu(net, alpha=0.01)

class acgan(object):
    def __init__(self):
        self.model = 'acgan'
        self.dir_logs        = 'models/' + self.model + '/logs'
        self.dir_checkpoints = 'models/' + self.model + '/checkpoints'
        self.dir_results     = 'models/' + self.model + '/results'
        
        utils.checkfolder(self.dir_checkpoints)
        utils.checkfolder(self.dir_logs)
        utils.checkfolder(self.dir_results)

        self.unstructured_noise_dim = 62
        self.lbls_dim = 10
 
    def __generator(self, inputs, weight_decay = 2.5e-5, is_training = True):
        """InfoGAN discriminator network on MNIST digits.

        Based on a paper https://arxiv.org/abs/1606.03657 and their code
        https://github.com/openai/InfoGAN.
        
        Args:
            inputs: A 2-tuple of Tensors (unstructured_noise, labels_onehot). `inputs[0]` and `inputs[2]` must be
                2D, and `inputs[1]` must be 1D. All must have the same first dimension.
            categorical_dim: Dimensions of the incompressible categorical noise.
            weight_decay: The value of the l2 weight decay.
            is_training: If `True`, batch norm uses batch statistics. If `False`, batch
                norm uses the exponential moving average collected from population 
                statistics.
        
        Returns:
            A generated image in the range [-1, 1].
        """

        unstructured_noise, lbls_onehot = inputs
        all_noise = tf.concat([unstructured_noise, lbls_onehot], axis=1)
    
        with framework.arg_scope(
            [layers.fully_connected, layers.conv2d_transpose],
            activation_fn=tf.nn.relu, normalizer_fn=layers.batch_norm,
            weights_regularizer=layers.l2_regularizer(weight_decay)),\
        framework.arg_scope([layers.batch_norm], is_training=is_training):
            net = layers.fully_connected(all_noise, 1024)
            net = layers.fully_connected(net, 7 * 7 * 128)
            net = tf.reshape(net, [-1, 7, 7, 128])
            net = layers.conv2d_transpose(net, 64, [4, 4], stride = 2)
            net = layers.conv2d_transpose(net, 32, [4, 4], stride = 2)
            # Make sure that generator output is in the same range as `inputs`
            # ie [-1, 1].
            net = layers.conv2d(net, 1, 4, normalizer_fn=None, activation_fn=tf.tanh)
    
            return net
    
    def __discriminator(self, img, unused_conditioning, weight_decay=2.5e-5, class_dim=10, is_training=True):
        """InfoGAN discriminator network on MNIST digits.
    
        Based on a paper https://arxiv.org/abs/1606.03657 and their code
        https://github.com/openai/InfoGAN.
    
        Args:
            img: Real or generated MNIST digits. Should be in the range [-1, 1].
                unused_conditioning: The TFGAN API can help with conditional GANs, which
                would require extra `condition` information to both the generator and the
                discriminator. Since this example is not conditional, we do not use this
                argument.
            weight_decay: The L2 weight decay.
            class_dim: Number of classes to classify.
            is_training: If `True`, batch norm uses batch statistics. If `False`, batch
                norm uses the exponential moving average collected from population statistics.
    
        Returns:
            Logits for the probability that the image is real, and logits for the probability
                that the images belong to each class
        """
        with framework.arg_scope(
            [layers.conv2d, layers.fully_connected],
            activation_fn=leaky_relu, normalizer_fn=None,
            weights_regularizer=layers.l2_regularizer(weight_decay),
            biases_regularizer=layers.l2_regularizer(weight_decay)):
            net = layers.conv2d(img,  64, [4, 4], stride = 2)
            net = layers.conv2d(net, 128, [4, 4], stride = 2)
            net = layers.flatten(net)
            net = layers.fully_connected(net, 1024, normalizer_fn=layers.layer_norm)
    
            logits_source = layers.fully_connected(net, 1, activation_fn=None)

            # Recognition network for latent variables has an additional layer
            with framework.arg_scope([layers.batch_norm], is_training=is_training):
                encoder = layers.fully_connected(
                    net, 128, normalizer_fn=layers.batch_norm)

            # Compute logits for each category of categorical latent.
            logits_class = layers.fully_connected(
                encoder, class_dim, activation_fn=None)

            return logits_source, logits_class
    

    def _create_inference(self):
        """ Define the inference model for the network
        Args:
    
        Returns:
        """

        # Create input placeholders
        self.input_images = tf.placeholder(
            dtype = tf.float32, 
            shape = [None,28,28,1], 
            name = 'input_images')
        self.input_lbls = tf.placeholder(
            dtype = tf.float32,   
            shape = [None, self.lbls_dim], 
            name = 'input_lbls')
        self.input_unstructured_noise = tf.placeholder(
            dtype = tf.float32, 
            shape = [None, self.unstructured_noise_dim], 
            name = 'input_unstructured_noise')

        # Create generator and discriminator functions and setup infoGAN model
        generator_fn = functools.partial(self.__generator)
        discriminator_fn = functools.partial(self.__discriminator, 
                                             class_dim = self.lbls_dim)

        self.acgan_model = tfgan.acgan_model(
            generator_fn = generator_fn,
            discriminator_fn = discriminator_fn,
            real_data = self.input_images,
            generator_inputs = [self.input_unstructured_noise, self.input_lbls],
            one_hot_labels = self.input_lbls
        )

    
    def _create_losses(self):
        """ Define loss function[s] for the network
        Args:
    
        Returns:
        """
        
        # Create acgan losses
        self.acgan_loss = tfgan.gan_loss(
            self.acgan_model,
            generator_loss_fn = tfgan.losses.acgan_generator_loss,
            discriminator_loss_fn = tfgan.losses.acgan_discriminator_loss,
            gradient_penalty_weight = 1.0
        )
        
    def _create_optimizer(self):
        """ Create optimizer for the network
        Args:
    
        Returns:
        """

        # Create optimizers and Create update operations
        generator_optimizer = tf.train.AdamOptimizer(learning_rate = 0.001, beta1 = 0.5)
        discriminator_optimizer = tf.train.AdamOptimizer(learning_rate = 0.00009, beta1 = 0.5)

        self.gan_train_ops = tfgan.gan_train_ops(
            self.acgan_model,
            self.acgan_loss,
            generator_optimizer = generator_optimizer,
            discriminator_optimizer = discriminator_optimizer)
        
    def _create_summaries(self):
        """ Create summaries for the network
        Args:
    
        Returns:
        """

        # Create image summaries to inspect the variation due to categorical latent codes
        with tf.name_scope("SummaryImages_ClassVariation"):
            grid_size = 10
            images_test = []
            for class_var in range(0,self.lbls_dim):
                with tf.variable_scope('Generator', reuse=True):
                    test_input = self._genTestInput(class_var, grid_size)
                    images_cat = self.acgan_model.generator_fn(test_input, is_training=False)
                    images_cat = tfgan.eval.image_reshaper(tf.concat(images_cat, 0), num_cols=grid_size)
                    images_test.append(images_cat[0,:,:,:])

            self.summary_imgClass_op = tf.summary.image('test_images', images_test, max_outputs = 20)

        ### Add loss summaries
        with tf.name_scope("SummaryLosses"):
            summary_gloss = tf.summary.scalar('loss_generator', self.acgan_loss.generator_loss)
            summary_dloss = tf.summary.scalar('loss_discriminator', self.acgan_loss.discriminator_loss)
            
            self.summary_loss_op = tf.summary.merge([summary_gloss, 
                                                     summary_dloss])
                                                                 
        
    def train(self, dataset_str, epoch_N, batch_size):
        """ Run training of the network
        Args:
    
        Returns:
        """
        
        # Use dataset for loading in datasamples from .tfrecord (https://www.tensorflow.org/programmers_guide/datasets#consuming_tfrecord_data)
        # The iterator will get a new batch from the dataset each time a sess.run() is executed on the graph.
        filenames = ['data/processed/' + dataset_str + '/train.tfrecords']
        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.map(process_dataset._decodeData)      # decoding the tfrecord
        dataset = dataset.map(self._genLatentCodes)
        dataset = dataset.shuffle(buffer_size = 10000, seed = None)
        dataset = dataset.batch(batch_size = batch_size)
        iterator = dataset.make_initializable_iterator()
        input_getBatch = iterator.get_next()        
        
        # Define model, loss, optimizer and summaries.
        self._create_inference()
        self._create_losses()
        self._create_optimizer()
        self._create_summaries()

        with tf.Session() as sess:
            # Initialize all model Variables.
            sess.run(tf.global_variables_initializer())
            
            # Create Saver object for loading and storing checkpoints
            saver = tf.train.Saver()
            
            # Create Writer object for storing graph and summaries for TensorBoard
            writer = tf.summary.FileWriter(self.dir_logs, sess.graph)

            # Reload Tensor values from latest checkpoint
            ckpt = tf.train.get_checkpoint_state(self.dir_checkpoints)
            epoch_start = 0
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
                epoch_start = int(ckpt_name.split('-')[-1]) + 1
            
            interationCnt = 0
            for epoch_n in range(epoch_start, epoch_N):

                # Test model output before any training
                if epoch_n == 0:
                    summaryImg = sess.run(self.summary_imgClass_op)
                    writer.add_summary(summaryImg, global_step=-1)



                # Initiate or Re-initiate iterator
                sess.run(iterator.initializer)
                
                ### ----------------------------------------------------------
                ### Update model
                print(datetime.datetime.now(),'- Running training epoch no:', epoch_n)
                while True:
                    try:
                        image_batch, lbl_batch, unst_noise_batch = sess.run(input_getBatch)

                        _ = sess.run(
                            [self.gan_train_ops.discriminator_train_op],
                             feed_dict={self.input_images:             image_batch,
                                        self.input_lbls:               lbl_batch, 
                                        self.input_unstructured_noise: unst_noise_batch})

                        _, summaryLoss = sess.run(
                            [self.gan_train_ops.generator_train_op, self.summary_loss_op],
                             feed_dict={self.input_images:             image_batch,
                                        self.input_lbls:               lbl_batch, 
                                        self.input_unstructured_noise: unst_noise_batch})

                        writer.add_summary(summaryLoss, global_step=interationCnt)
                        interationCnt += 1

                    except tf.errors.OutOfRangeError:
                        # Test current model
                        summaryImg = sess.run(self.summary_imgClass_op)
                        writer.add_summary(summaryImg, global_step=epoch_n)

                        break
                
                # Save model variables to checkpoint
                if epoch_n % 1 == 0:
                    saver.save(sess,os.path.join(self.dir_checkpoints, self.model + '.model'), global_step=epoch_n)
            
    
    def predict(self):
        """ Run prediction of the network
        Args:
    
        Returns:
        """
        
        # not implemented yet
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
    
    
    def _genLatentCodes(self, image_proto, lbl_proto):
        """ Augment dataset entries. Adds two continuous latent 
            codes for the network to estimate. Also generates a GAN noise
            vector per data sample.
        Args:

        Returns:
        """
    
        image = image_proto
        lbl = tf.one_hot(lbl_proto, self.lbls_dim)

        unstructured_noise = tf.random_normal([self.unstructured_noise_dim])
    
        return image, lbl, unstructured_noise
    
    def _genTestInput(self, class_code, grid_dim):
        """ Defines test code and noise generator. Generates laten codes based
            on a testCategory input.
        Args:
    
        Returns:
        """

        n_images = grid_dim ** 2

        test_unstructured_noise = np.random.normal(size=[n_images, self.unstructured_noise_dim])
        test_class_lbls = np.tile(class_code, n_images)
        test_class_lbls = tf.one_hot(test_class_lbls, self.lbls_dim)

        return test_unstructured_noise, test_class_lbls

model = acgan()
model.train('MNIST', 25, 32)
