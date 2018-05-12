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
import scipy
import argparse
import shlex

import src.utils as utils
import src.data.util_data as util_data

tfgan = tf.contrib.gan
layers = tf.contrib.layers
framework = tf.contrib.framework
ds = tf.contrib.distributions

leaky_relu = lambda net: tf.nn.leaky_relu(net, alpha=0.01)

def hparams_parser(hparams_string):
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr_discriminator', 
                        type=float, 
                        default='0.0002',
                        help='discriminator learning rate')
    
    parser.add_argument('--lr_generator', 
                        type=float, 
                        default='0.001',
                        help='generator learning rate')

    parser.add_argument('--n_testsamples', 
                        type=int, 
                        default='20',
                        help='number of samples in test images per class')

    parser.add_argument('--unstructured_noise_dim', 
                        type=int, 
                        default='62',
                        help='number of random input variables to the generator')

    parser.add_argument('--id',
                        type=str,
                        default = None,
                        help = 'Optional ID to distinguise experiments')

    return parser.parse_args(shlex.split(hparams_string))

class acgan_v01(object):
    def __init__(self, dataset, hparams_string):

        args = hparams_parser(hparams_string)

        self.model = 'acgan_v01'
        if args.id != None:
            self.model = self.model + '_' + args.id

        self.dir_logs        = 'models/' + self.model + '/logs'
        self.dir_checkpoints = 'models/' + self.model + '/checkpoints'
        self.dir_results     = 'models/' + self.model + '/results'
        
        utils.checkfolder(self.dir_checkpoints)
        utils.checkfolder(self.dir_logs)
        utils.checkfolder(self.dir_results)

        if dataset == 'MNIST':
            self.dateset_filenames =  ['data/processed/MNIST/train.tfrecord']
        else:
            raise ValueError('Selected Dataset is not supported by model: acgan_v01')

        self.unstructured_noise_dim = args.unstructured_noise_dim
        self.lbls_dim = 10
        self.image_dims = [28,28,1]

        self.d_learning_rate = args.lr_discriminator
        self.g_learning_rate = args.lr_generator

        self.n_testsamples = args.n_testsamples
 
    def __generator(self, noise, lbls_onehot, weight_decay = 2.5e-5, is_training = True, reuse=False):
        """InfoGAN discriminator network on MNIST digits.

        Based on a paper https://arxiv.org/abs/1606.03657 and their code
        https://github.com/openai/InfoGAN.
        
        Args:
            inputs: A 2-tuple of Tensors (unstructured_noise, labels_onehot).
                inputs[0] and inputs[1] is both 2D. All must have the same first dimension.
            weight_decay: The value of the l2 weight decay.
            is_training: If `True`, batch norm uses batch statistics. If `False`, batch
                norm uses the exponential moving average collected from population 
                statistics.
            reuse: If `True`, the variables in scope will be reused
        
        Returns:
            A generated image in the range [-1, 1].
        """
        with tf.variable_scope("generator", reuse=reuse):

            all_noise = tf.concat([noise, lbls_onehot], axis=1)
        
            with framework.arg_scope(
                [layers.fully_connected, layers.conv2d_transpose],
                activation_fn=tf.nn.relu, normalizer_fn=layers.batch_norm,
                weights_regularizer=layers.l2_regularizer(weight_decay)),\
            framework.arg_scope([layers.batch_norm], is_training=is_training):
                net = layers.fully_connected(all_noise, 1024)
                net = layers.fully_connected(net, 7 * 7 * 128)
                net = tf.reshape(net, [-1, 7, 7, 128])
                net = layers.conv2d_transpose(net, 64, [4, 4], stride = 2)
                net = layers.conv2d_transpose(net, self.image_dims[2], [4, 4], stride = 2, normalizer_fn = None, activation_fn = tf.tanh)
                # Make sure that generator output is in the same range as `inputs`
                # ie [-1, 1].
        
                return net
    
    def __discriminator(self, img, weight_decay=2.5e-5, is_training=True, reuse = False):
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
            reuse: If `True`, the variables in scope will be reused
    
        Returns:
            Logits for the probability that the image is real, and logits for the probability
                that the images belong to each class
        """
        with tf.variable_scope("discriminator", reuse=reuse):

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
                    encoder, self.lbls_dim, activation_fn=None)

                return logits_source, logits_class
    

    def _create_inference(self, images, labels, noise):
        """ Define the inference model for the network
        Args:
            images: Input batch of Real MNIST images
            labels: Input labels corresponding to each image (one_hot encoded) 
            z: input batch of unstructured noise vectors
    
        Returns:
            logits_source: 2D tuple with estimated probability of images being real. [0] output for real images, [1] output for fake images
            logits_class: 2D tuple with estimated probability of the image belonging to each class. [0] output for real images, [1] output for fake images
            test_images: output images to inspect the current performance of the generator
        """

        generated_images = self.__generator(noise, labels)
        logits_source_real, logits_class_real = self.__discriminator(images)
        logits_source_fake, logits_class_fake = self.__discriminator(generated_images, reuse = True)

        return [logits_source_real, logits_source_fake], [logits_class_real, logits_class_fake]
    
    def _create_losses(self, logits_source, logits_class, labels):
        """ Define loss function[s] for the network
        Args:
            logits_source: 2D tuple with estimated probability of images being real.
            logits_class: 2D tuple with estimated probability of the image belonging to each class.
            labels: groundtruth labels (one_hot encoded)
        Returns:
            loss_discriminator: calculated loss for the discriminator network
            loss_generator: calculated loss for the generator network
        """

        [logits_source_real, logits_source_fake] = logits_source
        [logits_class_real, logits_class_fake] = logits_class

        # Source losses discriminator
        loss_source_real_d = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels = tf.ones_like(logits_source_real), logits = logits_source_real
            ))
        
        loss_source_fake_d = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels = tf.zeros_like(logits_source_fake), logits = logits_source_fake
            ))
        
        # Source loss generator
        loss_source_fake_g = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels = tf.ones_like(logits_source_fake), logits = logits_source_fake
            ))

        # Class losses
        loss_class_real = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                labels = labels, logits = logits_class_real
        ))

        loss_class_fake = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                labels = labels, logits = logits_class_fake
        ))

        loss_discriminator = loss_source_real_d + loss_source_fake_d + loss_class_real + loss_class_fake
        loss_generator = loss_source_fake_g + loss_class_real + loss_class_fake

        return loss_discriminator, loss_generator

        
    def _create_optimizer(self, loss_discriminator, loss_generator):
        """ Create optimizer for the network
        Args:
            loss_discriminator: calculated loss for the discriminator network
            loss_generator: calculated loss for the generator network
    
        Returns:
            train_op_discriminator: tensorflow optimizer operation used to update the weigths of the discriminator network
            train_op_generator: tensorflow optimizer operation used to update the weigths of the generator network
        """
        # variables for discriminator
        d_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

        # variables for generator
        g_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

        # train discriminator
        optimizer_discriminator = tf.train.AdamOptimizer(learning_rate = self.d_learning_rate, beta1 = 0.5)
        train_op_discriminator = optimizer_discriminator.minimize(loss_discriminator, var_list=d_vars)

        # train generator
        optimizer_generator = tf.train.AdamOptimizer(learning_rate = self.g_learning_rate, beta1 = 0.5)
        train_op_generator = optimizer_generator.minimize(loss_generator, var_list=g_vars)

        return train_op_discriminator, train_op_generator
        
    def _create_summaries(self, loss_discriminator, loss_generator, test_noise, test_labels):
        """ Create summaries for the network
        Args:
    
        Returns:
        """

        test_images = self.__generator(test_noise, test_labels, is_training=False, reuse=True)

        # Create image summaries to inspect the variation due to categorical latent codes
        with tf.name_scope("SummaryImages_ClassVariation"):
            summary_img = tfgan.eval.image_reshaper(tf.concat(test_images, 0), num_cols=self.lbls_dim)
            summary_op_img = tf.summary.image('summary_images', summary_img, max_outputs = 20)

        ### Add loss summaries
        with tf.name_scope("SummaryLosses"):
            summary_op_dloss = tf.summary.scalar('loss_discriminator', loss_discriminator)
            summary_op_gloss = tf.summary.scalar('loss_generator', loss_generator)
            
        return summary_op_dloss, summary_op_gloss, summary_op_img, summary_img
                                                                 
        
    def train(self, epoch_N, batch_size):
        """ Run training of the network
        Args:
    
        Returns:
        """
        
        # Use dataset for loading in datasamples from .tfrecord (https://www.tensorflow.org/programmers_guide/datasets#consuming_tfrecord_data)
        # The iterator will get a new batch from the dataset each time a sess.run() is executed on the graph.
        dataset = tf.data.TFRecordDataset(self.dateset_filenames)
        dataset = dataset.map(util_data.decode_image)      # decoding the tfrecord
        dataset = dataset.map(self._genLatentCodes)
        dataset = dataset.shuffle(buffer_size = 10000, seed = None)
        dataset = dataset.batch(batch_size = batch_size)
        iterator = dataset.make_initializable_iterator()
        input_getBatch = iterator.get_next()

        # Create input placeholders
        input_images = tf.placeholder(
            dtype = tf.float32, 
            shape = [None] + self.image_dims, 
            name = 'input_images')
        input_lbls = tf.placeholder(
            dtype = tf.float32,   
            shape = [None, self.lbls_dim], 
            name = 'input_lbls')
        input_unstructured_noise = tf.placeholder(
            dtype = tf.float32, 
            shape = [None, self.unstructured_noise_dim], 
            name = 'input_unstructured_noise')
        input_test_lbls = tf.placeholder(
            dtype = tf.float32, 
            shape = [self.n_testsamples * self.lbls_dim, self.lbls_dim], 
            name = 'input_test_lbls')
        input_test_noise = tf.placeholder(
            dtype = tf.float32, 
            shape = [self.n_testsamples * self.lbls_dim, self.unstructured_noise_dim], 
            name = 'input_test_noise')        
        
        # Define model, loss, optimizer and summaries.
        logits_source, logits_class = self._create_inference(input_images, input_lbls, input_unstructured_noise)
        loss_discriminator, loss_generator = self._create_losses(logits_source, logits_class, input_lbls)
        train_op_discriminator, train_op_generator = self._create_optimizer(loss_discriminator, loss_generator)
        summary_op_dloss, summary_op_gloss, summary_op_img, summary_img = self._create_summaries(loss_discriminator, loss_generator, input_test_noise, input_test_lbls)

        # show network architecture
        utils.show_all_variables()

        # create constant test variable to inspect changes in the model
        test_noise, test_lbls = self._genTestInput(self.lbls_dim, n_samples = self.n_testsamples)
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
                    summaryImg_tb, summaryImg = sess.run(
                        [summary_op_img, summary_img],
                        feed_dict={input_test_noise:    test_noise,
                                   input_test_lbls:     test_lbls})

                    writer.add_summary(summaryImg_tb, global_step=-1)
                    self.save_image_local(summaryImg, 'Epoch_' + str(-1))

                # Initiate or Re-initiate iterator
                sess.run(iterator.initializer)
                
                ### ----------------------------------------------------------
                ### Update model
                print(datetime.datetime.now(),'- Running training epoch no:', epoch_n)
                while True:
                    try:
                        image_batch, lbl_batch, unst_noise_batch = sess.run(input_getBatch)

                        _, summary_dloss = sess.run(
                            [train_op_discriminator, summary_op_dloss],
                             feed_dict={input_images:             image_batch,
                                        input_lbls:               lbl_batch, 
                                        input_unstructured_noise: unst_noise_batch})
                                        
                        writer.add_summary(summary_dloss, global_step=interationCnt)

                        _, summary_gloss = sess.run(
                            [train_op_generator, summary_op_gloss],
                             feed_dict={input_images:             image_batch,
                                        input_lbls:               lbl_batch, 
                                        input_unstructured_noise: unst_noise_batch})

                        writer.add_summary(summary_gloss, global_step=interationCnt)
                        interationCnt += 1

                    except tf.errors.OutOfRangeError:
                        # Test current model
                        summaryImg_tb, summaryImg = sess.run(
                            [summary_op_img, summary_img],
                            feed_dict={input_test_noise:    test_noise,
                                        input_test_lbls:     test_lbls})

                        writer.add_summary(summaryImg_tb, global_step=epoch_n)
                        self.save_image_local(summaryImg, 'Epoch_' + str(epoch_n))

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
    
    
    def _genLatentCodes(self, image_proto, lbl_proto, class_proto, height_proto, width_proto, channels_proto, origin_proto):
        """ Augment dataset entries. Adds two continuous latent 
            codes for the network to estimate. Also generates a GAN noise
            vector per data sample.
        Args:

        Returns:
        """
    
        image = image_proto
        image = tf.image.resize_images(image, size = [28,28])
        lbl = tf.one_hot(lbl_proto, self.lbls_dim)

        unstructured_noise = tf.random_normal([self.unstructured_noise_dim])
    
        return image, lbl, unstructured_noise
    
    def _genTestInput(self, lbls_dim, n_samples):
        """ Defines test code and noise generator. Generates laten codes based
            on a testCategory input.
        Args:
    
        Returns:
        """

        # Create repeating noise vector, so a sample for each class is created using the same noise
        test_unstructured_noise = np.random.uniform(low = -1.0, high = 1, size = [n_samples, self.unstructured_noise_dim])
        test_unstructured_noise = np.repeat(test_unstructured_noise, lbls_dim, axis = 0)
        
        # Create one-hot encoded label for each class and tile along axis 0
        test_labels = np.eye(lbls_dim)
        test_labels = np.tile(test_labels,(n_samples,1))

        return test_unstructured_noise, test_labels

    def save_image_local(self, image, infostr):
        datestr = datetime.datetime.now().strftime('%Y%m%d_T%H%M%S')
        path = self.dir_results + '/' + datestr + '_' + infostr + '.png'
        image = np.squeeze(image)
        scipy.misc.imsave(path, image)
