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

leaky_relu = lambda net: tf.nn.leaky_relu(net, alpha=0.2)

def hparams_parser(hparams_string):
    parser = argparse.ArgumentParser()

    parser.add_argument('--id',
                        type=str,
                        default = None,
                        help = 'Optional ID to distinguise experiments')

    parser.add_argument('--lr_discriminator', 
                        type=float, 
                        default='0.0002',
                        help='discriminator learning rate')
    
    parser.add_argument('--lr_generator', 
                        type=float, 
                        default='0.001',
                        help='Generator learning rate')

    parser.add_argument('--n_testsamples', 
                        type=int, 
                        default='20',
                        help='Number of samples in test images per class')

    parser.add_argument('--unstructured_noise_dim', 
                        type=int, 
                        default='62',
                        help='Number of random input variables to the generator')
    
    parser.add_argument('--d_iter',
                        type = int,
                        default = '1',
                        help = 'Number of times the discriminator is trained each loop')
    
    parser.add_argument('--gp_lambda',
                        type = int,
                        default = '10',
                        help = 'Gradient penalty weight')

    return parser.parse_args(shlex.split(hparams_string))

class acgan_Wgp(object):
    def __init__(self, dataset, hparams_string):

        args = hparams_parser(hparams_string)

        self.model = 'acgan_Wgp'
        if args.id != None:
            self.model = self.model + '_' + args.id

        self.dir_base        = 'models/' + self.model
        self.dir_logs        = self.dir_base + '/logs'
        self.dir_checkpoints = self.dir_base + '/checkpoints'
        self.dir_results     = self.dir_base + '/results'
        
        utils.checkfolder(self.dir_checkpoints)
        utils.checkfolder(self.dir_logs)
        utils.checkfolder(self.dir_results)

        dir_configuration = self.dir_base + '/configuration.txt'
        with open(dir_configuration, "w") as text_file:
            print(str(args), file=text_file)

        if dataset == 'MNIST':
            self.dateset_filenames =  ['data/processed/MNIST/train.tfrecord']
            self.lbls_dim = 10
            self.image_dims = [128,128,1]

        elif dataset == 'PSD':
            self.dateset_filenames = ['data/processed/PSD/Nonsegmented.tfrecord']
            self.lbls_dim = 12
            self.image_dims = [128,128,3]
        else:
            raise ValueError('Selected Dataset is not supported by model: acgan_W')

        self.unstructured_noise_dim = args.unstructured_noise_dim
        
        self.d_learning_rate = args.lr_discriminator
        self.g_learning_rate = args.lr_generator

        self.d_iter = args.d_iter
        self.n_testsamples = args.n_testsamples

        self.gp_lambda = 10
        # self.d_lr = 0.00009
        # self.g_lr = 0.001
 
    def __generator(self, noise, lbls_onehot, weight_decay = 2.5e-5, is_training = True, reuse=False):
        """InfoGAN discriminator network on MNIST digits.

        Based on a paper https://arxiv.org/abs/1606.03657 and their code
        https://github.com/openai/InfoGAN.
        
        Args:
            inputs: A 2-tuple of Tensors (unstructured_noise, labels_onehot).
                inputs[0] and inputs[1] is both 2D. All must have the same first dimension.
            categorical_dim: Dimensions of the incompressible categorical noise.
            weight_decay: The value of the l2 weight decay.
            is_training: If `True`, batch norm uses batch statistics. If `False`, batch
                norm uses the exponential moving average collected from population 
                statistics.
        
        Returns:
            A generated image in the range [-1, 1].
        """
        with tf.variable_scope("generator", reuse=reuse):

            all_noise = tf.concat([noise, lbls_onehot], axis=1)
        
            with framework.arg_scope(
                [layers.fully_connected, layers.conv2d_transpose],
                activation_fn=tf.nn.relu, normalizer_fn=layers.batch_norm,
                weights_regularizer=layers.l2_regularizer(weight_decay)),\
            framework.arg_scope([layers.conv2d_transpose], padding = 'VALID'),\
            framework.arg_scope([layers.batch_norm], is_training=is_training):
                net = layers.fully_connected(all_noise, 768, normalizer_fn = None)
                net = tf.reshape(net, [-1, 1, 1, 768])
                net = layers.conv2d_transpose(net, 384, [5, 5], stride = 2)
                net = layers.conv2d_transpose(net, 256, [5, 5], stride = 2)
                net = layers.conv2d_transpose(net, 192, [5, 5], stride = 2)
                net = layers.conv2d_transpose(net,  64, [5, 5], stride = 2)
                net = layers.conv2d_transpose(net, self.image_dims[2], [8, 8], stride = 2, normalizer_fn = None, activation_fn = tf.tanh)

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
    
        Returns:
            Logits for the probability that the image is real, and logits for the probability
                that the images belong to each class
        """
        with tf.variable_scope("discriminator", reuse = reuse):

            with framework.arg_scope(
                [layers.conv2d, layers.fully_connected],
                activation_fn=leaky_relu, normalizer_fn=None,
                weights_regularizer=layers.l2_regularizer(weight_decay),
                biases_regularizer=layers.l2_regularizer(weight_decay)),\
            framework.arg_scope([layers.batch_norm, layers.dropout], is_training=is_training):
                net = layers.conv2d(img,  16, [3,3], stride = 2, normalizer_fn = None, padding='SAME')
                net = layers.dropout(net)
                net = layers.conv2d(net,  32, [3,3], stride = 1, padding='VALID')
                net = layers.dropout(net)
                net = layers.conv2d(net,  64, [3,3], stride = 2, padding='SAME')
                net = layers.dropout(net)
                net = layers.conv2d(net, 128, [3,3], stride = 1, padding='VALID')
                net = layers.dropout(net)
                net = layers.conv2d(net, 256, [3,3], stride = 2, padding='SAME')
                net = layers.dropout(net)
                net = layers.conv2d(net, 512, [3,3], stride = 1, padding='VALID')
                net = layers.dropout(net)
                net = tf.reshape(net,[-1, 13*13*512])

                logits_source = layers.fully_connected(net, 1, normalizer_fn = None, activation_fn = None)
                logits_class = layers.fully_connected(net, self.lbls_dim, normalizer_fn = None, activation_fn=None)

                return logits_source, logits_class
    

    def _create_inference(self, images, labels, noise):
        """ Define the inference model for the network
        Args:
    
        Returns:
        """
        generated_images = self.__generator(noise, labels)
        logits_source_real, logits_class_real = self.__discriminator(images)
        logits_source_fake, logits_class_fake = self.__discriminator(generated_images, reuse = True)

        epsilon = tf.random_uniform(shape = images.get_shape(),
                                    minval= 0.0,
                                    maxval= 1.0)
        interpolated_images = epsilon * images + (1-epsilon) * generated_images
        logits_source_interpolates, logits_class_interpolates = self.__discriminator(interpolated_images, reuse = True)

        logits_source = [logits_source_real, logits_source_fake, logits_source_interpolates]
        logits_class = [logits_class_real, logits_class_fake, logits_class_interpolates]
        artificial_images = [generated_images, interpolated_images]

        return logits_source, logits_class, artificial_images
    
    def _create_losses(self, logits_source, logits_class, artificial_images, labels):
        """ Define loss function[s] for the network
        Args:
    
        Returns:
        """

        [logits_source_real, logits_source_fake, logits_source_interpolates] = logits_source
        [logits_class_real, logits_class_fake, _] = logits_class
        [_ , interpolated_images] = artificial_images

        loss_source_real = tf.reduce_mean(
            logits_source_real)
        
        loss_source_fake = tf.reduce_mean(
            logits_source_fake)

        loss_source_d = -(loss_source_real - loss_source_fake)
        loss_source_g = -loss_source_fake

        gradients = tf.gradients(logits_source_interpolates, [interpolated_images])[0]
        gradients_l2 = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1]))
        gradient_penalty = tf.reduce_mean(tf.square(gradients_l2 - 1.0))

        loss_source_d += self.gp_lambda * gradient_penalty

        # Class losses
        loss_class_real = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                labels = labels, 
                logits = logits_class_real,
                name = 'Loss_class_real'
        ))

        loss_class_fake = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                labels = labels, 
                logits = logits_class_fake,
                name = 'Loss_class_fake'
        ))

        loss_class = loss_class_real + loss_class_fake

        loss_discriminator = loss_source_d + loss_class
        loss_generator = loss_source_g + loss_class

        return loss_discriminator, loss_generator

        
    def _create_optimizer(self, loss_discriminator, loss_generator):
        """ Create optimizer for the network
        Args:
    
        Returns:
        """
        # variables for discriminator
        d_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

        # variables for generator
        g_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')


        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            # create train discriminator operation
            optimizer_discriminator = tf.train.AdamOptimizer(learning_rate = self.d_learning_rate, beta1 = 0.5)
            train_op_discriminator = optimizer_discriminator.minimize(loss_discriminator, var_list=d_vars)

            # create train generator operation
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

        # images = tf.image.grayscale_to_rgb(input_images)
        images = input_images
        images = tf.image.resize_images(images, size = [128, 128])        
        
        # Define model, loss, optimizer and summaries.
        logits_source, logits_class, artificial_images = self._create_inference(images, input_lbls, input_unstructured_noise)
        loss_discriminator, loss_generator = self._create_losses(logits_source, logits_class, artificial_images, input_lbls)
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
                # for idx in range(0, num_batches):
                    try:
                        for _ in range(self.d_iter):
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
    
        # image = image_proto
        # scalar = tf.random_uniform([3],minval = 0, maxval = 1)
        # mask = tf.tile(scalar,[28*28])
        # mask = tf.reshape(mask,[28,28,3])
        # image = tf.image.grayscale_to_rgb(image_proto)
        # image = tf.multiply(image, mask)

        image = image_proto
        image = tf.image.resize_images(image, size = [128, 128])  

        lbl = tf.one_hot(lbl_proto, self.lbls_dim)

        # unstructured_noise = tf.random_normal([self.unstructured_noise_dim])
        unstructured_noise = tf.random_uniform([self.unstructured_noise_dim], minval = -1, maxval = 1)
    
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

