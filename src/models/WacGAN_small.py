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
class OutOfRangeError(Exception): pass

def hparams_parser_train(hparams_string):
    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch_max', 
                        type=int, default='100', 
                        help='Max number of epochs to run')

    parser.add_argument('--batch_size', 
                        type=int, default='64', 
                        help='Number of samples in each batch')

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
                        default='128',
                        help='Number of random input variables to the generator')
    
    parser.add_argument('--d_iter',
                        type = int,
                        default = '5',
                        help = 'Number of times the discriminator is trained each loop')
    
    parser.add_argument('--gp_lambda',
                        type = int,
                        default = '10',
                        help = 'Gradient penalty weight')
    
    parser.add_argument('--class_scale_d',
                        type = float,
                        default = '1',
                        help = 'Scale significance of discriminator class loss')

    parser.add_argument('--class_scale_g',
                        type = float,
                        default = '1',
                        help = 'Scale significance of generator class loss')
    
    parser.add_argument('--backup_frequency',
                        type = int,
                        default = '10',
                        help = 'Number of iterations between backup of network weights')

    return parser.parse_args(shlex.split(hparams_string))


def hparams_parser_evaluate(hparams_string):
    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch_no', 
                        type=int,
                        default=None, 
                        help='Epoch no to reload')

    return parser.parse_args(shlex.split(hparams_string))


class WacGAN_small(object):
    def __init__(self, dataset, id):

        self.model = 'WacGAN_small'
        if id != None:
            self.model = self.model + '_' + id

        self.dir_base        = 'models/' + self.model
        self.dir_logs        = self.dir_base + '/logs'
        self.dir_checkpoints = self.dir_base + '/checkpoints'
        self.dir_results     = self.dir_base + '/results'
        
        utils.checkfolder(self.dir_checkpoints)
        utils.checkfolder(self.dir_logs)
        utils.checkfolder(self.dir_results)

        if dataset == 'MNIST':
            self.dateset_filenames = ['data/processed/MNIST/train.tfrecord']
            self.lbls_dim = 10
            self.image_dims = [28,28,1]

        else:
            raise ValueError('Selected Dataset is not supported by model: ' + self.model)
        
        self.dataset = dataset

 
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

        # Source losses
        loss_source_real = tf.reduce_mean(
            logits_source_real)
        
        loss_source_fake = tf.reduce_mean(
            logits_source_fake)

        loss_source_discriminator = -(loss_source_real - loss_source_fake)
        loss_source_generator = -loss_source_fake

        # Discriminator Gradient Penalty
        gradients = tf.gradients(logits_source_interpolates, [interpolated_images])[0]
        gradients_l2 = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1]))
        gradient_penalty = tf.reduce_mean(tf.square(gradients_l2 - 1.0))

        loss_source_discriminator += self.gp_lambda * gradient_penalty

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

        loss_class_discriminator = loss_class_real + loss_class_fake
        loss_class_generator = loss_class_fake

        # Total losses, with scaled class loss
        loss_total_discriminator = loss_source_discriminator + self.class_scale_d * loss_class_discriminator
        loss_total_generator = loss_source_generator + self.class_scale_g * loss_class_generator

        loss_discriminator = [loss_total_discriminator, 
                              loss_source_discriminator, 
                              loss_class_discriminator]
        loss_generator     = [loss_total_generator, 
                              loss_source_generator,
                              loss_class_generator]

        return loss_discriminator, loss_generator

        
    def _create_optimizer(self, loss_discriminator, loss_generator):
        """ Create optimizer for the network
        Args:
    
        Returns:
        """
        
        # optimize on total loss
        d_loss = loss_discriminator[0]
        g_loss = loss_generator[0]

        # variables for discriminator
        d_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

        # variables for generator
        g_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')


        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            # create train discriminator operation
            optimizer_discriminator = tf.train.AdamOptimizer(learning_rate = self.d_learning_rate, beta1 = 0.5, beta2 = 0.9)
            train_op_discriminator = optimizer_discriminator.minimize(d_loss, var_list=d_vars)

            # create train generator operation
            optimizer_generator = tf.train.AdamOptimizer(learning_rate = self.g_learning_rate, beta1 = 0.5, beta2 = 0.9)
            train_op_generator = optimizer_generator.minimize(g_loss, var_list=g_vars)

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
            summary_op_img = tf.summary.image('In', summary_img, max_outputs = 1)

        ### Add loss summaries
        [loss_total_discriminator, loss_source_discriminator, loss_class_discriminator] = loss_discriminator
        [loss_total_generator, loss_source_generator, loss_class_generator] = loss_generator

        with tf.name_scope("SummaryLosses_Discriminator"):

            summary_dloss_source = tf.summary.scalar('loss_source', loss_source_discriminator)
            summary_dloss_class = tf.summary.scalar('loss_class', loss_class_discriminator)
            summary_dloss_tot = tf.summary.scalar('loss_total', loss_total_discriminator)
            summary_op_dloss = tf.summary.merge(
                [summary_dloss_source,
                 summary_dloss_class,
                 summary_dloss_tot], name = 'loss_discriminator')

        with tf.name_scope("SummaryLosses_Generator"):

            summary_gloss_source = tf.summary.scalar('loss_source', loss_source_generator)
            summary_gloss_class = tf.summary.scalar('loss_class', loss_class_generator)
            summary_gloss_tot = tf.summary.scalar('loss_tot', loss_total_generator)
            summary_op_gloss = tf.summary.merge(
                [summary_gloss_source,
                 summary_gloss_class,
                 summary_gloss_tot], name = 'loss_generator')
            
        return summary_op_dloss, summary_op_gloss, summary_op_img, summary_img
                                                                 
        
    def train(self, hparams_string):
        """ Run training of the network
        Args:
    
        Returns:
        """
        args_train = hparams_parser_train(hparams_string)

        self.batch_size = args_train.batch_size
        self.epoch_max = args_train.epoch_max

        self.unstructured_noise_dim = args_train.unstructured_noise_dim
        
        self.d_learning_rate = args_train.lr_discriminator
        self.g_learning_rate = args_train.lr_generator

        self.d_iter = args_train.d_iter
        self.n_testsamples = args_train.n_testsamples

        self.gp_lambda = args_train.gp_lambda
        self.class_scale_d = args_train.class_scale_d
        self.class_scale_g = args_train.class_scale_g

        self.backup_frequency = args_train.backup_frequency

        utils.save_model_configuration(args_train, self.dir_base)

        
        # Use dataset for loading in datasamples from .tfrecord (https://www.tensorflow.org/programmers_guide/datasets#consuming_tfrecord_data)
        # The iterator will get a new batch from the dataset each time a sess.run() is executed on the graph.
        dataset = tf.data.TFRecordDataset(self.dateset_filenames)
        dataset = dataset.map(util_data.decode_image)      # decoding the tfrecord
        dataset = dataset.map(self._genLatentCodes)
        dataset = dataset.shuffle(buffer_size = 10000, seed = None)
        dataset = dataset.batch(batch_size = self.batch_size)
        iterator = dataset.make_initializable_iterator()
        input_getBatch = iterator.get_next()

        # Create input placeholders
        input_images = tf.placeholder(
            dtype = tf.float32, 
            shape = [self.batch_size] + self.image_dims, 
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

        images = input_images       
        
        # Define model, loss, optimizer and summaries.
        logits_source, logits_class, artificial_images = self._create_inference(images, input_lbls, input_unstructured_noise)
        loss_discriminator, loss_generator = self._create_losses(logits_source, logits_class, artificial_images, input_lbls)
        train_op_discriminator, train_op_generator = self._create_optimizer(loss_discriminator, loss_generator)
        summary_op_dloss, summary_op_gloss, summary_op_img, summary_img = self._create_summaries(loss_discriminator, loss_generator, input_test_noise, input_test_lbls)

        # show network architecture
        utils.show_all_variables()

        # create constant test variable to inspect changes in the model
        test_noise, test_lbls = self._genTestInput(self.lbls_dim, n_samples = self.n_testsamples)

        dir_results_train = os.path.join(self.dir_results, 'Training')
        utils.checkfolder(dir_results_train)

        with tf.Session() as sess:
            # Initialize all model Variables.
            sess.run(tf.global_variables_initializer())
            
            # Create Saver object for loading and storing checkpoints
            saver = tf.train.Saver(max_to_keep=100)
            
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
            for epoch_n in range(epoch_start, self.epoch_max):

                # Test model output before any training
                if epoch_n == 0:
                    summaryImg_tb, summaryImg = sess.run(
                        [summary_op_img, summary_img],
                        feed_dict={input_test_noise:    test_noise,
                                   input_test_lbls:     test_lbls})

                    writer.add_summary(summaryImg_tb, global_step=-1)
                    utils.save_image_local(summaryImg, dir_results_train, 'Epoch_' + str(-1))

                # Initiate or Re-initiate iterator
                sess.run(iterator.initializer)
                
                ### ----------------------------------------------------------
                ### Update model
                utils.show_message('Running training epoch no: {0}'.format(epoch_n))
                while True:
                # for idx in range(0, num_batches):
                    try:
                        for _ in range(self.d_iter):
                            image_batch, lbl_batch, unst_noise_batch = sess.run(input_getBatch)
                            
                            if(image_batch.shape[0] != self.batch_size):
                                raise OutOfRangeError

                            _, summary_dloss = sess.run(
                                [train_op_discriminator, summary_op_dloss],
                                feed_dict={input_images:            image_batch,
                                        input_lbls:                 lbl_batch,
                                        input_unstructured_noise:   unst_noise_batch})
                                        
                        writer.add_summary(summary_dloss, global_step=interationCnt)

                        _, summary_gloss = sess.run(
                            [train_op_generator, summary_op_gloss],
                            feed_dict={input_images:            image_batch,
                                    input_lbls:                 lbl_batch,
                                    input_unstructured_noise:   unst_noise_batch})

                        writer.add_summary(summary_gloss, global_step=interationCnt)
                        interationCnt += 1

                    except (tf.errors.OutOfRangeError, OutOfRangeError):
                        # Test current model
                        summaryImg_tb, summaryImg = sess.run(
                            [summary_op_img, summary_img],
                            feed_dict={input_test_noise:    test_noise,
                                        input_test_lbls:     test_lbls})

                        writer.add_summary(summaryImg_tb, global_step=epoch_n)
                        utils.save_image_local(summaryImg, dir_results_train, 'Epoch_' + str(epoch_n))

                        break
                
                # Save model variables to checkpoint
                if (epoch_n +1) % self.backup_frequency == 0:
                    saver.save(sess,os.path.join(self.dir_checkpoints, self.model), global_step=epoch_n)
            
    
    def evaluate(self, hparams_string):
        """ Run experiments to evaluate the performance of the model
        Args:
    
        Returns:
        """
        args_train = utils.load_model_configuration(self.dir_base)
        args_evaluate = hparams_parser_evaluate(hparams_string)

        self.unstructured_noise_dim = args_train.unstructured_noise_dim

        input_lbls = tf.placeholder(
            dtype = tf.float32, 
            shape = [None, self.lbls_dim], 
            name = 'input_test_lbls')
        input_noise = tf.placeholder(
            dtype = tf.float32, 
            shape = [None, self.unstructured_noise_dim], 
            name = 'input_test_noise')

        _  = self.__generator(input_noise, input_lbls)
        generated_images = self.__generator(input_noise, input_lbls, is_training=False, reuse=True)

        num_samples = 200

        ckpt = tf.train.get_checkpoint_state(self.dir_checkpoints)
        
        if args_evaluate.epoch_no == None:
            checkpoint_path = ckpt.model_checkpoint_path
        else:
            all_checkpoint_paths = ckpt.all_model_checkpoint_paths[:]
            suffix_match = '-'+str(args_evaluate.epoch_no)
            ckpt_match = [f for f in all_checkpoint_paths if f.endswith(suffix_match)]
            
            if ckpt_match:
                checkpoint_path = ckpt_match[0]
            else:
                checkpoint_path = ckpt.model_checkpoint_path

        with tf.Session() as sess:
            # Initialize all model Variables.
            sess.run(tf.global_variables_initializer())
            
            # Create Saver object for loading and storing checkpoints
            saver = tf.train.Saver()

            # Reload Tensor values from latest or specified checkpoint
            saver.restore(sess, checkpoint_path)
        
            # Generate evaluation noise
            np.random.seed(seed = 0)
            eval_noise = np.random.uniform(low = -1.0, high = 1.0, size = [num_samples, self.unstructured_noise_dim])

            # Generate artificial images for each class
            for i in range(0,self.lbls_dim):
                utils.show_message('Generating images for class ' + str(i))
                
                eval_lbls = np.zeros(shape = [num_samples, self.lbls_dim])
                eval_lbls[:,i] = 1

                eval_images = sess.run(
                    generated_images, 
                    feed_dict={input_noise: eval_noise,
                               input_lbls:  eval_lbls})
                
                dir_results_eval = os.path.join(self.dir_results, 'Evaluation', str(i))
                utils.checkfolder(dir_results_eval)

                for j in range(0,num_samples):
                    utils.save_image_local(eval_images[j,:,:,:], dir_results_eval,'Sample_' + str(j))
    
    
    def _genLatentCodes(self, image_proto, lbl_proto, class_proto, height_proto, width_proto, channels_proto, origin_proto):
        """ Augment dataset entries. Adds two continuous latent 
            codes for the network to estimate. Also generates a GAN noise
            vector per data sample.
        Args:

        Returns:
        """

        image = image_proto

        # Dataset specific preprocessing
        if self.dataset == 'MNIST':
            pass

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

    

