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
import itertools

import src.utils as utils
import src.data.util_data as util_data
import src.data.datasets.psd as psd_dataset
import src.data.datasets.GAN_samples as GAN_samples
import src.data.preprocess_factory as preprocess_factory

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
                        default='15',
                        help='Number of samples in test images per class')

    parser.add_argument('--unstructured_noise_dim',
                        type=int, 
                        default='128',
                        help='Number of random inputn_testsamples')
    
    parser.add_argument('--d_iter',
                        type = int,
                        default = '5',
                        help = 'Number of times the n_testsamples')
    
    parser.add_argument('--gp_lambda',
                        type = int,
                        default = '10',
                        help = 'Gradient penalty weight')
    
    parser.add_argument('--class_scale_d',
                        type = float,
                        default = '7.5',
                        help = 'Scale significance of discriminator class loss')

    parser.add_argument('--class_scale_g',
                        type = float,
                        default = '7.5',
                        help = 'Scale significance of generator class loss')
    
    parser.add_argument('--backup_frequency',
                        type = int,
                        default = '100',
                        help = 'Number of iterations between backup of network weights')

    parser.add_argument('--info_var_dim',
                        type = int,
                        default = '2',
                        help = 'Dimensions of the latent info variables')

    parser.add_argument('--info_scale_d',
                        type = float,
                        default = '15',
                        help = 'Scale significance of discriminator info loss')

    parser.add_argument('--info_scale_g',
                        type = float,
                        default = '15',
                        help = 'Scale significance of generator info loss')

    parser.add_argument('--shards_idx_test',
                        nargs='+',
                        type = int,
                        default = 0,
                        help = '')

    return parser.parse_args(shlex.split(hparams_string))



def hparams_parser_evaluate(hparams_string):
    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch_no', 
                        type=int,
                        default=None, 
                        help='Epoch no to reload')

    parser.add_argument('--gen_samples',
                        action='store_true', 
                        help = 'Generates random samples from each class. [Defaults to False if argument is omitted]')

    parser.add_argument('--gen_summary',
                        action='store_true', 
                        help = 'Generates eval summary samples from the model. [Defaults to False if argument is omitted]')
    
    parser.add_argument('--gen_interpolations',
                        action='store_true', 
                        help = 'Generate interpolations between random samples. [Defaults to False if argument is omitted]')
    
    parser.add_argument('--num_samples',
                        type=int,
                        default=200,
                        help='Number of random samples to generate')

    parser.add_argument('--summary_samples',
                        type=int,
                        default=15,
                        help='Number of samples in the summary')

    parser.add_argument('--chunk_size',
                        type=int,
                        default=200,
                        help='Divide the total number of samples into smaller chunk, to lessen load in forward pass')

    parser.add_argument('--convert_samples',
                        action='store_true', 
                        help = 'Post evaluation command. Converts GAN samples to tfrecord')

    return parser.parse_args(shlex.split(hparams_string))


class WacGAN_info(object):
    def __init__(self, dataset, id):

        self.model = 'WacGAN_info'
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
            self.dataset_filenames = ['data/processed/MNIST/train.tfrecord']
            self.lbls_dim = 10
            self.image_dims = [128,128,1]

        elif dataset == 'PSD_Nonsegmented':
            self.dataset_filenames = ['data/processed/PSD_Nonsegmented/data_shard_{:03d}-of-{:03d}.tfrecord'.format(i,psd_dataset._NUM_SHARDS) for i in range(psd_dataset._NUM_SHARDS)]
            self.lbls_dim = 9
            self.image_dims = [128,128,3]

        elif dataset == 'PSD_Segmented':
            # self.dataset_filenames = ['data/processed/PSD_Segmented_Anders/data_shard_{:03d}-of-{:03d}.tfrecord'.format(i+1,psd_dataset._NUM_SHARDS) for i in range(psd_dataset._NUM_SHARDS)]
            self.dataset_filenames = ['data/processed/PSD_Segmented/data_shard_{:03d}-of-{:03d}.tfrecord'.format(i+1,psd_dataset._NUM_SHARDS) for i in range(psd_dataset._NUM_SHARDS)]
            self.lbls_dim = 9
            self.image_dims = [128,128,3]

        else:
            raise ValueError('Selected Dataset is not supported by model: '+ self.model)
        
        self.dataset = dataset

 
    def __generator(self, noise, lbls_onehot, info_noise, weight_decay = 2.5e-5, is_training = True, reuse=False):
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

            all_noise = tf.concat([noise, lbls_onehot, info_noise], axis=1)
        
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
                logits_class = layers.fully_connected(net, self.lbls_dim, normalizer_fn = None, activation_fn = None)
                logits_info = layers.fully_connected(net, self.info_var_dim, normalizer_fn = None, activation_fn = None)

                return logits_source, logits_class, logits_info
    

    def _create_inference(self, images, labels, noise, info_noise):
        """ Define the inference model for the network
        Args:
    
        Returns:
        """
        generated_images = self.__generator(noise, labels, info_noise)
        logits_source_real, logits_class_real, logits_info_real = self.__discriminator(images)
        logits_source_fake, logits_class_fake, logits_info_fake = self.__discriminator(generated_images, reuse = True)

        epsilon = tf.random_uniform(shape = images.get_shape(),
                                    minval= 0.0,
                                    maxval= 1.0)
        interpolated_images = epsilon * images + (1-epsilon) * generated_images
        logits_source_interpolates, logits_class_interpolates, logits_info_interpolates = self.__discriminator(interpolated_images, reuse = True)

        logits_source = [logits_source_real, logits_source_fake, logits_source_interpolates]
        logits_class = [logits_class_real, logits_class_fake, logits_class_interpolates]
        logits_info = [logits_info_real, logits_info_fake, logits_info_interpolates]
        artificial_images = [generated_images, interpolated_images]

        return logits_source, logits_class, logits_info, artificial_images

    
    def _create_losses(self, logits_source, logits_class, logits_info, artificial_images, labels, latent_info):
        """ Define loss function[s] for the network
        Args:
    
        Returns:
        """

        [logits_source_real, logits_source_fake, logits_source_interpolates] = logits_source
        [logits_class_real, logits_class_fake, _] = logits_class
        [_ , logits_info_fake, _] = logits_info
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

        # info loss
        loss_info = tf.reduce_mean(tf.reduce_sum(tf.square(latent_info - logits_info_fake), axis=1))

        # Total losses, with scaled class loss
        loss_total_discriminator = loss_source_discriminator + self.class_scale_d * loss_class_discriminator + self.info_scale_d * loss_info
        loss_total_generator = loss_source_generator + self.class_scale_g * loss_class_generator + self.info_scale_g * loss_info

        loss_discriminator = [loss_total_discriminator, 
                              loss_source_discriminator, 
                              loss_class_discriminator,
                              loss_info]
        loss_generator     = [loss_total_generator, 
                              loss_source_generator,
                              loss_class_generator,
                              loss_info]

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

        
    def _create_summaries(self, loss_discriminator, loss_generator, test_noise, test_labels, test_info):
        """ Create summaries for the network
        Args:
    
        Returns:
        """
        test_images = self.__generator(test_noise, test_labels, test_info, is_training=False, reuse=True)

        # Create image summaries to inspect the variation due to categorical latent codes
        with tf.name_scope("SummaryImages_ClassVariation"):
            summary_img = tfgan.eval.image_reshaper(tf.concat(test_images, 0), num_cols=self.n_testsamples)
            summary_op_img = tf.summary.image('In', summary_img, max_outputs = 1)

        ### Add loss summaries
        [loss_total_discriminator, loss_source_discriminator, loss_class_discriminator, loss_info_discriminator] = loss_discriminator
        [loss_total_generator, loss_source_generator, loss_class_generator, loss_info_generator] = loss_generator

        with tf.name_scope("SummaryLosses_Discriminator"):

            summary_dloss_source = tf.summary.scalar('loss_source', loss_source_discriminator)
            summary_dloss_class = tf.summary.scalar('loss_class', loss_class_discriminator)
            summary_dloss_info = tf.summary.scalar('loss_info', loss_info_discriminator)
            summary_dloss_tot = tf.summary.scalar('loss_total', loss_total_discriminator)
            summary_op_dloss = tf.summary.merge(
                [summary_dloss_source,
                 summary_dloss_class,
                 summary_dloss_info,
                 summary_dloss_tot], name = 'loss_discriminator')

        with tf.name_scope("SummaryLosses_Generator"):

            summary_gloss_source = tf.summary.scalar('loss_source', loss_source_generator)
            summary_gloss_class = tf.summary.scalar('loss_class', loss_class_generator)
            summary_gloss_info = tf.summary.scalar('loss_info', loss_info_generator)
            summary_gloss_tot = tf.summary.scalar('loss_tot', loss_total_generator)
            summary_op_gloss = tf.summary.merge(
                [summary_gloss_source,
                 summary_gloss_class,
                 summary_gloss_info,
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
        self.info_var_dim = args_train.info_var_dim
        self.n_testsamples = args_train.n_testsamples
        
        self.d_learning_rate = args_train.lr_discriminator
        self.g_learning_rate = args_train.lr_generator
        self.d_iter = args_train.d_iter

        self.gp_lambda = args_train.gp_lambda
        self.class_scale_d = args_train.class_scale_d
        self.class_scale_g = args_train.class_scale_g

        self.info_scale_d = args_train.info_scale_d
        self.info_scale_g = args_train.info_scale_g

        self.backup_frequency = args_train.backup_frequency

        self.shards_idx_test = args_train.shards_idx_test

        utils.save_model_configuration(args_train, self.dir_base)

        # Create folder for saving training results
        dir_results_train = os.path.join(self.dir_results, 'Training')
        utils.checkfolder(dir_results_train)

        for class_n in range(self.lbls_dim):
            dir_result_train_class = dir_results_train + '/' + str(class_n).zfill(2)
            utils.checkfolder(dir_result_train_class)

        if 0 in self.shards_idx_test:
            dataset_filenames = self.dataset_filenames
        else:
            self.shards_idx_test = np.subtract(self.shards_idx_test, 1)
            shards_idx_training = np.delete(range(len(self.dataset_filenames)), self.shards_idx_test)
            dataset_filenames = [self.dataset_filenames[i] for i in shards_idx_training]

            utils.show_message('Training Data:')
            print(dataset_filenames)

        # Setup preprocessing pipeline
        preprocessing = preprocess_factory.preprocess_factory()

        # Dataset specific preprocessing
        if self.dataset == 'MNIST':
            pass

        elif self.dataset == 'PSD_Nonsegmented':
            pass

        elif self.dataset == 'PSD_Segmented':
            preprocessing.prep_pipe_from_string("pad_to_size;{'height': 566, 'width': 566, 'constant': -1.0};random_rotation;{};crop_to_size;{'height': 400, 'width': 400};resize;{'height': 128, 'width': 128}")


        
        # Use dataset for loading in datasamples from .tfrecord (https://www.tensorflow.org/programmers_guide/datasets#consuming_tfrecord_data)
        # The iterator will get a new batch from the dataset each time a sess.run() is executed on the graph.
        dataset = tf.data.TFRecordDataset(dataset_filenames)
        dataset = dataset.shuffle(buffer_size = 10000, seed = None)
        dataset = dataset.map(util_data.decode_image)           # decoding the tfrecord
        dataset = dataset.map(self._genLatentCodes)             # preprocess data and perform augmentation
        dataset = dataset.map(preprocessing.pipe)
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
        input_info_noise = tf.placeholder(
            dtype = tf.float32, 
            shape = [None, self.info_var_dim], 
            name = 'input_info_noise')
        input_test_lbls = tf.placeholder(
            dtype = tf.float32, 
            shape = [self.n_testsamples**np.minimum(2, self.info_var_dim), self.lbls_dim], 
            name = 'input_test_lbls')
        input_test_noise = tf.placeholder(
            dtype = tf.float32, 
            shape = [self.n_testsamples**np.minimum(2, self.info_var_dim), self.unstructured_noise_dim], 
            name = 'input_test_noise')
        input_test_info_noise = tf.placeholder(
            dtype = tf.float32, 
            shape = [self.n_testsamples**np.minimum(2, self.info_var_dim), self.info_var_dim], 
            name = 'input_test_info_noise')

        
        # Define model, loss, optimizer and summaries.
        logits_source, logits_class, logits_info, artificial_images = self._create_inference(input_images, input_lbls, input_unstructured_noise, input_info_noise)
        loss_discriminator, loss_generator = self._create_losses(logits_source, logits_class, logits_info, artificial_images, input_lbls, input_info_noise)
        train_op_discriminator, train_op_generator = self._create_optimizer(loss_discriminator, loss_generator)
        summary_op_dloss, summary_op_gloss, summary_op_img, summary_img = self._create_summaries(loss_discriminator, loss_generator, input_test_noise, input_test_lbls, input_test_info_noise)

        # show network architecture
        utils.show_all_variables()

        # create constant test variable to inspect changes in the model
        self.combinations_info_var = itertools.combinations(range(self.info_var_dim),2)
        self.combinations_info_var = list(self.combinations_info_var)

        test_noise, test_info = self._genTestInput()

        with tf.Session() as sess:
            # Initialize all model Variables.
            sess.run(tf.global_variables_initializer())
            
            # Create Saver object for loading and storing checkpoints
            saver = tf.train.Saver(max_to_keep=500)
            
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
                    for class_n in range(self.lbls_dim):
                        test_lbls = np.zeros([self.n_testsamples**np.minimum(2, self.info_var_dim),self.lbls_dim])
                        test_lbls[:,class_n] = 1

                        for i in range(len(test_info)):
                            test_info_combi = test_info[i]

                            _ , summaryImg = sess.run(
                                [summary_op_img, summary_img],
                                feed_dict={input_test_noise:     test_noise,
                                        input_test_lbls:         test_lbls,
                                        input_test_info_noise:   test_info_combi})
                            
                            dir_result_train_class = dir_results_train + '/' + str(class_n).zfill(2)
                            if self.info_var_dim < 2:
                                filename_temp = 'Epoch_{0}_LatentVar_1'.format(epoch_n)
                            else:
                                filename_temp = 'Epoch_{0}_LatentCombi_{1}_{2}'.format(epoch_n,self.combinations_info_var[i][0],self.combinations_info_var[i][1])

                            # writer.add_summary(summaryImg_tb, global_step=epoch_n)
                            utils.save_image_local(summaryImg, dir_result_train_class,filename_temp)

                # Initiate or Re-initiate iterator
                sess.run(iterator.initializer)
                
                ### ----------------------------------------------------------
                ### Update model
                if (np.mod(epoch_n,100) == 0) or epoch_n < 25:
                    utils.show_message('Running training epoch no: {0}'.format(epoch_n))

                while True:
                # for idx in range(0, num_batches):
                    try:
                        for _ in range(self.d_iter):
                            image_batch, lbl_batch, unst_noise_batch, info_noise_batch = sess.run(input_getBatch)
                            
                            if(image_batch.shape[0] != self.batch_size):
                                raise OutOfRangeError

                            _, summary_dloss = sess.run(
                                [train_op_discriminator, summary_op_dloss],
                                feed_dict={input_images:            image_batch,
                                        input_lbls:                 lbl_batch,
                                        input_unstructured_noise:   unst_noise_batch,
                                        input_info_noise:           info_noise_batch})
                                        
                        writer.add_summary(summary_dloss, global_step=interationCnt)

                        _, summary_gloss = sess.run(
                            [train_op_generator, summary_op_gloss],
                            feed_dict={input_images:            image_batch,
                                    input_lbls:                 lbl_batch,
                                    input_unstructured_noise:   unst_noise_batch,
                                    input_info_noise:           info_noise_batch})

                        writer.add_summary(summary_gloss, global_step=interationCnt)
                        interationCnt += 1

                    except (tf.errors.OutOfRangeError, OutOfRangeError):
                        # Test current model
                        for class_n in range(self.lbls_dim):
                            test_lbls = np.zeros([self.n_testsamples**np.minimum(2, self.info_var_dim),self.lbls_dim])
                            test_lbls[:,class_n] = 1

                            for i in range(len(test_info)):
                                test_info_combi = test_info[i]

                                _, summaryImg = sess.run(
                                    [summary_op_img, summary_img],
                                    feed_dict={input_test_noise:     test_noise,
                                            input_test_lbls:         test_lbls,
                                            input_test_info_noise:   test_info_combi})
                                
                                dir_result_train_class = dir_results_train + '/' + str(class_n).zfill(2)
                                if self.info_var_dim < 2:
                                    filename_temp = 'Epoch_{0}_LatentVar_1'.format(epoch_n)
                                else:
                                    filename_temp = 'Epoch_{0}_LatentCombi_{1}_{2}'.format(epoch_n,self.combinations_info_var[i][0],self.combinations_info_var[i][1])

                                # writer.add_summary(summaryImg_tb, global_step=epoch_n)
                                utils.save_image_local(summaryImg, dir_result_train_class,filename_temp)

                        break
                
                # Save model variables to checkpoint
                if (epoch_n +1) % self.backup_frequency == 0:
                    saver.save(sess,os.path.join(self.dir_checkpoints, self.model + '.model'), global_step=epoch_n)
            
    
    def evaluate(self, hparams_string):
        """ Run experiments to evaluate the performance of the model
        Args:
    
        Returns:
        """

        args_train = utils.load_model_configuration(self.dir_base)
        args_evaluate = hparams_parser_evaluate(hparams_string)

        self.unstructured_noise_dim = args_train.unstructured_noise_dim
        self.info_var_dim = args_train.info_var_dim

        summary_samples = args_evaluate.summary_samples
        num_samples = args_evaluate.num_samples
        chunk_size = args_evaluate.chunk_size

        # setup inference
        input_lbls = tf.placeholder(
            dtype = tf.float32, 
            shape = [None, self.lbls_dim], 
            name = 'input_test_lbls')
        input_noise = tf.placeholder(
            dtype = tf.float32, 
            shape = [None, self.unstructured_noise_dim], 
            name = 'input_test_noise')
        input_info_noise = tf.placeholder(
            dtype = tf.float32, 
            shape = [None, self.info_var_dim], 
            name = 'input_test_noise')

        generated_images = self.__generator(input_noise, input_lbls, input_info_noise, is_training=False)
        logits_source, logits_class, _ = self.__discriminator(generated_images, is_training=False)

        summary_images = tf.placeholder(
            dtype = tf.float32, 
            shape = [self.lbls_dim*summary_samples] + self.image_dims, 
            name = 'summary_images')
        eval_summary_img = tfgan.eval.image_reshaper(tf.concat(summary_images, 0), num_cols=self.lbls_dim)

        # select check point file
        ckpt = tf.train.get_checkpoint_state(self.dir_checkpoints)
        
        if args_evaluate.epoch_no == None:
            checkpoint_path = ckpt.model_checkpoint_path
            dir_results_eval = os.path.join(self.dir_results, 'Evaluation')
        else:
            all_checkpoint_paths = ckpt.all_model_checkpoint_paths[:]
            suffix_match = '-'+str(args_evaluate.epoch_no)
            ckpt_match = [f for f in all_checkpoint_paths if f.endswith(suffix_match)]
            
            if ckpt_match:
                checkpoint_path = ckpt_match[0]
                dir_results_eval = os.path.join(self.dir_results, 'Evaluation_' + str(args_evaluate.epoch_no))
                
            else:
                checkpoint_path = ckpt.model_checkpoint_path
                dir_results_eval = os.path.join(self.dir_results, 'Evaluation')

        # Generate folders for evaluation samples
        utils.checkfolder(dir_results_eval)


        with tf.Session() as sess:
            # Initialize all model Variables.
            sess.run(tf.global_variables_initializer())
            
            # Create Saver object for loading and storing checkpoints
            saver = tf.train.Saver()

            # Reload Tensor values from latest or specified checkpoint
            utils.show_message('Restoring model parameters from: {0}'.format(checkpoint_path))
            saver.restore(sess, checkpoint_path)

            ## Generate summary image
            if args_evaluate.gen_summary:
                np.random.seed(seed = 0)
                eval_unstructured_noise = np.random.uniform(low = -1.0, high = 1, size = [summary_samples, self.unstructured_noise_dim])
                eval_unstructured_noise = np.repeat(eval_unstructured_noise, self.lbls_dim, axis = 0)

                eval_info_noise = np.random.uniform(low = -1.0, high = 1, size = [summary_samples, self.info_var_dim])
                eval_info_noise = np.repeat(eval_info_noise, self.lbls_dim, axis = 0)
        
                # Create one-hot encoded label for each class and tile along axis 1
                eval_lbls = np.eye(self.lbls_dim)
                eval_lbls = np.tile(eval_lbls,(summary_samples,1))

                summary_imgs = sess.run(
                    generated_images, 
                    feed_dict={input_noise:         eval_unstructured_noise,
                               input_info_noise:    eval_info_noise,
                               input_lbls:          eval_lbls})
                
                summary_img = sess.run(
                    eval_summary_img, 
                    feed_dict={summary_images:      summary_imgs})
                
                utils.save_image_local(summary_img, self.dir_results, 'evalSummary_{0}'.format(args_evaluate.epoch_no))


            ## Generate samples for each class
            if args_evaluate.gen_samples:
                # Seed RNG to reproduce results
                np.random.seed(seed = 0)
                eval_noise = np.random.uniform(low = -1.0, high = 1.0, size = [num_samples,self.unstructured_noise_dim])
                eval_noise_info = np.random.uniform(low = -1.0, high = 1.0, size = [num_samples,self.info_var_dim])

                chunks_eval_noise = [eval_noise[i:i + chunk_size] for i in range(0, num_samples, chunk_size)]
                chunks_eval_noise_info = [eval_noise_info[i:i + chunk_size] for i in range(0, num_samples, chunk_size)]

                for idx_class in range(self.lbls_dim):
                    utils.show_message('Generating samples for class ' + str(idx_class))

                    dir_results_eval_samples = os.path.join(dir_results_eval, 'Samples', str(idx_class))
                    utils.checkfolder(dir_results_eval_samples)

                    for idx_chunk in range(int(np.ceil(num_samples/chunk_size))):
                        eval_lbls = np.zeros(shape = [len(chunks_eval_noise[idx_chunk]), self.lbls_dim])
                        eval_lbls[:,idx_class] = 1

                        eval_images, logits_s, logits_c = sess.run(
                            [generated_images, logits_source, logits_class], 
                            feed_dict={input_noise:         chunks_eval_noise[idx_chunk],
                                       input_info_noise:    chunks_eval_noise_info[idx_chunk],
                                       input_lbls:          eval_lbls})

                        #print(logits_s)

                        f = open(dir_results_eval_samples+'/scores.txt', 'a')
                        for idx_sample in range(len(chunks_eval_noise[idx_chunk])):
                            utils.save_image_local(eval_images[idx_sample,:,:,:], dir_results_eval_samples, 'Sample_{0}'.format(idx_sample + idx_chunk*chunk_size))
                            f.write('Sample_{0},{1},{2}\n'.format(idx_sample + idx_chunk*chunk_size, logits_s[idx_sample], logits_c[idx_sample,:]))
                        f.close()

    def post_evaluation(self, hparams_string):
        """ Run post evaluation on the results of the model
        Args:
    
        Returns:
        """
        args_evaluate = hparams_parser_evaluate(hparams_string)

        if args_evaluate.epoch_no == None:
            setname = 'Evaluation'
        else:
            setname = 'Evaluation_' + str(args_evaluate.epoch_no)

        dir_results_eval = os.path.join(self.dir_results, setname)

        if not(os.path.isdir(dir_results_eval)):
            utils.show_message('{0} does not exist!'.format(dir_results_eval))
            return
        
        if args_evaluate.convert_samples:
            if os.path.isdir(os.path.join(dir_results_eval,'Samples')):
                GAN_samples.interim(self.model, setname)
            else:
                utils.show_message('No GAN samples found in {0}'.format(dir_results_eval))
                               
    
    
    def _genLatentCodes(self, image_proto, lbl_proto, class_proto, height_proto, width_proto, channels_proto, origin_proto):
        """ Augment dataset entries. Adds two continuous latent 
            codes for the network to estimate. Also generates a GAN noise
            vector per data sample.
        Args:

        Returns:
        """

        image = image_proto
        lbl = tf.one_hot(lbl_proto, self.lbls_dim)

        unstructured_noise = tf.random_uniform([self.unstructured_noise_dim], minval = -1, maxval = 1)
        info_noise = tf.random_uniform([self.info_var_dim], minval = -1, maxval = 1)
    
        return image, lbl, unstructured_noise, info_noise
    
    def _genTestInput(self):
        """ Defines test code and noise generator. Generates laten codes based
            on a testCategory input.
        Args:
    
        Returns:
        """

        test_info = []

        if self.info_var_dim < 2:
            test_unstructured_noise = np.random.uniform(low = -1.0, high = 1.0, size = [self.n_testsamples, self.unstructured_noise_dim])
            test_info_temp = np.expand_dims(np.linspace(-1,1,self.n_testsamples), axis=1)
            test_info.append(test_info_temp)

        else:
            test_unstructured_noise = np.random.uniform(low = -1.0, high = 1.0, size = [self.n_testsamples**2, self.unstructured_noise_dim])

            for combi in self.combinations_info_var:
                test_info_temp = np.zeros(shape = [self.n_testsamples**2,self.info_var_dim])
                test_info_temp[:,combi[0]] = np.repeat(np.linspace(-1,1,self.n_testsamples),self.n_testsamples)
                test_info_temp[:,combi[1]] = np.tile(np.linspace(-1,1,self.n_testsamples),self.n_testsamples)

                test_info.append(test_info_temp)

        return test_unstructured_noise, test_info

        # dim_latentVars = dim_latentVars
        # n_combinations = (dim_latentVars-1)*(dim_latentVars)/2

        # if n_combinations == 0:
        #     n_testsamples = n_testsamples
        # else:
        #     n_testsamples = n_testsamples^2 * n_combinations

        # test_unstructured_noise = np.random.uniform(low = -1.0, high = 1.0, size = [n_testsamples, self.unstructured_noise_dim])

        # test_info = np.zeros(shape = [n_testsamples,dim_latentVars])

        # # Create repeating noise vector, so a sample for each class is created using the same noise
        # test_unstructured_noise = np.random.uniform(low = -1.0, high = 1, size = [self.n_testsamples *self.n_testsamples, self.unstructured_noise_dim])
        
        # # Create one-hot encoded label for each class and tile along axis 0
        # test_labels = np.eye(self.lbls_dim)
        # test_labels = np.repeat(test_labels,self.n_testsamples*self.n_testsamples,axis = 0)

        # test_info = np.linspace(-1, 1, num = self.n_testsamples)
        # test_info = np.array(list(product(test_info,test_info)))

        # return test_unstructured_noise, test_labels, test_info

    

