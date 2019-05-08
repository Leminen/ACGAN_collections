
import sys
import os
import tensorflow as tf
import numpy as np

import src.utils as utils
import src.data.util_data as util_data
import src.data.datasets.psd as psd_dataset
import src.data.preprocess_factory as preprocess_factory


class OutOfRangeError(Exception): pass

def _genLatentCodes(image_proto, lbl_proto, class_proto, height_proto, width_proto, channels_proto, origin_proto):
    """ Augment dataset entries. Adds two continuous latent 
        codes for the network to estimate. Also generates a GAN noise
        vector per data sample.
    Args:

    Returns:
    """

    image = image_proto
    lbls_dim = 9
    lbl = tf.one_hot(lbl_proto, lbls_dim)

    return image, lbl

# Setup preprocessing pipeline
preprocessing = preprocess_factory.preprocess_factory()
preprocessing.prep_pipe_from_string("pad_to_size;{'height': 566, 'width': 566, 'constant': -1.0};random_rotation;{};crop_to_size;{'height': 400, 'width': 400};resize;{'height': 128, 'width': 128}")

dataset_filenames = ['data/processed/PSD_Segmented/data_shard_{:03d}-of-{:03d}.tfrecord'.format(i+1,psd_dataset._NUM_SHARDS) for i in range(psd_dataset._NUM_SHARDS)]
batch_size = 64


dataset = tf.data.TFRecordDataset(dataset_filenames)
dataset = dataset.shuffle(buffer_size = 10000, seed = None)
dataset = dataset.map(util_data.decode_image)       # decoding the tfrecord
dataset = dataset.map(_genLatentCodes)              # preprocess data and perform data augmentation
dataset = dataset.map(preprocessing.pipe)           # preprocess data and perform data augmentation
dataset = dataset.batch(batch_size = batch_size)
iterator = dataset.make_initializable_iterator()
input_getBatch = iterator.get_next()


n_epochs = 10

with tf.Session() as sess:
    # Initialize all model Variables.
    sess.run(tf.global_variables_initializer())

    for n in range(n_epochs):

        # Initiate or Re-initiate iterator
        sess.run(iterator.initializer)

        while True:
            try:
                image_batch, lbl_batch = sess.run(input_getBatch)
                if(image_batch.shape[0] != batch_size):
                    raise OutOfRangeError
            
            except (tf.errors.OutOfRangeError, OutOfRangeError):
                break



