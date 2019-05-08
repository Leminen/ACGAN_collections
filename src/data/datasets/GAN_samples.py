"""
Methods for downloading and converting the MNIST dataset to TF-records

implementation is heavily inspired by the slim.datasets implementation (https://github.com/tensorflow/models/tree/master/research/slim/datasets)
"""
import os
import sys
import shutil
import distutils.dir_util as dir_util

import numpy as np
import tensorflow as tf

import src.utils as utils
import src.data.util_data as util_data

_NUM_SHARDS = 1


def chunkify(lst,n):
    return [lst[i::n] for i in iter(range(n))]

class ImageReader(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        # Initializes function that decodes RGB PNG data.
        self._decode_png_data = tf.placeholder(dtype=tf.string)
        self._decode_png = tf.image.decode_png(self._decode_png_data, channels=3)
        self._encode_png = tf.image.encode_png(self._decode_png)

    def truncate_image(self, sess, image_data):
        image, reencoded_image = sess.run(
            [self._decode_png, self._encode_png],
            feed_dict={self._decode_png_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return reencoded_image, image.shape[0], image.shape[1], image.shape[2]

    def read_image_dims(self, sess, image_data):
        image = self.decode_png(sess, image_data)
        return image.shape[0], image.shape[1], image.shape[2]

    def decode_png(self, sess, image_data):
        image = sess.run(self._decode_png,
            feed_dict={self._decode_png_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image

    def encode_png(self, sess, image_data):
        image_data = sess.run(self._encode_png,
            feed_dict={self._decode_png_data: image_data})


def _get_filenames_and_classes(dataset_dir, setname):
    """Returns a list of filenames and inferred class names.

    Args:
      dataset_dir: A directory containing a set of subdirectories representing
        class names. Each subdirectory should contain PNG or JPG encoded images.

    Returns:
      A list of image file paths, relative to `dataset_dir` and the list of
      subdirectories, representing class names.
    """
    np.random.seed(0)
    data_root = os.path.join(dataset_dir, *setname)

    # list classes and class directories
    directories = [] 
    class_names = []
    for filename in os.listdir(data_root):
        path = os.path.join(data_root, filename)
        if os.path.isdir(path):
            directories.append(path)
            class_names.append(filename)

    # list filenames and split them into equal sized chunks for each class 
    photo_filenames = []
    for _ in range(_NUM_SHARDS):
        photo_filenames.append([])

    for directory in directories:
        filenames = os.listdir(directory)
        filenames = np.random.permutation(filenames) # shuffle list of filenames
        paths = [os.path.join(directory, filename) for filename in filenames]
            
        paths_split = chunkify(paths,_NUM_SHARDS)
        paths_split = np.random.permutation(paths_split) # shuffle splits to ensure equal sized shards

        for shard_n in range(_NUM_SHARDS):
            photo_filenames[shard_n].extend(paths_split[shard_n])

    return photo_filenames, sorted(class_names)


def _convert_to_tfrecord(filenames, class_dict, tfrecord_writer):
    """Loads data from the binary MNIST files and writes files to a TFRecord.

    Args:
        data_filename: The filename of the MNIST images.
        labels_filename: The filename of the MNIST labels.
        num_images: The number of images in the dataset.
        tfrecord_writer: The TFRecord writer to use for writing.
    """
    
    num_images = len(filenames)

    image_reader = ImageReader()

    with tf.Session('') as sess:
        for i in range(num_images):
            sys.stdout.write('\r>> Converting image %d/%d' % (i + 1, num_images))
            sys.stdout.flush()

            # Read the filename:
            encoded_img = tf.gfile.FastGFile(filenames[i], 'rb').read()
            encoded_img, height, width, channels = image_reader.truncate_image(sess, encoded_img)

            class_name = os.path.basename(os.path.dirname(filenames[i]))
            label = class_dict[class_name]

            example = util_data.encode_image(
                image_data = encoded_img,
                image_format = 'png'.encode(),
                class_lbl = label,
                class_text = class_name.encode(),
                height = height,
                width = width,
                channels = channels,
                origin = filenames[i].encode()
                )

            tfrecord_writer.write(example.SerializeToString())
        print('\n', end = '')
        

def _get_output_filename(dataset_dir, shard_id):
    """Creates the output filename.

    Args:
      dataset_dir: The directory where the temporary files are stored.
      split_name: The name of the train/test split.

    Returns:
      An absolute file path.
    """
    return '%s/data_shard_%03d-of-%03d.tfrecord' % (dataset_dir, shard_id+1, _NUM_SHARDS)


def download(dataset_part):
    """Downloads PSD locally
    """

    _dir_data = os.path.join('models', dataset_part,'results')
    if not(os.path.exists(_dir_data)):
        print('{0} directory does not exist'.format(dataset_part))
        return

    lst_evalFolders = os.listdir(_dir_data)
    lst_evalFolders = list(filter(lambda folder: folder.startswith('Evaluation'), lst_evalFolders))

    if len(lst_evalFolders)== 0:
        print('No model evaluations found for: {0}'.format(dataset_part))
        return
    
    for folder in lst_evalFolders:
        dir_src = os.path.join(_dir_data, folder)
        dir_dest = os.path.join('data/raw', 'GAN_samples_' + dataset_part, folder)
        print('>> Copying {0} to {1}'.format(dir_src, dir_dest))
        dir_util.copy_tree(dir_src, dir_dest)
        #shutil.copytree(dir_src, dir_dest)


def process(dataset_part):
    """Runs the conversion operation.

    Args:
      dataset_part: The dataset part to be converted.
    """

    _dir_raw = os.path.join('data/raw', 'GAN_samples_' + dataset_part)
    _dir_processed = os.path.join('data/processed', 'GAN_samples_' + dataset_part)
    setname = 'Samples'

    lst_evalFolders = os.listdir(_dir_raw)
    
    for folder in lst_evalFolders:
        dir_raw = os.path.join(_dir_raw,folder)
        dir_processed = os.path.join(_dir_processed,folder)
        utils.checkfolder(dir_processed)

        utils.show_message('Processing samples from {0}'.format(folder))

        # list filenames and classes. Also divides filenames into equally sized shards
        filenames, class_names = _get_filenames_and_classes(dir_raw, [setname])

        # save class dictionary
        class_dict = dict(zip(class_names, range(len(class_names))))

        # convert images to tf records based on the list of filenames
        shard_n = 0
        tf_filename = _get_output_filename(dir_processed, shard_n)
        with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
            _convert_to_tfrecord(filenames[shard_n], class_dict, tfrecord_writer)

        print('\nFinished converting the PSD %s dataset!' % setname)



  


    
