"""
Methods for downloading and converting the MNIST dataset to TF-records

implementation is heavily inspired by the slim.datasets implementation (https://github.com/tensorflow/models/tree/master/research/slim/datasets)
"""
import os
import sys

import numpy as np
from PIL import Image
from six.moves import urllib
import gzip
import zipfile
import tensorflow as tf

import src.utils as utils
import src.data.util_data as util_data

# The URLs where the PSD data can be downloaded.
_DATA_URL = 'https://vision.eng.au.dk/?download=/data/WeedData/'
_NONSEGMENTED = 'NonsegmentedV2.zip'
_SEGMENTED = 'Segmented.zip'

_DATA_URL_NONSEGMENTED = 'https://vision.eng.au.dk/?download=/data/WeedData/NonsegmentedV2.zip'
_DATA_URL_SEGMENTED = 'https://vision.eng.au.dk/?download=/data/WeedData/Segmented.zip'

# Local directories to store the dataset
_DIR_RAW = 'data/raw/PSD'
_DIR_PROCESSED = 'data/processed/PSD'

_DIR_RAW_NONSEGMENTED = 'data/raw/PSD_Nonsegmented/NonsegmentedV2.zip'
_DIR_PROCESSED_NONSEGMENTED = 'data/processed/PSD_Nonsegmented/'

_DIR_RAW_SEGMENTED = 'data/raw/PSD_Segmented/Segmented.zip'
_DIR_PROCESSED_SEGMENTED = 'data/processed/PSD_Segmented/'


_EXCLUDED_GRASSES = True
_EXCLUDE_LARGE_IMAGES = True
_LARGE_IMAGE_DIM = 400
_NUM_SHARDS = 4 #10




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


def _get_filenames_and_classes(dataset_dir, setname, exclude_list):
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
            if not any(x in filename for x in exclude_list):
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

        if _EXCLUDE_LARGE_IMAGES:
            paths_temp = paths.copy()
            for path in paths_temp:
                img_temp = Image.open(path)
                if any(np.array(img_temp.size) > _LARGE_IMAGE_DIM):
                    paths.remove(path)
            
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
    if dataset_part == 'Nonsegmented':
        _data_url = _DATA_URL_NONSEGMENTED
        filepath = os.path.join(_DIR_RAW_NONSEGMENTED)
    else:
        _data_url = _DATA_URL_SEGMENTED
        filepath = os.path.join(_DIR_RAW_SEGMENTED)

    if not os.path.exists(filepath):
        print('Downloading dataset...')
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %.1f%%' % (
                float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(_data_url, filepath, _progress)
        print('\n', end = '')

        print()
        with tf.gfile.GFile(filepath) as f:
            size = f.size()
        print('Successfully downloaded', size, 'bytes.')



def process(dataset_part):
    """Runs the conversion operation.

    Args:
      dataset_part: The dataset part to be converted [Nonsegmented, Segmented].
    """
    if dataset_part == 'Nonsegmented':
        _dir_raw = _DIR_RAW_NONSEGMENTED
        _dir_processed = _DIR_PROCESSED_NONSEGMENTED
        setname = 'Nonsegmented'
    else:
        _dir_raw = _DIR_RAW_SEGMENTED
        _dir_processed = _DIR_PROCESSED_SEGMENTED
        setname = 'Segmented' 

    if _EXCLUDED_GRASSES:
        exclude_list = ['Black-grass', 'Common wheat', 'Loose Silky-bent']
    else:
        exclude_list = []

    # extract raw data
    data_filename = os.path.join(_dir_raw)
    archive = zipfile.ZipFile(data_filename)
    archive.extractall(_dir_processed)

    # list filenames and classes. Also divides filenames into equally sized shards
    filenames, class_names = _get_filenames_and_classes(_dir_processed, [setname], exclude_list)

    # save class dictionary
    class_dict = dict(zip(class_names, range(len(class_names))))
    utils.save_dict(class_dict, _dir_processed, 'class_dict.json')

    # convert images to tf records based on the list of filenames
    for shard_n in range(_NUM_SHARDS):
        utils.show_message('Processing shard %d/%d' % (shard_n+1,_NUM_SHARDS))
        tf_filename = _get_output_filename(_dir_processed, shard_n)

        with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
            _convert_to_tfrecord(filenames[shard_n], class_dict, tfrecord_writer)

    # clean up
    tmp_dir = os.path.join(_dir_processed, setname)
    tf.gfile.DeleteRecursively(tmp_dir)

    print('\nFinished converting the PSD %s dataset!' % setname)



  


    
