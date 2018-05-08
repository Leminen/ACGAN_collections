import os
import tensorflow as tf
import scipy
import numpy as np
import gzip

def int64_feature(values):
    """Returns a TF-Feature of int64s.
    Args:
      values: A scalar or list of values.
    
    Returns:
      A TF-Feature.
    """
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def bytes_feature(values):
    """Returns a TF-Feature of bytes.
    
    Args:
      values: A string.

    Returns:
      A TF-Feature.
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def float_feature(values):
    """Returns a TF-Feature of floats.

    Args:
      values: A scalar of list of values.

    Returns:
      A TF-Feature.
    """
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def encode_image(image_data, image_format, class_lbl, height, width, class_text = None):
    """ Encodes image data and label and returns a tfrecord example
    Args:
      image_data:   Encoded image (eg. tf.image.encode_png)
      image_format: Format in which the image is encoded
      height:       Image height
      width:        Image width
      class_lbl:    Class label to which the image belong
      class_text:   Readable class label, if not avaliable 
                    defaults to str(class_lbl)
    
    Returns:
      A tfrecord example
    """

    if class_text == None:
        class_text = str(class_lbl).encode()

    features = tf.train.Features(
        feature = {
            'image/encoded':    bytes_feature(image_data),
            'image/format':     bytes_feature(image_format),
            'image/class/label':int64_feature(class_lbl),
            'image/class/text': bytes_feature(class_text),
            'image/height':     int64_feature(height),
            'image/width':      int64_feature(width),
        })

    return tf.train.Example(features = features)


def decode_image(example_proto):
    """ decodes a tfrecord example and returns an image and label
    Args:
      example_proto: A tfrecord example
    
    Returns:
      image:      A decoded image tensor with type float32 
                  and shape [height, width, num_channels]. 
                  The image is normalized to be in range: 
                  -1.0 to 1.0   
      class_lbl:  Class label to which the image belong
      class_text: Readable class label
      height:     Image height
      width:      Image width
    """

    features = {
        'image/encoded':    tf.FixedLenFeature([], tf.string),
        'image/format':     tf.FixedLenFeature([], tf.string),
        'image/class/label':tf.FixedLenFeature([], tf.int64),
        'image/class/text': tf.FixedLenFeature([], tf.string),
        'image/height':     tf.FixedLenFeature([], tf.int64),
        'image/width':      tf.FixedLenFeature([], tf.int64)
    }

    # parsed_example = tf.parse_example(example_proto, features)
    parsed_example = tf.parse_single_example(example_proto, features)

    image_format = parsed_example['image/format']

    image = parsed_example['image/encoded']

    image = tf.case(
        pred_fn_pairs = [
            (tf.equal(image_format,b'jpeg'), lambda : tf.image.decode_jpeg(image)),
            (tf.equal(image_format,b'bmp'), lambda : tf.image.decode_bmp(image))],
            default = lambda : tf.image.decode_png(image))

    image = (tf.to_float(image) - 128.0) / 128.0

    
    class_lbl = parsed_example['image/class/label']
    class_text = parsed_example['image/class/text']
    height = parsed_example['image/height']
    width = parsed_example['image/width']

    return image, class_lbl, class_text, height, width