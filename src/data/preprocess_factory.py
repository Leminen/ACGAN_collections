# Helper for building a preprocessing pipeline to use with the tf.Dataset class
#
# Author: Anders Krogh Mortensen
# GitHub: https://github.com/anderskm
#

import ast
import inspect
import tensorflow as tf


class preprocess_factory(object):

    def __init__(self):
        self._pipe_params = [(self.nop,{})]

    def prep_pipe(self, pipe_params):
        self._pipe_params = pipe_params
    
    def prep_pipe_from_string(self, pipe_params_string):
        pipe_params_parts = pipe_params_string.split(';')
        #TODO: Assert length? Must be a multiple of 2
        pipe_params = list(zip(*[iter(pipe_params_parts)]*2))

        # Convert string names to method references
        pipe_params = [([m_ref for m_name, m_ref in inspect.getmembers(self) if m_name == param_pair[0]][0], ast.literal_eval(param_pair[1])) for param_pair in pipe_params]

        self.prep_pipe(pipe_params)

        return


    def pipe(self, tf_image, tf_labels, *args):
        pipe_params = self._pipe_params
        for process_tuple in pipe_params:
            _process, param_dict = process_tuple
            tf_image, tf_labels, *args = _process(param_dict, tf_image, tf_labels, *args)
        return (tf_image, tf_labels, *args)

    def nop(self, param_dict, tf_image, tf_label, *args):
        # Dummy preprocessing function performing no operation
        pass

        return (tf_image, tf_label, *args)

    def pad_to_size(self, param_dict, tf_image, tf_label, *args):
        # Pad image with zeros on all four sides to produce an output image of size [target_height, target_width]
        # The input image will be approximately centered in the output image.
        #
        # input:
        #   param_dict      dict with the following key-value pairs:
        #                       'target_height': height in pixels of output image (int)
        #                       'target_width': width in pixels of output image (int)
        #                       'constant': padding value (float)
        # 

        target_height = param_dict['height']
        target_width = param_dict['width']
        constant_value = param_dict['constant']

        with tf.name_scope('pad_to_size_' + '{:d}'.format(target_height) + '_' + '{:d}'.format(target_width)):
            tf_image_shape = tf.shape(tf_image)
            image_height = tf.cond(tf.equal(tf.rank(tf_image),4),lambda: tf_image_shape[1], lambda: tf_image_shape[0])
            image_width = tf.cond(tf.equal(tf.rank(tf_image),4),lambda: tf_image_shape[2],lambda: tf_image_shape[1])

            height_diff = target_height - image_height
            width_diff = target_width - image_width

            # def pad_image_single(image,height_d,width_d,constant):
            #     paddings = [[height_d, height_d], [width_d, width_d], [0,0]]
            #     image = tf.pad(image, 
            #                       tf.floor_div(paddings,2),
            #                       mode='CONSTANT', 
            #                       name=None, 
            #                       constant_values=constant
            #                       )
            #     return image

            # def pad_image_batch(images,height_d,width_d,constant):
            #     paddings = [[0,0], [height_d, height_d], [width_d, width_d], [0,0]]
            #     image = tf.pad(images, 
            #                       tf.floor_div(paddings,2),
            #                       mode='CONSTANT', 
            #                       name=None, 
            #                       constant_values=constant
            #                       )
            #     return image

            # tf_image = tf.cond(tf.equal(tf.rank(tf_image),4),
            #                    lambda: pad_image_single(tf_image,height_diff,width_diff,constant_value),
            #                    lambda: pad_image_batch(tf_image,height_diff,width_diff,constant_value))

            paddings = [[height_diff, height_diff], [width_diff, width_diff], [0,0]]
            paddings = tf.floor_div(paddings,2)
            
            tf_image = tf.pad(tf_image, paddings, mode='CONSTANT', name=None, constant_values=constant_value)


        return (tf_image, tf_label, *args)

    def crop_to_size(self, param_dict, tf_image, tf_label, *args):
        # Crop or pad image with zeros on all four sides to produce an output image of size [height, width]
        #
        # input:
        #   param_dict      dict with the following key-value pairs:
        #                       'height': height in pixels of output image (int)
        #                       'width': width in pixels of output image (int)
        # 
        
        target_height = param_dict['height']
        target_width = param_dict['width']

        with tf.name_scope('crop_to_size_' + '{:d}'.format(target_height) + '_' + '{:d}'.format(target_width)):
            
            tf_image_shape = tf.shape(tf_image)
            image_height = tf.cond(tf.equal(tf.rank(tf_image),4),lambda: tf_image_shape[1], lambda: tf_image_shape[0])
            image_width = tf.cond(tf.equal(tf.rank(tf_image),4),lambda: tf_image_shape[2],lambda: tf_image_shape[1])

            height_diff = image_height - target_height
            width_diff = image_width - target_width

            tf_image = tf.image.crop_to_bounding_box(tf_image, 
                                                     tf.floor_div(height_diff,2), 
                                                     tf.floor_div(width_diff,2), 
                                                     target_height, 
                                                     target_width)

        return (tf_image, tf_label, *args)

    def crop_pad_to_size(self, param_dict, tf_image, tf_label, *args):
        # Crop or pad image with zeros on all four sides to produce an output image of size [height, width]
        #
        # input:
        #   param_dict      dict with the following key-value pairs:
        #                       'height': height in pixels of output image (int)
        #                       'width': width in pixels of output image (int)
        # 
        
        height = param_dict['height']
        width = param_dict['width']

        with tf.name_scope('crop_pad_to_size_' + '{:d}'.format(height) + '_' + '{:d}'.format(width)):
            tf_image = tf.image.resize_image_with_crop_or_pad(tf_image, target_height=height, target_width=width)
        return (tf_image, tf_label, *args)

    def random_rotation(self, param_dict, tf_image, tf_label, *args):
        # Rotate image by a random angle drawn from a uniform distribution in range [min_angle, max_angle]
        #
        #   param_dict      dict with the following key-value pairs:
        #                       'min_angle': lower limit of random angle in radians (float, default = 0)
        #                       'max_angle': upper limit of random angle in radians (float, default = 2*pi)
        #
        # NOTE: Only works on single images!
        # TODO: Extend to batch of images
        
        
        min_angle = param_dict.get('min_angle', 0)
        max_angle = param_dict.get('max_angle', 6.28318530718)
        with tf.name_scope('random_rotation_'+ '{:5.3f}'.format(min_angle) + '_' + '{:5.3f}'.format(max_angle) + ''):
            rotations = tf.random_uniform([1,],minval=min_angle,maxval=max_angle)
            tf_image = tf.contrib.image.rotate(tf_image, angles=rotations)

        return (tf_image, tf_label, *args)

    def resize(self, param_dict, tf_image, tf_label, *args):
        # Resize image to height and width
        #
        #   param_dict      dict with the following key-value pairs:
        #                       'height': height in pixels of output image (int)
        #                       'width': width in pixels of output image (int)
        #                       'resize_method': method used for interpolating between pixels.
        #                                        Must be one of the following strings:
        #                                        'bilinear' (default), 'nearest', 'bicubic' or 'area'
        #
        # See also: tf.image.resize_images()

        height = param_dict['height']
        width = param_dict['width']
        resize_method_string = param_dict.get('resize_method', 'bilinear')
        if (resize_method_string == 'bilinear'):
            resize_method = tf.image.ResizeMethod.BILINEAR
        elif (resize_method_string == 'nearest'):
            resize_method = tf.image.ResizeMethod.NEAREST_NEIGHBOR
        elif (resize_method_string == 'bicubic'):
            resize_method = tf.image.ResizeMethod.BICUBIC
        elif (resize_method_string == 'area'):
            resize_method = tf.image.ResizeMethod.AREA
        else:
            raise ValueError('Unknown resize method (' + resize_method_string + '). Expected on of the following \'bilinear\' (default), \'nearest\', \'bicubic\' or \'area\'')

        with tf.name_scope('resize_' + '{:d}'.format(height) + '_' + '{:d}'.format(width)):
            tf_image = tf.image.resize_images(tf_image,
                                            size=[height, width],
                                            method=resize_method
                                            )

        return (tf_image, tf_label, *args)