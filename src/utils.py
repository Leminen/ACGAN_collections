"""
Most codes from https://github.com/carpedm20/DCGAN-tensorflow
"""

import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy
import numpy as np
import gzip



def checkfolder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)
