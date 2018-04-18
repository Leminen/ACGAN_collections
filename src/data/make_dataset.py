# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, '/home/leminen/Documents/RoboWeedMaps/GAN/weed-gan-v1')

import os
import urllib

from tensorflow.contrib.learn.python.learn.datasets import mnist

import src.utils as utils


def make_dataset(dataset):
    dir_rawData = 'data/raw/'+ dataset
    utils.checkfolder(dir_rawData)
    
    if dataset == 'MNIST':
        # Download the MNIST dataset from source and save it in 'data/raw/mnist'
        _ = mnist.read_data_sets('data/raw/MNIST', one_hot=True)

    if dataset == 'SVHN':
        urllib.request.urlretrieve('http://ufldl.stanford.edu/housenumbers/train_32x32.mat',
                                   'data/raw/SVHN/train.mat')
        urllib.request.urlretrieve('http://ufldl.stanford.edu/housenumbers/test_32x32.mat',
                                    'data/raw/SVHN/test.mat')
        # urllib.request.urlretrieve('http://ufldl.stanford.edu/housenumbers/extra_32x32.mat',
        #                             'data/raw/SVHN/extra.mat')
    
    # if dataset == 'PSD_Segmented':
    #     # Download the Plant Seedlings Dataset Segmented from source (http://vision.eng.au.dk/data/WeedData/Segmented.zip) 
    #     # and save it in data/raw/Plant_Seedlings_Dataset_Segmented
        
    #     urllib.request.urlretrieve('http://vision.eng.au.dk/data/WeedData/Segmented.zip',
    #                                'data/raw/PSD_Segmented/data.zip')
    
    # if dataset == 'PSD_NonSegmented':
    #     # Download the Plant Seedlings Dataset NonSegmented from source (http://vision.eng.au.dk/data/WeedData/Nonsegmented.zip) 
    #     # and save it in data/raw/Plant_Seedlings_Dataset_NonSegmented
        
    #     urllib.request.urlretrieve('http://vision.eng.au.dk/data/WeedData/Nonsegmented.zip',
    #                                'data/raw/PSD_NonSegmented/data.zip')