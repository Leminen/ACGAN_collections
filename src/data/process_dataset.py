
import os
import sys
import numpy as np
import scipy.io
import zipfile
import types
import PIL
from PIL import Image, ImageOps
import math

import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets import mnist

sys.path.append('/home/leminen/Documents/RoboWeedMaps/GAN/weed-gan-v1')
import src.utils as utils

def process_dataset(dataset):
    dir_processedData = 'data/processed/'+ dataset
    utils.checkfolder(dir_processedData)
    
    if dataset == 'MNIST':
        # Download the MNIST dataset from source and save it in 'data/raw/mnist'
        data = mnist.read_data_sets('data/raw/MNIST', reshape=False)

        train_images = (data.train.images - 0.5) / 0.5
        _convert_to_tfrecord(train_images, data.train.labels, dataset, 'train')
        validation_images = (data.validation.images - 0.5) / 0.5
        _convert_to_tfrecord(validation_images, data.validation.labels, dataset, 'validation')
        test_images = (data.test.images - 0.5) / 0.5
        _convert_to_tfrecord(test_images, data.test.labels, dataset, 'test')
    
    if dataset == 'SVHN':
        dirRaw = 'data/raw/SVHN/'
        for file in os.listdir(dirRaw):
            if file.endswith('.mat'):
                mat = scipy.io.loadmat(dirRaw + file)
                images = np.transpose(mat['X'], (3, 0, 1, 2))
                images = np.array(images, dtype = 'f') / 255
                images = (images - 0.5) / 0.5
                labels = np.mod(mat['y'][:,0],10)
                _convert_to_tfrecord(images, labels, dataset, file[:-4])


    if dataset == 'PSD_Segmented':
        dirRaw = 'data/raw/PSD_Segmented'
        for file in os.listdir(dirRaw):
            if file.endswith('.zip'):
                excludeList = ['Black-grass', 'Common Chickweed', 'Loose Silky-bent']    

                images, labels, _ = _getCompressedDataset('data/raw/PSD_Segmented/data.zip',excludeList)
                images = _scaleImages(images)
                images = np.array([np.asarray(img, dtype = 'f') / 255 for img in images])
                _convert_to_tfrecord(images,labels,dataset,file[:-4])
        
    if dataset == 'PSD_NonSegmented':
        print('Needs implementation to process PSD_NonSegmented data and save into tfrecord')
#        dirRaw = 'data/raw/PSD_NonSegmented'
#        for file in os.listdir(dirRaw):
#            if file.endswith('.zip'):
#                data = _getCompressedDataset(os.path.join(dirRaw,file))
#                _convert_to_tfrecord(data.images,data.labels,dataset,file[:-4])
        


def __int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def __bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_tfrecord(images, labels, dataset, name):
    """Converts a dataset to tfrecords."""
    num_examples = len(labels)

    if images.shape[0] != num_examples:
        raise ValueError('Images size %d does not match label size %d.' %
                         (images.shape[0], num_examples))

    filename = os.path.join('data/processed/',dataset, name + '.tfrecords')
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)
      
    for index in range(num_examples):
        example = _encodeData(images[index], labels[index])

        writer.write(example.SerializeToString())
    writer.close()
    
### Define data encoder and decoder for the .tfrecord file[s]. The metodes must be reverse of each other,
### Encoder will be used by process_dataset directly whereas the decoder is used by the Model[s] to load data
### Look at this guide to format the tfrecord features: http://www.machinelearninguru.com/deep_learning/tensorflow/basics/tfrecord/tfrecord.html
    
def _encodeData(image,lbl):
    image_raw = image.tostring()
    shape = np.array(image.shape, np.int32)
    shape = shape.tobytes()
    
    features = {'label'     : __int64_feature(int(lbl)),
                'shape'     : __bytes_feature(shape),
                'image_raw' : __bytes_feature(image_raw)}
    
    example = tf.train.Example(features=tf.train.Features(feature=features))
    
    return example

def _decodeData(example_proto):
    features = {'label'     : tf.FixedLenFeature([], tf.int64),
                'shape'     : tf.FixedLenFeature([], tf.string),
                'image_raw' : tf.FixedLenFeature([], tf.string)}
   
    parsed_features = tf.parse_single_example(example_proto, features)

    shape = tf.decode_raw(parsed_features['shape'], tf.int32)
    image = tf.decode_raw(parsed_features['image_raw'], tf.float32)
    image = tf.reshape(image, shape)

    label = parsed_features['label']
    
    return image, label


###
def _getCompressedDataset(dirCompressedData,excludeList = None):
    archive = zipfile.ZipFile(dirCompressedData)
    
    images =list()
    labels = list()
    
    for file in archive.namelist():
        file_info = archive.getinfo(file)

        if not(file_info.is_dir()):
            ## if file contains xyz skip
            file_data = archive.open(file)
            img = Image.open(file_data)
            
            if excludeList == None:
                images.append(img)
                labels.append(file.split('/')[-2])
            else:
                if not any(x in file for x in excludeList):
                    images.append(img)
                    labels.append(file.split('/')[-2])
    
    dictionary, labels = np.unique(labels,return_inverse=True) # Convert to numbers

    return images, labels, dictionary

def _scaleImages(listImages, desired_size = 224):

    listImagesNew = list()
    for image in listImages:
        x, y = image.size
        delta_w = desired_size - x
        delta_h = desired_size - y
        padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
        image = ImageOps.expand(image, padding)
        listImagesNew.append(image)
    
    return listImagesNew

# process_dataset('MNIST')

#excludeList = ['Black-grass', 'Common Chickweed', 'Loose Silky-bent']    
#
#images, labels, dictionary = _getCompressedDataset('data/raw/PSD_Segmented/data.zip',excludeList)
#images = _scaleImages(images)
#
##images2 = [np.array(img) for img in images]
#images3 = np.array([np.array(img) for img in images])
#
#lbl = labels
#lbl = labels

## https://stackoverflow.com/questions/19371860/python-open-file-from-zip-without-temporary-extracting-it
#import io, pygame, zipfile
#archive = zipfile.ZipFile('images.zip', 'r')
#
## read bytes from archive
#img_data = archive.read('img_01.png')
#
## create a pygame-compatible file-like object from the bytes
#bytes_io = io.BytesIO(img_data)
#
#img = pygame.image.load(bytes_io)
    


### https://stackoverflow.com/questions/33166316/how-to-read-an-image-inside-a-zip-file-with-pil-pillow
#import sys
#from zipfile import ZipFile
#from PIL import Image # $ pip install pillow
#
#filename = sys.argv[1]
#with ZipFile(filename) as archive:
#    for entry in archive.infolist():
#        with archive.open(entry) as file:
#            img = Image.open(file)
#            print(img.size, img.mode, len(img.getdata()))
    

####https://stackoverflow.com/questions/8934335/python-zipfile-how-do-i-know-an-item-is-a-directory
    
    
