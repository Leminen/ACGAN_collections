import sys
import os

import src.utils as utils
import src.data.util_data as util_data

import src.data.datasets.mnist as mnist
import src.data.datasets.psd as psd
import src.data.datasets.GAN_samples as GAN_samples

def make_dataset(dataset):
    dir_rawData = 'data/raw/'+ dataset
    utils.checkfolder(dir_rawData)

    if dataset == 'MNIST':
        mnist.download()

    elif dataset == 'PSD_Nonsegmented': 
        psd.download('Nonsegmented')

    elif dataset == 'PSD_Segmented':
        psd.download('Segmented')
    
    elif dataset.startswith('GAN_samples_'):
        dataset_part = dataset[12:]
        GAN_samples.download(dataset_part)

    else:
        print('No matching dataset for: {0}'.format(dataset))
        pass

def process_dataset(dataset):
    dir_processedData = 'data/processed/'+ dataset
    utils.checkfolder(dir_processedData)

    if dataset == 'MNIST':
        mnist.process()

    elif dataset == 'PSD_Nonsegmented':
        psd.process('Nonsegmented')

    elif dataset == 'PSD_Segmented':
        psd.process('Segmented')

    elif dataset.startswith('GAN_samples_'):
        dataset_part = dataset[12:]
        GAN_samples.process(dataset_part)

    else:
        print('No matching dataset for: {0}'.format(dataset))
        pass
