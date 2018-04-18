"""
This file is used to run the project.
Notes:
- The structure of this file (and the entire project in general) is made with emphasis on flexibility for research
purposes, and the pipelining is done in a python file such that newcomers can easily use and understand the code.
- Remember that relative paths in Python are always relative to the current working directory.

Hence, if you look at the functions in make_dataset.py, the file paths are relative to the path of
this file (main.py)
"""

__author__ = "Simon Leminen Madsen"
__email__ = "slm@eng.au.dk"

import os
import argparse
import datetime

from src.data import make_dataset
from src.data import process_dataset
from src.models.BasicModel import BasicModel
from src.models.infoGAN import infoGAN
from src.models.infoGAN_rbg import infoGAN_rgb
from src.models.infoGAN_32x32 import infoGAN_32x32
from src.visualization import visualize


"""parsing and configuration"""
def parse_args():
    
# ----------------------------------------------------------------------------------------------------------------------
# Define default pipeline
# ----------------------------------------------------------------------------------------------------------------------

    desc = "Pipeline for running Tensorflow implementation of infoGAN"
    parser = argparse.ArgumentParser(description=desc)
    
    parser.add_argument('--make_dataset', 
                        action='store_true', 
                        help = 'Fetch dataset from remote source into /data/raw/. Or generate raw dataset [Defaults to False if argument is omitted]')
    
    parser.add_argument('--process_dataset', 
                        action='store_true', 
                        help = 'Run preprocessing of raw data. [Defaults to False if argument is omitted]')

    parser.add_argument('--train_model', 
                        action='store_true', 
                        help = 'Run configuration and training network [Defaults to False if argument is omitted]')

    parser.add_argument('--visualize', 
                        action='store_true', 
                        help = 'Run visualization of results [Defaults to False if argument is omitted]')
    
# ----------------------------------------------------------------------------------------------------------------------
# Define the arguments used in the entire pipeline
# ----------------------------------------------------------------------------------------------------------------------

    parser.add_argument('--model', 
                        type=str, 
                        default='infoGAN', 
                        choices=['infoGAN', 
                                 'infoGAN_rgb',
                                 'infoGAN_32x32'],
                        #required = True,
                        help='The name of the network model')

    parser.add_argument('--dataset', 
                        type=str, default='MNIST', 
                        choices=['MNIST',
                                 'SVHN'],
                        #required = True,
                        help='The name of dataset')

    parser.add_argument('--epoch_max', 
                        type=int, default='20', 
                        help='The name of dataset')

    parser.add_argument('--batch_size', 
                        type=int, default='32', 
                        help='The name of dataset')                    
    
# ----------------------------------------------------------------------------------------------------------------------
# Define the arguments for the training
# ----------------------------------------------------------------------------------------------------------------------

    # parser.add_argument('--hparams',
    #                     type=str,
    #                     help='Comma separated list of "name=value" pairs.')


    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    
    # Assert if training parameters are provided, when training is selected
#    if args.train_model:
#        try:
#            assert args.hparams is ~None
#        except:
#            print('hparams not provided for training')
#            exit()
        
    return args


"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
        exit()
    
    # Make dataset
    if args.make_dataset:
        print('%s - Fetching raw dataset: %s'  % (datetime.datetime.now(), args.dataset))
        make_dataset.make_dataset(args.dataset)
        
    # Make dataset
    if args.process_dataset:
        print('%s - Processing raw dataset: %s' % (datetime.datetime.now(), args.dataset))
        process_dataset.process_dataset(args.dataset)
        
    # Build and train model
    if args.train_model:
        print('%s - Configuring and training Network: %s' % (datetime.datetime.now(), args.model))
        

        if args.model == 'BasicModel':
            model = BasicModel()
            model.train(dataset_str = args.dataset, epoch_N = args.epoch_max, batch_N = 64)
               
        elif args.model == 'infoGAN':
            model = infoGAN()
            model.train(dataset_str = args.dataset, 
                        epoch_N = args.epoch_max, 
                        batch_size = args.batch_size)
        
        elif args.model == 'infoGAN_rgb':
            model = infoGAN_rgb()
            model.train(dataset_str = args.dataset, 
                        epoch_N = args.epoch_max, 
                        batch_size = args.batch_size)
        
        elif args.model == 'infoGAN_32x32':
            model = infoGAN_32x32()
            model.train(dataset_str = args.dataset, 
                        epoch_N = args.epoch_max, 
                        batch_size = args.batch_size)
        
        # elif args.model == 'weedGAN':
        #     model = weedGAN()
        #     model.train(dataset_str = args.dataset, epoch_N = 25, batch_N = 64)
    
    # Visualize results
    if args.visualize:
        print('Visualizing Results')
        #################################
        ####### To Be Implemented #######
        #################################
    

if __name__ == '__main__':
    main()
