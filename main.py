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
import GPUtil
import argparse

import src.utils as utils
from src.data import dataset_manager
from src.models.BasicModel import BasicModel
from src.models.acgan import acgan
from src.models.WacGAN import WacGAN
from src.models.WacGAN_small import WacGAN_small
from src.models.WacGAN_info import WacGAN_info
from src.visualization import visualize

# DEVICE_ID_LIST = GPUtil.getFirstAvailable(attempts = 100, interval = 120)
# os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID_LIST[0])

"""parsing and configuration"""
def parse_args():
    
# ----------------------------------------------------------------------------------------------------------------------
# Define default pipeline
# ----------------------------------------------------------------------------------------------------------------------

    desc = "Pipeline for running Tensorflow implementation of ACGAN"
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

    parser.add_argument('--evaluate_model', 
                        action='store_true', 
                        help = 'Run evaluation of the model by computing the results [Defaults to False if argument is omitted]')
    
    parser.add_argument('--posteval_model', 
                        action='store_true', 
                        help = 'Run post evaluation of the model by visualizing or reformatting the results [Defaults to False if argument is omitted]')
    
# ----------------------------------------------------------------------------------------------------------------------
# Define the arguments used in the entire pipeline
# ----------------------------------------------------------------------------------------------------------------------

    parser.add_argument('--model', 
                        type=str, 
                        default='acgan', 
                        choices=['acgan', 
                                 'WacGAN',
                                 'WacGAN_small',
								 'WacGAN_info'],
                        #required = True,
                        help='The name of the network model')

    parser.add_argument('--dataset', 
                        type=str, default='MNIST', 
                        choices=['MNIST',
                                 'PSD_Nonsegmented',
                                 'PSD_Segmented'],
                        #required = True,
                        help='The name of dataset')                   
    
# ----------------------------------------------------------------------------------------------------------------------
# Define the arguments for the training
# ----------------------------------------------------------------------------------------------------------------------

    parser.add_argument('--id',
                        type= str,
                        default = None,
                        help = 'Optional ID, to distinguise experiments')

    parser.add_argument('--hparams',
                        type=str, default = '',
                        help='CLI arguments for the model wrapped in a string')

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
        utils.show_message('Fetching raw dataset: {0}'.format(args.dataset), lvl = 1)
        dataset_manager.make_dataset(args.dataset)
        
    # Make dataset
    if args.process_dataset:
        utils.show_message('Processing raw dataset: {0}'.format(args.dataset), lvl = 1)
        dataset_manager.process_dataset(args.dataset)
        
    # Build and train model
    if args.train_model:
        utils.show_message('Configuring and Training Network: {0}'.format(args.model), lvl = 1)        

        if args.model == 'BasicModel':
            model = BasicModel(
                dataset = args.dataset,
                id = args.id)
            model.train(hparams_string = args.hparams)
        
        elif args.model == 'acgan':
            model = acgan(
                dataset = args.dataset,
                id = args.id)
            model.train(hparams_string = args.hparams)

        elif args.model == 'WacGAN':
            model = WacGAN(
                dataset = args.dataset,
                id = args.id)
            model.train(hparams_string = args.hparams)

        elif args.model == 'WacGAN_small':
            model = WacGAN_small(
                dataset = args.dataset,
                id = args.id)
            model.train(hparams_string = args.hparams)
            
        elif args.model == 'WacGAN_info':
            model = WacGAN_info(
                dataset = args.dataset,
                id = args.id)
            model.train(hparams_string = args.hparams)
        
        
  
    # Evaluate model
    if args.evaluate_model:
        utils.show_message('Evaluating Network: {0}'.format(args.model), lvl = 1)

        if args.model == 'BasicModel':
            model = BasicModel(
                dataset = args.dataset,
                id = args.id)
            model.evaluate(hparams_string = args.hparams)
        
        elif args.model == 'acgan':
            model = acgan(
                dataset = args.dataset,
                id = args.id)
            model.evaluate(hparams_string = args.hparams)

        elif args.model == 'WacGAN':
            model = WacGAN(
                dataset = args.dataset,
                id = args.id)
            model.evaluate(hparams_string = args.hparams)

        elif args.model == 'WacGAN_small':
            model = WacGAN_small(
                dataset = args.dataset,
                id = args.id)
            model.evaluate(hparams_string = args.hparams)
			
        elif args.model == 'WacGAN_info':
            model = WacGAN_info(
                dataset = args.dataset,
                id = args.id)
            model.evaluate(hparams_string = args.hparams)

    if args.posteval_model:
        utils.show_message('Running Post Evaluation on Network: {0}'.format(args.model), lvl = 1)

        if args.model == 'WacGAN':
            model = WacGAN(
                dataset = args.dataset,
                id = args.id)
            model.post_evaluation(hparams_string = args.hparams)

        if args.model == 'WacGAN_info':
            model = WacGAN_info(
                dataset = args.dataset,
                id = args.id)
            model.post_evaluation(hparams_string = args.hparams)

        #################################
        ####### To Be Implemented #######
        #################################
    

if __name__ == '__main__':
    main()
