#!/usr/bin/env python

import functools
import numpy as np
import matplotlib.pyplot as plt

from models.conv_vae import ConvVae
import pickle
from data.clutterMNIST import load_data_cluttered_MNIST, generate_cluttered_MNIST
from data.MNIST import load_data_MNIST, generate_MNIST
from data.SAGE import generate_SAGE


# Global Dictionary of Flags
flags = {
    'data_directory': '../../../Data/',  # in relationship to the code_directory
    'previous_processed_directory': '1_Cropped/',
    'aux_directory': 'aux/',
    'model_directory': 'conv_vae/',
    'datasets': ['SAGE'],
    'restore': False,
    'restore_file': 'starting_point.ckpt',
    'seed': 143,
    'image_dim': 28,
    'hidden_size': 64,
    'batch_size': 128,
    'display_step': 10,
    'lr_iters': [(0.001, 1000), (0.005, 1000), (0.001, 1000), (0.0005, 1000), (0.0001, 1000)]
}


def main():

    if 'Clutter_MNIST' in flags['datasets']:
        train_set, valid_set, test_set = load_data_cluttered_MNIST(flags['data_directory'] + flags['datasets'][0] + '/mnist.pkl.gz')
        bgf = functools.partial(generate_cluttered_MNIST, dims=[flags['image_dim'], flags['image_dim']],
                                nImages=flags['batch_size'], clutter=0.5, numbers=[8], prob=0.5,
                                train_set=train_set)
    elif 'MNIST' in flags['datasets']:
        mnist = load_data_MNIST()
        bgf = functools.partial(generate_MNIST, mnist, flags['batch_size'])
    elif 'SAGE' in flags['datasets']:
        image_dict = pickle.load(open(flags['aux_directory'] + 'preprocessed_image_dict.pickle', 'rb'))
        bgf = functools.partial(generate_SAGE, flags, image_dict)
    else:
        bgf = None
        print('Dataset not defined for batch generation')
        exit()
    model = ConvVae(flags)
    model.train(bgf, lr_iters=flags['lr_iters'], run_num=2)


if __name__ == "__main__":
    main()
