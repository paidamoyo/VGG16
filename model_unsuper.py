#!/usr/bin/env python

import tensorflow as tf
import sys
import functools
import matplotlib.pyplot as plt

import numpy as np
from functions.record import record_metrics, print_log, setup_metrics
from functions.data import generate_lr
from models.conv_vae import ConvVae
from data.clutterMNIST import load_data, generate_cluttered_MNIST


# Global Dictionary of Flags
flags = {
    'data_directory': '../../../Data/',  # in relationship to the code_directory
    'aux_directory': 'aux/',
    'model_directory': 'conv_vae/',
    'datasets': ['MNIST'],
    'restore': False,
    'restore_file': 'starting_point.ckpt'
}


params = {
    'image_dim': 128,
    'batch_size': 32,  # must be divisible by 8
    'hidden_size': 128,
    'display_step': 5,
    'training_iters': 500
}


def main():

    seed = 14
    # batch = int(sys.argv[1])
    batch = 32
    params['batch_size'] = batch
    params['lr'] = 0.0001  # generate_lr(int(sys.argv[2]))
    lr_str = str(params['lr'])

    folder = str(batch) + '/'
    aux_filenames = 'lr_' + lr_str + '_batch_%d' % params['batch_size']
    logging = setup_metrics(flags, aux_filenames, folder)
    train_set, valid_set, test_set = load_data(flags['data_directory'] + flags['datasets'][0] + '/mnist.pkl.gz')
    model = ConvVae(params, flags, logging, seed)

    bgf = functools.partial(generate_cluttered_MNIST, dims=[params['image_dim'], params['image_dim']],
                                                  nImages=params['batch_size'], clutter=0.0, numbers=[8], prob=1,
                            train_set=train_set)
    model.print_variable(var='x_reconst')
    model.train(bgf, aux_filenames)


if __name__ == "__main__":
    main()
