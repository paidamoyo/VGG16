#!/usr/bin/env python

import functools
import pickle
import sys

import numpy as np

from data.BREAST import generate_SAGE
from data.MNIST import load_data_MNIST, generate_MNIST
from data.clutterMNIST import load_data_cluttered_MNIST, generate_cluttered_MNIST
from functions.record import print_log
from models.conv_auto import ConvAuto


# Global Dictionary of Flags
flags = {
    'data_directory': '../../../Data/',  # in relationship to the code_directory
    'previous_processed_directory': 'Smart_Crop/',
    'aux_directory': 'aux/',
    'model_directory': 'conv_vae/',
    'datasets': ['Clutter_MNIST'],
    'restore': False,
    'restore_file': 'starting_point.ckpt',
    'seed': 13,
    'image_dim': 28,
    'hidden_size': 16,
    'batch_size': 16,
    'display_step': 50,
    'lr_iters': [(0.001, 10000)]
}


def main():
    o = np.random.randint(1, 1000, 1)
    flags['seed'] = o[0]
    # a = np.random.uniform(-5.5, -4.5, 1)
    # lr = np.power(10, a[0])
    # flags['lr_iters'] = [(lr, 10000)]
    run_num = sys.argv[1]

    if 'Clutter_MNIST' in flags['datasets']:
        train_set, valid_set, test_set = load_data_cluttered_MNIST(flags['data_directory'] + flags['datasets'][0] + '/mnist.pkl.gz')
        bgf = functools.partial(generate_cluttered_MNIST, dims=[flags['image_dim'], flags['image_dim']],
                                nImages=flags['batch_size'], clutter=0.1, numbers=[], prob=0.5,
                                train_set=train_set)
    elif 'MNIST' in flags['datasets']:
        mnist = load_data_MNIST()
        bgf = functools.partial(generate_MNIST, mnist, flags['batch_size'])
    elif 'SAGE' in flags['datasets']:
        print('Using SAGE dataset')
        image_dict = pickle.load(open(flags['aux_directory'] + 'preprocessed_image_dict.pickle', 'rb'))
        bgf = functools.partial(generate_SAGE, flags, image_dict)
    else:
        bgf = None
        print('Dataset not defined for batch generation')
        exit()
    model_auto = ConvAuto(flags, model=run_num)
    # model.save_x(bgf)
    # x_recon = model.output_shape()
    # print(x_recon.shape)
    print_log("Seed: %d" % flags['seed'])

    model_auto.train(bgf, lr_iters=flags['lr_iters'], model=2)


if __name__ == "__main__":
    main()
