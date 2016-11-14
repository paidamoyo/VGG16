#!/usr/bin/env python

import functools

from models.conv_vae import ConvVae
import pickle
import numpy as np
from functions.record import print_log
from data.clutterMNIST import load_data_cluttered_MNIST, generate_cluttered_MNIST
from data.MNIST import load_data_MNIST, generate_MNIST
from data.SAGE import generate_breast_patch
import sys
import scipy.misc


# Global Dictionary of Flags
flags = {
    'data_directory': '../../../Data/',  # in relationship to the code_directory
    'previous_processed_directory': 'Smart_Crop/',
    'aux_directory': 'aux/',
    'model_directory': 'conv_vae/',
    'datasets': ['INbreast'],
    'restore': False,
    'restore_file': 'INbreast.ckpt',
    'recon': 10000000,
    'vae': 0.00000001,
    'image_dim': 128,
    'hidden_size': 128,
    'batch_size': 32,
    'display_step': 50,
    'lr_iters': [(0.001, 10000), (0.00075, 10000)]
}


def main():
    o = np.random.randint(1, 1000, 1)
    flags['seed'] = 107#o[0]
    # a = np.random.uniform(-5.5, -3.5, 1)
    # lr = 0.0001 #np.power(10, a[0])
    #flags['lr_iters'] = [(lr, 10000)]
    run_num = sys.argv[1]

    if 'Clutter_MNIST' in flags['datasets']:
        train_set, valid_set, test_set = load_data_cluttered_MNIST(flags['data_directory'] + flags['datasets'][0] + '/mnist.pkl.gz')
        bgf = functools.partial(generate_cluttered_MNIST, dims=[flags['image_dim'], flags['image_dim']],
                                nImages=flags['batch_size'], clutter=0.2, numbers=[], prob=0.5,
                                train_set=train_set)
    elif 'MNIST' in flags['datasets']:
        mnist = load_data_MNIST()
        bgf = functools.partial(generate_MNIST, mnist, flags['batch_size'])
    elif 'SAGE' or 'INbreast' in flags['datasets']:
        print('Using Breast dataset')
        image_dict = pickle.load(open(flags['aux_directory'] + 'preprocessed_image_dict.pickle', 'rb'))
        bgf = functools.partial(generate_breast_patch, flags, image_dict)
    else:
        bgf = None
        print('Dataset not defined for batch generation')
        exit()
    model_vae = ConvVae(flags, model=run_num)
    # model.save_x(bgf)
    # x_recon = model_vae.output_shape()
    # print(x_recon.shape)
    print_log("Seed: %d" % flags['seed'])
    print_log("Vae Weights: %d" % flags['vae'])
    print_log("Recon Weight: %d" % flags['recon'])
    model_vae.train(bgf, lr_iters=flags['lr_iters'], model=1)
    #model_vae.restore()
    #model_vae.save_x_gen(bgf, 15)



if __name__ == "__main__":
    main()
