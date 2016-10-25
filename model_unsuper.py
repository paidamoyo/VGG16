#!/usr/bin/env python

import functools

from models.conv_vae import ConvVae
from data.clutterMNIST import load_data_cluttered_MNIST, generate_cluttered_MNIST
from data.MNIST import load_data_MNIST, generate_MNIST


# Global Dictionary of Flags
flags = {
    'data_directory': '../../../Data/',  # in relationship to the code_directory
    'aux_directory': 'aux/',
    'model_directory': 'conv_vae/',
    'datasets': ['Clutter_MNIST'],
    'restore': False,
    'restore_file': 'starting_point.ckpt',
    'seed': 14,
    'image_dim': 28,
    'hidden_size': 10,
    'batch_size': 128,
    'display_step': 100,
    'lr_iters': [(0.001, 5000), (0.005, 5000), (0.001, 7500), (0.0005, 10000), (0.0001, 10000)]
}


def main():

    if 'Clutter_MNIST' in flags['datasets']:
        train_set, valid_set, test_set = load_data_cluttered_MNIST(flags['data_directory'] + flags['datasets'][0] + '/mnist.pkl.gz')
        bgf = functools.partial(generate_cluttered_MNIST, dims=[flags['image_dim'], flags['image_dim']],
                                nImages=flags['batch_size'], clutter=0.5, numbers=[], prob=1,
                                train_set=train_set)
    if 'MNIST' in flags['datasets']:
        mnist = load_data_MNIST()
        bgf = functools.partial(generate_MNIST, mnist, flags['batch_size'])
    model = ConvVae(flags)
    # print(model.print_variable(var='x_reconst').shape)

    model.train(bgf, lr_iters=flags['lr_iters'], run_num=1)
    model.x_recon()


if __name__ == "__main__":
    main()
