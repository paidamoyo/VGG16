#!/usr/bin/env python

import functools

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
    'seed': 14,
    'image_dim': 28,
    'hidden_size': 15,
    'batch_size': 128,
    'display_step': 10,
    'lr_iters': [(0.01, 500), (0.005, 500), (0.001, 750), (0.0005, 1000), (0.0001, 1000)]
}


def main():

    train_set, valid_set, test_set = load_data(flags['data_directory'] + flags['datasets'][0] + '/mnist.pkl.gz')
    model = ConvVae(params, flags)

    bgf = functools.partial(generate_cluttered_MNIST, dims=[params['image_dim'], params['image_dim']],
                                                  nImages=params['batch_size'], clutter=0.0, numbers=range(9), prob=1,
                            train_set=train_set)
    # print(model.print_variable(var='x_reconst').shape)

    model.train(bgf, lr_iters=params['lr_iters'], run_num=2)


if __name__ == "__main__":
    main()
