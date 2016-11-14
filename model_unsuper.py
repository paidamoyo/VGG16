#!/usr/bin/env python

import pickle
import sys

import numpy as np

from conv_vae import ConvVae
from functions.record import print_log

# Global Dictionary of Flags
flags = {
    'data_directory': '../../../Data/',  # in relationship to the code_directory
    'previous_processed_directory': 'Smart_Crop/',
    'aux_directory': 'aux/',
    'model_directory': 'conv_vae/',
    'datasets': ['INbreast'],
    'restore': False,
    'restore_file': 'INbreast.ckpt',
    'recon': 1,
    'vae': 1,
    'image_dim': 128,
    'hidden_size': 128,
    'batch_size': 128,
    'display_step': 50,
    'lr_iters': [(0.001, 2000), (0.00075, 2000), (0.0005, 2000), (0.00025, 2000), (0.0001, 2000), (0.00005, 10000)]
}


def main():
    o = np.random.randint(1, 1000, 1)
    flags['seed'] = 107#o[0]
    # a = np.random.uniform(-5.5, -3.5, 1)
    # lr = 0.0001 #np.power(10, a[0])
    #flags['lr_iters'] = [(lr, 10000)]
    run_num = sys.argv[1]

    # train_set, valid_set, test_set = load_data_cluttered_MNIST(flags['data_directory'] + flags['datasets'][0] + '/mnist.pkl.gz')
    # bgf = functools.partial(generate_cluttered_MNIST, dims=[flags['image_dim'], flags['image_dim']], nImages=flags['batch_size'], clutter=0.2, numbers=[], prob=0.5, train_set=train_set)
    print('Using Breast dataset')
    image_dict = pickle.load(open(flags['aux_directory'] + 'preprocessed_image_dict.pickle', 'rb'))
    model_vae = ConvVae(flags, model=run_num)
    # model.save_x(bgf)
    # x_recon = model_vae.output_shape()
    # print(x_recon.shape)
    print_log("Seed: %d" % flags['seed'])
    print_log("Vae Weights: %d" % flags['vae'])
    print_log("Recon Weight: %d" % flags['recon'])
    model_vae.train(image_dict, model=1)
    #model_vae.restore()
    #model_vae.save_x_gen(bgf, 15)



if __name__ == "__main__":
    main()
