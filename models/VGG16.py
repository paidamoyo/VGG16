#!/usr/bin/env python

import h5py
import tensorflow as tf
import re

from functions.tf import conv2d, maxpool2d


class Vgg16:
    def __init__(self):
        self.weights, self.biases = self.init_params()

    def init_params(self):
        weights, biases = self.load_pretrained_params()
        return weights, biases

    def load_pretrained_params(self):
        filename = self.flags['weights_directory'] + 'vgg16_ImageNet_tf.h5'
        pretrained = h5py.File(filename, 'r')
        weights = {}
        biases = {}
        for layer_key in list(pretrained.keys()):  # 'block' and 'conv' to define layer
            m = re.search('block(.+?)_conv(.+?)', layer_key)
            if m is not None:
                for nums_key in list(pretrained[layer_key].keys()):
                    exp = re.search('block' + m.group(1) + '_conv' + m.group(2) + '_(.+?)', nums_key)
                    if exp.group(1) == 'W':
                        weights['b' + m.group(1) + '_c' + m.group(2)] = \
                            tf.Variable(pretrained[layer_key][nums_key].value, trainable=False)
                    if exp.group(1) == 'b':
                        biases['b' + m.group(1) + '_c' + m.group(2)] = \
                            tf.Variable(pretrained[layer_key][nums_key].value, trainable=False)
                    else:
                        ValueError('Could not locate weights in pretrained dictionary!')
            else:
                continue  # skip fc weights and pool layers
        return weights, biases

    def model_vgg16(self, x):
        weights = self.weights
        biases = self.biases
        block1_conv1 = conv2d(x, weights['b1_c1'], biases['b1_c1'])
        block1_conv2 = conv2d(block1_conv1, weights['b1_c2'], biases['b1_c2'])
        block1_pool = maxpool2d(block1_conv2, k=2)

        block2_conv1 = conv2d(block1_pool, weights['b2_c1'], biases['b2_c1'])
        block2_conv2 = conv2d(block2_conv1, weights['b2_c2'], biases['b2_c2'])
        block2_pool = maxpool2d(block2_conv2, k=2)

        block3_conv1 = conv2d(block2_pool, weights['b3_c1'], biases['b3_c1'])
        block3_conv2 = conv2d(block3_conv1, weights['b3_c2'], biases['b3_c2'])
        block3_conv3 = conv2d(block3_conv2, weights['b3_c3'], biases['b3_c3'])
        block3_pool = maxpool2d(block3_conv3, k=2)

        block4_conv1 = conv2d(block3_pool, weights['b4_c1'], biases['b4_c1'])
        block4_conv2 = conv2d(block4_conv1, weights['b4_c2'], biases['b4_c2'])
        block4_conv3 = conv2d(block4_conv2, weights['b4_c3'], biases['b4_c3'])
        block4_pool = maxpool2d(block4_conv3, k=2)

        block5_conv1 = conv2d(block4_pool, weights['b5_c1'], biases['b5_c1'])
        block5_conv2 = conv2d(block5_conv1, weights['b5_c2'], biases['b5_c2'])
        block5_conv3 = conv2d(block5_conv2, weights['b5_c3'], biases['b5_c3'])
        return block5_conv3

    def run(self, x):
        map_stack = self.model_vgg16(x=x)
        return map_stack

if __name__ == '__main__':
    print('Not a callable function.')
