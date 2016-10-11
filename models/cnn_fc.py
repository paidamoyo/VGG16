#!/usr/bin/env python

import tensorflow as tf

from functions.tf import conv2d, maxpool2d, weight_variable, bias_variable


class CnnFc:
    def __init__(self):
        self.weights, self.biases = self.init_params()

    def get_params(self):
        return self.weights, self.biases

    def init_params(self):  # (204 / 6) * (96 / 6) * 8 = 4352
        weights = dict()
        biases = dict()
        weights['conv1'] = weight_variable([3, 3, 512, 64])
        weights['conv2'] = weight_variable([3, 3, 64, 8])
        weights['fc1'] = weight_variable([4352, 2])
        biases['conv1'] = bias_variable([64])
        biases['conv2'] = bias_variable([8])
        biases['fc1'] = bias_variable([2])
        return weights, biases

    def cnn(self, x):
        conv1 = conv2d(x, w=self.weights['conv1'], b=self.biases['conv1'], strides=1)
        pool1 = maxpool2d(conv1, k=2)
        conv2 = conv2d(pool1, w=self.weights['conv2'], b=self.biases['conv2'], strides=1)
        output = maxpool2d(conv2, k=3)
        return output

    def fc(self, x):
        flattened = tf.reshape(x, [-1, 4352])
        fc = tf.add(tf.matmul(flattened, self.weights['fc1']), self.biases['fc1'])
        output = tf.nn.relu(fc)
        return output

    def run(self, x):
        output = self.cnn(x)
        output = self.fc(output)
        return output
