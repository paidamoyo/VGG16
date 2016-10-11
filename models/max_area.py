#!/usr/bin/env python

import tensorflow as tf

from functions.tf import weight_variable, bias_variable


class Max_area:
    def __init__(self):
        self.weights, self.biases = self.init_params()

    def init_params(self): # (204 / 6) * (96 / 6) * 8 = 4352
        weights = dict()
        biases = dict()
        weights['fc1'] = weight_variable([512, 256])
        weights['fc2'] = weight_variable([256, 2])
        biases['fc1'] = bias_variable([256])
        biases['fc2'] = bias_variable([2])
        return weights, biases

    def fc(self, x):
        flattened = tf.reshape(x, [-1, 512])
        fc1 = tf.add(tf.matmul(flattened, self.weights['fc1']), self.biases['fc1'])
        fc1 = tf.nn.relu(fc1)
        fc2 = tf.add(tf.matmul(fc1, self.weights['fc2']), self.biases['fc2'])
        fc2 = tf.nn.relu(fc2)
        return fc2

    def max_pool(self, x):
        return tf.nn.max_pool(x, ksize=[1, 204, 96, 1], strides=[1, 1, 1, 1], padding='VALID')

    def run(self, x):
        output = self.max_pool(x)
        return self.fc(output)


