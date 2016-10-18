#!/usr/bin/env python

import tensorflow as tf

from functions.tf import conv2d, deconv2d, fc, weight_variable, bias_variable


class ConvVae:
    def __init__(self, params, seed):
        tf.set_random_seed(seed)
        self.weights = dict()
        self.biases = dict()
        self.hidden_size = params['hidden_size']
        self.batch_size = params['batch_size']
        self.depth_conv = [1, 32, 32, 64, 64, 128, 128, 256]
        self.num_conv = len(self.depth_conv)-1
        self.depth_fc = [3*3*256, 256, self.hidden_size*2]
        self.num_fc = len(self.depth_fc)-1
        self.depth_deconv = [self.hidden_size, 256, 128, 128, 64, 64, 32, 32, 1]
        self.num_deconv = len(self.depth_deconv)-1
        self.init_params()
        # self.summary()

    def summary(self):
        for k in self.weights.keys():
            tf.histogram_summary("weights_" + k, self.weights[k])
        for k in self.biases.keys():
            tf.histogram_summary("biases_" + k, self.biases[k])

    def init_params(self):
        for c in range(self.num_conv):
            self.weights['conv' + str(c)] = weight_variable([3, 3, self.depth_conv[c], self.depth_conv[c+1]])
            self.biases['conv' + str(c)] = bias_variable([self.depth_conv[c+1]])
        for f in range(self.num_fc):
            self.weights['fc' + str(f)] = weight_variable([self.depth_fc[f], self.depth_fc[f+1]])
            self.biases['fc' + str(f)] = bias_variable([self.depth_fc[f+1]])
        for d in range(self.num_deconv):
            self.weights['deconv' + str(d)] = weight_variable([3, 3, self.depth_deconv[d+1], self.depth_deconv[d]])

    def decoder(self, z, epsilon):
        if z is None:
            mean = None
            stddev = None
            input_sample = epsilon
        else:
            mean, stddev = tf.split(1, 2, z)
            stddev = tf.sqrt(tf.exp(stddev))
            input_sample = mean + epsilon * stddev
        y = tf.expand_dims(tf.expand_dims(input_sample, 1), 1)
        for d in range(self.num_deconv):
            key = 'deconv' + str(d)
            if d != 7:
                y = deconv2d(y, w=self.weights[key], stride=2, padding='VALID')
            else:
                y = deconv2d(y, w=self.weights[key], stride=2, padding='SAME')  # to return even number (510 x 510)
        return tf.pad(y, [[0, 0], [1, 1], [1, 1], [0, 0]]), mean, stddev

    def encoder(self, x, keep_prob):
        for c in range(self.num_conv):
            key = 'conv' + str(c)
            x = conv2d(x, w=self.weights[key], b=self.biases[key], stride=2, padding='VALID')
        x = tf.reshape(x, [-1, 3*3*256])
        for f in range(self.num_fc):
            key = 'fc' + str(f)
            x = fc(x, w=self.weights[key], b=self.biases[key])
            x = tf.nn.dropout(x, keep_prob=keep_prob)
        return x

    def init_cost(self, output_tensor, target_tensor, mean, stddev, epsilon=1e-8):
        vae = tf.reduce_sum(-target_tensor * tf.log(output_tensor + epsilon) -
                            (1.0 - target_tensor) * tf.log(1.0 - output_tensor + epsilon))
        recon = tf.reduce_sum(0.5 * (tf.square(mean) + tf.square(stddev) - 2.0 * tf.log(stddev + epsilon) - 1.0))
        return vae + recon

    def run(self, x, keep_prob, epsilon):
        y, mean, stddev = self.decoder(self.encoder(x, keep_prob), epsilon=epsilon)
        print_y = tf.Print(y, [y])
        cost = self.init_cost(y, x, mean, stddev)
        return y, cost, print_y

    def get_params(self):
        return self.weights, self.biases
