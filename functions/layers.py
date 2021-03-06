#!/usr/bin/env python

"""
Author: Dan Salo
Last Edit: 11/11/2016

Purpose: Class for convolutional model creation similar to Keras with layer-by-layer formulation.
Example:
    x ### is a numpy 4D array
    encoder = Layers(input)
    encoder.conv2d(3, 64)
    encoder.conv2d(3, 64)
    encoder.maxpool()
    ...
    decoder = Layers(z)
    decoder.deconv2d(4, 156, padding='VALID')
    decoder.deconv2d(3, 144, stride=2)
    decoder.deconv2d(5, 128, stride=2)
    ...
"""

import tensorflow as tf
import logging
import tensorflow.contrib.layers as init


class Layers:
    def __init__(self, x):
        """
        Initialize model Layers.
        .input = numpy array
        .input_shape = [batch_size, height, width, channels]
        .count = dictionary to keep count of number of certain types of layers for naming purposes
        """
        self.input = x
        self.input_shape = tf.shape(x)
        self.count = {'conv': 1, 'deconv': 1, 'fc': 1, 'flat': 1, 'mp': 1, 'up': 1, 'ap': 1}

    def conv2d(self, filter_size, output_channels, stride=1, padding='SAME', activation_fn=tf.nn.relu, b_value=0.0, s_value=1.0):
        """
        :param filter_size: int. assumes square filter
        :param output_channels: int
        :param stride: int
        :param padding: 'VALID' or 'SAME'
        :param activation_fn: tf.nn function
        :param b_value: float
        :param s_value: float
        """
        scope = 'conv_' + str(self.count['conv'])
        with tf.variable_scope(scope):
            input_channels = self.input.get_shape()[3]
            output_shape = [filter_size, filter_size, input_channels, output_channels]
            w = self.weight_variable(name='weights', shape=output_shape)
            self.input = tf.nn.conv2d(self.input, w, strides=[1, stride, stride, 1], padding=padding)
            if s_value is not None:
                s = self.const_variable(name='scale', shape=[output_channels], value=s_value)
                self.input = self.batch_norm(self.input, s)
            if b_value is not None:
                b = self.const_variable(name='bias', shape=[output_channels], value=b_value)
                self.input = tf.add(self.input, b)
            if activation_fn is not None:
                self.input = activation_fn(self.input)
        self.print_log(scope + ' output: ' + str(self.input.get_shape()))
        
        self.count['conv'] += 1

    def deconv2d(self, filter_size, output_channels, stride=1, padding='SAME', activation_fn=tf.nn.relu, b_value=0.0, s_value=1.0):
        """
        :param filter_size: int. assumes square filter
        :param output_channels: int
        :param stride: int
        :param padding: 'VALID' or 'SAME'
        :param activation_fn: tf.nn function
        :param b_value: float
        :param s_value: float
        """
        scope = 'deconv_' + str(self.count['deconv'])
        with tf.variable_scope(scope):
            input_channels = self.input.get_shape()[3]
            output_shape = [filter_size, filter_size, output_channels, input_channels]
            w = self.weight_variable(name='weights', shape=output_shape)

            batch_size = tf.shape(self.input)[0]
            input_height = tf.shape(self.input)[1]
            input_width = tf.shape(self.input)[2]
            filter_height = tf.shape(w)[0]
            filter_width = tf.shape(w)[1]
            out_channels = tf.shape(w)[2]
            row_stride = stride
            col_stride = stride

            if padding == "VALID":
                out_rows = (input_height - 1) * row_stride + filter_height
                out_cols = (input_width - 1) * col_stride + filter_width
            else:  # padding == "SAME":
                out_rows = input_height * row_stride
                out_cols = input_width * col_stride

            out_shape = tf.pack([batch_size, out_rows, out_cols, out_channels])

            self.input = tf.nn.conv2d_transpose(self.input, w, out_shape, [1, stride, stride, 1], padding)
            if s_value is not None:
                s = self.const_variable(name='scale', shape=[output_channels], value=s_value)
                self.input = self.batch_norm(self.input, s)
            if b_value is not None:
                b = self.const_variable(name='bias', shape=[output_channels], value=b_value)
                self.input = tf.add(self.input, b)
            if activation_fn is not None:
                self.input = activation_fn(self.input)
        self.print_log(scope + ' output: ' + str(self.input.get_shape()))
        
        self.count['deconv'] += 1

    def flatten(self, keep_prob=1):
        """
        :param keep_prob: int. set to 1 for no dropout
        """
        scope = 'flat_' + str(self.count['flat'])
        with tf.variable_scope(scope):
            input_nodes = tf.Dimension(self.input.get_shape()[1] * self.input.get_shape()[2] * self.input.get_shape()[3])
            output_shape = tf.pack([-1, input_nodes])
            self.input = tf.reshape(self.input, output_shape)
            if keep_prob != 1:
                self.input = tf.nn.dropout(self.input, keep_prob=keep_prob)
        self.print_log(scope + ' output: ' + str(self.input.get_shape()))
        self.count['flat'] += 1

    def fc(self, output_nodes, keep_prob=1, activation_fn=tf.nn.relu, b_value=0.0):
        """
        :param output_nodes: int
        :param keep_prob: int. set to 1 for no dropout
        :param activation_fn: tf.nn function
        """
        scope = 'fc_' + str(self.count['fc'])
        with tf.variable_scope(scope):
            input_nodes = self.input.get_shape()[1]
            output_shape = [input_nodes, output_nodes]
            w = self.weight_variable(name='weights', shape=output_shape)
            self.input = tf.matmul(self.input, w)
            if b_value is not None:
                b = self.const_variable(name='bias', shape=[output_nodes], value=0.0)
                self.input = tf.add(self.input, b)
            if activation_fn is not None:
                self.input = activation_fn(self.input)
            if keep_prob != 1:
                self.input = tf.nn.dropout(self.input, keep_prob=keep_prob)
        self.print_log(scope + ' output: ' + str(self.input.get_shape()))
        self.count['fc'] += 1

    def unpool(self, k=2):
        """
        :param k: int
        """
        # Source: https://github.com/tensorflow/tensorflow/issues/2169
        # Not Yet Tested
        bottom_shape = tf.shape(self.input)
        top_shape = [bottom_shape[0], bottom_shape[1] * 2, bottom_shape[2] * 2, bottom_shape[3]]

        batch_size = top_shape[0]
        height = top_shape[1]
        width = top_shape[2]
        channels = top_shape[3]

        argmax_shape = tf.to_int64([batch_size, height, width, channels])
        output_list = [k // (argmax_shape[2] * argmax_shape[3]),
                       k % (argmax_shape[2] * argmax_shape[3]) // argmax_shape[3]]
        argmax = tf.pack(output_list)

        t1 = tf.to_int64(tf.range(channels))
        t1 = tf.tile(t1, [batch_size * (width // 2) * (height // 2)])
        t1 = tf.reshape(t1, [-1, channels])
        t1 = tf.transpose(t1, perm=[1, 0])
        t1 = tf.reshape(t1, [channels, batch_size, height // 2, width // 2, 1])
        t1 = tf.transpose(t1, perm=[1, 0, 2, 3, 4])

        t2 = tf.to_int64(tf.range(batch_size))
        t2 = tf.tile(t2, [channels * (width // 2) * (height // 2)])
        t2 = tf.reshape(t2, [-1, batch_size])
        t2 = tf.transpose(t2, perm=[1, 0])
        t2 = tf.reshape(t2, [batch_size, channels, height // 2, width // 2, 1])

        t3 = tf.transpose(argmax, perm=[1, 4, 2, 3, 0])

        t = tf.concat(4, [t2, t3, t1])
        indices = tf.reshape(t, [(height // 2) * (width // 2) * channels * batch_size, 4])

        x1 = tf.transpose(self.input, perm=[0, 3, 1, 2])
        values = tf.reshape(x1, [-1])

        delta = tf.SparseTensor(indices, values, tf.to_int64(top_shape))
        return tf.sparse_tensor_to_dense(tf.sparse_reorder(delta))

    def maxpool(self, k=2, globe=False):
        """
        :param k: int
        :param globe:  int, whether to pool over each feature map in its entirety
        """
        scope = 'maxpool_' + str(self.count['mp'])
        if globe is True:  # self.input must be a 4D image stack
            k1 = self.input.get_shape()[1]
            k2 = self.input.get_shape()[2]
            s1 = 1
            s2 = 1
            padding = 'VALID'
        else:
            k1 = k
            k2 = k
            s1 = k
            s2 = k
            padding = 'SAME'
        with tf.variable_scope(scope):
            self.input = tf.nn.max_pool(self.input, ksize=[1, k1, k2, 1], strides=[1, s1, s2, 1], padding=padding)
        self.print_log(scope + ' output: ' + str(self.input.get_shape()))
        self.count['mp'] += 1

    def avgpool(self, k=2, globe=False):
        """
         :param k: int
         :param globe: int, whether to pool over each feature map in its entirety
         """
        scope = 'avgpool_' + str(self.count['mp'])
        if globe is True:  # self.input must be a 4D image stack
            k1 = self.input.get_shape()[1]
            k2 = self.input.get_shape()[2]
            s1 = 1
            s2 = 1
            padding = 'VALID'
        else:
            k1 = k
            k2 = k
            s1 = k
            s2 = k
            padding = 'SAME'
        with tf.variable_scope(scope):
            self.input = tf.nn.avg_pool(self.input, ksize=[1, k1, k2, 1], strides=[1, s1, s2, 1], padding=padding)
        self.print_log(scope + ' output: ' + str(self.input.get_shape()))
        self.count['ap'] += 1

    def weight_variable(self, name, shape):
        """
        :param name: string
        :param shape: 4D array
        :return: tf variable
        """
        w = tf.get_variable(name=name, shape=shape, initializer=init.variance_scaling_initializer())
        weights_norm = tf.reduce_sum(tf.nn.l2_loss(w), name=name + '_norm')
        tf.add_to_collection('weight_losses', weights_norm)
        return w

    def get_output(self):
        """
        call at the last layer of the network.
        """
        return self.input

    @staticmethod
    def batch_norm(x, s, epsilon=1e-3):
        """
        :param x: input feature map stack
        :param s: constant tf variable
        :param epsilon: float
        :return: output feature map stack
        """
        # Calculate batch mean and variance
        batch_mean1, batch_var1 = tf.nn.moments(x, [0], keep_dims=True)

        # Apply the initial batch normalizing transform
        z1_hat = (x - batch_mean1) / tf.sqrt(batch_var1 + epsilon)
        z1_hat = z1_hat * s
        return z1_hat

    @staticmethod
    def print_log(message):
        print(message)
        logging.info(message)

    @staticmethod
    def const_variable(name, shape, value):
        """
        :param name: string
        :param shape: 1D array
        :param value: float
        :return: tf variable
        """
        return tf.get_variable(name, shape, initializer=tf.constant_initializer(value))
