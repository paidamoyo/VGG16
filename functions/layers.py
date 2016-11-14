#!/usr/bin/env python

"""
Author: Dan Salo
Last Edit: 11/11/2016

Purpose: Class for convolutional model creation similar to Keras layer-by-layer formulation.
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
        self.count = {'conv': 1, 'deconv': 1, 'fc': 1, 'flat': 1, 'mp': 1, 'up': 1}
        self.Functions = Functions()

    def conv2d(self, filter_size, output_channels, stride=1, padding='SAME', activation_fn=tf.nn.relu, b_value=0.0, s_value=0.0):
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
            w = self.Functions.weight_variable(name='weights', shape=output_shape,)
            b = self.Functions.const_variable(name='bias', shape=[output_channels], value=b_value)
            s = self.Functions.const_variable(name='scale', shape=[output_channels], value=s_value)
            self.input = self.Functions.conv2d(self.input, w, s, b, stride, padding, activation_fn)
        self.print_log(scope + ' output: ' + str(self.input.get_shape()))
        self.input_shape = output_shape
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
            w = self.Functions.weight_variable(name='weights', shape=output_shape)
            b = self.Functions.const_variable(name='bias', shape=[output_channels], value=b_value)
            s = self.Functions.const_variable(name='scale', shape=[output_channels], value=s_value)
            self.input = self.Functions.deconv2d(self.input, w, s, b, stride, padding, activation_fn)
        self.print_log(scope + ' output: ' + str(self.input.get_shape()))
        self.input_shape = output_shape
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
                self.input = self.Functions.dropout(self.input, keep_prob=keep_prob)
        self.print_log(scope + ' output: ' + str(self.input.get_shape()))
        self.input_shape = output_shape
        self.count['flat'] += 1

    def fc(self, output_nodes, keep_prob=1, activation_fn=tf.nn.relu):
        """
        :param output_nodes: int
        :param keep_prob: int. set to 1 for no dropout
        :param activation_fn: tf.nn function
        """
        scope = 'fc_' + str(self.count['fc'])
        with tf.variable_scope(scope):
            input_nodes = self.input.get_shape()[1]
            output_shape = [input_nodes, output_nodes]
            w = self.Functions.weight_variable(name='weights', shape=output_shape)
            b = self.Functions.const_variable(name='bias', shape=[output_nodes], value=0.0)
            self.input = self.Functions.fc(self.input, w, b, act_fn=activation_fn)
            if keep_prob != 1:
                self.input = self.Functions.dropout(self.input, keep_prob=keep_prob)
        self.print_log(scope + ' output: ' + str(self.input.get_shape()))
        self.input_shape = output_shape
        self.count['fc'] += 1

    def maxpool(self, k=2):
        """
        :param k: int
        """
        scope = 'maxpool_' + str(self.count['mp'])
        with tf.variable_scope(scope):
            self.input = self.Functions.maxpool(self.input, k)
        self.print_log(scope + ' output: ' + str(self.input.get_shape()))
        self.count['mp'] += 1

    def unpool2x2(self, argmax=2):
        """
        :param argmax: int
        """
        scope = 'unpool2x2_' + str(self.count['up'])
        with tf.variable_scope(scope):
            self.input = self.Functions.unpool2x2(argmax, self.input)
        self.print_log(scope + ' output: ' + str(self.input.get_shape()))
        self.count['up'] += 1

    def get_output(self):
        """
        call at the top of the network.
        """
        return self.input

    def print_log(self, message):
        print(message)
        logging.info(message)


class Functions:
    def __init__(self):
        self.x = 1

    def weight_variable(self, name, shape):
        """
        :param name: string
        :param shape: 4D array
        :return: tf variable
        """
        v = tf.get_variable(name=name, shape=shape, initializer=init.variance_scaling_initializer())
        return v

    def const_variable(self, name, shape, value):
        """
        :param name: string
        :param shape: 1D array
        :param value: float
        :return: tf variable
        """
        v = tf.get_variable(name, shape, initializer=tf.constant_initializer(value))
        return v

    def maxpool(self, x, k):
        """
        :param x: input feature maps
        :param k: int. k=2 for 2x2 maxpool
        :return: output feature maps
        """
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                              padding='SAME')

    def conv2d(self, x, w, b, s, stride, padding, act_fn):
        """
        :param x: input feature map stack
        :param w: 4D tf variable
        :param b: constant tf variable
        :param s: constant tf variable
        :param stride: int
        :param padding: 'VALID' or 'SAME' for zero padding
        :param act_fn: tf.nn function
        :return: output feature map stack
        """
        x = tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding=padding)
        if b is not None or s is not None:
            x = self.batch_norm(x, s)
        x = tf.add(x, b)
        if act_fn is not None:
            x = act_fn(x)
        return x

    def deconv2d(self, x, w, b, s, stride, padding, act_fn):
        """
        :param x: input feature map stack
        :param w: 4D tf variable
        :param b: constant tf variable
        :param s: constant tf variable
        :param stride: int
        :param padding: 'VALID' or 'SAME' for zero padding
        :param act_fn: tf.nn function
        :return: output feature map stack
        """
        batch_size = tf.shape(x)[0]
        input_height = tf.shape(x)[1]
        input_width = tf.shape(x)[2]
        filter_height = tf.shape(w)[0]
        filter_width = tf.shape(w)[1]
        out_channels = tf.shape(w)[2]
        row_stride = stride
        col_stride = stride

        if padding == "VALID":
            out_rows = (input_height - 1) * row_stride + filter_height
            out_cols = (input_width - 1) * col_stride + filter_width
        elif padding == "SAME":
            out_rows = input_height * row_stride
            out_cols = input_width * col_stride
        else:
            out_rows = 0
            out_cols = 0
            print("check the padding on the deconv2d")
            exit()

        output_shape = tf.pack([batch_size, out_rows, out_cols, out_channels])
        x = tf.nn.conv2d_transpose(x, w, output_shape, [1, stride, stride, 1], padding)
        if b is not None or s is not None:
            x = self.batch_norm(x, s)
        x = tf.add(x, b)
        if act_fn is not None:
            x = act_fn(x)
        return x

    def unpool2x2(self,argmax, bottom):
        # Source: https://github.com/tensorflow/tensorflow/issues/2169
        ### Not yet implemented
        bottom_shape = tf.shape(bottom)
        top_shape = [bottom_shape[0], bottom_shape[1] * 2, bottom_shape[2] * 2, bottom_shape[3]]

        batch_size = top_shape[0]
        height = top_shape[1]
        width = top_shape[2]
        channels = top_shape[3]

        argmax_shape = tf.to_int64([batch_size, height, width, channels])
        output_list = [argmax // (argmax_shape[2] * argmax_shape[3]),
                       argmax % (argmax_shape[2] * argmax_shape[3]) // argmax_shape[3]]
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

        x1 = tf.transpose(bottom, perm=[0, 3, 1, 2])
        values = tf.reshape(x1, [-1])

        delta = tf.SparseTensor(indices, values, tf.to_int64(top_shape))
        return tf.sparse_tensor_to_dense(tf.sparse_reorder(delta))

    def batch_norm(self, x, s, epsilon=1e-3):
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

    def fc(self, x, w, b, act_fn):
        """
        :param x: input flattened vector
        :param w: 4D tf variable
        :param b: constant tf variable
        :param act_fn: tf.nn function
        :return: output flattened vector
        """
        x = tf.matmul(x, w)
        x = tf.add(x, b)
        if act_fn is not None:
            x = act_fn(x)
        return x

    def dropout(self, x, keep_prob):
        """
        :param x: input vector
        :param keep_prob: float. probability of keeping an element of the input vector
        :return: output vector
        """
        return tf.nn.dropout(x, keep_prob=keep_prob)