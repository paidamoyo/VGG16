#!/usr/bin/env python

import tensorflow as tf
import numpy as np


def weight_variable(name, shape):
    if len(shape) == 4:
        fan_in = shape[0] * shape[1] * shape[2]
        fan_out = shape[0] * shape[1] * shape[3]
    else:  # len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
    initial = xavier_init(fan_in, fan_out, shape)
    return tf.Variable(initial, name=name)


def deconv_weight_variable(name, shape):
    if len(shape) == 4:
        fan_in = shape[0] * shape[1] * shape[3]
        fan_out = shape[0] * shape[1] * shape[2]
    else:  # len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
    initial = xavier_init(fan_in, fan_out, shape)
    return tf.Variable(initial, name=name)


def bias_variable(name, shape, value=0.0):
    initial = tf.constant(value, shape=shape, dtype=tf.float32)
    return tf.Variable(initial, name)


def xavier_init(fan_in, fan_out, shape, constant=1):
    """ Xavier initialization of network weights"""
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    low = -constant*np.sqrt(6.0/(fan_in + fan_out))
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform(shape, minval=low, maxval=high, dtype=tf.float32)


def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


def conv2d(x, w, b, s, stride=1, padding='SAME', act_fn=tf.nn.tanh):
    x = tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding=padding)
    if b is not None or s is not None:
        x = batch_norm(x, s)
    x = tf.add(x, b)
    x = tf.nn.tanh(x)
    return x


def deconv2d(x, w, b, s, stride=2, padding='SAME', act_fn=tf.nn.tanh):
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
        print("check the padding on the deconv2d")
        exit()

    # batch_size, rows, cols, number of channels #
    output_shape = tf.pack([batch_size, out_rows, out_cols, out_channels])
    x = tf.nn.conv2d_transpose(x, w, output_shape, [1, stride, stride, 1], padding)
    if b is not None or s is not None:
        x = batch_norm(x, s)
    x = tf.add(x, b)
    x = tf.nn.tanh(x)
    return x


def batch_norm(x, s, epsilon=1e-3):
    # Calculate batch mean and variance
    batch_mean1, batch_var1 = tf.nn.moments(x, [0], keep_dims=True)

    # Apply the initial batch normalizing transform
    z1_hat = (x - batch_mean1) / tf.sqrt(batch_var1 + epsilon)
    z1_hat = z1_hat * s
    return z1_hat


def fc(x, w, b):
    return tf.nn.tanh(tf.add(tf.matmul(x, w), b))


def spp_layer(mapstack, dims, poolnum):
    maps = tf.unpack(mapstack, axis=3)
    flattened = []
    for mapnum in range(len(maps)):
        for p in poolnum:
            div = tf.constant(p, dtype=tf.int32)
            size = tf.reshape(
                tf.pack([tf.constant([0], dtype=tf.int32), tf.to_int32(dims[0] / div), tf.to_int32(dims[1] / div)]),
                [3, ])
            for i in range(p):
                i = tf.constant(i, dtype=tf.int32)
                for j in range(p):
                    j = tf.constant(j, dtype=tf.int32)
                    x = tf.to_int32(dims[0] / div)
                    y = tf.to_int32(dims[0] / div)
                    begin = tf.reshape(
                        tf.pack([tf.constant([0], dtype=tf.int32), tf.to_int32(i * x), tf.to_int32(j * y)]), [3, ])
                    flattened.append(tf.reduce_max(tf.slice(maps[0], begin=begin, size=size)))
    return tf.reshape(tf.convert_to_tensor(flattened), [1, 2560])