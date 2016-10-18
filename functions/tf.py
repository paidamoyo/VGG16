#!/usr/bin/env python

import tensorflow as tf
from tensorflow.python.framework import tensor_shape


def weight_variable(shape, mean=0, stddev=0.1):
    initial = tf.truncated_normal(shape, mean=mean, stddev=stddev, dtype=tf.float32)
    return tf.Variable(initial)


def bias_variable(shape, value=0.1):
    initial = tf.constant(value, shape=shape, dtype=tf.float32)
    return tf.Variable(initial)


def conv2d(img, w, b, stride=1, padding='SAME'):
    img = tf.nn.conv2d(img, w, strides=[1, stride, stride, 1], padding=padding)
    img = tf.nn.bias_add(img, b)
    return tf.nn.relu(img)


def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

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


def deconv2d(x, w, stride=2, padding='VALID'):

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
    else:  # padding == "SAME"
        out_rows = input_height * row_stride
        out_cols = input_width * col_stride

    # batch_size, rows, cols, number of channels #
    output_shape = tf.pack([batch_size, out_rows, out_cols, out_channels])
    a = tf.Print(out_rows, [out_rows], 'hi')
    sess = tf.Session()
    a.eval(session=sess)
    y = tf.nn.conv2d_transpose(x, w, output_shape, [1, stride, stride, 1], padding)
    return tf.nn.relu(y)


def fc(x, w, b):
    return tf.nn.relu(tf.add(tf.matmul(x, w), b))