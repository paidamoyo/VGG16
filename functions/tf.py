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


def deconv2d(x, w, b, stride=2, padding='VALID'):

    print(x.get_shape())
    print(w.get_shape())
    input_height = x.get_shape()[1]
    input_width = x.get_shape()[2]
    filter_height = w.get_shape()[0]
    filter_width = w.get_shape()[1]
    row_stride = stride
    col_stride = stride

    out_rows, out_cols = get2d_deconv_output_size(input_height, input_width, filter_height,
                                                  filter_width, row_stride, col_stride, padding)
    # batch_size, rows, cols, number of channels #

    print(out_rows)
    print(out_cols)
    output_shape = tf.pack([tf.shape(x)[0], out_rows, out_rows, tf.shape(w)[2]])
    print(output_shape)
    y = tf.nn.conv2d_transpose(x, w, output_shape, [1, stride, stride, 1], padding)
    print(y.get_shape())
    y = tf.nn.bias_add(y, b)
    return tf.nn.relu(y)


def get2d_deconv_output_size(input_height, input_width, filter_height,
                         filter_width, row_stride, col_stride, padding_type):
    """Returns the number of rows and columns in a convolution/pooling output."""
    # Compute number of rows in the output, based on the padding.
    if input_height.value is None or filter_height.value is None:
        out_rows = None
    elif padding_type == "VALID":
        out_rows = (input_height.value - 1) * row_stride + filter_height.value
    elif padding_type == "SAME":
        out_rows = input_height.value * row_stride
    else:
        raise ValueError("Invalid value for padding: %r" % padding_type)

    # Compute number of columns in the output, based on the padding.
    if input_width.value is None or filter_width.value is None:
        out_cols = None
    elif padding_type == "VALID":
        out_cols = (input_width.value - 1) * col_stride + filter_width.value
    elif padding_type == "SAME":
        out_cols = input_width.value * col_stride

    return out_rows, out_cols


def fc(x, w, b):
    return tf.nn.relu(tf.add(tf.matmul(x, w), b))