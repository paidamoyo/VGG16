from functions.tf import conv2d, deconv2d, fc, conv_weight_variable, weight_variable, const_variable, dropout
import tensorflow as tf


class Layers:

    def __init__(self, x):
        self.input = x
        self.input_shape = tf.shape(x)
        self.count = {
            'conv': 1,
            'deconv': 1,
            'fc': 1,
            'flat': 1
        }
        self.params = dict()

    def conv2d(self, filter_size, output_channels, stride=1, padding='SAME', activation_fn=tf.nn.relu):
        scope = 'conv_' + str(self.count['conv'])
        with tf.variable_scope(scope):
            input_channels = self.input.get_shape()[3]
            output_shape = [filter_size, filter_size, input_channels, output_channels]
            w = conv_weight_variable(name='weights', shape=output_shape)
            b = const_variable(name='bias', shape=[output_channels], value=0.0)
            s = const_variable(name='scale', shape=[output_channels], value=1.0)
            self.input = conv2d(self.input, w, s, b, stride, padding, activation_fn)
        self.input_shape = output_shape
        self.count['conv'] += 1

    def deconv2d(self, filter_size, output_channels, stride=1, padding='SAME', activation_fn=tf.nn.relu):
        scope = 'deconv_' + str(self.count['deconv'])
        with tf.variable_scope(scope):
            input_channels = self.input.get_shape()[3]
            output_shape = [filter_size, filter_size, output_channels, input_channels]
            w = conv_weight_variable(name='weights', shape=output_shape)
            b = const_variable(name='bias', shape=[output_channels], value=0.0)
            s = const_variable(name='scale', shape=[output_channels], value=1.0)
            self.input = deconv2d(self.input, w, s, b, stride, padding, activation_fn)
        self.input_shape = output_shape
        self.count['deconv'] += 1

    def flatten(self, keep_prob=1):
        scope = 'flat_' + str(self.count['flat'])
        with tf.variable_scope(scope):
            input_nodes = 9 * self.input.get_shape()[3]
            output_shape = tf.pack([tf.constant(-1), input_nodes])
            self.input = tf.reshape(self.input, output_shape)
            if keep_prob != 1:
                self.input = dropout(self.input, keep_prob=keep_prob)
        self.input_shape = output_shape
        self.count['flat'] += 1

    def fc(self, output_nodes, keep_prob=1, activation_fn=tf.nn.relu):
        scope = 'fc_' + str(self.count['fc'])
        with tf.variable_scope(scope):
            input_nodes = self.input.get_shape()[1]
            output_shape = [input_nodes, output_nodes]
            w = weight_variable(name='weights', shape=output_shape)
            b = const_variable(name='bias', shape=[output_nodes], value=0.0)
            self.input = fc(self.input, w, b, act_fn=activation_fn)
            if keep_prob != 1:
                self.input = dropout(self.input, keep_prob=keep_prob)
        self.input_shape = output_shape
        self.count['fc'] += 1

    def get_output(self):
        return self.input

