#!/usr/bin/env python

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.misc

from functions.tf import conv2d, deconv2d, fc, weight_variable, bias_variable, deconv_weight_variable, batch_norm
from functions.record import record_metrics, print_log, setup_metrics


class ConvVae:
    def __init__(self, params, flags):

        self.params = params
        self.flags = flags
        self.x = tf.placeholder(tf.float32, [None, params['image_dim'], params['image_dim'], 1], name='x')  # input patches
        self.keep_prob = tf.placeholder(tf.float32, name='dropout')
        self.epsilon = tf.placeholder(tf.float32, [None, params['hidden_size']], name='epsilon')
        self.lr = tf.placeholder(tf.float32, name='learning_rate')

        self._set_seed()
        self._define_layers(params)
        self.weights, self.biases = self._initialize_variables()
        self.x_reconst, self.mean, self.stddev, self.gen = self._create_network()
        self.vae, self.recon, self.cost, self.optimizer = self._create_loss_optimizer()
        self.saver = tf.train.Saver()
        self.summary()
        self.merged = tf.merge_all_summaries()
        self.sess = tf.InteractiveSession()

    def _define_layers(self, params):
        if params['image_dim'] == 28:
            self.conv = {'input': 1,
                         'layers': [(32, 5, 2, 'SAME'), (64, 5, 2, 'SAME'), (128, 5, 1, 'VALID')]}
            self.conv_num = len(self.conv['layers'])
            self.fc = {'reshape': [-1, 3*3*128],
                       'layers': [3*3*128, params['hidden_size'] * 2]}
            self.fc_num = len(self.fc['layers'])-1
            self.deconv = {'input': params['hidden_size'],
                           'layers': [(128, 3, 1, 'VALID'), (64, 5, 1, 'VALID'), (32, 5, 2, 'SAME'), (1, 5, 2, 'SAME')]}
            self.deconv_num = len(self.deconv['layers'])

    def _set_seed(self):
        tf.set_random_seed(self.params['seed'])
        np.random.seed(self.params['seed'])

    def summary(self):
        for k in self.weights.keys():
            tf.histogram_summary("weights_" + k, self.weights[k])
        for k in self.biases.keys():
            tf.histogram_summary("biases_" + k, self.biases[k])
        tf.scalar_summary("Total Loss", self.cost)
        tf.scalar_summary("Reconstruction Loss", self.recon)
        tf.scalar_summary("VAE Loss", self.vae)
        tf.image_summary("x", self.x)
        tf.image_summary("x_reconst", self.x_reconst)

    def _initialize_variables(self):
        weights, biases = dict(), dict()

        for c in range(self.conv_num):
            i = self.conv['layers'][c]
            if c == 0:
                previous = self.conv['input']
            else:
                previous = self.conv['layers'][c-1][0]
            weights['conv' + str(c)] = weight_variable('conv' + str(c), [i[1], i[1], previous, i[0]])
            biases['conv' + str(c)] = bias_variable('conv' + str(c), [i[0]])
            biases['conv' + str(c) + '_scale'] = bias_variable('conv' + str(c) + '_scale', [i[0]], value=1.0)
        for f in range(self.fc_num):
            weights['fc' + str(f)] = weight_variable('fc' + str(f), [self.fc['layers'][f], self.fc['layers'][f+1]])
            biases['fc' + str(f)] = bias_variable('fc' + str(f), [self.fc['layers'][f+1]])
        for d in range(self.deconv_num):
            i = self.deconv['layers'][d]
            if d == 0:
                previous = self.deconv['input']
            else:
                previous = self.deconv['layers'][d-1][0]
            weights['deconv' + str(d)] = deconv_weight_variable('deconv' + str(d), [i[1], i[1], i[0], previous])
            biases['deconv' + str(d)] = bias_variable('deconv' + str(d), [i[0]])
            biases['deconv' + str(d) + '_scale'] = bias_variable('deconv' + str(d) + '_scale', [i[0]], value=1.0)
        return weights, biases

    def encoder(self):
        x = self.x
        for c in range(self.conv_num):
            key = 'conv' + str(c)
            x = conv2d(x, w=self.weights[key], b=self.biases['conv' + str(c)],
                       s=self.biases['conv' + str(c) + '_scale'], stride=self.conv['layers'][c][2],
                       padding=self.conv['layers'][c][3])
        x = tf.reshape(x, self.fc['reshape'])
        for f in range(self.fc_num):
            key = 'fc' + str(f)
            x = tf.nn.dropout(x, keep_prob=self.keep_prob)
            x = fc(x, w=self.weights[key], b=self.biases[key])
        return x

    def decoder(self, z):
        if z is None:
            mean = None
            stddev = None
            input_sample = self.epsilon
        else:
            mean, stddev = tf.split(1, 2, z)
            stddev = tf.sqrt(tf.exp(stddev))
            input_sample = mean + self.epsilon * stddev
        x_reconst = tf.expand_dims(tf.expand_dims(input_sample, 1), 1)
        for d in range(self.deconv_num):
            key = 'deconv' + str(d)
            x_reconst = deconv2d(x_reconst, w=self.weights[key], b=self.biases['deconv' + str(d)],
                         s=self.biases['deconv' + str(d) + '_scale'], stride=self.deconv['layers'][d][2],
                         padding=self.deconv['layers'][d][3])
        x_reconst = tf.nn.sigmoid(x_reconst)
        return x_reconst, mean, stddev

    def _create_network(self):
        print('Creating Network')
        x_reconst, mean, stddev = self.decoder(z=self.encoder())
        gen, _, _ = self.decoder(z=None)
        return x_reconst, mean, stddev, gen

    def print_variable(self, var):
        if var == 'x_reconst':
            print_var = tf.Print(self.x_reconst, [self.x_reconst])
            norm = np.random.normal(size=[self.params['batch_size'], self.params['hidden_size']])
            x = np.zeros([self.params['batch_size'], self.params['image_dim'], self.params['image_dim'], 1])
        else:
            print('Print Variable not defined .... printing x_reconst')
            return tf.Print(self.x_reconst, [self.x_reconst])
        return self.sess.run(print_var, feed_dict={self.x: x, self.keep_prob: 0.5, self.epsilon: norm})

    def _create_loss_optimizer(self, epsilon=1e-8):
        vae = tf.reduce_sum(-self.x * tf.log(self.x_reconst + epsilon) -
                            (1.0 - self.x) * tf.log(1.0 - self.x_reconst + epsilon))
        recon = tf.reduce_sum(0.5 * (tf.square(self.mean) + tf.square(self.stddev) - 2.0 * tf.log(self.stddev + epsilon) - 1.0))
        cost = tf.reduce_mean(vae + recon)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(cost)
        return vae, recon, cost, optimizer

    def generate_x_reconst(self):
        norm = np.random.normal(size=[10, self.params['hidden_size']])
        images = self.sess.run(self.gen, feed_dict={self.epsilon: norm})
        for i in range(len(images)):
            plt.imshow(np.squeeze(images[i]), cmap='gray')
            plt.savefig(self.flags['logging_directory'] + 'image' + str(i))

    def generate_x(self, image_generating_fxn):
        images = image_generating_fxn()
        for i in range(len(images)):
            plt.imshow(np.squeeze(images[i]), cmap='gray')
            plt.savefig(self.flags['logging_directory'] + 'image' + str(i))

    def partial_fit(self, x):
        """Train model based on mini-batch of input data.

        Return cost of mini-batch.
        """
        opt, cost = self.sess.run((self.optimizer, self.cost),
                                  feed_dict={self.x: x})
        return cost

    def transform(self, x):
        """Transform data by mapping it into the latent space."""
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return self.sess.run(self.mean, feed_dict={self.x: x})

    def reconstruct(self, x):
        """ Use VAE to reconstruct given data. """
        return self.sess.run(self.x_reconst,
                             feed_dict={self.x: x})

    def train(self, batch_generating_fxn, lr_iters, run_num):

        setup_metrics(self.flags, self.params, lr_iters, run_num)
        self.writer = tf.train.SummaryWriter(self.flags['logging_directory'], self.sess.graph)
        if self.flags['restore'] is True:
            self.saver.restore(self.sess, self.flags['restore_directory'] + self.flags['restore_file'])
            print_log("Model restored from %s" % self.flags['restore_file'])
        else:
            self.sess.run(tf.initialize_all_variables())
            print_log("Model training from scratch.")

        for i in range(len(lr_iters)):
            lr = lr_iters[i][0]
            iters = lr_iters[i][1]
            print_log('Learning Rate: %d' % lr)
            print_log('Iterations: %d' % iters)
            step = 1
            while step < iters:

                print('Batch number: %d' % step)
                batch_x = batch_generating_fxn()
                norm = np.random.standard_normal([self.params['batch_size'], self.params['hidden_size']])
                summary, _ = self.sess.run([self.merged, self.optimizer], feed_dict={self.x: batch_x, self.keep_prob: 0.9, self.epsilon: norm, self.lr: lr})

                if step % self.params['display_step'] == 0:
                    summary, loss, _ = self.sess.run([self.merged, self.cost, self.optimizer], feed_dict={self.x: batch_x, self.keep_prob: 0.9, self.epsilon: norm, self.lr: lr})
                    record_metrics(loss=loss, acc=None, batch_y=None, step=step, split=None, params=self.params)
                self.writer.add_summary(summary=summary, global_step=step)
                step += 1

            print("Optimization Finished!")
            checkpoint_name = self.flags['logging_directory'] + 'Run' + str(run_num) + 'epoch_%d' % i + '.ckpt'
            save_path = self.saver.save(self.sess, checkpoint_name)
            print("Model saved in file: %s" % save_path)

    def test(self, test_data, test_labels):


    def generate(self):
        norm = np.random.standard_normal([self.params['batch_size'], self.params['hidden_size']])
        imgs = self.sess.run(self.gen, feed_dict={self.epsilon: norm})
        for k in range(self.params['batch_size']):
            scipy.misc.imsave(self.flags['logging_directory'] + 'image_%d.png' % k, imgs[k].reshape(28, 28))