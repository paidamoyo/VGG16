#!/usr/bin/env python

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from functions.tf import conv2d, deconv2d, fc, weight_variable, bias_variable
from functions.record import print_log, record_metrics


class ConvVae:
    def __init__(self, params, flags, logging, seed):
        tf.set_random_seed(seed)

        self.params = params
        self.logging = logging
        self.flags = flags
        self.x = tf.placeholder(tf.float32, [None, params['image_dim'], params['image_dim'], 1], name='x')  # input patches
        self.keep_prob = tf.placeholder(tf.float32, name='dropout')
        self.epsilon = tf.placeholder(tf.float32, [None, params['hidden_size']], name='epsilon')
        self.lr = params['lr']

        self._define_layers(params)
        self.weights, self.biases = self._initialize_variables()
        self.x_reconst, self.mean, self.stddev, self.gen = self._create_network()
        self.cost, self.optimizer = self._create_loss_optimizer()
        self.saver = tf.train.Saver()
        self.summary()
        self.merged = tf.merge_all_summaries()


        # Launch the session
        self.sess = tf.InteractiveSession()

        self.writer = tf.train.SummaryWriter(self.flags['logging_directory'], self.sess.graph)
        if self.flags['restore'] is True:
            self.saver.restore(self.sess, self.flags['restore_directory'] + self.flags['restore_file'])
            print_log("Model restored from %s" % self.flags['restore_file'], self.logging)
        else:
            self.sess.run(tf.initialize_all_variables())
            print_log("Mode training from scratch.", logging)

    def _define_layers(self, params):
        if params['image_dim'] == 512:
            self.depth_conv = [1, 32, 32, 64, 64, 128, 128, 256]
            self.num_conv = len(self.depth_conv) - 1
            self.depth_fc = [3 * 3 * 256, 256, params['hidden_size'] * 2]
            self.num_fc = len(self.depth_fc) - 1
            self.depth_deconv = [params['hidden_size'], 256, 128, 128, 64, 64, 32, 32, 1]
            self.num_deconv = len(self.depth_deconv) - 1
        if params['image_dim'] == 128:
            self.depth_conv = [1, 32, 64, 64, 128, 128]
            self.num_conv = len(self.depth_conv) - 1
            self.depth_fc = [3 * 3 * 128, 1024, params['hidden_size'] * 2]
            self.num_fc = len(self.depth_fc) - 1
            self.depth_deconv = [params['hidden_size'], 128, 128, 64, 64, 32, 1]
            self.num_deconv = len(self.depth_deconv) - 1
            self.fc_reshape = [-1, 3*3*128]
        if params['image_dim'] == 32:
            self.depth_conv = [1, 32, 64, 128]
            self.num_conv = len(self.depth_conv) - 1
            self.depth_fc = [64*9, params['hidden_size'] * 2]
            self.num_fc = len(self.depth_fc) - 1
            self.depth_deconv = [params['hidden_size'], 64, 32, 16, 1]
            self.num_deconv = len(self.depth_deconv) - 1
            self.fc_reshape = [-1, 64*9]

    def summary(self):
        for k in self.weights.keys():
            tf.histogram_summary("weights_" + k, self.weights[k])
        for k in self.biases.keys():
            tf.histogram_summary("biases_" + k, self.biases[k])
        tf.scalar_summary("cost", self.cost)
        tf.image_summary("x", self.x)
        tf.image_summary("x_reconst", self.x_reconst)

    def _initialize_variables(self):
        weights, biases = dict(), dict()
        for c in range(self.num_conv):
            weights['conv' + str(c)] = weight_variable('conv' + str(c), [3, 3, self.depth_conv[c], self.depth_conv[c+1]])
            biases['conv' + str(c)] = bias_variable('conv' + str(c), [self.depth_conv[c+1]])
        for f in range(self.num_fc):
            weights['fc' + str(f)] = weight_variable('fc' + str(f), [self.depth_fc[f], self.depth_fc[f+1]])
            biases['fc' + str(f)] = bias_variable('fc' + str(f), [self.depth_fc[f+1]])
        for d in range(self.num_deconv):
            weights['deconv' + str(d)] = weight_variable('deconv' + str(d), [3, 3, self.depth_deconv[d+1], self.depth_deconv[d]])
        return weights, biases

    def decoder(self, z):
        if z is None:
            mean = None
            stddev = None
            input_sample = self.epsilon
        else:
            mean, stddev = tf.split(1, 2, z)
            stddev = tf.sqrt(tf.exp(stddev))
            print(mean.get_shape())
            print(stddev.get_shape())
            input_sample = mean + self.epsilon * stddev
        y = tf.expand_dims(tf.expand_dims(input_sample, 1), 1)
        for d in range(self.num_deconv):
            key = 'deconv' + str(d)
            if d != self.num_deconv - 1:
                y = deconv2d(y, w=self.weights[key], stride=2, padding='VALID')
            else:
                y = deconv2d(y, w=self.weights[key], stride=2, padding='SAME')  # to return even number (510 x 510)
        return tf.pad(y, [[0, 0], [1, 1], [1, 1], [0, 0]]), mean, stddev

    def encoder(self):
        x = self.x
        for c in range(self.num_conv):
            key = 'conv' + str(c)
            x = conv2d(x, w=self.weights[key], b=self.biases[key], stride=2, padding='SAME')
        x = tf.reshape(x, self.fc_reshape)
        for f in range(self.num_fc):
            key = 'fc' + str(f)
            print(key)
            print(x.get_shape())
            x = fc(x, w=self.weights[key], b=self.biases[key])
            x = tf.nn.dropout(x, keep_prob=self.keep_prob)
        return x

    def _create_network(self):
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
        return cost, optimizer

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

    def train(self, batch_generating_fxn, aux_filenames):
        step = 1
        while step < self.params['training_iters']:

            print('Begin batch number: %d' % step)
            batch_x = batch_generating_fxn()
            norm = np.random.standard_normal([self.params['batch_size'], self.params['hidden_size']])
            summary, _ = self.sess.run([self.merged, self.optimizer], feed_dict={self.x: batch_x, self.keep_prob: 0.5, self.epsilon: norm})

            if step % self.params['display_step'] == 0:
                summary, loss, _ = self.sess.run([self.merged, self.cost, self.optimizer], feed_dict={self.x: batch_x, self.keep_prob: 0.5, self.epsilon: norm})
                record_metrics(loss=loss, acc=None, batch_y=None, logging=self.logging, step=step, split=None)
            self.writer.add_summary(summary=summary, global_step=step)
            step += 1

        print("Optimization Finished!")
        checkpoint_name = self.flags['logging_directory'] + aux_filenames + '.ckpt'
        save_path = self.saver.save(self.sess, checkpoint_name)
        print("Model saved in file: %s" % save_path)
