#!/usr/bin/env python

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from functions.record import record_metrics, print_log, setup_metrics
from functions.layers import Layers

class ConvVae:
    def __init__(self, flags):

        self.flags = flags
        self.x = tf.placeholder(tf.float32, [None, flags['image_dim'], flags['image_dim'], 1], name='x')  # input patches
        self.y = tf.placeholder(tf.int32, shape=[1])
        self.keep_prob = tf.placeholder(tf.float32, name='dropout')
        self.epsilon = tf.placeholder(tf.float32, [None, flags['hidden_size']], name='epsilon')
        self.lr = tf.placeholder(tf.float32, name='learning_rate')

        self._set_seed()
        self.x_recon, self.mean, self.stddev, self.gen = self._create_network()
        self.vae, self.recon, self.cost, self.optimizer = self._create_loss_optimizer()

        self._summary()
        self.merged = tf.merge_all_summaries()

        self.saver = tf.train.Saver()
        self.sess = tf.InteractiveSession()

    def _set_seed(self):
        tf.set_random_seed(self.flags['seed'])
        np.random.seed(self.flags['seed'])

    def _summary(self):
        for var in tf.trainable_variables():
            tf.histogram_summary(var.name, var)
        tf.scalar_summary("Total Loss", self.cost)
        tf.scalar_summary("Reconstruction Loss", self.recon)
        tf.scalar_summary("VAE Loss", self.vae)
        tf.histogram_summary("Mean", self.mean)
        tf.histogram_summary("Stddev", self.stddev)
        tf.image_summary("x", self.x)
        tf.image_summary("x_recon", self.x_recon)

    def _encoder(self, x):
        encoder = Layers(x)
        encoder.conv2d(5, 32, stride=2)
        encoder.conv2d(5, 64, stride=2)
        encoder.conv2d(5, 128, padding='VALID')
        encoder.flatten(self.keep_prob)
        encoder.fc(self.flags['hidden_size'] * 2, activation_fn=None)
        return encoder.get_output()

    def _decoder(self, z):
        if z is None:
            mean = None
            stddev = None
            input_sample = self.epsilon
        else:
            mean, stddev = tf.split(1, 2, z)
            stddev = tf.sqrt(tf.exp(stddev))
            input_sample = mean + self.epsilon * stddev
        decoder = Layers(tf.expand_dims(tf.expand_dims(input_sample, 1), 1))
        decoder.deconv2d(3, 128, padding='VALID')
        decoder.deconv2d(5, 64, padding='VALID')
        decoder.deconv2d(5, 32, stride=2)
        decoder.deconv2d(5, 1, stride=2, activation_fn=tf.nn.sigmoid)
        return decoder.get_output(), mean, stddev

    def _create_network(self):
        with tf.variable_scope("model"):
            latent = self._encoder(x=self.x)
            x_recon, mean, stddev = self._decoder(z=latent)
        with tf.variable_scope("model", reuse=True):
            x_gen, _, _ = self._decoder(z=None)
        return x_recon, mean, stddev, x_gen

    def _create_loss_optimizer(self, epsilon=1e-8):
        recon = tf.reduce_sum(-self.x * tf.log(self.x_recon + epsilon) - (1.0 - self.x) * tf.log(1.0 - self.x_recon + epsilon))
        vae = tf.reduce_sum(0.5 * (tf.square(self.mean) + tf.square(self.stddev) - 2.0 * tf.log(self.stddev + epsilon) - 1.0))
        cost = tf.reduce_mean(vae + recon)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(cost)
        return vae, recon, cost, optimizer

    def print_variable(self, var):
        if var == 'x_recon':
            print_var = tf.Print(self.x_recon, [self.x_recon])
            norm = np.random.normal(size=[self.flags['batch_size'], self.flags['hidden_size']])
            x = np.zeros([self.flags['batch_size'], self.flags['image_dim'], self.flags['image_dim'], 1])
        else:
            print('Print Variable not defined .... printing x_recon')
            return tf.Print(self.x_recon, [self.x_recon])
        return self.sess.run(print_var, feed_dict={self.x: x, self.keep_prob: 0.5, self.epsilon: norm})

    def x_recon(self, n):
        norm = np.random.normal(size=[n, self.flags['hidden_size']])
        images = self.sess.run(self.gen, feed_dict={self.epsilon: norm})
        for i in range(len(images)):
            plt.imshow(np.squeeze(images[i]), cmap='gray')
            plt.savefig(self.flags['logging_directory'] + 'x_recon_' + str(i))

    def x(self, image_generating_fxn):
        images = image_generating_fxn()
        for i in range(len(images)):
            plt.imshow(np.squeeze(images[i]), cmap='gray')
            plt.savefig(self.flags['logging_directory'] + 'x_' + str(i))

    def transform(self, x):
        """Transform data by mapping it into the latent space."""
        return self.sess.run(self.mean, feed_dict={self.x: x, self.keep_prob: 1.0})

    def train(self, batch_generating_fxn, lr_iters, run_num):

        setup_metrics(self.flags, lr_iters, run_num)
        writer = tf.train.SummaryWriter(self.flags['logging_directory'], self.sess.graph)
        if self.flags['restore'] is True:
            self.saver.restore(self.sess, self.flags['restore_directory'] + self.flags['restore_file'])
            print_log("Model restored from %s" % self.flags['restore_file'])
        else:
            self.sess.run(tf.initialize_all_variables())
            print_log("Model training from scratch.")

        global_step = 1
        for i in range(len(lr_iters)):
            lr = lr_iters[i][0]
            iters = lr_iters[i][1]
            print_log('Learning Rate: %d' % lr)
            print_log('Iterations: %d' % iters)
            step = 1
            while step < iters:

                print('Batch number: %d' % step)
                labels_x, batch_x = batch_generating_fxn()
                norm = np.random.standard_normal([self.flags['batch_size'], self.flags['hidden_size']])
                summary, _ = self.sess.run([self.merged, self.optimizer], feed_dict={self.x: batch_x, self.keep_prob: 0.9, self.epsilon: norm, self.lr: lr})

                if step % self.flags['display_step'] == 0:
                    summary, loss, _ = self.sess.run([self.merged, self.cost, self.optimizer], feed_dict={self.x: batch_x, self.keep_prob: 0.9, self.epsilon: norm, self.lr: lr})
                    record_metrics(loss=loss, acc=None, batch_y=None, step=step, split=None, flags=self.flags)
                writer.add_summary(summary=summary, global_step=global_step)
                step += 1
                global_step += 1

            print("Optimization Finished!")
            checkpoint_name = self.flags['logging_directory'] + 'Run' + str(run_num) + 'epoch_%d' % i + '.ckpt'
            save_path = self.saver.save(self.sess, checkpoint_name)
            print("Model saved in file: %s" % save_path)

    # def test(self, test_data, test_labels):