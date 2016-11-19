#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import logging
import scipy.misc
import pickle
import sys

from functions.record import record_metrics, print_log, setup_metrics
from functions.data import make_directory
from functions.layers import Layers
from data.BREAST import BreastData


# Global Dictionary of Flags
flags = {
    'data_directory': '../../../Data/',  # in relationship to the code_directory
    'previous_processed_directory': 'Smart_Crop/',
    'aux_directory': 'aux/',
    'model_directory': 'conv_vae/',
    'datasets': ['SAGE'],
    'restore': False,
    'restore_file': 'INbreast.ckpt',
    'recon': 1,
    'vae': 1e-10,
    'image_dim': 128,
    'hidden_size': 256,
    'batch_size': 32,
    'display_step': 100,
    'weight_decay': 5e-5,
    'lr_decay': 0.99,
    'lr_iters': [(0.01, 750), (0.005, 1000), (0.001, 1500), (0.0005, 2000), (0.0001, 3000)]
}


class ConvVae:
    def __init__(self, flags, model):
        print(flags)
        self.flags = flags
        self.x = tf.placeholder(tf.float32, [None, flags['image_dim'], flags['image_dim'], 1], name='x')  # input patches
        self.y = tf.placeholder(tf.int32, shape=[1])
        self.keep_prob = tf.placeholder(tf.float32, name='dropout')
        self.epsilon = tf.placeholder(tf.float32, [None, flags['hidden_size']], name='epsilon')
        self.lr = tf.placeholder(tf.float32, name='learning_rate')
        folder = 'Model' + str(model) + '/'
        flags['restore_directory'] = flags['aux_directory'] + flags['model_directory']
        flags['logging_directory'] = flags['restore_directory'] + folder
        make_directory(flags['logging_directory'])
        logging.basicConfig(filename=flags['logging_directory'] + 'ModelInformation.log', level=logging.INFO)

        self._set_seed()
        if flags['image_dim'] == 28:
            print_log('Using 28x28 Architecture')
            self.x_recon, self.mean, self.stddev, self.x_gen, self.latent = self._create_network_MNIST()
        else:  # breast patches of 128
            print_log('Using 128x128 Architecture')
            self.x_recon, self.mean, self.stddev, self.x_gen, self.latent = self._create_network_BREAST()
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

    def _encoder_BREAST(self, x):
        encoder = Layers(x)
        encoder.conv2d(5, 64)
        encoder.maxpool()
        encoder.conv2d(3, 64)
        encoder.conv2d(3, 64)
        encoder.conv2d(3, 128, stride=2)
        encoder.conv2d(3, 128)
        encoder.conv2d(3, 256, stride=2)
        encoder.conv2d(3, 256)
        encoder.conv2d(3, 512, stride=2)
        encoder.conv2d(3, 512)
        encoder.conv2d(3, 1024, stride=2)
        encoder.conv2d(3, 1024)
        encoder.conv2d(1, self.flags['hidden_size'] * 2, activation_fn=None)
        encoder.avgpool(globe=True)
        return encoder.get_output()

    def _decoder_BREAST(self, z):
        if z is None:
            mean = None
            stddev = None
            input_sample = self.epsilon
        else:
            print(z.get_shape())
            mean, stddev = tf.split(1, 2, z)
            stddev = tf.sqrt(tf.exp(stddev))
            input_sample = mean + self.epsilon * stddev
        decoder = Layers(tf.expand_dims(tf.expand_dims(input_sample, 1), 1))
        decoder.deconv2d(4, 1024, padding='VALID')
        decoder.deconv2d(3, 1024)
        decoder.deconv2d(3, 512, stride=2)
        decoder.deconv2d(3, 512)
        decoder.deconv2d(3, 128, stride=2)
        decoder.deconv2d(3, 128)
        decoder.deconv2d(3, 64, stride=2)
        decoder.deconv2d(3, 64)
        decoder.deconv2d(3, 64, stride=2)
        decoder.deconv2d(5, 1, stride=2, activation_fn=tf.nn.tanh, s_value=None)
        return decoder.get_output(), mean, stddev

    def _create_network_BREAST(self):
        with tf.variable_scope("model"):
            latent = self._encoder_BREAST(x=self.x)
            x_recon, mean, stddev = self._decoder_BREAST(z=latent)
        with tf.variable_scope("model", reuse=True):
            x_gen, _, _ = self._decoder_BREAST(z=None)
        return x_recon, mean, stddev, x_gen, latent

    def _create_loss_optimizer(self, epsilon=1e-8):
        const = 1/self.flags['batch_size'] * 1/(self.flags['image_dim'] * self.flags['image_dim'])
        recon = const * self.flags['recon'] * tf.reduce_sum(tf.squared_difference(self.x, self.x_recon))
        vae = const * self.flags['vae'] * -0.5 * tf.reduce_sum(1.0 - tf.square(self.mean) - tf.square(self.stddev) + 2.0 * tf.log(self.stddev + epsilon))
        weight = self.flags['weight_decay'] * tf.add_n(tf.get_collection('weight_losses'))
        cost = tf.reduce_sum(vae + recon + weight)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(cost)
        return vae, recon, cost, optimizer

    def print_variable(self, var):
        self.sess.run(tf.initialize_all_variables())
        if var == 'x_recon':
            print_var = tf.Print(self.x_recon, [self.x_recon])
            norm = np.random.normal(size=[self.flags['batch_size'], self.flags['hidden_size']])
            x = np.zeros([self.flags['batch_size'], self.flags['image_dim'], self.flags['image_dim'], 1])
        else:
            print('Print Variable not defined .... printing x_recon')
            return tf.Print(self.x_recon, [self.x_recon])
        return self.sess.run(print_var, feed_dict={self.x: x, self.keep_prob: 0.5, self.epsilon: norm})

    def output_shape(self):
        self.sess.run(tf.initialize_all_variables())
        norm = np.random.normal(size=[self.flags['batch_size'], self.flags['hidden_size']])
        x = np.zeros([self.flags['batch_size'], self.flags['image_dim'], self.flags['image_dim'], 1])
        return self.sess.run(self.x_recon, feed_dict={self.x: x, self.keep_prob: 0.5, self.epsilon: norm})

    def save_x_recon(self, image_generating_fxn):
        labels, x = image_generating_fxn()
        for i in range(len(x)):
            scipy.misc.imsave(self.flags['logging_directory'] + 'x_' + str(i) +'.png', np.squeeze(x[i]))
        norm = np.zeros(shape=[len(x), self.flags['hidden_size']])
        images = self.sess.run(self.x_recon, feed_dict={self.x: x, self.keep_prob: 1.0, self.epsilon: norm})
        for i in range(len(images)):
            scipy.misc.imsave(self.flags['logging_directory'] + 'x_recon' + str(i) + '.png', np.squeeze(images[i]))
        return x

    def save_x_gen(self, image_generating_fxn, num):
        labels, x = image_generating_fxn()
        print(self.flags['logging_directory'])
        for i in range(num):
            scipy.misc.imsave(self.flags['logging_directory'] + 'x_' + str(i) +'.png', np.squeeze(x[i]))
            means, stddevs = self.transform(x[0:num, :, :, :])
            print(stddevs)
            norm = np.random.normal(loc=means)
        images = self.sess.run(self.x_gen, feed_dict={self.x: x[1:num, :, :, :], self.keep_prob: 1.0, self.epsilon: norm})
        for i in range(num):
            scipy.misc.imsave(self.flags['logging_directory'] + 'x_gen' + str(i) + '.png', np.squeeze(images[i]))

    def transform(self, x):
        norm = np.random.normal(size=[x.shape[0], self.flags['hidden_size']])
        return self.sess.run([self.mean, self.epsilon], feed_dict={self.x: x, self.epsilon: norm, self.keep_prob: 1.0})

    def restore(self):
        self.saver.restore(self.sess, self.flags['restore_directory'] + self.flags['restore_file'])
        print_log("Model restored from %s" % self.flags['restore_file'])

    def train(self, image_dict, model):

        setup_metrics(self.flags, self.flags['lr_iters'])
        writer = tf.train.SummaryWriter(self.flags['logging_directory'], self.sess.graph)
        if self.flags['restore'] is True:
            self.restore()
        else:
            self.sess.run(tf.initialize_all_variables())
            print_log("Model training from scratch.")

        global_step = 1
        data = BreastData(self.flags, image_dict)
        for i in range(len(self.flags['lr_iters'])):
            lr = self.flags['lr_iters'][i][0]
            iters = self.flags['lr_iters'][i][1]
            print_log('Learning Rate: %d' % lr)
            print_log('Iterations: %d' % iters)
            step = 1
            while step < iters:

                print('Batch number: %d' % step)
                labels_x, batch_x = data.generate_training_batch(global_step)
                norm = np.random.standard_normal([self.flags['batch_size'], self.flags['hidden_size']])

                if step % self.flags['display_step'] != 0:
                    summary, _ = self.sess.run([self.merged, self.optimizer],
                                               feed_dict={self.x: batch_x, self.keep_prob: 1.0, self.epsilon: norm,
                                                          self.lr: lr * self.flags['lr_decay']})
                else:
                    summary, loss, x_recon, latent, _ = self.sess.run([self.merged, self.cost, self.x_recon, self.latent, self.optimizer], feed_dict={self.x: batch_x, self.keep_prob: 0.9, self.epsilon: norm, self.lr: lr})
                    for j in range(1):
                        scipy.misc.imsave(self.flags['logging_directory'] + 'x_' + str(step) + '.png', np.squeeze(batch_x[j]))
                        scipy.misc.imsave(self.flags['logging_directory'] + 'x_recon_' + str(step) + '.png', np.squeeze(x_recon[j]))
                    record_metrics(loss=loss, acc=None, batch_y=None, step=step, split=None, flags=self.flags)
                    print_log("Max of x: %f" % batch_x[1].max())
                    print_log("Min of x: %f" % batch_x[1].min())
                    print_log("Mean of x: %f" % batch_x[1].mean())
                    print_log("Max of x_recon: %f" % x_recon[1].max())
                    print_log("Min of x_recon: %f" % x_recon[1].min())
                    print_log("Mean of x_recon: %f" % x_recon[1].mean())

                writer.add_summary(summary=summary, global_step=global_step)
                step += 1
                global_step += 1

            print("Optimization Finished!")
            checkpoint_name = self.flags['logging_directory'] + 'Model' + str(model) + 'epoch_%d' % i + '.ckpt'
            save_path = self.saver.save(self.sess, checkpoint_name)
            print("Model saved in file: %s" % save_path)

    # def test(self, test_data, test_labels):


def main():
    o = np.random.randint(1, 1000, 1)
    flags['seed'] = 324  # o[0]
    # a = np.random.uniform(-5.5, -3.5, 1)
    # lr = 0.0001 #np.power(10, a[0])
    # flags['lr_iters'] = [(lr, 10000)]
    run_num = sys.argv[1]
    image_dict = pickle.load(open(flags['aux_directory'] + 'preprocessed_image_dict.pickle', 'rb'))
    model_vae = ConvVae(flags, model=run_num)
    # model.save_x(bgf)
    # x_recon = model_vae.output_shape()
    # print(x_recon.shape)
    print_log("Seed: %d" % flags['seed'])
    print_log("Vae Weights: %f" % flags['vae'])
    print_log("Recon Weight: %d" % flags['recon'])
    model_vae.train(image_dict, model=1)
    # model_vae.restore()
    # model_vae.save_x_gen(bgf, 15)

if __name__ == "__main__":
    main()