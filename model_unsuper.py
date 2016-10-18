#!/usr/bin/env python

import tensorflow as tf
import sys
import scipy.misc
import matplotlib.pyplot as plt

import numpy as np
from functions.record import record_metrics, print_log, setup_metrics
from functions.data import generate_lr
from models.conv_vae import ConvVae
from data.clutterMNIST import load_data, generate_cluttered_MNIST


# Global Dictionary of Flags
flags = {
    'data_directory': '../../../Data/',  # in relationship to the code_directory
    'aux_directory': 'aux/',
    'model_directory': 'conv_vae/',
    'datasets': ['MNIST'],
    'restore': True,
    'restore_file': 'starting_point.ckpt'
}


params = {
    'image_dim': 512,
    'batch_size': 64,  # must be divisible by 8
    'hidden_size': 10,
    'display_step': 5,
    'training_iters': 500
}


def main():
    seed = 14
    batch = int(sys.argv[1])
    params['batch_size'] = batch
    params['lr'] = generate_lr(int(sys.argv[2]))
    lr_str = str(params['lr'])

    folder = str(batch) + '/'
    aux_filenames = 'lr_' + lr_str + '_batch_%d' % params['batch_size']
    logging = setup_metrics(flags, aux_filenames, folder)

    # Import Data
    train_set, valid_set, test_set = load_data(flags['data_directory'] + flags['datasets'][0] + '/mnist.pkl.gz')

    # tf Graph input
    x = tf.placeholder(tf.float32, [None, params['image_dim'], params['image_dim'], 1], name='x')  # input patches
    keep_prob = tf.placeholder(tf.float32, name='dropout')
    epsilon = tf.placeholder(tf.float32, [None, params['hidden_size']], name='epsilon')

    # Construct model and initialize
    model = ConvVae(params, seed)
    generated_img, cost, gen = model.run(x=x, keep_prob=keep_prob, epsilon=epsilon)
    optimizer = tf.train.AdamOptimizer(learning_rate=params['lr']).minimize(cost)
    tf.scalar_summary("cost", cost)
    merged = tf.merge_all_summaries()
    init = tf.initialize_all_variables()
    saver = tf.train.Saver()

    # Launch the graph
    with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
        sess.run(init)
        step = 1
        writer = tf.train.SummaryWriter(flags['logging_directory'], sess.graph)
        if flags['restore'] is True:
            saver.restore(sess, flags['restore_directory'] + flags['restore_file'])
            print_log("Model restored from %s" % flags['restore_file'], logging)
        else:
            sess.run(init)
            print_log("Mode training from scratch.", logging)

        norm = np.random.normal(size=[10, params['hidden_size']])
        image = sess.run([gen], feed_dict={epsilon: norm})
        for i in range(10):
            plt.imshow(np.ndarray.squeeze(image[i]), cmap='gray')
            plt.savefig(flags['logging_directory'] + 'image' + str(i))
        '''
        while step < params['training_iters']:

            print('Begin batch number: %d' % step)
            batch_x = generate_cluttered_MNIST(dims=[params['image_dim'], params['image_dim']],
                                               nImages=params['batch_size'], clutter=0.1, numbers=[8],
                                               prob=0.1, train_set=train_set)
            norm = np.random.standard_normal([params['batch_size'], params['hidden_size']])
            summary, _ = sess.run([merged, optimizer], feed_dict={x: batch_x, keep_prob: 0.5, epsilon: norm})
            writer.add_summary(summary=summary, global_step=step)

            if step % params['display_step'] == 0:
                loss = sess.run([cost], feed_dict={x: batch_x, keep_prob: 0.5, epsilon: norm})
                record_metrics(loss=loss, acc=None, batch_y=None, logging=logging, step=step, split=None)
            step += 1

        print("Optimization Finished!")
        checkpoint_name = flags['logging_directory'] + aux_filenames + '.ckpt'
        save_path = saver.save(sess, checkpoint_name)
        print("Model saved in file: %s" % save_path)
        '''

if __name__ == "__main__":
    main()
