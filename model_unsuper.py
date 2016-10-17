#!/usr/bin/env python

import tensorflow as tf

from functions.record import record_metrics, print_log, setup_metrics
from models.conv_vae import ConvVae
from data.clutterMNIST import load_data, generate_cluttered_MNIST


# Global Dictionary of Flags
flags = {
    'data_directory': '../../../Data/',  # in relationship to the code_directory
    'aux_directory': 'aux/',
    'model_directory': 'conv_vae/',
    'datasets': ['MNIST'],
    'restore': False,
    'restore_file': 'starting_point.ckpt'
}


params = {
    'batch_size': 32,  # must be divisible by 8
    'hidden_size': 20,
    'display_step': 2,
    'training_iters': 50
}


def main():
    seed = 3
    params['lr'] = 0.001
    lr_str = '1e-3'

    folder = str(seed) + '/'
    aux_filenames = 'lr_' + lr_str + '_batch_%d' % params['batch_size']
    logging = setup_metrics(flags, aux_filenames, folder)

    # Import Data
    train_set, valid_set, test_set =  load_data(flags['data_directory'] + flags['datasets'][0] + '/mnist.pkl.gz')

    # tf Graph input
    x = tf.placeholder(tf.int32, [None, 256, 256, 1], name='256x256_input')  # input patches

    # Construct model and initialize
    model = ConvVae(params)
    generated_img, cost = model.run(x=x)
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
        while step < params['training_iters']:

            print('Begin batch number: %d' % step)
            batch_x = generate_cluttered_MNIST([256, 256], params['batch_size'], 0.1, 8, 0.1, train_set)
            summary, _ = sess.run([merged, optimizer], feed_dict={x: batch_x})
            writer.add_summary(summary=summary, global_step=step)

            if step % params['display_step'] == 0:
                loss = sess.run([cost], feed_dict={x: batch_x})
                record_metrics(loss=loss, acc=None, batch_y=None, logging=logging, step=step, split=None)
            step += 1

        print("Optimization Finished!")
        checkpoint_name = flags['logging_directory'] + aux_filenames + '.ckpt'
        save_path = saver.save(sess, checkpoint_name)
        print("Model saved in file: %s" % save_path)

if __name__ == "__main__":
    main()
