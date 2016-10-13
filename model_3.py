#!/usr/bin/env python

import numpy as np
import tensorflow as tf
import pickle
import sys

from functions.data import split_data, generate_minibatch_dict, generate_one_test_index, generate_lr, make_directory, generate_split
from functions.record import record_metrics, print_log, setup_metrics
from models.cnn_fc import CnnFc


# Global Dictionary of Flags
flags = {
    'data_directory': '../../../Data/',  # in relationship to the code_directory
    'aux_directory': 'aux/',
    'model_directory': 'cnn_fc/',
    'previous_processed_directory': '2_VGG/',
    'datasets': {'SAGE'},
    'restore': False,
    'restore_file': 'starting_point.ckpt'
}


params = {
    'batch_size': 16,  # must be divisible by 8
    'display_step': 10,
    'training_iters': 0
}


def main():
    seed = int(sys.argv[1])  # ranges from 3- 7
    lr_num = int(sys.argv[2])
    params['lr'] = generate_lr(lr_num) # ranges from 1 - 4
    batch = int(sys.argv[3])
    params['batch_size'] = batch # includes 12, 24

    folder = str(seed) + '/'
    aux_filenames = 'lr_%d' % lr_num + '_batch_%d' % batch
    logging = setup_metrics(flags, aux_filenames, folder)
    image_dict = pickle.load(open(flags['aux_directory'] + 'preprocessed_image_dict.pickle', 'rb'))
    dict_train, dict_test, index_train, index_test = split_data(flags, image_dict, seed)

    # tf Graph input
    x = tf.placeholder(tf.float32, [None, 204, 96, 512], name='VGG_output')
    y = tf.placeholder(tf.int64, shape=[None], name='Labels')

    # Construct model
    model = CnnFc()
    logits = model.run(x=x)

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y))
    train_prediction = tf.nn.softmax(logits)
    optimizer = tf.train.AdamOptimizer(learning_rate=params['lr']).minimize(cost)
    tf.scalar_summary("cost", cost)

    # Initializing the variables
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

            split = generate_split(step)
            print('Begin batch number: %d' % step, ", split:", split)
            batch_x, batch_y = generate_minibatch_dict(flags, dict_train, params['batch_size'], split)
            summary, _ = sess.run([merged, optimizer], feed_dict={x: batch_x, y: batch_y})
            writer.add_summary(summary=summary, global_step=step)

            if step % params['display_step'] == 0:
                loss, acc, _ = sess.run([cost, train_prediction], feed_dict={x: batch_x, y: batch_y})
                record_metrics(loss, acc, batch_y, logging, step, split, params)
            step += 1

        print("Optimization Finished!")
        checkpoint_name = flags['logging_directory'] + aux_filenames + '.ckpt'
        save_path = saver.save(sess, checkpoint_name)
        print("Model saved in file: %s" % save_path)

        print_log("Begin scoring the test images.", logging)
        preds = list()
        trues = list()
        for inds in index_test:
            test_x, test_y = generate_one_test_index(flags, inds, image_dict)
            acc = sess.run(train_prediction, feed_dict={x: test_x, y: test_y})
            trues.extend(test_x)
            preds.extend(acc)
        preds = np.array(preds)
        trues = np.array(trues)
        print_log("Scored a total of %d images " % len(preds) + "in test dataset.", logging)
        record_metrics(loss=None, acc=preds, batch_y=trues, logging=logging, step=None, split=None, params=params)

if __name__ == "__main__":
    main()
