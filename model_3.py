#!/usr/bin/env python

import numpy as np
import tensorflow as tf
import pickle
import sklearn.metrics
import sys
import logging

from functions.data import split_data, generate_minibatch_dict, organize_test_index, generate_split, generate_lr
from models.cnn_fc import CnnFc



# Global Dictionary of Flags
flags = {
    'data_directory': '../../../Data/',  # in relationship to the code_directory
    'aux_directory': 'aux/',
    'model_directory': 'cnn_fc',
    'previous_processed_directory': '2_VGG/',
    'datasets': ['SAGE', 'INbreast'],
}


params = {
    'lr': 0.00001,
    'training_iters': 11,
    'batch_size': 8,  # must be divisible by 2
    'display_step': 5
}


def error_rate(predictions, labels):
    """Return the error rate based on dense predictions and sparse labels."""
    return 100.0 - (
        100.0 *
        np.sum(np.argmax(predictions, 1) == labels) /
        predictions.shape[0])


def auc_roc(predictions, labels):
    try:
        return sklearn.metrics.roc_auc_score(np.array(labels), np.argmax(predictions, 1))
    except ValueError:  # if all predicted labels are the same
        print('All predicted labels are the same')
        return -1

def main():
    params['lr'] = generate_lr(sys.argv[1]) # ranges from 1 - 3
    split_num = sys.argv[2]  # ranges from 1 - 7
    seed = sys.argv[3]  # ranges from 1 - 10

    logging_file = 'split_%d' % split_num + '_lr_%d' % sys.argv[1] + '_seed_%d' % seed + '.log'
    logging.basicConfig(filename=logging_file, level=logging.INFO)
    image_dict = pickle.load(open(flags['aux_directory'] + 'preprocessed_image_dict.pickle', 'rb'))
    dict_train, dict_test, index_train, index_test = split_data(image_dict, seed=seed)

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
        writer = tf.train.SummaryWriter(flags['aux_directory'] + flags['model_directory'], sess.graph)
        while step < params['training_iters']:

            split = generate_split(split_num, step)
            print('Begin batch number: %d' % step)
            batch_x, batch_y = generate_minibatch_dict(flags, dict_train, params['batch_size'], split)
            summary, _ = sess.run([merged, optimizer], feed_dict={x: batch_x, y: batch_y})
            writer.add_summary(summary=summary, global_step=step)

            if step % params['display_step'] == 0:
                loss, acc, login = sess.run([cost, train_prediction, logits], feed_dict={x: batch_x,
                                                                          y: batch_y})
                print("Batch Number " + str(step) + ", Image Loss= " +
                      "{:.6f}".format(loss) + ", Error: %.1f%%" % error_rate(acc, batch_y) +
                      ", AUC= %.3f" % auc_roc(acc, batch_y))
                logging.info("Batch Number " + str(step) + ", Image Loss= " +
                      "{:.6f}".format(loss) + ", Error: %.1f%%" % error_rate(acc, batch_y) +
                      ", AUC= %.3f" % auc_roc(acc, batch_y))
                print("Predicted Labels: ", np.argmax(acc, 1).tolist())
                print("True Labels: ", batch_y)
                print("Training Split: ", split)
                print("Fraction of Positive Predictions: %d / %d" %
                      (np.count_nonzero(np.argmax(acc, 1)), params['batch_size']))
            step += 1
        print("Optimization Finished!")
        checkpoint_name = 'split_%d' % split_num + '_lr_%d' % sys.argv[1] + '_seed_%d' % seed + '.ckpt'
        save_path = saver.save(sess, flags['aux_directory'] + flags['model_directory'] + checkpoint_name)
        print("Model saved in file: %s" % save_path)

        print("Scoring %d total images " % len(index_test) + "in test dataset.")
        X_test, y_test = organize_test_index(flags, index_test, image_dict)
        acc = sess.run(train_prediction, feed_dict={x: X_test, y: y_test})
        print("For Test Data ... Error: %.1f%%" % error_rate(acc, y_test) + ", AUC= %.3f" % auc_roc(acc, y_test))
        logging.info("For Test Data ... Error: %.1f%%" % error_rate(acc, y_test) + ", AUC= %.3f" % auc_roc(acc, y_test))

if __name__ == "__main__":
    main()
