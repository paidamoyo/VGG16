#!/usr/bin/env python

import numpy as np
import tensorflow as tf
import pickle
import sklearn.metrics
import sys
import logging

from functions.data import split_data, generate_minibatch_dict, generate_one_test_index, generate_lr, make_directory
from models.cnn_fc import CnnFc


# Global Dictionary of Flags
flags = {
    'data_directory': '../../../Data/',  # in relationship to the code_directory
    'aux_directory': 'aux/',
    'model_directory': 'cnn_fc/',
    'previous_processed_directory': '2_VGG/',
    'datasets': ['SAGE', 'INbreast'],
    'restore': False
}


params = {
    'batch_size': 12,  # must be divisible by 2
    'display_step': 10,
    'training_iters': 500
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
    lr_num = int(sys.argv[1])
    params['lr'] = generate_lr(lr_num) # ranges from 1 - 4
    batch = int(sys.argv[2])
    params['batch_size'] = batch # includes 12, 24
    seed = int(sys.argv[3])  # ranges from 3- 7


    folder = 'lr_%d' % lr_num + '_batch_%d' % batch + '/'
    flags['restore_directory'] = flags['aux_directory'] + flags['model_directory']
    flags['logging_directory'] = flags['restore_directory'] + folder
    make_directory(flags['logging_directory'])
    logging.basicConfig(filename=flags['logging_directory'] + str(seed) + '.log', level=logging.INFO)
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
        writer = tf.train.SummaryWriter(flags['logging_directory'], sess.graph)
        if flags['restore'] is True:
            saver.restore(sess, flags['restore_directory'] + 'starting_point.ckpt')
            print("Model restored.")
            logging.info("Model restored from starting_point.ckpt")
        else:
            sess.run(init)
            print("Mode training from scratch.")
            logging.info("Model training from scratch.")
        while step < params['training_iters']:

            if step % 3:
                split = [0.75, 0.25]
            else:
                split = [0, 1]

            print('Begin batch number: %d' % step, ", split:", split)
            batch_x, batch_y = generate_minibatch_dict(flags, dict_train, params['batch_size'], split)
            print(params['batch_size'])
            print(len(batch_y))
            print(len(batch_x))
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
                logging.info("Fraction of Positive Predictions: %d / %d" %
                      (np.count_nonzero(np.argmax(acc, 1)), params['batch_size']))

            step += 1
        print("Optimization Finished!")
        checkpoint_name = flags['logging_directory'] + str(seed) + '.ckpt'
        save_path = saver.save(sess, checkpoint_name)
        print("Model saved in file: %s" % save_path)

        print("Scoring %d total images " % len(index_test) + "in test dataset.")
        preds = list()
        trues = list()
        for inds in index_test:
            X_test, y_test = generate_one_test_index(flags, inds, image_dict)
            acc = sess.run(train_prediction, feed_dict={x: X_test, y: y_test})
            trues.extend(y_test)
            preds.extend(acc)
        preds = np.array(preds)
        trues = np.array(trues)
        print("For Test Data ... Error: %.1f%%" % error_rate(preds, trues) + ", AUC= %.3f" % auc_roc(preds, trues))
        logging.info("For Test Data ... Error: %.1f%%" % error_rate(preds, trues) + ", AUC= %.3f" % auc_roc(preds, trues))

if __name__ == "__main__":
    main()
