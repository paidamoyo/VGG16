#!/usr/bin/env python

import numpy as np
import tensorflow as tf
import pickle
import sklearn.metrics

from functions.data import split_data, generate_minibatch
from functions.tf import model_CNN_FC



# Global Dictionary of Flags
flags = {
    'data_directory': '../../../Data/',  # in relationship to the code_directory
    'aux_directory': 'aux/',
    'previous_processed_directory': '2_VGG/',
    'datasets': ['SAGE', 'INbreast'],
}


params = {
    'lr': 0.001,
    'training_iters': 1000,
    'batch_size': 32,
    'display_step': 10,
    'dropout': 0.5
}


def error_rate(predictions, labels):
    """Return the error rate based on dense predictions and sparse labels."""
    return 100.0 - (
        100.0 *
        np.sum(np.argmax(predictions, 1) == labels) /
        predictions.shape[0])


def auc_roc(predictions, labels):
    return sklearn.metrics.roc_auc_score(labels, np.argmax(predictions, 1), average='macro', sample_weight=None)


def main():
    image_dict = pickle.load(open(flags['aux_directory'] + 'preprocessed_image_dict.pickle', 'rb'))
    dict_train, dict_test, index_train, index_test = split_data(image_dict, seed=1234)
    batch_x, batch_y = generate_minibatch(flags, dict_train, batch_size=2)
    print('Imported Images have dimension: %s' % str(batch_x[0].shape))

    # tf Graph input
    x = tf.placeholder(tf.float32, [params['batch_size'], 204, 96, 512], name='VGG_output')
    y = tf.placeholder(tf.int64, shape=(32,), name='Labels')
    keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

    # Construct model
    logits, weights, biases = model_CNN_FC(x=x, params=params)
    tf.histogram_summary("weights_conv1", weights['conv1'])
    tf.histogram_summary("weights_conv2", weights['conv2'])
    tf.histogram_summary("weights_fc1", weights['fc1'])
    tf.histogram_summary("biases_conv1", biases['conv1'])
    tf.histogram_summary("biases_conv2", biases['conv2'])
    tf.histogram_summary("biases_fc1", biases['fc1'])

    # Define loss and optimizer

    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y))
    train_prediction = tf.nn.softmax(logits)
    optimizer = tf.train.AdamOptimizer(learning_rate=params['lr']).minimize(cost)
    cost_summ = tf.scalar_summary("cost", cost)

    # Initializing the variables
    merged = tf.merge_all_summaries()
    init = tf.initialize_all_variables()
    saver = tf.train.Saver()

    # Launch the graph
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        sess.run(init)
        step = 1
        writer = tf.train.SummaryWriter(flags['aux_directory'] + "summary_logs", sess.graph_def)
        while step < params['training_iters']:
            batch_x, batch_y = generate_minibatch(flags, dict_train, params['batch_size'])
            print('Begin batch number: %d' % step)
            summary = sess.run([merged, optimizer], feed_dict={x: batch_x, y: batch_y, keep_prob: params['dropout']})
            writer.add_summary(summary=summary)
            if step % params['display_step'] == 0:
                loss, acc = sess.run([cost, train_prediction], feed_dict={x: batch_x,
                                                                          y: batch_y,
                                                                          keep_prob: 1.})
                tf.scalar_summary("Accuracy",)
                print("Batch Number " + str(step) + ", Image Loss= " +
                      "{:.6f}".format(loss) + ", Error: %.1f%%" % error_rate(acc, batch_y) +
                      ", AUC= %d" % auc_roc(acc, batch_y))
                save_path = saver.save(sess, flags['aux_directory'] + 'model.ckpt')
                print("Model saved in file: %s" % save_path)
            step += 1
        print("Optimization Finished!")

        '''
        labels = {}
        for i in range(2):
            labels[i] = []
            print("Processing %d total images " % len(dict_test[i]) + "for label %d" % i)
            for b in range(len(dict_test[i])):
                X_test, y_test = generate_minibatch(flags['save_directory'], dict_test)
                acc = sess.run(train_prediction, feed_dict={x: X_test, y: y_test, keep_prob: 1.})
                print("Image %d" % b + " of %d" % len(dict_test[i]) + ", Error: %.1f%%" % error_rate(acc, y_test) + \
                      ", Label= %d" % y_test[0])
                labels[i].append(error_rate(acc, y_test))
        print("True Positive: %f" % np.mean(labels[1]) + ", True Negative: %f" % np.mean(labels[0]))
        '''

if __name__ == "__main__":
    main()
