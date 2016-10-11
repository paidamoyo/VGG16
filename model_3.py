#!/usr/bin/env python

import numpy as np
import tensorflow as tf
import pickle
import sklearn.metrics

from functions.data import split_data, generate_minibatch_dict, generate_minibatch_index
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
    'training_iters': 100,
    'batch_size': 16,  # must be divisible by 2
    'display_step': 3
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
    dict_train, dict_test, index_train, index_test = split_data(image_dict, seed=134)

    # tf Graph input
    x = tf.placeholder(tf.float32, [None, 204, 96, 512], name='VGG_output')
    y = tf.placeholder(tf.int64, shape=[None], name='Labels')

    # Construct model
    logits, weights, biases = model_CNN_FC(x=x)
    tf.histogram_summary("weights_conv1", weights['conv1'])
    tf.histogram_summary("weights_fc1", weights['fc1'])
    tf.histogram_summary("biases_conv1", biases['conv1'])
    tf.histogram_summary("biases_fc1", biases['fc1'])

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y))
    train_prediction = tf.round(tf.log(logits))
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
        writer = tf.train.SummaryWriter(flags['aux_directory'] + "summary_logs5", sess.graph)
        while step < params['training_iters']:
            batch_x, batch_y, batch_dataset = generate_minibatch_dict(flags, dict_train, image_dict, params['batch_size'])
            print('Begin batch number: %d' % step)
            summary, _ = sess.run([merged, optimizer], feed_dict={x: batch_x, y: batch_y})
            writer.add_summary(summary=summary, global_step=step)

            if step % params['display_step'] == 0:
                loss, acc, login = sess.run([cost, train_prediction, logits], feed_dict={x: batch_x,
                                                                          y: batch_y})
                print("Batch Number " + str(step) + ", Image Loss= " +
                      "{:.6f}".format(loss) + ", Error: %.1f%%" % error_rate(acc, batch_y) +
                      ", AUC= %d" % auc_roc(acc, batch_y))
                print(login)
                print(batch_dataset)
                save_path = saver.save(sess, flags['aux_directory'] + 'model.ckpt')
                print("Model saved in file: %s" % save_path)
            step += 1
        print("Optimization Finished!")

        labels = {}
        for i in range(2):
            labels[i] = []
            print("Processing %d total images " % len(dict_test[i]) + "for label %d" % i)
            for b in range(len(dict_test[i])):
                X_test, y_test = generate_minibatch_index(flags['save_directory'], dict_test)
                acc = sess.run(train_prediction, feed_dict={x: X_test, y: y_test})
                print("Image %d" % b + " of %d" % len(dict_test[i]) + ", Error: %.1f%%" % error_rate(acc, y_test) + \
                      ", Label= %d" % y_test[0])
                labels[i].append(error_rate(acc, y_test))
        print("True Positive: %f" % np.mean(labels[1]) + ", True Negative: %f" % np.mean(labels[0]))


if __name__ == "__main__":
    main()
