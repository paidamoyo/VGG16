#!/usr/bin/env python

import pickle as cp
import numpy as np
import tensorflow as tf
import pickle

from functions.data import split_data, generate_minibatch
from functions.tf import model_CNN_FC



# Global Dictionary of Flags
flags = {
    'data_directory': '../../../Data/', # in relationship to the code_directory
    'aux_directory': 'aux/',
    'previous_processed_directory': '2_VGG/',
    'datasets': ['SAGE', 'INbreast'],
}


params = {
    'lr': 0.0001,
    'training_iters': 3000,
    'batch_size': 12*12,
    'display_step': 10,
    'dropout': 0.5
}


def error_rate(predictions, labels):
    """Return the error rate based on dense predictions and sparse labels."""
    return 100.0 - (
        100.0 *
        np.sum(np.argmax(predictions, 1) == labels) /
        predictions.shape[0])


def main():
    previous = str.split(flags['previous_processed_directory'], '/')[0]
    image_dict = pickle.load(open(flags['aux_directory'] + previous + '_image_dict.pickle', 'rb'))
    dict_train, dict_test, index_train, index_test = split_data(image_dict, seed=1234)
    batch_x, batch_y = generate_minibatch(flags, dict_train)
    print(batch_x[0].shape)
    exit()

    # tf Graph input
    x = tf.placeholder(tf.float32, [params['batch_size'], 272, 128, 3])
    y = tf.placeholder(tf.int64, shape=(1,))
    keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

    # Construct model
    logits = model_CNN_FC(x=x, flags=flags, params=params)

    # Define loss and optimizer

    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y))
    train_prediction = tf.nn.softmax(logits)
    optimizer = tf.train.AdamOptimizer(learning_rate=params['lr']).minimize(cost)

    # Initializing the variables
    init = tf.initialize_all_variables()
    saver = tf.train.Saver()

    # Launch the graph
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        sess.run(init)
        step = 1
        while step < params['training_iters']:
            if step % 3 == 0:
                pos_neg = 1
            else:
                pos_neg = 0
            batch_x, batch_y = generate_minibatch(flags['save_directory'], dict_train, pos_neg)

            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: params['dropout']})
            loss, acc = sess.run([cost, train_prediction], feed_dict={x: batch_x,
                                                              y: batch_y,
                                                              keep_prob: 1.})
            print("Image Number " + str(step) + ", Image Loss= " + \
                  "{:.6f}".format(loss) + ", Error: %.1f%%" % error_rate(acc, batch_y) + \
                    ", Label= %d" % batch_y[0])

            if step % params['display_step'] == 0:
                save_path = saver.save(sess, '../aux/model.ckpt')
                print("Model saved in file: %s" % save_path)
            step += 1
        print("Optimization Finished!")

        labels = {}
        for i in range(2):
            labels[i] = []
            print("Processing %d total images " % len(dict_test[i]) + "for label %d" % i)
            for b in range(len(dict_test[i])):
                X_test, y_test = generate_minibatch_test(flags['save_directory'], dict_test, pos_neg=i, batch_ind=b)
                acc = sess.run(train_prediction, feed_dict={x: X_test, y: y_test, keep_prob: 1.})
                print("Image %d" % b + " of %d" % len(dict_test[i]) + ", Error: %.1f%%" % error_rate(acc, y_test) + \
                      ", Label= %d" % y_test[0])
                labels[i].append(error_rate(acc, y_test))
        print("True Positive: %f" % np.mean(labels[1]) + ", True Negative: %f" % np.mean(labels[0]))

if __name__ == "__main__":
    main()
