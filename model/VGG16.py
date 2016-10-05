#!/usr/bin/env python

import pickle as cp
import numpy as np
import tensorflow as tf

from functions_data import split_data, generate_minibatch_dict_small, get_image_data
from functions_tf import build_model


# Global Dictionary of Flags
flags = {
    'save_directory': '../../../../Data/Processed/SAGE/',
    'aux_directory': '../aux/',
    'code_directory': '../',
}


params = {
    'pyramid_pool_layers': [1, 2],
    'lr': 0.001,
    'training_iters': 4000,
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
    image_dict = cp.load(open(flags['aux_directory'] + 'vgg_image_dict.pickle', 'rb'))
    dict_test, dict_train, index_test, index_train = split_data(image_dict)

    # tf Graph input
    x = tf.placeholder(tf.float32, [params['batch_size'], 272, 128, 3])
    y = tf.placeholder(tf.int64, shape=(1,))
    keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

    # Construct model
    logits = build_model(x=x, flags=flags, params=params)

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
        pos_neg = 1
        while step < params['training_iters']:
            if step > 1000:
                if pos_neg == 1:
                    pos_neg = 0
                else:
                    pos_neg = 1
            batch_x, batch_y = generate_minibatch_dict_small(flags['save_directory'], dict_train, pos_neg)

            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: params['dropout']})
            loss, acc = sess.run([cost, train_prediction], feed_dict={x: batch_x,
                                                              y: batch_y,
                                                              keep_prob: 1.})
            print("Image Number " + str(step) + ", Image Loss= " + \
                  "{:.6f}".format(loss) + 'Error: %.1f%%' % error_rate(acc, batch_y) + \
                    ", Label= " + 'Error: %d' % batch_y[0])

            if step % params['display_step'] == 0:
                # Calculate batch loss and accuracy
                save_path = saver.save(sess, '../aux/model.ckpt')
                print("Model saved in file: %s" % save_path)
            step += 1
        print("Optimization Finished!")

        X_test, y_test = get_image_data(index_train, flags['save_directory'], image_dict)
        print("Testing Accuracy:",
              sess.run(accuracy, feed_dict={x: X_test,
                                            y: y_test,
                                            keep_prob: 1.}))


if __name__ == "__main__":
    main()
