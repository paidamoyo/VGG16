#!/usr/bin/env python

import numpy as np
import tensorflow as tf
import pickle
import sklearn.metrics

from functions.data import split_data, generate_minibatch_dict, organize_test_index
from models.max_area import MaxArea



# Global Dictionary of Flags
flags = {
    'data_directory': '../../../Data/',  # in relationship to the code_directory
    'aux_directory': 'aux/',
    'model_directory': 'max_area/',
    'previous_processed_directory': '2_VGG/',
    'datasets': ['SAGE', 'INbreast'],
}


params = {
    'lr': 0.0001,
    'training_iters': 10,
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
        return sklearn.metrics.roc_auc_score(labels, np.argmax(predictions, 1).tolist())
    except ValueError:  # if all predicted labels are the same
        return 0


def main():
    image_dict = pickle.load(open(flags['aux_directory'] + 'preprocessed_image_dict.pickle', 'rb'))
    dict_train, dict_test, index_train, index_test = split_data(image_dict, seed=134)

    # tf Graph input
    x = tf.placeholder(tf.float32, [None, 204, 96, 512], name='VGG_output')
    y = tf.placeholder(tf.int64, shape=[None], name='Labels')

    # Construct model
    model = MaxArea()
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
        bool = True
        split = [0, 1]
        writer = tf.train.SummaryWriter(flags['aux_directory'] + flags['model_directory'], sess.graph)
        while step < params['training_iters']:
            batch_x, batch_y = generate_minibatch_dict(flags, dict_train, params['batch_size'], split)
            print('Begin batch number: %d' % step)
            summary, _ = sess.run([merged, optimizer], feed_dict={x: batch_x, y: batch_y})
            writer.add_summary(summary=summary, global_step=step)

            if step % params['display_step'] == 0:
                loss, acc, login = sess.run([cost, train_prediction, logits], feed_dict={x: batch_x,
                                                                          y: batch_y})
                print("Batch Number " + str(step) + ", Image Loss= " +
                      "{:.6f}".format(loss) + ", Error: %.1f%%" % error_rate(acc, batch_y) +
                      ", AUC= %d" % auc_roc(acc, batch_y))
                # print("Training split is %f negatives and %d positive" % (int(split[0] * 100), int(split[1]*100)))
                print(type(np.argmax(acc, 1)))
                print(split)
                print("Number of Positive Predictions: %d" % np.count_nonzero(np.argmax(acc, 1)))
                save_path = saver.save(sess, flags['aux_directory'] + flags['model_directory'] +'model.ckpt')
                print("Model saved in file: %s" % save_path)
                if bool is True:
                    split = [0.25, 0.75]
                    bool = False
                else:
                    split = [0, 1]
                    bool = True
            step += 1
        print("Optimization Finished!")

        print("Scoring %d total images " % len(index_test) + "in test dataset.")
        X_test, y_test = organize_test_index(flags, index_test, image_dict)
        acc = sess.run(train_prediction, feed_dict={x: X_test, y: y_test})
        print("For Test Data ... Error: %.1f%%" % error_rate(acc, y_test) + ", AUC= %d" % auc_roc(acc, y_test))

if __name__ == "__main__":
    main()
