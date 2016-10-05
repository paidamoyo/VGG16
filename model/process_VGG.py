#!/usr/bin/env python

import pickle
import numpy as np
import tensorflow as tf

from functions_data import split_data, one_tiled_image, get_image_data, reconstruct
from functions_tf import compute_VGG


# Global Dictionary of Flags
flags = {
    'save_directory': '../../../../Data/Processed/SAGE/',
    'aux_directory': '../aux/',
    'code_directory': '../',
    'save_pickled_dictionary': True,
    'save_pickled_images': True,
}


params = {
    'batch_size': 12*12,
}

def check_str(obj):
    if isinstance(obj, str):
        return obj
    if isinstance(obj, float):
        return str(int(obj))
    else:
        return str(obj)


image_dict = pickle.load(open(flags['aux_directory'] + 'vgg_image_dict.pickle', 'rb'))
dict_train, dict_test, index_train, index_test = split_data(image_dict)

# tf Graph input
x = tf.placeholder(tf.float32, [params['batch_size'], 272, 128, 3])
y = tf.placeholder(tf.int64, shape=(1,))

logits = compute_VGG(x=x, flags=flags)

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    # Construct model
    for i in range(2):
        for b in range(len(dict_train[i])):
            batch_x, batch_y = one_tiled_image(flags['save_directory'], dict_test, pos_neg=i, batch_ind=b)
            volume = sess.run(logits, feed_dict={x: batch_x, y: batch_y})
            image = reconstruct(volume)
            if flags['save_pickled_images'] is True:  # save image array as .pickle file in appropriate directory
                save_path = flags['save_directory'] + '/' + check_str(dict_train[i][b]) + '_' + check_str(dict_train[i][b]) + '.pickle'
                with open(save_path, "wb") as f:
                    pickle.dump(image, f, protocol=2)
        for b in range(len(dict_test[i])):
            batch_x, batch_y = one_tiled_image(flags['save_directory'], dict_test, pos_neg=i, batch_ind=b)
            volume = sess.run(logits, feed_dict={x: batch_x, y: batch_y})
            image = reconstruct(volume)
            if flags['save_pickled_images'] is True:  # save image array as .pickle file in appropriate directory
                save_path = flags['save_directory'] + '/' + check_str(dict_test[i][b]) + '_' + check_str(dict_test[i][b]) + '.pickle'
                with open(save_path, "wb") as f:
                    pickle.dump(image, f, protocol=2)

if flags['save_pickled_dictionary'] is True:
    save_path = '../aux/vgg_train_dict.pickle'
    with open(save_path, "wb") as f:
        pickle.dump(dict_train, f, protocol=2)
    save_path = '../aux/vgg_test_dict.pickle'
    with open(save_path, "wb") as f:
        pickle.dump(dict_test, f, protocol=2)
