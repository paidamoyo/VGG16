#!/usr/bin/env python

import pickle as cp
import numpy as np
import tensorflow as tf

from functions_data import split_data, generate_minibatch_dict_small, generate_minibatch_test_small, get_image_data
from functions_tf import compute_VGG


# Global Dictionary of Flags
flags = {
    'save_directory': '../../../../Data/Processed/SAGE/',
    'aux_directory': '../aux/',
    'code_directory': '../',
}


params = {
    'batch_size': 12*12,
}


image_dict = cp.load(open(flags['aux_directory'] + 'vgg_image_dict.pickle', 'rb'))
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
            batch_x, batch_y = generate_minibatch_test_small(flags['save_directory'], dict_test, pos_neg=i, batch_ind=b)
            volume = sess.run(logits, feed_dict={x: batch_x, y: batch_y})
            print(type(volume))
            image_list = []
            # reconstruct images
            for j in range(12):
                for i in range(12):
                    if i == 0:
                        image_list[j] = volume[0, :, :, :]
                        continue
                    image_list[j] = np.concatenate((image_list[j], volume[i+j*12,:,:,:]), axis=0)
            for l in range(12):
                if l == 0:
                    image = image_list[l]
                image = np.concatenate((image, image_list[l]), axis=1)
            print(image.shape)
        for b in range(len(dict_test[i])):
            batch_x, batch_y = generate_minibatch_test_small(flags['save_directory'], dict_test, pos_neg=i, batch_ind=b)
            volume = sess.run(logits, feed_dict={x: batch_x, y: batch_y})

