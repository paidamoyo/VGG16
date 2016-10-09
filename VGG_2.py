#!/usr/bin/env python

import pickle
import numpy as np
import tensorflow as tf

from functions.data import split_data, one_tiled_image, reconstruct
from functions.tf import compute_VGG
from functions.aux import check_directories, check_str


# Global Dictionary of Flags
flags = {
    'data_directory': '../../../../Data/',  # in relationship to the code_directory
    'aux_directory': '../aux/',
    'weights_directory': '../weights/',
    'datasets': ['SAGE/'],
    'previous_processed_directory': '1_Cropped/',
    'processed_directory': '2_VGG/',
    'save_pickled_dictionary': True,
    'save_pickled_images': True,
}


params = {
    'batch_size': 12*12,
}


check_directories(flags)
image_dict = pickle.load(open(flags['aux_directory'] + 'vgg_image_dict.pickle', 'rb'))

# tf Graph input
x = tf.placeholder(tf.float32, [params['batch_size'], 272, 128, 3])
y = tf.placeholder(tf.int64, shape=(1,))

logits = compute_VGG(x=x, flags=flags)

# Initializing the variables
init = tf.initialize_all_variables()
image_dump_path = flags['data_directory'] + 'SAGE/Preprocessed/' + flags['processed_directory']

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    # Construct model
    counter = 0
    for d in image_dict:
        batch_x, batch_y = one_tiled_image(flags, image_dict, d)
        volume = sess.run(logits, feed_dict={x: batch_x, y: batch_y})
        image = reconstruct(volume)
        if flags['save_pickled_images'] is True:  # save image array as .pickle file in appropriate directory
            save_path = image_dump_path + check_str(d[1]) + '_' + check_str(d[2]) + '_vgg.pickle'
            with open(save_path, "wb") as f:
                pickle.dump(image, f, protocol=2)
        counter += 1
        print("Processed Image %d" % counter + ' of %d total images' % len(image_dict))


if flags['save_pickled_dictionary'] is True:
    save_path = flags['aux_directory'] + 'vgg_train_dict.pickle'
    with open(save_path, "wb") as f:
        pickle.dump(image_dict, f, protocol=2)
