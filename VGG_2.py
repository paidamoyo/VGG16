#!/usr/bin/env python

import pickle
import tensorflow as tf

from functions.data import one_tiled_image, reconstruct, save_image
from functions.tf import model_VGG16
from functions.aux import check_directories


# Global Dictionary of Flags
flags = {
    'data_directory': '../../../Data/',  # in relationship to the code_directory
    'aux_directory': 'aux/',
    'weights_directory': 'weights/',
    'datasets': ['SAGE', 'INbreast'],
    'previous_processed_directory': '1_Cropped/',
    'processed_directory': '2_VGG/',
    'save_pickled_dictionary': True,
    'save_pickled_images': True,
}


check_directories(flags)
previous = str.split(flags['previous_processed_directory'],'/')[0]
image_dict = pickle.load(open(flags['aux_directory'] + previous + '_image_dict.pickle', 'rb'))

# tf Graph input
x = tf.placeholder(tf.float32, [12*12, 272, 128, 3])
y = tf.placeholder(tf.int64, shape=(1,))

logits = model_VGG16(x=x, flags=flags)
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    counter = 0
    for d in image_dict:
        batch_x, batch_y = one_tiled_image(flags, image_dict, d)
        volume = sess.run(logits, feed_dict={x: batch_x, y: batch_y})
        image = reconstruct(volume)
        save_image(image)
        counter += 1
        print("Processed Image %d" % counter + ' of %d total images' % len(image_dict))

if flags['save_pickled_dictionary'] is True:
    current = str.split(flags['processed_directory'], '/')[0]
    save_path = flags['aux_directory'] + current + '_image_dict.pickle'
    with open(save_path, "wb") as f:
        pickle.dump(image_dict, f, protocol=2)
