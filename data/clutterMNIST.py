# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 10:54:22 2016

@author: Kevin Liang

Creates much larger MNIST images

Embeds MNIST handwritten in a much larger image. Option to add clutter, to an
adjustable degree. Clutter generated from random pieces of MNIST digits
"""

import gzip
import numpy as np
import os
import pickle


def load_data_cluttered_MNIST(dataset):
    ''' Loads the dataset
    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''
    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        from six.moves import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print('Downloading data from %s' % origin)
        urllib.request.urlretrieve(origin, dataset)

    print('... loading data')

    # Load the dataset
    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)

    print('Successfully loaded MNIST data')
    return [train_set, valid_set, test_set]


def generate_cluttered_MNIST(dims, nImages, numbers, prob, clutter, train_set):
    ''' Generates cluttered MNIST images

    Args:
        dims: dimensions of output cluttered MNIST images
        nImages: number of images to output
        numbers: array of MNIST digits (28x28) to insert into each image
        prob: probability of generating a number from numbers
        clutter: degree of clutter in image. Clutter generated by inserting 8x8
                    patches of other digits into image. Clutter ratio determined
                    by proportion of 8x8 patches to have MNIST piece
        train_set: MNIST training set tuple (data,labels)

    Return:
        images: 3D stack of cluttered MNIST images (dim[0],dim[1],nImages)
    '''
    images = np.zeros((nImages, dims[0], dims[1]))
    labels = np.ones((nImages))

    # Calculate number of patches of clutter to add in
    clutterPatches = int(clutter * dims[0] * dims[1] / (8 * 8))

    for k in range(nImages):
        # Add in MNIST digits
        for i in numbers:
            if prob < np.random.uniform(low=0, high=1):
                continue
            else:
                # Randomly select MNIST data of the correct digit
                index = np.where(train_set[1] == i)[0]
                digit = train_set[0][np.random.choice(index), :]
                digit = np.reshape(digit, (28, 28))

                if dims[0] != 28 or dims[1] != 28:
                    # Randomly choose location
                    x = np.random.randint(low=0, high=dims[0] - 28)
                    y = np.random.randint(low=0, high=dims[1] - 28)

                    # Insert digit
                    images[k, x:x + 28, y:y + 28] += digit
                else:
                    images[k, :, :] = digit
                labels[k] = -1

        # Add in clutter
        if clutterPatches != 0:
            for j in range(clutterPatches):
                # Randomly select MNIST digit
                index = np.random.choice(len(train_set[1]))
                digit = np.reshape(train_set[0][index, :], (28, 28))

                # Randomly select patch of selected digit
                px = np.random.randint(low=0, high=28 - 8)
                py = np.random.randint(low=0, high=28 - 8)

                # Randomly choose location to insert clutter
                x = np.random.randint(low=0, high=dims[0] - 8)
                y = np.random.randint(low=0, high=dims[1] - 8)

                # Insert digit fragment
                images[k, x:x + 8, y:y + 8] += digit[px:px + 8, py:py + 8]
                print(images[k, :, :].max())
                #for x1 in range(x, x + 8):
                #    for y1 in range(y, y+8):
                #        if images[k, x1, y1] > 255:
                #            images[k, x1, y1] = 255

        # Renormalize image
        save = images[k, :, :] - images[k, :, :].mean()
        images[k, :, :] = save / save.max()

    return labels, np.expand_dims(images, 3)


if __name__ == '__main__':
    train_set, valid_set, test_set = load_data('mnist.pkl.gz')
    images = generate_cluttered_MNIST([200, 200], 10, [5, 8, 2], 0.1, train_set)