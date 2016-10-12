#!/usr/bin/env python

import numpy as np
import math
import pandas as pd
import pickle
import scipy.misc
import os
from random import shuffle

from functions.aux import check_str, make_directory


def split_data(image_dict, seed):
    np.random.seed(seed=seed)
    index_test = list()
    index_train = list()
    dict_test = dict()
    dict_train = dict()
    dict_image = pd.DataFrame(image_dict)
    labels = dict_image.iloc[4]
    for i in ['0', '1']:
        patients = labels[labels == i].index.values
        partition = int(math.floor(len(patients) * .85))  # 70% of data goes to training
        indexes = np.random.choice(range(len(patients)), size=len(patients))
        dict_test[int(i)] = list(patients[indexes[partition:]])
        dict_train[int(i)] = list(patients[indexes[:partition]])
        index_test = index_test + list(patients[indexes[partition:]])
        index_train = index_train + list(patients[indexes[:partition]])
        print("Training split has %d mammograms" % len(dict_train[int(i)]) + " with label %d" % int(i))
        print("Testing split has %d mammograms" % len(dict_test[int(i)]) + " with label %d" % int(i))
    return dict_train, dict_test, index_train, index_test


def generate_minibatch_dict(flags, dict_name, batch_size, split):
    unshuffled_batch = []
    for i in range(2):
        bsize = int(split[i] * batch_size)
        if bsize == 0:
            continue
        batch_ind = np.random.randint(low=0, high=len(dict_name[i]), size=bsize).tolist()
        for b in batch_ind:
            inds = dict_name[i][b]
            data_directory = flags['data_directory'] + inds[0] + '/Preprocessed/' + flags['previous_processed_directory']
            image_path = data_directory + check_str(inds[1]) + '_' + check_str(inds[2]) + '.pickle'
            with open(image_path, 'rb') as basefile:
                map_stack = pickle.load(basefile)
                unshuffled_batch.append((map_stack, i))
    shuffle(unshuffled_batch)
    batch_data = [map_stack for (map_stack, i) in unshuffled_batch]
    batch_labels = [i for (map_stack, i) in unshuffled_batch]
    return batch_data, batch_labels


def generate_one_test_index(flags, inds, image_dict):
    unshuffled_batch = []
    i = image_dict[inds][4]
    data_directory = flags['data_directory'] + inds[0] + '/Preprocessed/' + flags['previous_processed_directory']
    image_path = data_directory + check_str(inds[1]) + '_' + check_str(inds[2]) + '.pickle'
    with open(image_path, 'rb') as basefile:
        map_stack = pickle.load(basefile)
        unshuffled_batch.append((map_stack, i))
    shuffle(unshuffled_batch)
    batch_data = [map_stack for (map_stack, i) in unshuffled_batch]
    batch_labels = [int(i) for (map_stack, i) in unshuffled_batch]
    return batch_data, batch_labels


def one_tiled_image(flags, image_dict, inds, tile):
    batch_data = []
    batch_labels = [image_dict[inds][4]]
    path = flags['data_directory'] + inds[0] + '/Preprocessed/' + flags['previous_processed_directory']
    image_path = path + check_str(inds[1]) + '_' + check_str(inds[2]) + '.pickle'
    with open(image_path, 'rb') as basefile:
        image = pickle.load(basefile)
        for i in range(tile):
            for j in range(tile):
                img = image[int(i*(3264/tile)):int((i+1)*(3264/tile)), int(j*(1536/tile)):int((j+1)*(1536/tile))]
                img = np.expand_dims(img, axis=2)
                batch_data.append(np.concatenate((img, img, img), axis=2))
    return batch_data, batch_labels


def reconstruct(volume, tile):
    image_list = []
    # reconstruct images
    for j in range(tile):
        for i in range(tile):
            if i == 0:
                image_list.append(volume[i + j * tile, :, :, :])
                continue
            additive = volume[i + j * tile, :, :, :]
            image_list[j] = np.concatenate((image_list[j], additive), axis=0)
    for l in range(tile):
        if l == 0:
            image = image_list[l]
            continue
        image = np.concatenate((image, image_list[l]), axis=1)
    return image


def save_image(flags, dataset, image_original, image_processed, inds):
    preprocessed_directory = flags['data_directory'] + dataset + '/Preprocessed/' + flags['processed_directory']
    original_directory = flags['data_directory'] + dataset + '/Originals/'
    image_filename = check_str(inds[1]) + '_' + check_str(inds[2])

    if flags['save_original_jpeg'] is True:  # save processed and cropped jpeg
        save_path = original_directory + '/original_jpeg_images/' + image_filename + '.jpg'
        scipy.misc.imsave(save_path, image_original)

    if flags['save_processed_jpeg'] is True:  # save large jpeg file for viewing/presentation purposes
        save_path = preprocessed_directory + '/processed_jpeg_images/'
        if len(image_processed.shape) > 2:  # if is a volume
            directory = save_path + image_filename + '/'
            make_directory(directory)
            for l in range(image_processed.shape[2]):
                scipy.misc.imsave(directory + ('map_%d' % l) + '.jpg', image_processed[:, :, l])
        else:
            scipy.misc.imsave(save_path + image_filename + '.jpg', image_processed)

    if flags['save_pickled_images'] is True:  # save image array as .pickle file in appropriate directory
        save_path = preprocessed_directory + image_filename + '.pickle'
        with open(save_path, "wb") as file:
            pickle.dump(image_processed, file, protocol=2)
    return image_filename


def find_dicom_files(path):  # currently not used but may be useful later.
    """
    Args: folder
        folder_path: Folder location of SAGE competition files
    Returns: A list of all file names with ".dcm.gz" in the name
    """
    print('Searching for DICOM images in %s' % path)
    list_of_dicom_files = []  # create an empty list
    bol = False
    for dirName, subdirList, fileList in os.walk(path):
        print('Found a total of %d files.' % len(fileList))
        for filename in fileList:
            if ".dcm" in filename.lower():  # check whether the file's DICOM\
                bol = True
                list_of_dicom_files.append(os.path.join(dirName, filename))
    if bol is False:
        print('Warning! No Dicom Files found in %s' % path + '!')
        exit()
    return list_of_dicom_files


def generate_split(split_num, step):

    if split_num == 1:
        if step % 3:
            split = [0.5, 0.5]
        else:
            split = [0, 1]

    elif split_num == 2:
        if step % 3:
            split = [0.75, 0.25]
        else:
            split = [0, 1]

    elif split_num == 3:
        if step % 3:
            split = [0, 1]
        else:
            split = [(1/3), (2/3)]

    elif split_num == 4:
        if step % 3:
            split = [0.5, 0.5]
        else:
            split = [0, 1]
    else:
        split = [0, 1]

    return split


def generate_lr(sys_arg):
    if sys_arg == 1:
        return 0.001
    elif sys_arg == 2:
        return 0.0001
    elif sys_arg == 3:
        return 0.00001
    else:
        return 0.001