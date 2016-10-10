#!/usr/bin/env python

import os

def make_directory(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def check_str(obj):
    if isinstance(obj, str):
        return obj
    if isinstance(obj, float):
        return str(int(obj))
    else:
        return str(obj)


def check_directories(flags):
    make_directory(flags['aux_directory'])
    for dataset in flags['datasets']:
        preprocessed_directory = flags['data_directory'] + dataset + '/Preprocessed/' + flags['processed_directory']
        make_directory(preprocessed_directory)
        try:
            if flags['save_processed_jpeg'] is True:
                make_directory(preprocessed_directory + 'processed_jpeg_images')
            else:
                print('Not saving Processed JPEGs')
        except KeyError:
            print('Not saving Processed JPEGs')

        try:
            if flags['save_original_jpeg'] is True:
                make_directory(preprocessed_directory + 'original_jpeg_images')
            else:
                print('Not saving Original JPEGs')
        except KeyError:
            print('Not saving Original JPEGs')

        try:
            if flags['save_pickled_images'] is True:
                make_directory(preprocessed_directory)
            else:
                print('Not saving pickled images')
        except KeyError:
            print('Not saving pickled images')
