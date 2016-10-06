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
        if flags['save_pickled_images'] is True:  # ensure pickle directory exists. if not, create it.
            make_directory(preprocessed_directory)