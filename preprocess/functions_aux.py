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


def check_directory(flags):
    if flags['save_processed_jpeg'] is True:  # ensure jpeg directory exists. if not, create it.
        directory_processed_jpeg = flags['save_directory'] + '/processed_jpeg_images'
        make_directory(directory_processed_jpeg)

    if flags['save_original_jpeg'] is True:  # ensure jpeg directory exists. if not, create it.
        directory_original_jpeg = flags['save_directory'] + '/original_jpeg_images'
        make_directory(directory_original_jpeg)

    if flags['save_pickled_images'] is True:  # ensure pickle directory exists. if not, create it.
        directory_pickle = flags['save_directory']
        make_directory(directory_pickle)


def find_dicom_files(flags):  # currently not used but may be useful later.
    """
    Args: folder
        folder_path: Folder location of SAGE competition files
    Returns: A list of all file names with ".dcm.gz" in the name
    """
    folder_path = flags['project_directory']
    print('Searching for DICOM images in %s' % folder_path)
    list_of_dicom_files = []  # create an empty list
    for dirName, subdirList, fileList in os.walk(folder_path + '/trainingData'):
        for filename in fileList:
            if ".dcm.gz" in filename.lower():  # check whether the file's DICOM
                list_of_dicom_files.append(os.path.join(dirName, filename))
    return list_of_dicom_files