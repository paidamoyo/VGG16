#!/usr/bin/env python

import cv2
import numpy as np
import dicom as di
import scipy.misc
import pickle
import os

from functions.aux import check_str


def smart_crop(image):
    """
    Args: image
        image: numpy array 2D with pixel values, orientated so chest wall is on left side of image: (:,0)
    Returns: numpy 2D array with pixels values cropped at calculated column index... keep (:,0:i). If no
    suitable index can be found, return the original image.
    """
    for brightness in [10, 5, 2]:  # loop over different brightness levels
        for i in range(0, image.shape[1]-1, 64):
            if not sum(image[:, i] + image[:, i+1]) < 256*2*brightness:
                continue
            else:
                return image[:, 0:i]
    return image


def dumb_crop(image, rows, cols):
    return image[0:rows, 0:cols]


def adjust_gamma(image, gamma):
    invgamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invgamma) * 255
                      for i in np.arange(0, 256)]).astype('uint8')
    return cv2.LUT(image, table)


def read_image(base_file):
    while True:
        try:
            ds = di.read_file(base_file)
            # image_dims = (int(ds.Rows), int(ds.Columns))
            return ds.pixel_array
        except ValueError:  # couldn't read it
            return None
        except FileNotFoundError:  # couldn't find it
            return None


def clean_image(image, dict_entry, dumb_crop_dims):
    image = cv2.convertScaleAbs(image, alpha=(255.0 / image.max()))  # convert to uint8
    if dict_entry[3] == 'R':  # flip all images into left orientation
        image = np.fliplr(image)
    if dumb_crop_dims is None:
        image = smart_crop(image)
    else:
        image = dumb_crop(image, dumb_crop_dims[0], dumb_crop_dims[1])
    image = adjust_gamma(image, gamma=3)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe.apply(image)
    return image


def save_image(flags, dataset, image_original, image_processed, d):
    preprocessed_directory = flags['data_directory'] + dataset + '/Preprocessed/' + flags['processed_directory']
    original_directory = flags['data_directory'] + dataset + '/Originals/'
    image_filename = check_str(d[1]) + '_' + check_str(d[2]) + '.pickle'

    if flags['save_original_jpeg'] is True:  # save processed and cropped jpeg
        save_path = original_directory + '/original_jpeg_images/' + check_str(d[0]) + '_' + check_str(d[1]) + '.jpg'
        scipy.misc.imsave(save_path, image_original)

    if flags['save_processed_jpeg'] is True:  # save large jpeg file for viewing/presentation purposes
        save_path = preprocessed_directory + '/processed_jpeg_images/' + check_str(d[0]) + '_' + check_str(d[1]) + '.jpg'
        scipy.misc.imsave(save_path, image_processed)

    if flags['save_pickled_images'] is True:  # save image array as .pickle file in appropriate directory
        save_path = preprocessed_directory + image_filename
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
        for filename in fileList:
            print(filename)
            if ".dcm." in filename.lower():  # check whether the file's DICOM\
                bol = True
                print('Found file: %s' % filename.lower())
                list_of_dicom_files.append(os.path.join(dirName, filename))
    if bol is False:
        print('Warning! No Dicom Files found in %s' % path + '!')
        exit()
    return list_of_dicom_files