#!/usr/bin/env python

import cv2
import numpy as np
import dicom as di


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

def dumb_crop(image):
    return image[0:3264, 0:1536] #  total size of 1536 x 3264


def adjust_gamma(image, gamma):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invgamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invgamma) * 255
                      for i in np.arange(0, 256)]).astype('uint8')

    # apply gamma correction using the lookup table
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


def clean_image(image, dict_entry):
    image = cv2.convertScaleAbs(image, alpha=(255.0 / image.max()))  # convert to uint8
    if dict_entry[3] == 'R':  # flip all images into left orientation
        image = np.fliplr(image)
    image = dumb_crop(image)
    image = adjust_gamma(image, gamma=3)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe.apply(image)
    return image