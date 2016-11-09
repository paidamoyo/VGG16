#!/usr/bin/env python

import cv2
import numpy as np
import dicom as di


def smart_crop(image, dataset):
    """
    Args: image
        image: numpy array 2D with pixel values, orientated so chest wall is on left side of image: (:,0)
    Returns: numpy 2D array with pixels values cropped at calculated column index... keep (:,0:i). If no
    suitable index can be found, return the original image.
    """
    if dataset is 'SAGE':
        bright_range = [20, 10, 4]
    elif dataset is 'INbreast':
        bright_range = [10, 5, 2]
    else:
        bright_range = [0]
        print('Dataset not defined for smart_crop')
        exit()

    for brightness in bright_range:  # loop over different brightness levels
        for i in range(0, image.shape[1]-1, 64):
            if not sum(image[:, i] + image[:, i+1]) < 255*brightness:
                continue
            else:
                if i < 512:
                    exit()
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


def clean_image(image, image_data_dict, d, dumb_crop_dims=None):
    image = cv2.convertScaleAbs(image, alpha=(255.0 / image.max()))  # convert to uint8
    if image_data_dict[d][2] == 'R':  # flip all images into left orientation
        image = np.fliplr(image)
        print('Image Flipped.')
    if dumb_crop_dims is None:
        image = smart_crop(image, d[0])
    else:
        image = dumb_crop(image, dumb_crop_dims[0], dumb_crop_dims[1])
    if d[0] == 'SAGE':
        gamma = 3.0
    elif d[0] == 'INbreast':
        gamma = 2.0
    image = adjust_gamma(image, gamma=gamma)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    image = clahe.apply(image)
    return image
