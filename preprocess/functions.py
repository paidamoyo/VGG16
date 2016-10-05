#!/usr/bin/env python

import gzip
import pickle
import scipy.misc
import csv

from functions_images import clean_image, read_image
from functions_aux import check_directory, check_str, save_image


def process_images_SAGE(image_data_dict, flags, dataset):
    """
    Args: image_data_dict, folder_path
        image_data_dict: Dictionary keyed by a tuple containing (subjectId, image#). Each value is a list of:
            [Exam Number, Image Index (not used), View, Side,  DCM Filename, Binary Label]
        folder_path: Folder location of SAGE competition files
    Performs: reads and pickles the dicom images individually. Option to save images as jpegs.
    """
    print('Processing all %d images' % len(image_data_dict))
    counter = 0
    check_directory(flags)

    for d in image_data_dict:  # loop through all images in dictionary.
        filename = flags['data_directory'] + '/' + image_data_dict[d][4] + '.gz' # extract image filename
        with gzip.open(filename) as f:  # open gzipped images, only in pilot image set
            image_original = read_image(f)
            if image_original is None:
                print("File Type Cannot be Read! Skipping Image...")
                continue
            image_processed = clean_image(image_original, image_data_dict[d])
            save_image(flags, dataset, image_original, image_processed, d)

            if counter % 25 == 0 and counter != 0:
                print('Finished processing %d images' % counter)
            counter += 1


# Text Processing Functions
def process_text(flags):
    """
    Args: folder_path
        folder_path: Folder location of SAGE competition files
    Returns: image_data_dict: Dictionary keyed by a tuple containing (subjectId, image#). Each value is a list of:
            [Exam Number, Image Index (not used), View, Side,  DCM Filename, Binary Label]
    """
    # hardwired filenames for .tsv files
    folder_path = flags['data_directory']
    crosswalk_tsv_path = folder_path + "metadata/images_crosswalk.tsv"
    metadata_tsv_path = folder_path + "metadata/exams_metadata.tsv"

    with open(crosswalk_tsv_path) as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter="\t")
        next(tsvreader, None)  # skip the one headerline
        bol = False
        counter = 0
        image_data_dict = {}
        for line in tsvreader:
            if bol is True:
                if subjectId == line[0]:
                    counter += 1
                else:
                    subjectId = line[0]
                    counter = 0
                image_data_dict[(subjectId, counter)] = line[1:]
            else:
                subjectId = line[0]
                image_data_dict[(subjectId, counter)] = line[1:]
                bol = True

    with open(metadata_tsv_path) as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter="\t")
        next(tsvreader, None)  # skip the one headerline
        for line in tsvreader:
            for c in image_data_dict:
                if line[0] == c[0] and line[1] == image_data_dict[c][0]:  # match patient ID and exam number
                    if image_data_dict[c][3] == 'R':
                        image_data_dict[c].append(line[4])
                    if image_data_dict[c][3] == 'L':
                        image_data_dict[c].append(line[3])

    if flags['save_pickled_dictionary'] is True:
        save_path = '../aux/vgg_image_dict.pickle'
        with open(save_path, "wb") as f:
            pickle.dump(image_data_dict, f, protocol=2)

    return image_data_dict
