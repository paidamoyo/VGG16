#!/usr/bin/env python

"""
Preprocessing Code for Mammograms
Summary: Reads, crops, image processing, pickles dicom Images
Author: Dan Salo
Created: 9/12/16
Last Edit: DCS, 10/05/16
"""

import gzip
import csv
import pickle

from functions.images import read_image, clean_image, save_image
from functions.aux import check_directories

# Global Dictionary of Flags
flags = {
    'data_directory': '../../../Data/', # in relationship to the code_directory
    'aux_directory': 'aux/',
    'processed_directory': '1_Cropped/',
    'datasets': ['SAGE'],
    'save_processed_jpeg': True,
    'save_original_jpeg': False,
    'save_pickled_dictionary': True,
    'save_pickled_images': True,
}



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

    for d in image_data_dict:  # loop through all images in dictionary.
        filename = flags['data_directory'] + dataset + '/Originals/' + image_data_dict[d][4] + '.gz'
        with gzip.open(filename) as f:  # open gzipped images, only in pilot image set
            image_original = read_image(f)
            if image_original is None:
                print("File Type Cannot be Read! Skipping Image...")
                continue
            image_processed = clean_image(image_original, image_data_dict[d], dumb_crop_dims=[3264, 1536])
            new_filename = save_image(flags, dataset, image_original, image_processed, d)
            image_data_dict[d][4] = new_filename

            if counter % 25 == 0 and counter != 0:
                print('Finished processing %d images' % counter)
            counter += 1


# Text Processing Functions
def process_text_SAGE(flags, dataset):
    """
    Args: folder_path
        folder_path: Folder location of SAGE competition files
    Returns: image_data_dict: Dictionary keyed by a tuple containing (subjectId, image#). Each value is a list of:
            [Exam Number, Image Index (not used), View, Side,  DCM Filename, Binary Label]
    """
    metadata_path = flags['data_directory'] + 'SAGE' + "/Metadata/"
    crosswalk_tsv_path = metadata_path + "images_crosswalk.tsv"

    with open(crosswalk_tsv_path) as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter="\t")
        next(tsvreader, None)  # skip the one headerline
        bol = False
        counter = 0
        patient_num = 0
        image_data_dict = {}
        for line in tsvreader:
            if bol is True:
                if subjectId == line[0]:
                    counter += 1
                else:
                    subjectId = line[0]
                    patient_num += 1
                    counter = 0
                line2 = line[1:]
                del line2[1]
                image_data_dict[('SAGE', patient_num, counter)] = line2
            else:
                subjectId = line[0]
                line2 = line[1:]
                del line2[1]
                image_data_dict[('SAGE', patient_num, counter)] = line2
                bol = True
    return image_data_dict


def main():
    check_directories(flags)
    for d in flags['datasets']:
        if d == 'SAGE':
            image_data_dict = process_text_SAGE(flags, d)
            process_images_SAGE(image_data_dict, flags, d)

    if flags['save_pickled_dictionary'] is True:
        save_path = flags['aux_directory'] + '1_cropped_image_dict.pickle'
        with open(save_path, "wb") as f:
            pickle.dump(image_data_dict, f, protocol=2)


if __name__ == "__main__":
    main()
