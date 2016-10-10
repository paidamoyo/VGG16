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
import pandas as pd

from functions.images import read_image, clean_image, save_image, find_dicom_files
from functions.aux import check_directories

# Global Dictionary of Flags
flags = {
    'data_directory': '../../../Data/', # in relationship to the code_directory
    'aux_directory': 'aux/',
    'processed_directory': '1_Cropped/',
    'datasets': ['SAGE', 'INbreast'],
    'save_processed_jpeg': True,
    'save_original_jpeg': False,
    'save_pickled_dictionary': True,
    'save_pickled_images': True,
}


def process_images(image_data_dict, flags, dataset):
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
        filename = flags['data_directory'] + dataset + '/Originals/' + image_data_dict[d][3]
        print('Processing %s' % filename)
        image_original = read_image(filename)
        if image_original is None:
            print("File Type Cannot be Read! Skipping Image...")
            continue
        image_processed = clean_image(image_original, image_data_dict[d], dumb_crop_dims=[3264, 1536])
        new_filename = save_image(flags, dataset, image_original, image_processed, d)
        image_data_dict[d][3] = new_filename

        if counter % 25 == 0 and counter != 0:
            print('Finished processing %d images' % counter)
        counter += 1


def process_text_SAGE(flags):
    crosswalk_tsv_path = flags['data_directory'] + 'SAGE' + "/Metadata/images_crosswalk.tsv"
    indices = [1,3,4,5,6]

    with open(crosswalk_tsv_path) as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter="\t")
        next(tsvreader, None)  # skip the one headerline
        img_counter = 0
        patient_num = -1
        subjectId = -1
        image_data_dict = {}
        for line in tsvreader:
            if subjectId == line[0]:
                img_counter += 1
            else:
                subjectId = line[0]
                patient_num += 1
                img_counter = 0
            image_data_dict[('SAGE', patient_num, img_counter)] = [line[i] for i in indices]
    return image_data_dict


def process_text_INbreast(flags):
    indices = [None, 3, 2, 5, 7]
    crosswalk_tsv_path = flags['data_directory'] + 'INbreast' + "/Metadata/images_crosswalk.csv"
    originals_directory = flags['data_directory'] + 'INbreast' + '/Originals'
    list_dicom_files = find_dicom_files(originals_directory)
    print('Found a total of %d DICOM Images.' % len(list_dicom_files))
    list_file_pat_names = [(str.split(l, '_')[2], str.split(l, '_')[1]) for l in list_dicom_files]
    names = pd.DataFrame(list_file_pat_names)

    with open(crosswalk_tsv_path) as tsvfile:
        csvreader = csv.reader(tsvfile, delimiter=';')
        next(csvreader, None)  # skip the one headerline
        img_counter = 0
        patient_num = -1
        subjectId = -1
        image_data_dict = {}
        for line in csvreader:
            print(line)
            file = line[5]
            new_subjectId = int(names[names[1] == file].index.values)
            print(new_subjectId)
            print(type(new_subjectId))
            if subjectId == new_subjectId:
                img_counter += 1
            else:
                subjectId = new_subjectId
                patient_num += 1
                img_counter = 0
            index = [i for i, x in enumerate(list_file_pat_names) if x == (subjectId, file)]
            line[5] = list_dicom_files[index[0]]
            if line[7] in {'0', '1', '2', '3'}:
                line[7] = 0
            else:
                line[7] = 1
            image_data_dict[('INbreast', patient_num, img_counter)] = [line[i] for i in indices]
    return image_data_dict

def main():
    check_directories(flags)
    dict_SAGE = [process_text_SAGE(flags) for d in flags['datasets'] if d == 'SAGE']
    dict_INbreast = [process_text_INbreast(flags) for d in flags['datasets'] if d == 'INbreast']
    image_data_dict = {**dict_SAGE[0], **dict_INbreast[0]}
    process_images(image_data_dict, flags)

    if flags['save_pickled_dictionary'] is True:
        save_path = flags['aux_directory'] + '1_cropped_image_dict.pickle'
        with open(save_path, "wb") as f:
            pickle.dump(image_data_dict, f, protocol=2)


if __name__ == "__main__":
    main()
