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

from functions.images import read_image, clean_image
from functions.data import save_image, find_dicom_files
from functions.aux import check_directories

# Global Dictionary of Flags
flags = {
    'data_directory': '../../../Data/',  # in relationship to the code_directory
    'aux_directory': 'aux/',
    'processed_directory': 'Smart_Crop/',
    'datasets': ['INbreast', 'SAGE'],
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
        if d[0] == dataset:
            filename = flags['data_directory'] + dataset + '/Originals/' + image_data_dict[d][3]
            print('Processing %s' % image_data_dict[d][3], ', with identifiers: ', image_data_dict[d][2], d)
            image_original = read_image(filename)
            if image_original is None:
                print("File Type Cannot be Read! Skipping Image...")
                continue
            image_processed = clean_image(image_original, image_data_dict, d, dumb_crop_dims=None)
            new_filename = save_image(flags, dataset, image_original, image_processed, d)
            image_data_dict[d][3] = new_filename

            if counter % 25 == 0 and counter != 0:
                print('Finished processing %d images' % counter)
            counter += 1


def process_text_SAGE(flags):
    crosswalk_tsv_path = flags['data_directory'] + 'SAGE' + "/Metadata/images_crosswalk.tsv"
    indices = [1, 3, 4, 5, 6]

    with open(crosswalk_tsv_path) as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter="\t")
        next(tsvreader, None)  # skip the one headerline
        img_counter = 0
        patient_num = -1
        subjectId = -1
        total = 0
        image_data_dict = {}
        for line in tsvreader:
            if subjectId == line[0]:
                img_counter += 1
            else:
                subjectId = line[0]
                patient_num += 1
                img_counter = 0
            image_data_dict[('SAGE', patient_num, img_counter)] = [line[i] for i in indices]
            total += 1
        print('Collated organized %d patients and %d images from the %s dataset.' % (patient_num, total, 'SAGE'))
    return image_data_dict


def process_text_INbreast(flags):
    indices = [3, 2, 5, 7]  # add 0 for all
    crosswalk_tsv_path = flags['data_directory'] + 'INbreast' + "/Metadata/images_crosswalk.csv"
    originals_directory = flags['data_directory'] + 'INbreast' + '/Originals'
    list_dicom_files = find_dicom_files(originals_directory)
    print('Found a total of %d DICOM Images.' % len(list_dicom_files))
    list_file_pat_names = [(str.split(l, '_')[1], str.split(str.split(l, '_')[0],'/')[-1]) for l in list_dicom_files]
    names = pd.DataFrame(list_file_pat_names)

    with open(crosswalk_tsv_path) as tsvfile:
        csvreader = csv.reader(tsvfile, delimiter=';')
        next(csvreader, None)  # skip the one headerline
        img_counter = 0
        total = 0
        patient_num = -1
        subjectId = -1
        image_data_dict = {}
        for line in csvreader:
            file = line[5]
            new_subjectId = names[names[1] == file].values[0][0]
            if subjectId == new_subjectId:
                img_counter += 1
            else:
                subjectId = new_subjectId
                patient_num += 1
                img_counter = 0
            index = [i for i, x in enumerate(list_file_pat_names) if x == (subjectId, file)]
            if len(index) == 0:
                IndexError('No DICOM Files match the line in the CSV file! Exiting...')
                exit()
            line[5] = str.split(list_dicom_files[index[0]], '/')[-1]
            if line[7] in {'0', '1', '2', '3'}:
                line[7] = '0'
            else:
                line[7] = '1'
            image_data_dict[('INbreast', patient_num, img_counter)] = ['0'] + [line[i] for i in indices]
            total += 1
        print('Collated %d patients and %d images from the %s dataset.' % (patient_num, total,'INbreast'))
    return image_data_dict


def main():
    check_directories(flags)
    list_dict = list()
    list_dict.extend([process_text_SAGE(flags) for d in flags['datasets'] if d == 'SAGE'])
    list_dict.extend([process_text_INbreast(flags) for d in flags['datasets'] if d == 'INbreast'])
    if len(list_dict) == 2:
        image_data_dict = {**list_dict[0], **list_dict[1]}
    else:
        image_data_dict = list_dict[0]
    example = pd.DataFrame(image_data_dict, index=['ExamNumber', 'View', 'Laterality', 'Filename', 'Label'])

    if any([flags['save_pickled_images'], flags['save_processed_jpeg']]) is True:
        [process_images(image_data_dict, flags, d) for d in flags['datasets']]

    if flags['save_pickled_dictionary'] is True:
        save_path = flags['aux_directory'] + 'preprocessed_image_dict.pickle'
        with open(save_path, "wb") as f:
            pickle.dump(image_data_dict, f, protocol=2)


if __name__ == "__main__":
    main()
