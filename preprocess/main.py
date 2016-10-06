#!/usr/bin/env python

"""
Preprocessing Code for Mammograms
Summary: Reads, crops, image processing, pickles dicom Images
Author: Dan Salo
Created: 9/12/16
Last Edit: DCS, 10/05/16
"""

from functions import process_text_SAGE, process_images_SAGE
from functions_aux import check_directories

# Global Dictionary of Flags
flags = {
    'data_directory': '../../../../Data/', # in relationship to the code_directory
    'aux_directory': '../aux/',
    'processed_directory': '1_Cropped/',
    'datasets': ['SAGE'],
    'save_processed_jpeg': True,
    'save_original_jpeg': False,
    'save_pickled_dictionary': True,
    'save_pickled_images': True,
}


def main():
    check_directories(flags)
    for d in flags['datasets']:
        if d == 'SAGE':
            image_data_dict = process_text_SAGE(flags, d)
            process_images_SAGE(image_data_dict, flags, d)

if __name__ == "__main__":
    main()
