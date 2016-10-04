#!/usr/bin/env python

"""
Preprocessing Code for Pilot Images of SAGE Competition Mammogram dataset.
Summary: Given the path of the folder contataining the image and text data, generate
python structures to organize and hold the data. JPEG images can be saved from the
dicom images (adjusted by changing the global dictionary flags).
Author: Dan Salo
Created: 9/12/16
Last Edit: DCS, 9/12/16
"""


from functions import process_text, process_images

# Global Dictionary of Flags
flags = {
    'data_directory': '../../../Data/Raw/SAGE/',
    'save_directory': '../../../Data/Processed/SAGE',
    'code_directory': '.',
    'save_processed_jpeg': False,
    'save_original_jpeg': False,
    'save_pickled_dictionary': True,
    'save_pickled_images': False,
    'sub_challenge': '1',
}


def main():
    image_data_dict = process_text(flags)
    process_images(image_data_dict, flags)

if __name__ == "__main__":
    main()
