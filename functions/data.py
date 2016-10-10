import numpy as np
import math
import pandas as pd
import pickle
import scipy.misc

from functions.aux import check_str


def split_data(image_dict, seed):
    np.random.seed(seed=seed)
    index_test = []
    index_train = []
    dict_test = {}
    dict_train = {}
    dict_image = pd.DataFrame(image_dict)
    labels = dict_image.iloc[5]
    print(labels)
    for i in ['0', '1']:
        patients = labels[labels == i].index.values
        partition = int(math.floor(len(patients) * .15))  # 70% of data goes to training
        indexes = np.random.choice(range(len(patients)), size=len(patients))
        dict_test[int(i)] = list(patients[indexes[partition:]])
        dict_train[int(i)] = list(patients[indexes[:partition]])
        index_test = index_test + list(patients[indexes[partition:]])
        index_train = index_train + list(patients[indexes[:partition]])
        print("Training split has %d mammograms" % len(dict_train[int(i)]) + " with label %d" % int(i))
        print("Testing split has %d mammograms" % len(dict_train[int(i)]) + " with label %d" % int(i))
    return dict_train, dict_test, index_train, index_test


def generate_minibatch(data_directory, dict_name, batch_size):
    batch_data = []
    batch_labels = []
    factor = [0.75, 0.25]
    for i in range(2):
        batch_ind = np.random.randint(low=0, high=len(dict_name[i]) - 1, size=[1])
        for b in batch_ind:
            inds = dict_name[i][b]
            image_path = data_directory + '/' + inds[0] + '_' + ('%d' % inds[1]) + '.pickle'
            with open(image_path, 'rb') as basefile:
                image = pickle.load(basefile)
                img = np.expand_dims(image,axis=2)
                batch_data.append(np.concatenate((img,img,img),axis=2))
                batch_labels.append(i)
    a = np.asarray(batch_data)
    b = np.asarray(batch_labels)
    return batch_data, batch_labels


def one_tiled_image(flags, image_dict, inds):
    batch_data = []
    batch_labels = [image_dict[inds][4]]
    path = flags['data_directory'] + inds[0] + '/Preprocessed/' + flags['previous_processed_directory']
    image_path = path + check_str(inds[1]) + '_' + check_str(inds[2]) + '.pickle'
    with open(image_path, 'rb') as basefile:
        image = pickle.load(basefile)
        for i in range(12):
            for j in range(12):
                img = image[i*(3264/12):(i+1)*(3264/12), j*(1536/12):(j+1)*(1536/12)]
                img = np.expand_dims(img, axis=2)
                batch_data.append(np.concatenate((img, img, img), axis=2))
    return batch_data, batch_labels


def reconstruct(volume):
    image_list = []
    # reconstruct images
    for j in range(12):
        for i in range(12):
            if i == 0:
                image_list.append(volume[i + j * 12, :, :, :])
                continue
            additive = volume[i + j * 12, :, :, :]
            image_list[j] = np.concatenate((image_list[j], additive), axis=0)
    for l in range(12):
        if l == 0:
            image = image_list[l]
            continue
        image = np.concatenate((image, image_list[l]), axis=1)
    return image


def save_image(flags, dataset, image_original, image_processed, inds):
    preprocessed_directory = flags['data_directory'] + dataset + '/Preprocessed/' + flags['processed_directory']
    original_directory = flags['data_directory'] + dataset + '/Originals/'
    image_filename = check_str(inds[1]) + '_' + check_str(inds[2])

    if flags['save_original_jpeg'] is True:  # save processed and cropped jpeg
        save_path = original_directory + '/original_jpeg_images/' + image_filename + '.jpg'
        scipy.misc.imsave(save_path, image_original)

    if flags['save_processed_jpeg'] is True:  # save large jpeg file for viewing/presentation purposes
        save_path = preprocessed_directory + '/processed_jpeg_images/' + image_filename + '.jpg'
        scipy.misc.imsave(save_path, image_processed)

    if flags['save_pickled_images'] is True:  # save image array as .pickle file in appropriate directory
        save_path = preprocessed_directory + image_filename + '.pickle'
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
        print('Found a total of %d files.' % len(fileList))
        for filename in fileList:
            if ".dcm" in filename.lower():  # check whether the file's DICOM\
                bol = True
                list_of_dicom_files.append(os.path.join(dirName, filename))
    if bol is False:
        print('Warning! No Dicom Files found in %s' % path + '!')
        exit()
    return list_of_dicom_files