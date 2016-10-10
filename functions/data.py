import numpy as np
import math
import pandas as pd
import pickle

from functions.aux import check_str

def get_image_data(index_name, data_directory, image_dict):
    data = []
    labels = []
    counter = 0
    for inds in index_name:
        image_path = data_directory + '/' + ('%d' % inds[0]) + '_' + ('%d' % inds[1]) + '.cpickle'
        with open(image_path, 'rb') as filename:
            data.append(pickle.load(filename))
            labels.append(int(image_dict[inds][5]))
            counter += 1
    return data, labels


def split_data(image_dict, seed):
    np.random.seed(seed=seed)
    index_test = []
    index_train = []
    dict_test = {}
    dict_train = {}
    dict_image = pd.DataFrame(image_dict)
    labels = dict_image.iloc[5]
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


def generate_minibatch_dict(data_directory, dict_name, batch_size):
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


def generate_minibatch_dict_small(data_directory, dict_name, pos_neg):
    batch_data = []
    l = pos_neg
    batch_labels = [l]
    batch_ind = np.random.randint(low=0, high=len(dict_name[l]) - 1, size=1)
    inds = dict_name[l][batch_ind]
    image_path = data_directory + '/' + inds[0] + '_' + ('%d' % inds[1]) + '.pickle'
    with open(image_path, 'rb') as basefile:
        image = pickle.load(basefile)
        for i in range(12):
            for j in range(12):
                img = image[i*(3264/12):(i+1)*(3264/12), j*(1536/12):(j+1)*(1536/12)]
                img = np.expand_dims(img, axis=2)
                batch_data.append(np.concatenate((img, img, img), axis=2))
    return batch_data, batch_labels


def one_tiled_image(flags, inds):
    batch_data = []
    batch_labels = [l]
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