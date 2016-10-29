import pickle
import numpy as np
import pandas as pd
from functions.aux import check_str, make_directory


def generate_SAGE(flags, image_dict):
    pdict = pd.DataFrame(image_dict)
    sage = pdict['SAGE']
    all_inds = sage.columns.values
    batch_ind = np.random.randint(low=0, high=len(all_inds), size=1)
    inds = all_inds[batch_ind][0]
    data_directory = flags['data_directory'] + 'SAGE/Preprocessed/' + flags['previous_processed_directory']
    image_path = data_directory + check_str(inds[0]) + '_' + check_str(inds[1]) + '.pickle'
    with open(image_path, 'rb') as basefile:
        patches = np.zeros((flags['batch_size'], 28, 28, 1))
        labels = np.zeros((flags['batch_size']))
        image = pickle.load(basefile)
        label = sage[inds][4]
        dims = image.shape
        for b in range(flags['batch_size']):
            # Randomly select 28x28 patch from breast image
            x = np.random.randint(low=0 + dims[0]/4, high=dims[0] - dims[0]/4)
            y = np.random.randint(low=0 + dims[1]/4, high=dims[1] - dims[1]/4)
            patches[b, :, :, 0] = image[x:x + 28, y:y + 28] / image[x:x + 28, y:y + 28].max()
            labels[b] = label
    return labels, patches
