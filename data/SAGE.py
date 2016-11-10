import pickle
import numpy as np
import pandas as pd
from functions.aux import check_str, make_directory


def generate_breast_patch(flags, image_dict):
    pdict = pd.DataFrame(image_dict)
    for dataset in flags['datasets']:
        breast = pdict[dataset]
        all_inds = breast.columns.values
        batch_ind = np.random.randint(low=0, high=len(all_inds), size=1)
        inds = all_inds[batch_ind][0]
        data_directory = flags['data_directory'] + dataset +'/Preprocessed/' + flags['previous_processed_directory']
        image_path = data_directory + check_str(inds[0]) + '_' + check_str(inds[1]) + '.pickle'
        with open(image_path, 'rb') as basefile:
            patches = np.zeros((flags['batch_size'], flags['image_dim'], flags['image_dim'], 1))
            labels = np.zeros((flags['batch_size']))
            image = pickle.load(basefile)
            label = breast[inds][4]
            dims = image.shape
            for b in range(flags['batch_size']):
                # Randomly select 28x28 patch from breast image
                successful = False
                while not successful:
                    x = np.random.randint(low=0, high=dims[0] - flags['image_dim'])
                    y = np.random.randint(low=0, high=dims[1] - flags['image_dim'])
                    img = image[x:x + flags['image_dim'], y:y + flags['image_dim']]
                    if img.max() > 5:
                        successful = True
                        print(img.max())
                img = img / img.max()
                patches[b, :, :, 0] = img - img.mean()
                labels[b] = label
    return labels, patches
