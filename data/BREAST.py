import pickle
import numpy as np
import pandas as pd


class BreastData:
    def __init__(self, flags, image_dict):
        self.flags = flags
        self.data_info(image_dict)
        self.split_data()

    def data_info(self, image_dict):
        self.pdict = pd.DataFrame(image_dict)
        self.dataset = self.flags['datasets'][0]
        self.breast = self.pdict[self.dataset]
        self.all_inds = self.breast.columns.values
        self.data_directory = self.flags['data_directory'] + self.dataset + '/Preprocessed/'\
                              + self.flags['previous_processed_directory']

    def split_data(self):
        self.training_num = len(self.all_inds) - 40

    def generate_training_batch(self, global_step):
        inds = self.all_inds[global_step % self.training_num]
        image_path = self.data_directory + self.check_str(inds[0]) + '_' + self.check_str(inds[1]) + '.pickle'
        with open(image_path, 'rb') as basefile:
            patches = np.zeros((self.flags['batch_size'], self.flags['image_dim'], self.flags['image_dim'], 1))
            labels = np.zeros((self.flags['batch_size']))
            image = pickle.load(basefile)
            label = self.breast[inds][4]
            dims = image.shape
            for b in range(self.flags['batch_size']):
                # Randomly select 28x28 patch from breast image
                successful = False
                while not successful:
                    x = np.random.randint(low=0, high=dims[0] - self.flags['image_dim'])
                    y = np.random.randint(low=0, high=dims[1] - self.flags['image_dim'])
                    img = image[x:x + self.flags['image_dim'], y:y + self.flags['image_dim']]
                    if img.max() > 5:
                        successful = True
                img = img / img.max()
                patches[b, :, :, 0] = img - img.mean()
                labels[b] = label
        return labels, patches

    @staticmethod
    def check_str(obj):
        if isinstance(obj, str):
            return obj
        if isinstance(obj, float):
            return str(int(obj))
        else:
            return str(obj)