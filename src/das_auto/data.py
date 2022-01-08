import tensorflow
import numpy as np


class PairGen(tensorflow.keras.utils.Sequence):

    def __init__(self, data_gen, augs):
        self.data_gen = data_gen
        self.augs = augs

    def __len__(self):
        return len(self.data_gen)

    def __getitem__(self, idx):
        batch = self.data_gen[0]
        data, labels = batch[0], batch[1]

        # augment data
        data1 = self.augs(data)
        data2 = self.augs(data)
        labels = np.ones((data.shape[0],))

        return [data1, data2], labels
