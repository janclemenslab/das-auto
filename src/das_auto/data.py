import tensorflow
import numpy as np
from typing import Callable, Optional


class PairGen(tensorflow.keras.utils.Sequence):

    def __init__(self, data_gen, augs: Callable = None, thres: Optional[float] = None):
        self.data_gen = data_gen

        if augs is None:
            augs = lambda x: x  # identity transform
        self.augs = augs
        self.thres = thres

    def __len__(self):
        return len(self.data_gen)

    def _keep_loud(self, thres):
        batch = self.data_gen[0]
        data, labels = batch[0], batch[1]
        amps = np.max(np.abs(data), axis=(1, 2))
        while np.any(amps < self.thres):
            # get new batch
            batch_tmp = self.data_gen[0]
            data_tmp, labels_tmp = batch_tmp[0], batch_tmp[1]
            amps_tmp = np.max(np.abs(data_tmp), axis=(1, 2))

            # get the loud ones from the new batch
            loud = np.where(amps_tmp >= self.thres)[0]

            # get the too-soft-ones for the original batch
            soft = np.where(amps < self.thres)[0]
            min_len = min(len(loud), len(soft))

            # replace the too-soft-ones with the loud ones
            data[soft[:min_len], ...] = data_tmp[loud[:min_len], ...]
            labels[soft[:min_len], ...] = labels_tmp[loud[:min_len], ...]

            # new amps
            amps = np.max(np.abs(data), axis=(1, 2))
        return data, labels

    def __getitem__(self, idx):
        if self.thres is None:
            batch = self.data_gen[0]
            data, labels = batch[0], batch[1]
        else:
            data, labels = self._keep_loud(self.thres)

        # augment data
        data1 = self.augs(data)
        data2 = self.augs(data)
        labels = np.ones((data.shape[0],))

        return [data1, data2], labels
