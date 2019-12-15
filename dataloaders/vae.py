import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from dataloaders.base import AbstractDataloader


class VAEDataset(Dataset):

    def __init__(self, u2seq, sc, partition_ratio=0.0, partition_random_seed=None):
        self.u2seq = u2seq
        self.uids = sorted(u2seq.keys())
        self.uc = len(u2seq)
        self.sc = sc
        self.partition_random_seed = partition_random_seed
        self.partition_ratio = partition_ratio

    def __len__(self):
        return self.uc

    def __getitem__(self, index):
        uid = self.uids[index]
        seq = np.array(self.u2seq[uid])

        if self.partition_ratio > 0:
            indices = seq - 1
            if len(seq) >= 1 / self.partition_ratio:
                np.random.seed(self.partition_random_seed)
                test_indices = indices[np.random.choice(len(indices), size=int(self.partition_ratio * len(indices)), replace=False)]
            else:
                test_indices = np.array([])
            true_indices = np.setdiff1d(indices, test_indices)

            true_array, test_array = np.zeros(self.sc, dtype='bool'), np.zeros(self.sc, dtype='bool')
            true_array[true_indices] = True
            test_array[test_indices] = True
            return (
                torch.FloatTensor(true_array),
                torch.LongTensor(test_array),
            )
        else:
            array = np.zeros(self.sc, dtype='bool')
            array[seq-1] = True
            return (torch.FloatTensor(array), )


class VAEDataloader(AbstractDataloader):

    @classmethod
    def code(cls):
        return 'vae'

    def get_meta(self):
        return {
            'item_count': self.item_count
        }

    def get_pytorch_dataloaders(self):
        return (
            # train
            DataLoader(dataset=VAEDataset(self.train, self.item_count), batch_size=self.args.train_batch_size, pin_memory=True),

            # validation
            DataLoader(dataset=VAEDataset(self.val, self.item_count, partition_ratio=self.args.partition_ratio, partition_random_seed=self.args.partition_random_seed), batch_size=self.args.val_batch_size, pin_memory=True),

            # test
            DataLoader(dataset=VAEDataset(self.test, self.item_count, partition_ratio=self.args.partition_ratio, partition_random_seed=self.args.partition_random_seed), batch_size=self.args.test_batch_size, pin_memory=True),
        )
