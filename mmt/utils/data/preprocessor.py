from __future__ import absolute_import
import os
import os.path as osp
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import math
from PIL import Image

class Preprocessor(Dataset):
    def __init__(self, dataset, root=None, transform=None, mutual=False, cluster=False, pretrain=False):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform
        self.mutual = mutual
        self.cluster = cluster
        self.pretrain = pretrain

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if self.mutual:
            return self._get_mutual_item(indices)
        elif self.pretrain:                                # zdy
            return self._get_pretrain_item(indices)
        elif self.cluster:                                 # zdy
            return self._get_single_item_cluster(indices)  # zdy
        else:
            return self._get_single_item(indices)

    def _get_pretrain_item(self, index):
        _, fname, pid, camid = self.dataset[index]  ## zdy add _,
        #fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)

        img = Image.open(fpath).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, fname, pid, camid

    def _get_single_item(self, index):
        #_, fname, pid, camid = self.dataset[index]  ## zdy add _,
        fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)

        img = Image.open(fpath).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, fname, pid, camid

    # zdy
    def _get_single_item_cluster(self, index):
        #_, fname, pid, camid = self.dataset[index]  ## zdy add _,
        _, fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)

        img = Image.open(fpath).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, fname, pid, camid
    # zdy

    def _get_mutual_item(self, index):
        memo_index, fname, pid, camid = self.dataset[index]  ## zdy add _,
        #fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)

        img_1 = Image.open(fpath).convert('RGB')
        img_2 = img_1.copy()

        if self.transform is not None:
            img_1 = self.transform(img_1)
            img_2 = self.transform(img_2)

        return memo_index, img_1, img_2, pid
