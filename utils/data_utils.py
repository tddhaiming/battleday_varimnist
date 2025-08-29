""" utils/data_utils.py
功能：读取.pt特征/标签并返回 DataLoader；Dataset支持 lazy load在 DataLoader worker中把 tensor 加载到内存中以避免主进程瓶颈
""" #新代码
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class FeatureDataset(Dataset):
    def __init__(self, feat_path, label_path=None):
        self.feat_path = feat_path
        self.label_path = label_path
        self._feats = None
        self._labels = None

    def load_data(self): # lazy loading in worker
        if self._feats is None:
            self._feats = torch.load(self.feat_path).float()
        if self.label_path and self._labels is None:
            self._labels = torch.load(self.label_path).long()

    def __len__(self):
        self.load_data()
        return self._feats.size(0)

    def __getitem__(self, idx):
        self.load_data()
        feat = self._feats[idx]
        if self._labels is not None:
            label = self._labels[idx]
            return feat, label
        return feat

def make_dataloader(feat_path, label_path=None, batch_size=256, num_workers=8, shuffle=False, drop_last=False,prefetch_factor=4,pin_memory=True):
    ds = FeatureDataset(feat_path, label_path)
    # persistent_workers在非空DataLoader下能避免反复创建/销毁worker进程开销
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                    pin_memory=pin_memory, drop_last=drop_last, persistent_workers=(num_workers > 0), prefetch_factor=prefetch_factor)
    return dl
