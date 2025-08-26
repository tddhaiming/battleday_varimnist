# utils/data_utils.py
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms
import json

def load_processed_data():
    """加载预处理后的数据"""
    images_path = "/content/gdrive/MyDrive/battleday_varimnist/data/processed/images.npy"
    softlabels_path = "/content/gdrive/MyDrive/battleday_varimnist/data/processed/softlabels.npy"
    
    if os.path.exists(images_path) and os.path.exists(softlabels_path):
        images = np.load(images_path)
        softlabels = np.load(softlabels_path)
        return images, softlabels
    else:
        return None, None

def load_splits():
    """加载数据划分"""
    splits_path = "/content/gdrive/MyDrive/battleday_varimnist/data/splits.json"
    if os.path.exists(splits_path):
        with open(splits_path, "r") as f:
            return json.load(f)
    return None

def get_train_val_features(model_name='resnet18'):
    """获取训练和验证集的特征"""
    # 加载特征
    try:
        features = np.load(f"/mnt/dataset0/thm/code/battleday_varimnist/features/{model_name}_features.npy")
        softlabels = np.load("/mnt/dataset0/thm/code/battleday_varimnist/data/processed/softlabels.npy")
        splits = load_splits()
        
        if splits is None:
            raise FileNotFoundError("请先运行数据预处理脚本")
            
        # 划分数据
        train_idx = splits["train"]
        val_idx = splits["val"]
        
        X_train = features[train_idx]
        y_train = softlabels[train_idx]
        X_val = features[val_idx]
        y_val = softlabels[val_idx]
        
        return X_train, y_train, X_val, y_val
        
    except Exception as e:
        print(f"加载特征时出错: {e}")
        return None, None, None, None