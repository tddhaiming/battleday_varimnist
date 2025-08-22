# models/prototype.py
import torch
import torch.nn as nn
import numpy as np

class PrototypeModel(nn.Module):
    def __init__(self, num_classes=10, feature_dim=512, mode="classic"):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.mode = mode
        
        # 初始化原型向量
        self.prototypes = nn.Parameter(torch.randn(num_classes, feature_dim))
        
        # 根据模式设置协方差矩阵
        if mode == "linear":
            self.Sigma_inv = nn.Parameter(torch.ones(feature_dim))
        elif mode == "quadratic":
            self.Sigma_inv = nn.Parameter(torch.ones(num_classes, feature_dim))
        else:  # classic
            self.register_buffer("Sigma_inv", torch.ones(feature_dim))
        
        self.gamma = nn.Parameter(torch.tensor(1.0))

    def mahalanobis_distance(self, x, prototype, class_idx=None):
        """计算马氏距离"""
        diff = x - prototype  # (B, D)
        
        if self.mode == "quadratic" and class_idx is not None:
            weighted = diff * self.Sigma_inv[class_idx]
        elif self.mode == "linear":
            weighted = diff * self.Sigma_inv
        else:
            weighted = diff  # Euclidean distance
        
        return torch.sum(weighted * diff, dim=1)  # (B,)

    def forward(self, x):
        # x: (B, D)
        distances = []
        for i in range(self.num_classes):
            dist = self.mahalanobis_distance(x, self.prototypes[i], i)
            distances.append(dist)
        
        distances = torch.stack(distances, dim=1)  # (B, C)
        logits = -self.gamma * distances
        return torch.softmax(logits, dim=1)