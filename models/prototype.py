# # models/prototype.py
# import torch
# import torch.nn as nn
# import numpy as np

# class PrototypeModel(nn.Module):
#     def __init__(self, num_classes=10, feature_dim=512, mode="classic"):
#         super().__init__()
#         self.num_classes = num_classes
#         self.feature_dim = feature_dim
#         self.mode = mode
        
#         # 初始化原型向量
#         self.prototypes = nn.Parameter(torch.randn(num_classes, feature_dim))
        
#         # 根据模式设置协方差矩阵
#         if mode == "linear":
#             self.Sigma_inv = nn.Parameter(torch.ones(feature_dim))
#         elif mode == "quadratic":
#             self.Sigma_inv = nn.Parameter(torch.ones(num_classes, feature_dim))
#         else:  # classic
#             self.register_buffer("Sigma_inv", torch.ones(feature_dim))
        
#         self.gamma = nn.Parameter(torch.tensor(1.0))

#     def mahalanobis_distance(self, x, prototype, class_idx=None):
#         """计算马氏距离"""
#         diff = x - prototype  # (B, D)
        
#         if self.mode == "quadratic" and class_idx is not None:
#             weighted = diff * self.Sigma_inv[class_idx]
#         elif self.mode == "linear":
#             weighted = diff * self.Sigma_inv
#         else:
#             weighted = diff  # Euclidean distance
        
#         return torch.sum(weighted * diff, dim=1)  # (B,)

#     def forward(self, x):
#         # x: (B, D)
#         distances = []
#         for i in range(self.num_classes):
#             dist = self.mahalanobis_distance(x, self.prototypes[i], i)
#             distances.append(dist)
        
#         distances = torch.stack(distances, dim=1)  # (B, C)
#         logits = -self.gamma * distances
#         return torch.softmax(logits, dim=1)
# models/prototype.py
"""
Prototype models (Classic / Linear / Quadratic) implemented per Battleday et al. (2020).

Place this file in: models/prototype.py

Notes:
- prototypes: tensor of shape (num_classes, feature_dim) (empirical means)
- All new code blocks are delimited with "# === 新代码开始 ===" and "# === 新代码结束 ==="
- If you want prototypes to be learnable, set flags (not done by default).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# === 旧代码注释（如果之前仓库中有旧实现，可在此处保留/粘贴并注释） ===
# class PrototypeModelOld:
#     def __init__(...):
#         ...
#     def forward(self, x):
#         ...
# === 旧代码结束 ===


class BasePrototype(nn.Module):
    def __init__(self, prototypes: torch.Tensor, feature_dim: int, num_classes: int):
        """
        prototypes: tensor (num_classes, feature_dim)  -- empirical prototypes (means)
        """
        super().__init__()
        self.register_buffer("prototypes", prototypes)  # fixed empirical prototypes
        self.feature_dim = feature_dim
        self.num_classes = num_classes

    def mahalanobis_diag(self, x, inv_diag):
        """
        compute Mahalanobis distance (squared) with diagonal inverse covariance.
        x: (batch, feature_dim)
        inv_diag: shape (feature_dim,) or (num_classes, feature_dim)
        returns: d (batch, num_classes)
        """
        B = x.shape[0]
        C = self.prototypes.shape[0]
        # expand to (B, C, D)
        x_exp = x.unsqueeze(1).expand(B, C, self.feature_dim)   # (B,C,D)
        p_exp = self.prototypes.unsqueeze(0).expand(B, C, self.feature_dim)
        diff = x_exp - p_exp  # (B,C,D)
        # inv_diag: (D,) or (C,D)
        if inv_diag.dim() == 1:
            # (D,) -> broadcast
            d2 = (diff * diff) * inv_diag.view(1, 1, -1)
        else:
            # (C,D) -> expand to (B,C,D)
            d2 = (diff * diff) * inv_diag.unsqueeze(0)
        d = d2.sum(dim=2)  # (B, C) squared Mahalanobis distance
        return d


# === 新代码开始 ===
class ClassicPrototype(BasePrototype):
    """
    Classic: Σ = σ^2 I (identity) -> Euclidean distance.
    No learnable variance parameters. Only gamma (response scaling) provided at forward.
    """
    def __init__(self, prototypes, feature_dim, num_classes):
        super().__init__(prototypes, feature_dim, num_classes)

    def forward(self, x, gamma=1.0):
        inv_diag = torch.ones(self.feature_dim, device=x.device)  # identity inverse diag
        d = self.mahalanobis_diag(x, inv_diag)  # (B, C)
        logits = -gamma * d  # similarity ~ exp(-d); logits = -gamma * d
        return logits


class LinearPrototype(BasePrototype):
    """
    Linear: shared diagonal inverse covariance Σ^{-1} = diag(c) across classes.
    Learnable parameter: c (D,) constrained >0.
    """
    def __init__(self, prototypes, feature_dim, num_classes, init_c=None):
        super().__init__(prototypes, feature_dim, num_classes)
        if init_c is None:
            init_c = torch.ones(feature_dim)
        # parameterize c via inverse-softplus: store raw and apply softplus
        self.c_raw = nn.Parameter(torch.log(torch.exp(init_c) - 1.0))

    def inv_diag(self):
        # returns positive vector (D,)
        return F.softplus(self.c_raw)

    def forward(self, x, gamma=1.0):
        inv_d = self.inv_diag()  # (D,)
        d = self.mahalanobis_diag(x, inv_d)  # (B,C)
        logits = -gamma * d
        return logits


class QuadraticPrototype(BasePrototype):
    """
    Quadratic: per-class diagonal inverse covariance Σ_C^{-1} = diag(ci) for each class i.
    Learnable parameter: ci for each class (C x D), constrained >0.
    """
    def __init__(self, prototypes, feature_dim, num_classes, init_ci=None):
        super().__init__(prototypes, feature_dim, num_classes)
        if init_ci is None:
            # default initialize to ones
            init_ci = torch.ones(num_classes, feature_dim)
        self.ci_raw = nn.Parameter(torch.log(torch.exp(init_ci) - 1.0))  # (C, D)

    def inv_diag(self):
        # returns (C, D)
        return F.softplus(self.ci_raw)

    def forward(self, x, gamma=1.0):
        inv_diags = self.inv_diag()  # (C, D)
        d = self.mahalanobis_diag(x, inv_diags)  # (B, C)
        logits = -gamma * d
        return logits
# === 新代码结束 ===
