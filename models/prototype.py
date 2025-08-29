# models/prototype.py
"""
Prototype models (Classic / Linear / Quadratic) per Battleday et al., 2020.

实现要点（与论文对应）：
- Classic: Σ = σ^2 I（等价于欧氏距离），只有 gamma 可学（这里实现 gamma 可学或可固定）
- Linear: Σ = diag(1/c)（论文中写成 Σ^-1 = diag(c)），共享对角精度向量 c（全局可学）
- Quadratic: 每类 Σ_c = diag(1/c_c)，即每类有独立对角精度向量 c_c

接口：
- 初始化传入 feature_dim, num_classes
- build_prototypes_from_features(features, labels) : 从训练集中计算 μ_c（prototype）并保存
- forward(x) -> logits (B, C)  输出为可以直接用于 CrossEntropyLoss 的 logits（越大越倾向该类）

注意：
- c 参数采用正值参数化（softplus），确保正定
- gamma 为缩放参数（>0），参数化为 softplus 以保证正
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-9

class BasePrototype(nn.Module):
    def __init__(self, feature_dim, num_classes, learn_gamma=False, init_gamma=1.0):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        # prototypes μ_c stored as buffer (C, D)
        self.register_buffer('prototypes', torch.zeros(num_classes, feature_dim))
        # gamma: response scaling parameter (positive)
        if learn_gamma:
            self.gamma_unconstrained = nn.Parameter(torch.log(torch.exp(torch.tensor(init_gamma)) - 1.0))  # inverse softplus init
            self.learn_gamma = True
        else:
            self.register_buffer('gamma_const', torch.tensor(float(init_gamma)))
            self.learn_gamma = False

    def get_gamma(self):
        if self.learn_gamma:
            return F.softplus(self.gamma_unconstrained) + EPS
        else:
            return self.gamma_const

    def build_prototypes_from_features(self, features, labels):
        """
        features: Tensor (N, D)
        labels: LongTensor (N,)
        -> compute per-class mean prototypes and store to self.prototypes (on cpu buffer)
        """
        C = self.num_classes
        D = self.feature_dim
        device = features.device
        prot = torch.zeros(C, D, device=device)
        for c in range(C):
            mask = (labels == c)
            if mask.any():
                prot[c] = features[mask].mean(dim=0)
            else:
                prot[c] = torch.zeros(D, device=device)
        # store on cpu buffer for consistency
        self.prototypes = prot.cpu()

    def forward(self, x):
        """
        x: (B, D)
        returns logits (B, C)
        Default implementation: negative Mahalanobis distance using prototypes
        Subclasses override distance computation (e.g., with learned c vectors)
        """
        raise NotImplementedError("Use subclass implementations")


class ClassicPrototype(BasePrototype):
    """
    Classic prototype: Euclidean distance (Σ = σ^2 I fixed).
    Only gamma (response scaling) optionally learned here.
    distance = ||x - mu||^2  (we use squared or euclid; we use squared)
    logits = gamma * (-distance)
    """
    def __init__(self, feature_dim, num_classes, learn_gamma=True, init_gamma=1.0):
        super().__init__(feature_dim, num_classes, learn_gamma=learn_gamma, init_gamma=init_gamma)

    def forward(self, x):
        # check dimension
        if x.dim() != 2 or x.shape[1] != self.feature_dim:
            raise ValueError(f'Input feature dim mismatch {x.shape} vs {self.feature_dim}')
        proto = self.prototypes.to(x.device)  # (C, D)
        # squared euclid distances: (B, C)
        d2 = torch.cdist(x, proto, p=2) ** 2
        gamma = self.get_gamma()
        logits = - gamma * d2
        return logits


class LinearPrototype(BasePrototype):
    """
    Linear prototype: shared diagonal Mahalanobis (Σ^-1 = diag(c)), learn c (D,)
    Parameters learned: c (D,) positive, gamma
    distance = sum_k c_k * (x_k - mu_k)^2
    """
    def __init__(self, feature_dim, num_classes, learn_gamma=True, init_gamma=1.0, init_c=1.0):
        super().__init__(feature_dim, num_classes, learn_gamma=learn_gamma, init_gamma=init_gamma)
        # parameterize c via softplus to ensure positive
        self.c_unconstrained = nn.Parameter(torch.ones(feature_dim) * float(init_c))  # # 新代码: 可学对角精度向量（共享）
        # note: no prototypes stored here initially; build from data as usual

    def get_c(self):
        # positive vector
        return F.softplus(self.c_unconstrained) + EPS

    def forward(self, x):
        if x.dim() != 2 or x.shape[1] != self.feature_dim:
            raise ValueError(f'Input feature dim mismatch {x.shape} vs {self.feature_dim}')
        proto = self.prototypes.to(x.device)  # (C, D)
        # compute per-class Mahalanobis-like squared distance efficiently:
        # (x - mu)^2 -> expand: (B, C, D)
        # compute weighted sum over D by c
        c = self.get_c().to(x.device)  # (D,)
        # compute squared diffs then weight
        # x: (B, D), proto: (C, D)
        # use broadcasting
        dif = x.unsqueeze(1) - proto.unsqueeze(0)  # (B, C, D)
        d2w = (dif * dif) * c.unsqueeze(0).unsqueeze(0)  # (B, C, D)
        d = d2w.sum(dim=2)  # (B, C)
        gamma = self.get_gamma()
        logits = - gamma * d
        return logits


class QuadraticPrototype(BasePrototype):
    """
    Quadratic prototype: per-class diagonal Mahalanobis (Σ_c^-1 = diag(c_c)), learn c_c (C, D)
    Parameters learned: c_c (C, D), gamma
    distance for class c: sum_k c_c[k] * (x_k - mu_c[k])^2
    """
    def __init__(self, feature_dim, num_classes, learn_gamma=True, init_gamma=1.0, init_cc=1.0):
        super().__init__(feature_dim, num_classes, learn_gamma=learn_gamma, init_gamma=init_gamma)
        # per-class per-dim unconstrained parameters
        # shape (C, D)
        self.cc_unconstrained = nn.Parameter(torch.ones(num_classes, feature_dim) * float(init_cc))  # # 新代码: per-class diagonal precision

    def get_cc(self):
        return F.softplus(self.cc_unconstrained) + EPS  # (C, D)

    def forward(self, x):
        if x.dim() != 2 or x.shape[1] != self.feature_dim:
            raise ValueError(f'Input feature dim mismatch {x.shape} vs {self.feature_dim}')
        proto = self.prototypes.to(x.device)  # (C, D)
        cc = self.get_cc().to(x.device)  # (C, D)
        # compute (B, C, D) diffs
        dif = x.unsqueeze(1) - proto.unsqueeze(0)
        d2w = (dif * dif) * cc.unsqueeze(0)  # broadcast (B,C,D)
        d = d2w.sum(dim=2)  # (B,C)
        gamma = self.get_gamma()
        logits = - gamma * d
        return logits
