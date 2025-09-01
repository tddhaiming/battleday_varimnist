# models/exemplar.py
""" Exemplar models:
- ExemplarNoAttention: 缩放欧氏距离(beta * I)
- ExemplarAttention: 对角 Mahalanobis with attention weights w(shared across classes), plus beta scaling

Implementation notes:
- For efficiency we vectorize distance calculations with torch.cdist when possible.
- For attention model we compute weighted squared distance: sum_k w_k * (x_k - exemplar_k)^2
- We provide a method to cache exemplar bank on GPU to avoid repeated host->device copies(recommended on A100)
- Support chunking over exemplars to reduce peak memory if exemplar bank is huge
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-9

class ExemplarBase(nn.Module):
    def __init__(self, feature_dim, num_classes, exemplar_feats, exemplar_labels):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        # expect exemplar_feats: Tensor(N_ex, D)
        self.register_buffer('exemplar_feats', exemplar_feats) # may be on cpu initially
        self.register_buffer('exemplar_labels', exemplar_labels.long())
        # gamma(response scaling) always learnable in paper; beta is distance scaling(learnable)
        self.gamma_unconstrained = nn.Parameter(torch.tensor(1.0)) # param -> softplus
        self.beta_unconstrained = nn.Parameter(torch.tensor(1.0))
        # convenience
        self._cached_on_device = None

    def get_gamma(self):
        return F.softplus(self.gamma_unconstrained) + EPS

    def get_beta(self):
        return F.softplus(self.beta_unconstrained) + EPS

    def cache_exemplars_to_device(self, device):
        # 新代码：把 exemplar bank 一次性搬到 GPU（若显存允许）
        if self.exemplar_feats.numel() == 0:
            return
        self.exemplar_feats = self.exemplar_feats.to(device)
        self.exemplar_labels = self.exemplar_labels.to(device)
        self._cached_on_device = device

    def _ensure_exemplars_device(self, device):
        if self.exemplar_feats.device != device:
            # don't modify buffer in-place(they are registered buffers) but move a local copy
            return self.exemplar_feats.to(device), self.exemplar_labels.to(device)
        return self.exemplar_feats, self.exemplar_labels

class ExemplarNoAttention(ExemplarBase):
    """ Exemplar(no attention): scaled Euclidean distance with single beta
    S(y, C) = sum_{x in C} exp(-beta * ||y - x||^2)
    logits = gamma * log( S(y, C) + eps)
    """
    def __init__(self, feature_dim, num_classes, exemplar_feats, exemplar_labels):
        super().__init__(feature_dim, num_classes, exemplar_feats, exemplar_labels)

    def forward(self, x, chunk_size=None, use_gpu=True):
        """ x:(B, D)
        chunk_size: if not None, iterate over exemplar bank in chunks of this size to reduce memory
        returns logits(B, C)
        """
        device = x.device if use_gpu else torch.device('cpu')
        beta = self.get_beta()
        gamma = self.get_gamma()

        N = self.exemplar_feats.shape[0]
        B = x.shape[0]
        C = self.num_classes

        # prepare storage for class sums
        class_sum = torch.zeros(B, C, device=device)

        # ensure exemplar tensors on device(but avoid in-place move if possible)
        ex_feats, ex_labels = self._ensure_exemplars_device(device)

        if chunk_size is None:
            # compute pairwise squared euclid distances using cdist
            # cdist gives euclidean distances; we square them
            d = torch.cdist(x, ex_feats.to(device), p=2) # (B, N)
            d2 = d * d
            sim = torch.exp(- beta * d2) # (B, N)
            # aggregate per-class: scatter_add
            idx = ex_labels.unsqueeze(0).expand(B, -1)
            class_sum = class_sum.scatter_add_(1, idx, sim)
        else:
            for i in range(0, N, chunk_size):
                j = min(N, i + chunk_size)
                chunk = ex_feats[i:j].to(device)
                chunk_labels = ex_labels[i:j].to(device)
                d = torch.cdist(x, chunk, p=2)
                d2 = d * d
                sim = torch.exp(- beta * d2)
                idx = chunk_labels.unsqueeze(0).expand(B, j-i)
                class_sum = class_sum.scatter_add_(1, idx, sim)

        logits = gamma * torch.log(class_sum + EPS)
        return logits

class ExemplarAttention(ExemplarBase):
    """ Exemplar with attention weights w(shared across classes) and beta scaling.

    Attention weights: param w_unconstrained(D,) -> softmax -> positive weights that sum to 1(as in paper)
    Distance: sum_k w_k*(x_k - exemplar_k)^2
    S(y, C) = sum_{x in C} exp(- beta * weighted_sq_dist)
    logits = gamma * log(S + eps)

    优化点：改进 w_unconstrained 的初始化，使其不偏向均匀分布，可能有助于学习。
    修复点：使用适用于 1D 张量的初始化方法，避免 Xavier 初始化错误。
    """
    def __init__(self, feature_dim, num_classes, exemplar_feats, exemplar_labels):
        super().__init__(feature_dim, num_classes, exemplar_feats, exemplar_labels)
        # --- 修改点：使用适用于 1D 参数的初始化方法 ---
        # Xavier 初始化不适用于 1D 张量。
        # 改用均匀分布初始化，范围可以是 [-bound, bound]，其中 bound 是一个小的正值。
        # sqrt(1/feature_dim) 是 Xavier 初始化中用于均匀分布的 bound 的一种启发式。
        # 或者使用更简单的固定小范围，例如 0.1
        self.w_unconstrained = nn.Parameter(torch.randn(feature_dim) * 0.01)
        # 使用 Xavier 风格的 bound
        bound = 1.0 / (feature_dim ** 0.5)
        nn.init.uniform_(self.w_unconstrained, -bound, bound)
        # 或者使用固定的较小范围 (例如注释掉上面两行，取消注释下面一行)
        # nn.init.uniform_(self.w_unconstrained, -0.1, 0.1)
        # --- 结束修改点 ---

    def get_w(self):
        # softmax to get positive weights sum to 1
        return F.softmax(self.w_unconstrained, dim=0) + EPS # EPS added after softmax is less critical but kept for consistency

    def forward(self, x, chunk_size=None, use_gpu=True):
        device = x.device if use_gpu else torch.device('cpu')
        beta = self.get_beta()
        gamma = self.get_gamma()
        w = self.get_w().to(device) # (D,)

        N = self.exemplar_feats.shape[0]
        B = x.shape[0]
        C = self.num_classes

        class_sum = torch.zeros(B, C, device=device)
        ex_feats, ex_labels = self._ensure_exemplars_device(device)

        if chunk_size is None:
            # compute weighted squared distances in vectorized way: (x-ex)^2 * w
            # expand x(B,1,D) and ex_feats(1,N,D)
            dif = x.unsqueeze(1) - ex_feats.to(device).unsqueeze(0) # (B, N, D)
            d2w = (dif * dif) * w.unsqueeze(0).unsqueeze(0) # (B, N, D)
            d = d2w.sum(dim=2) # (B, N)
            sim = torch.exp(- beta * d)
            idx = ex_labels.unsqueeze(0).expand(B, -1)
            class_sum = class_sum.scatter_add_(1, idx, sim)
        else:
            for i in range(0, N, chunk_size):
                j = min(N, i + chunk_size)
                chunk = ex_feats[i:j].to(device)
                chunk_labels = ex_labels[i:j].to(device)
                dif = x.unsqueeze(1) - chunk.unsqueeze(0)
                d2w = (dif * dif) * w.unsqueeze(0).unsqueeze(0)
                d = d2w.sum(dim=2)
                sim = torch.exp(- beta * d)
                idx = chunk_labels.unsqueeze(0).expand(B, j-i)
                class_sum = class_sum.scatter_add_(1, idx, sim)

        logits = gamma * torch.log(class_sum + EPS)
        return logits
