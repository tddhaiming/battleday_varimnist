# # models/exemplar.py
# import torch
# import torch.nn as nn
# import numpy as np

# class ExemplarModel(nn.Module):
#     def __init__(self, exemplars, labels, feature_dim=512):
#         super().__init__()
#         # 使用 register_buffer 更合适，因为它明确表示这是模型的一部分但不是可学习参数
#         # 并且允许更灵活地管理其设备位置（例如，可以放在 CPU 上）
#         # 初始时可以放在 CPU 上以节省 GPU 内存
#         self.register_buffer('exemplars', torch.tensor(exemplars, dtype=torch.float32)) # (N, D) - 默认在 CPU
#         self.register_buffer('labels', torch.tensor(labels, dtype=torch.long))          # (N,) - 默认在 CPU
#         self.feature_dim = feature_dim
#         self.beta = nn.Parameter(torch.tensor(1.0))
#         self.gamma = nn.Parameter(torch.tensor(1.0))
#         self.Sigma_inv = nn.Parameter(torch.ones(feature_dim))
#         # 定义内部批处理大小，用于计算与 exemplars 的距离
#         # 你可以根据服务器 GPU 的显存大小调整这个值
#         # 例如，如果服务器 GPU 显存很大 (如 24GB, 32GB)，可以设置得更大 (如 4096, 8192)
#         self.internal_batch_size = 2048 # 这是一个可以调整的超参数

#     def set_internal_batch_size(self, batch_size):
#         """允许在运行时调整内部批处理大小"""
#         self.internal_batch_size = batch_size

#     def forward(self, x):
#         """
#         前向传播，计算输入 x 与所有 exemplars 的相似度，并聚合得到类别 logits。
        
#         Args:
#             x (torch.Tensor): 输入查询样本, 形状 (B, D)
            
#         Returns:
#             torch.Tensor: 输出概率分布, 形状 (B, num_classes=10)
#         """
#         device = x.device
#         batch_size = x.shape[0]
#         num_exemplars = self.exemplars.shape[0]
#         num_classes = 10 # 假设是 10 个数字类别

#         # 确保 exemplars 和 labels 在计算时与输入 x 在同一设备上
#         # 注意：如果 exemplars 很大，全部移动到 GPU 仍然可能 OOM
#         # 更稳健的做法是在内部循环中分批移动
#         # 但为了简化，这里假设 exemplars 已经被管理得当（例如，大小适中或也进行了分批处理）
#         # 如果 exemplars 本身太大，应该在初始化时就考虑只加载部分或使用外部存储
        
#         # 为了安全，确保 exemplars 和 labels 在正确的设备上（通常是 CPU，计算时再移动）
#         # 但如果它们已经被移到 GPU 了，这不会有问题
#         # 这里我们假设它们在 CPU 上，或者至少可以被处理
        
#         logits = []
#         for i in range(batch_size):
#             query = x[i:i+1] # (1, D) - 取出单个查询样本
#             similarities = [] # 存储该查询样本与所有 exemplars 的相似度

#             # --- 关键修改：分批处理与 exemplars 的距离计算 ---
#             for j in range(0, num_exemplars, self.internal_batch_size):
#                 # 1. 分批将一部分 exemplars 移动到 query 所在的设备 (通常是 GPU)
#                 exemplar_batch = self.exemplars[j:j+self.internal_batch_size].to(device) # (B_e, D)
#                 label_batch = self.labels[j:j+self.internal_batch_size].to(device)       # (B_e,)
                
#                 # 2. 计算查询样本与这一批 exemplars 的距离
#                 diff = query - exemplar_batch  # (1, B_e, D)
#                 # 注意：Sigma_inv 也需要在相同设备上
#                 weighted_diff = diff * self.Sigma_inv.to(device)  # (1, B_e, D)
#                 # print(f"weighted_diff shape: {weighted_diff.shape}")
#                 # print(f"diff shape: {diff.shape}")
#                 #distances = torch.sum(weighted_diff * diff, dim=2)  # (1, B_e)
#                 product = weighted_diff * diff
#                 # 使用安全的维度索引
#                 sum_dim = min(2, product.dim() - 1)
#                 distances = torch.sum(product, dim=sum_dim)
#                 # 3. 计算相似度 (注意力权重)
#                 batch_similarities = torch.exp(-self.beta.to(device) * distances.squeeze(0))  # (B_e,)
                
#                 # 4. 保存这批的相似度
#                 similarities.append(batch_similarities)
            
#             # --- 聚合所有批次的相似度 ---
#             # 将所有批次计算出的相似度拼接起来
#             all_similarities = torch.cat(similarities, dim=0) # (N,)
            
#             # --- 按类别聚合 ---
#             batch_logits = []
#             # 注意：self.labels 也需要在 device 上进行比较
#             # 但由于它可能很大，我们只移动需要的部分（在循环内部已完成）
#             # 或者，我们可以预先将它移动到 device（如果内存允许）
#             # 更安全的方式是在内部循环中处理，如上所示
            
#             # 为了效率，可以预先将 labels 移动到 device (如果不大)
#             # 但在这里我们假设已经在内部循环中处理了 device 问题
#             labels_on_device = self.labels.to(device)
            
#             for c in range(num_classes):
#                 # 创建类别 c 的掩码
#                 mask = (labels_on_device == c) # (N,)
#                 # 聚合属于类别 c 的所有 exemplar 的相似度
#                 # all_similarities 是 (N,), mask 是 (N,)
#                 score = torch.sum(all_similarities[mask]) # ()
#                 batch_logits.append(score)
            
#             # 将该查询样本的 logits 收集起来
#             logits.append(torch.stack(batch_logits)) # [(10,), (10,), ...]
        
#         # 将 batch 中所有查询样本的 logits 堆叠成一个张量
#         logits = torch.stack(logits) # (B, 10)
        
#         # 应用 softmax 得到最终概率
#         return torch.softmax(self.gamma.to(device) * logits, dim=1)

#     # --- 可选：提供一个方法来将 exemplars 移动到指定设备 ---
#     # 注意：对于非常大的 exemplars，直接移动可能不现实
#     def to_device(self, device):
#         """将模型的可学习参数和缓冲区移动到指定设备"""
#         # nn.Module.to() 会自动处理 parameters 和 buffers
#         # 但由于 exemplars 可能很大，这个方法可能需要特殊处理
#         # 默认的 to() 应该足够处理 beta, gamma, Sigma_inv 和 buffers (如果它们能放得下)
#         return super().to(device)

# # --- 重要说明 ---
# # 1. `internal_batch_size` 是关键。你需要根据你的 GPU 显存来调整它。
# #    显存越大，可以设置得越大，计算效率越高。
# #    例如，在 Colab 的 15GB GPU 上可能用 1024 或 2048，
# #    在 24GB 的服务器 GPU 上可以尝试 4096 或更高。
# # 2. 这种修改后的 `forward` 方法显著降低了单次计算的内存峰值，
# #    因为它不会同时创建 (B, N, D) 这样巨大的中间张量。
# # 3. 代价是计算速度可能会稍微慢一些，因为引入了内部循环。
# #    但在显存受限的情况下，这是必要的权衡。
# # 4. 确保在训练和评估脚本中，如果需要，可以调用 `model.set_internal_batch_size(new_size)`
# #    来动态调整以适应不同的硬件环境。
# models/exemplar.py
# models/exemplar.py
"""
Exemplar models implemented per Battleday et al. (2020).

Contains:
- ExemplarNoAttention: uniform feature weights, learnable beta
- ExemplarAttention: learnable attention weights over feature dimensions w, and beta

Notes:
- logits returned are of shape (batch, num_classes), corresponding to γ * log S(y, C)
- For very large exemplar sets, use the block-based wrapper in scripts/train_exemplar.py to avoid OOM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# === 旧代码注释（如果你之前有版本，可以粘贴并注释） ===
# class ExemplarModelOld:
#     def __init__(...):
#         ...
#     def forward(self, x):
#         ...
# === 旧代码结束 ===


class ExemplarBase(nn.Module):
    def __init__(self, exemplar_features: torch.Tensor, exemplar_labels: torch.Tensor,
                 feature_dim: int, num_classes: int):
        """
        exemplar_features: (Ne, D)
        exemplar_labels: (Ne,) long
        """
        super().__init__()
        self.register_buffer("exemplars", exemplar_features)  # (Ne, D)
        self.register_buffer("exemplar_labels", exemplar_labels)  # (Ne,)
        self.feature_dim = feature_dim
        self.num_classes = num_classes

    def pairwise_sqdist(self, x, weight_diag):
        """
        compute (batch x Ne) squared Mahalanobis-like distances,
        weight_diag: (D,) or (Ne, D)
        returns: (B, Ne)
        """
        B = x.shape[0]
        Ne = self.exemplars.shape[0]
        # Expand
        x_exp = x.unsqueeze(1).expand(B, Ne, self.feature_dim)  # (B, Ne, D)
        e_exp = self.exemplars.unsqueeze(0).expand(B, Ne, self.feature_dim)
        diff2 = (x_exp - e_exp) ** 2  # (B, Ne, D)
        # weight_diag: (D,) or (Ne,D)
        if weight_diag.dim() == 1:
            d2 = (diff2 * weight_diag.view(1, 1, -1)).sum(dim=2)  # (B, Ne)
        else:
            d2 = (diff2 * weight_diag.unsqueeze(0)).sum(dim=2)
        return d2  # squared distances


# === 新代码开始 ===
class ExemplarNoAttention(ExemplarBase):
    def __init__(self, exemplar_features, exemplar_labels, feature_dim, num_classes, init_beta=1.0):
        super().__init__(exemplar_features, exemplar_labels, feature_dim, num_classes)
        # beta scalar >0 via softplus
        self.beta_raw = nn.Parameter(torch.log(torch.exp(torch.tensor(init_beta)) - 1.0))

    def beta(self):
        return F.softplus(self.beta_raw)

    def forward(self, x, gamma=1.0):
        """
        Compute logits = γ * log S(y, C), where
        S(y,C) = sum_{e in C} exp(-β * d(y,e))
        - weight diag is identity (uniform)
        """
        device = x.device
        weight_diag = torch.ones(self.feature_dim, device=device)
        d2 = self.pairwise_sqdist(x, weight_diag)  # (B, Ne)
        beta = self.beta()
        sims = torch.exp(-beta * d2)  # (B, Ne)

        # aggregate per class using scatter_add
        labels = self.exemplar_labels.to(device).long()  # (Ne,)
        labels_exp = labels.unsqueeze(0).expand(x.shape[0], labels.shape[0])  # (B, Ne)
        class_sims = torch.zeros(x.shape[0], self.num_classes, device=device)
        class_sims = class_sims.scatter_add_(1, labels_exp, sims)  # sum sims per class

        logits = gamma * torch.log(class_sims + 1e-12)
        return logits


class ExemplarAttention(ExemplarBase):
    def __init__(self, exemplar_features, exemplar_labels, feature_dim, num_classes,
                 init_beta=1.0, init_w=None):
        super().__init__(exemplar_features, exemplar_labels, feature_dim, num_classes)
        if init_w is None:
            init_w = torch.ones(feature_dim)
        # parameterize w via raw; apply softmax to get positive sum-to-1 weights
        self.w_raw = nn.Parameter(torch.log(init_w + 1e-6))
        self.beta_raw = nn.Parameter(torch.log(torch.exp(torch.tensor(init_beta)) - 1.0))

    def beta(self):
        return F.softplus(self.beta_raw)

    def w(self):
        return F.softmax(self.w_raw, dim=0)  # (D,)

    def forward(self, x, gamma=1.0):
        """
        Weighted squared distance: sum_d w_d * (x_d - e_d)^2
        Then same aggregation as no-attention: S = sum exp(-beta * d)
        """
        device = x.device
        w = self.w().to(device)  # (D,)
        d2 = self.pairwise_sqdist(x, w)  # (B, Ne)
        beta = self.beta()
        sims = torch.exp(-beta * d2)  # (B, Ne)

        labels = self.exemplar_labels.to(device).long()
        labels_exp = labels.unsqueeze(0).expand(x.shape[0], labels.shape[0])
        class_sims = torch.zeros(x.shape[0], self.num_classes, device=device)
        class_sims = class_sims.scatter_add_(1, labels_exp, sims)
        logits = gamma * torch.log(class_sims + 1e-12)
        return logits
# === 新代码结束 ===



