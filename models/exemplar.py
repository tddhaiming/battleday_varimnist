# models/exemplar.py
import torch
import torch.nn as nn
import numpy as np

class ExemplarModel(nn.Module):
    def __init__(self, exemplars, labels, feature_dim=512):
        super().__init__()
        self.exemplars = nn.Parameter(torch.tensor(exemplars), requires_grad=False)  # (N, D)
        self.labels = torch.tensor(labels, dtype=torch.long)  # (N,)
        self.feature_dim = feature_dim
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.gamma = nn.Parameter(torch.tensor(1.0))
        self.Sigma_inv = nn.Parameter(torch.ones(feature_dim))

    def forward(self, x):
        # x: (B, D)
        # 计算查询样本与所有样本的距离
        diff = x.unsqueeze(1) - self.exemplars.unsqueeze(0)  # (B, N, D)
        weighted_diff = diff * self.Sigma_inv  # (B, N, D)
        distances = torch.sum(weighted_diff * diff, dim=2)  # (B, N)
        
        # 计算注意力权重
        attention_weights = torch.exp(-self.beta * distances)  # (B, N)
        
        # 聚合同类样本的权重
        logits = []
        for c in range(10):
            mask = (self.labels == c).to(x.device)  # (N,)
            score = torch.sum(attention_weights[:, mask], dim=1)  # (B,)
            logits.append(score)
        
        logits = torch.stack(logits, dim=1)  # (B, 10)
        return torch.softmax(self.gamma * logits, dim=1)