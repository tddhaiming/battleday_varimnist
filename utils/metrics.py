# utils/metrics.py
import torch
import numpy as np
from scipy.stats import spearmanr

def compute_nll(preds, targets):
    """计算负对数似然"""
    preds = np.clip(preds, 1e-8, 1 - 1e-8)
    return -np.mean(np.sum(targets * np.log(preds), axis=1))

def compute_spearman(preds, targets):
    """计算 Spearman 相关系数"""
    preds_max = np.argmax(preds, axis=1)
    targets_max = np.argmax(targets, axis=1)
    if len(preds_max) > 1 and len(np.unique(preds_max)) > 1 and len(np.unique(targets_max)) > 1:
        correlation, _ = spearmanr(preds_max, targets_max)
        return correlation
    return 0.0

def compute_aic(nll, k):
    """计算 AIC"""
    return 2 * k - 2 * (-nll)  # NLL 是负对数似然

def evaluate_model(model, features, softlabels, device):
    """评估模型性能"""
    model.eval()
    with torch.no_grad():
        X = torch.tensor(features).float().to(device)
        preds = model(X).cpu().numpy()
    
    nll = compute_nll(preds, softlabels)
    spearman = compute_spearman(preds, softlabels)
    
    # 计算参数数量
    k = sum(p.numel() for p in model.parameters())
    aic = compute_aic(nll, k)
    
    return {
        "nll": nll,
        "spearman": spearman,
        "aic": aic,
        "params": k
    }