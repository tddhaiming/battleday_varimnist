# utils/metrics.py
import torch
import numpy as np
from scipy.stats import spearmanr

def compute_nll(preds, targets):
    """计算负对数似然"""
    preds = np.clip(preds, 1e-8, 1 - 1e-8)
    return -np.mean(np.sum(targets * np.log(preds), axis=1))

# def compute_spearman(preds, targets):
#     """计算 Spearman 相关系数"""
#     preds_max = np.argmax(preds, axis=1)
#     targets_max = np.argmax(targets, axis=1)
#     if len(preds_max) > 1 and len(np.unique(preds_max)) > 1 and len(np.unique(targets_max)) > 1:
#         correlation, _ = spearmanr(preds_max, targets_max)
#         return correlation
#     return 0.0
def compute_spearman_per_image(preds, targets):
    if preds.shape != targets.shape:
        raise ValueError("preds and targets must have the same shape")
    
    n_samples = preds.shape[0]
    spearman_coeffs = []
    
    for i in range(n_samples):
        pred_dist = preds[i]      # 形状 (C,) - 模型对样本 i 的预测分布
        target_dist = targets[i]  # 形状 (C,) - 人类对样本 i 的响应分布
        
        # 检查是否有足够的变化来计算相关性
        # 如果所有值都相同，则无法计算有意义的相关系数
        if len(np.unique(pred_dist)) > 1 and len(np.unique(target_dist)) > 1:
            try:
                # 计算两个分布向量之间的 Spearman 相关系数
                corr, _ = spearmanr(pred_dist, target_dist)
                # 检查相关系数是否有效（避免 NaN 或 inf）
                if np.isfinite(corr):
                    spearman_coeffs.append(corr)
            except Exception:
                # 如果计算出错（例如数据问题），则跳过该样本
                pass
    
    if spearman_coeffs:
        # 返回所有样本的平均 Spearman 相关系数
        return np.mean(spearman_coeffs)
    else:
        # 如果没有有效的系数（例如所有分布都完全相同），返回 0
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
