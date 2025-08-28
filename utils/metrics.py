# # utils/metrics.py
# import torch
# import numpy as np
# from scipy.stats import spearmanr
# def compute_top1_accuracy(preds, targets):
#     """
#     计算 Top-1 Accuracy (基于 argmax)。
#     衡量模型预测的最可能类别与人类最倾向类别匹配的比例。
#     """
#     if preds.shape != targets.shape:
#         raise ValueError("preds and targets must have the same shape")

#     # 获取模型预测的类别 (每个样本预测概率最高的类别)
#     predicted_classes = np.argmax(preds, axis=1)  # (N,)
    
#     # 获取人类最倾向的类别 (每个样本人类响应概率最高的类别)
#     true_classes = np.argmax(targets, axis=1)     # (N,)
    
#     # 计算预测正确的样本数
#     correct_predictions = np.sum(predicted_classes == true_classes)
    
#     # 计算准确率
#     total_samples = preds.shape[0]
#     accuracy = (correct_predictions / total_samples) * 100.0
    
#     return accuracy

# def compute_expected_accuracy(preds, targets):
#     """
#     计算 Expected Accuracy。
#     衡量模型预测分布与人类响应分布的平均重合程度。
#     """
#     if preds.shape != targets.shape:
#         raise ValueError("preds and targets must have the same shape")
        
#     # 计算每个样本的预测分布与真实分布的点积
#     sample_accuracies = np.sum(preds * targets, axis=1) # (N,)
    
#     # 计算平均准确率
#     expected_accuracy = np.mean(sample_accuracies) * 100.0
    
#     return expected_accuracy

# def compute_nll(preds, targets):
#     """计算负对数似然"""
#     preds = np.clip(preds, 1e-8, 1 - 1e-8)
#     return -np.mean(np.sum(targets * np.log(preds), axis=1))

# # def compute_spearman(preds, targets):
# #     """计算 Spearman 相关系数"""
# #     preds_max = np.argmax(preds, axis=1)
# #     targets_max = np.argmax(targets, axis=1)
# #     if len(preds_max) > 1 and len(np.unique(preds_max)) > 1 and len(np.unique(targets_max)) > 1:
# #         correlation, _ = spearmanr(preds_max, targets_max)
# #         return correlation
# #     return 0.0
# def compute_spearman_per_image(preds, targets):
#     if preds.shape != targets.shape:
#         raise ValueError("preds and targets must have the same shape")
    
#     n_samples = preds.shape[0]
#     spearman_coeffs = []
    
#     for i in range(n_samples):
#         pred_dist = preds[i]      # 形状 (C,) - 模型对样本 i 的预测分布
#         target_dist = targets[i]  # 形状 (C,) - 人类对样本 i 的响应分布
        
#         # 检查是否有足够的变化来计算相关性
#         # 如果所有值都相同，则无法计算有意义的相关系数
#         if len(np.unique(pred_dist)) > 1 and len(np.unique(target_dist)) > 1:
#             try:
#                 # 计算两个分布向量之间的 Spearman 相关系数
#                 corr, _ = spearmanr(pred_dist, target_dist)
#                 # 检查相关系数是否有效（避免 NaN 或 inf）
#                 if np.isfinite(corr):
#                     spearman_coeffs.append(corr)
#             except Exception:
#                 # 如果计算出错（例如数据问题），则跳过该样本
#                 pass
    
#     if spearman_coeffs:
#         # 返回所有样本的平均 Spearman 相关系数
#         return np.mean(spearman_coeffs)
#     else:
#         # 如果没有有效的系数（例如所有分布都完全相同），返回 0
#         return 0.0

# def compute_aic(nll, k):
#     """计算 AIC"""
#     return 2 * k - 2 * (-nll)  # NLL 是负对数似然

# def evaluate_model(model, features, softlabels, device):
#     """评估模型性能"""
#     model.eval()
#     with torch.no_grad():
#         X = torch.tensor(features).float().to(device)
#         preds = model(X).cpu().numpy()
#     top1_acc = compute_top1_accuracy(preds, softlabels)
#     exp_acc = compute_expected_accuracy(preds, softlabels)
#     nll = compute_nll(preds, softlabels)
#     spearman = compute_spearman_per_image(preds, softlabels)
    
#     # 计算参数数量
#     k = sum(p.numel() for p in model.parameters())
#     aic = compute_aic(nll, k)
    
#     return {
#         "top1_accuracy": top1_acc,
#         "expected_accuracy": exp_acc,
#         "nll": nll,
#         "spearman": spearman,
#         "aic": aic,
#         "params": k
#     }
# utils/metrics.py
# utils/metrics.py
"""
Evaluation metrics used in experiments:
- accuracy (top-1)
- topk_accuracy (top-k)
- negative log-likelihood (NLL)
- AIC from NLL
- second-best accuracy (SBA)
- spearman per-image (requires scipy; fallback available)
"""

import torch
import torch.nn.functional as F
import numpy as np

# try to import spearman from scipy, fallback to numpy-based approx if not available
try:
    from scipy.stats import spearmanr
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


def accuracy(outputs, targets):
    """
    outputs: logits (B, C)
    targets: (B,)
    returns scalar in [0,1]
    """
    preds = outputs.argmax(dim=1)
    correct = (preds == targets).sum().item()
    return correct / targets.size(0)


def topk_accuracy(outputs, targets, k=5):
    topk = outputs.topk(k, dim=1).indices  # (B, k)
    correct = (topk == targets.view(-1, 1)).any(dim=1).float().mean().item()
    return correct


def nll_loss(outputs, targets):
    log_probs = F.log_softmax(outputs, dim=1)
    return F.nll_loss(log_probs, targets, reduction='mean').item()


def aic_from_nll(nll, k_params):
    """
    AIC = 2k - 2 ln L; nll = - ln L
    => AIC = 2k + 2 * nll
    """
    return 2 * k_params + 2.0 * nll


def sba(outputs, targets):
    """
    Second-best accuracy: proportion where model's 2nd choice equals target.
    (Mostly useful if 'target' is second-most-common human label; here targets is generic.)
    """
    top2 = outputs.topk(2, dim=1).indices  # (B,2)
    second = top2[:, 1]
    correct = (second == targets).sum().item()
    return correct / targets.size(0)


def spearman_per_image(model_probs, human_counts):
    """
    model_probs: (N_images, C) array-like (probabilities)
    human_counts: (N_images, C) array-like (counts or proportions)
    returns average Spearman rho across images.

    If scipy is not available, returns 0 for images where spearman can't be computed.
    """
    model_probs = np.asarray(model_probs)
    human_counts = np.asarray(human_counts)
    n = model_probs.shape[0]
    rhos = []
    for i in range(n):
        mp = model_probs[i]
        hc = human_counts[i]
        if _HAS_SCIPY:
            try:
                rho, _ = spearmanr(mp, hc)
                if np.isnan(rho):
                    rho = 0.0
            except Exception:
                rho = 0.0
        else:
            # fallback: use numpy rank correlation (Spearman) manual
            try:
                mp_rank = np.argsort(np.argsort(mp))
                hc_rank = np.argsort(np.argsort(hc))
                # compute Pearson on ranks
                mpr = mp_rank - mp_rank.mean()
                hcr = hc_rank - hc_rank.mean()
                denom = np.sqrt((mpr**2).sum() * (hcr**2).sum())
                if denom == 0:
                    rho = 0.0
                else:
                    rho = (mpr * hcr).sum() / denom
            except Exception:
                rho = 0.0
        rhos.append(rho)
    return float(np.mean(rhos))
