# utils/metrics.py
""" 简单评估指标：accuracy, topk_accuracy, negative log likelihood, AIC, SBA, Spearman
注意：spearman_per_image 需要 model_probs 和 human_counts 作为 numpy array，
与其他函数的 torch tensor 输入不同。
"""
import torch
import torch.nn.functional as F
import numpy as np

# 尝试从 scipy 导入 spearman，如果不可用则回退到基于 numpy 的近似
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
    """
    计算平均负对数似然损失 (NLL Loss)。
    outputs: logits (B, C)
    targets: (B,)
    returns scalar (float)
    """
    # 注意：F.cross_entropy 内部执行了 log_softmax，等效于 NLL Loss
    # 如果 outputs 已经是 log_probs，应使用 F.nll_loss
    # 这里假设 outputs 是 logits
    return F.cross_entropy(outputs, targets, reduction='mean').item()


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
    outputs: logits (B, C)
    targets: (B,)
    returns scalar in [0,1]
    """
    if outputs.size(1) < 2:
        # 如果类别数少于2，则无法计算第二最佳
        return 0.0
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
