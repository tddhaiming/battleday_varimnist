# scripts/train_exemplar.py
import sys
import os
import json
# 将项目根目录添加到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

import torch
import torch.nn as nn
import numpy as np
import os
from models.exemplar import ExemplarModel
from utils.data_utils import get_train_val_features
from utils.metrics import (
    compute_top1_accuracy,
    compute_expected_accuracy,
    compute_nll,
    compute_spearman_per_image,
    compute_aic
)
import time

# --- 内存管理设置 ---
# 如果你的 PyTorch 版本支持，可以设置以下环境变量来减少内存碎片
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
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

# def compute_spearman_per_image(preds, targets):
#     """计算每张图像的 Spearman 相关系数 (符合 Battleday 原文定义)"""
#     from scipy.stats import spearmanr
#     if preds.shape != targets.shape:
#         raise ValueError("preds and targets must have the same shape")
    
#     n_samples = preds.shape[0]
#     spearman_coeffs = []
    
#     for i in range(n_samples):
#         pred_dist = preds[i]
#         target_dist = targets[i]
        
#         if len(np.unique(pred_dist)) > 1 and len(np.unique(target_dist)) > 1:
#             try:
#                 corr, _ = spearmanr(pred_dist, target_dist)
#                 if np.isfinite(corr):
#                     spearman_coeffs.append(corr)
#             except Exception:
#                 pass
    
#     if spearman_coeffs:
#         return np.mean(spearman_coeffs)
#     else:
#         return 0.0

# def compute_aic(nll, k):
#     """计算 AIC"""
#     return 2 * k - 2 * (-nll)

def train_exemplar_model(model_name='resnet18', epochs=80):
    """
    训练 Exemplar 模型，使用所有可用的训练样本，并采用内存优化策略。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # --- 1. 加载特征 ---
    X_train, y_train, X_val, y_val = get_train_val_features(model_name)
    if X_train is None:
        print("请先运行特征提取脚本!")
        return
    
    original_train_size = X_train.shape[0]
    feature_dim = X_train.shape[1]
    print(f"特征维度: {feature_dim}")
    print(f"原始训练样本数: {original_train_size}")
    print(f"验证样本数: {X_val.shape[0]}")

    # --- 2. 准备 Exemplar 集合 (使用所有样本) ---
    X_train_exemplars = X_train
    y_train_exemplars = y_train
    train_labels_exemplars = np.argmax(y_train_exemplars, axis=1)
    
    print(f"使用全部 {X_train_exemplars.shape[0]} 个样本作为 Exemplars")

    # --- 3. 初始化模型 ---
    model = ExemplarModel(
        exemplars=X_train_exemplars,
        labels=train_labels_exemplars,
        feature_dim=feature_dim
    ).to(device)
    
    # 只优化模型的可学习参数
    optimizer = torch.optim.Adam([model.beta, model.gamma, model.Sigma_inv], lr=1e-2)
    criterion = nn.KLDivLoss(reduction="batchmean")
    
    # --- 4. 内存优化设置 ---
    train_batch_size = 16
    eval_batch_size = 32
    print(f"训练批处理大小: {train_batch_size}")
    print(f"评估批处理大小: {eval_batch_size}")
    
    # --- 5. 训练循环 ---
    model.train()
    print("开始训练...")
    for epoch in range(epochs):
        epoch_loss = 0
        num_batches = 0
        start_time = time.time()
        
        # 分批处理训练数据
        for i in range(0, len(X_train_exemplars), train_batch_size):
            X_batch = X_train_exemplars[i:i+train_batch_size]
            y_batch = y_train_exemplars[i:i+train_batch_size]
            
            X_batch_tensor = torch.tensor(X_batch).float().to(device)
            y_batch_tensor = torch.tensor(y_batch).float().to(device)
            
            optimizer.zero_grad()
            preds = model(X_batch_tensor)
            loss = criterion(torch.log(preds + 1e-8), y_batch_tensor)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        epoch_time = time.time() - start_time
        if epoch % 5 == 0 or epoch == epochs - 1:
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
            print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s")
    
    print("训练完成!")

    # --- 6. 保存模型 ---
    result_dir = "/mnt/dataset0/thm/code/battleday_varimnist/results"
    os.makedirs(result_dir, exist_ok=True)
    model_save_path = f"{result_dir}/exemplar_{model_name}_full.pth"
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'exemplar_count': len(X_train_exemplars),
        'feature_dim': feature_dim,
        'beta': model.beta.item(),
        'gamma': model.gamma.item(),
        'Sigma_inv': model.Sigma_inv.detach().cpu().numpy()
    }, model_save_path)
    print(f"模型元数据已保存到: {model_save_path}")

    # --- 7. 评估模型 ---
    print("开始评估模型...")
    model.eval()
    all_preds = []
    all_targets = []
    
    # 释放缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    with torch.no_grad():
        # 分批处理验证数据
        for i in range(0, len(X_val), eval_batch_size):
            X_val_batch = X_val[i:i+eval_batch_size]
            y_val_batch = y_val[i:i+eval_batch_size]
            
            X_val_tensor = torch.tensor(X_val_batch).float().to(device)
            preds_batch = model(X_val_tensor)
            
            all_preds.append(preds_batch.cpu().numpy())
            all_targets.append(y_val_batch)
    
    preds = np.vstack(all_preds)
    targets = np.vstack(all_targets)
    
    # --- 8. 计算最终指标（使用 metrics 中的函数）---
    top1_acc = compute_top1_accuracy(preds, targets)
    exp_acc = compute_expected_accuracy(preds, targets)  # 来自 metrics
    nll = compute_nll(preds, targets)  # 来自 metrics
    spearman = compute_spearman_per_image(preds, targets)  # 来自 metrics
    k = sum(p.numel() for p in model.parameters())
    aic = compute_aic(nll, k)  # 来自 metrics
    
    print(f"验证集性能 ({model_name} + exemplar - Full Dataset):")
    print(f"  Top-1 Accuracy: {top1_acc:.2f}%")
    print(f"  Expected Accuracy: {exp_acc:.2f}%")
    print(f"  NLL: {nll:.4f}")
    print(f"  Spearman (per-image): {spearman:.4f}")
    print(f"  AIC: {aic:.4f}")
    print(f"  可学习参数数量: {k}")
    total_storage_cost = X_train_exemplars.shape[0] * X_train_exemplars.shape[1] + train_labels_exemplars.shape[0]
    print(f"  总存储成本 (Exemplars + Labels): {total_storage_cost}") 
    
    result_dict = {
        "top1_accuracy": top1_acc,
        "expected_accuracy": exp_acc,
        "nll": nll,
        "spearman": spearman,
        "aic": aic,
        "trainable_params": k,
        "total_storage": total_storage_cost,
        "num_exemplars": X_train_exemplars.shape[0]
    }


     # --- 9. 保存结果到 JSON 文件 ---
    results_file = "/mnt/dataset0/thm/code/battleday_varimnist/results/model_comparison_results.json"
    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    # 读取现有结果（如果存在）
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            all_results = json.load(f)
    else:
        # 初始化结果结构（与 visualize_results.py 一致）
        all_results = {"prototype": {}, "exemplar": {}}

    # 更新 exmeplar 部分的结果（使用模型名+训练方式作为键，如 'resnet18_full'）
    result_key = f"{model_name}_full"  # 区分不同模型和训练配置
    all_results["exemplar"][result_key] = result_dict

    # 写入文件
    with open(results_file, 'w') as f:
        json.dump(all_results, indent=2, fp=f)
    print(f"评估结果已保存到: {results_file}")

    return result_dict

if __name__ == "__main__":
    train_exemplar_model('resnet18')