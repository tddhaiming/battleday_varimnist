# scripts/train_exemplar.py
import sys
import os

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
import time

# --- 内存管理设置 ---
# 如果你的 PyTorch 版本支持，可以设置以下环境变量来减少内存碎片
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def compute_nll(preds, targets):
    """计算负对数似然"""
    preds = np.clip(preds, 1e-8, 1 - 1e-8)
    return -np.mean(np.sum(targets * np.log(preds), axis=1))

def compute_spearman_per_image(preds, targets):
    """计算每张图像的 Spearman 相关系数 (符合 Battleday 原文定义)"""
    from scipy.stats import spearmanr
    if preds.shape != targets.shape:
        raise ValueError("preds and targets must have the same shape")
    
    n_samples = preds.shape[0]
    spearman_coeffs = []
    
    for i in range(n_samples):
        pred_dist = preds[i]
        target_dist = targets[i]
        
        if len(np.unique(pred_dist)) > 1 and len(np.unique(target_dist)) > 1:
            try:
                corr, _ = spearmanr(pred_dist, target_dist)
                if np.isfinite(corr):
                    spearman_coeffs.append(corr)
            except Exception:
                pass
    
    if spearman_coeffs:
        return np.mean(spearman_coeffs)
    else:
        return 0.0

def compute_aic(nll, k):
    """计算 AIC"""
    return 2 * k - 2 * (-nll)

def train_exemplar_model(model_name='resnet18', epochs=30): # 进一步减少 epochs
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
    # 不再进行采样
    X_train_exemplars = X_train
    y_train_exemplars = y_train
    train_labels_exemplars = np.argmax(y_train_exemplars, axis=1)
    
    print(f"使用全部 {X_train_exemplars.shape[0]} 个样本作为 Exemplars")

    # --- 3. 初始化模型 ---
    # 将 Exemplar 数据移动到 CPU，模型在前向传播时再按需处理
    model = ExemplarModel(
        exemplars=X_train_exemplars, # 存储在 CPU 上
        labels=train_labels_exemplars, # 存储在 CPU 上
        feature_dim=feature_dim
    ).to(device)
    
    # 只优化模型的可学习参数 (beta, gamma, Sigma_inv)
    optimizer = torch.optim.Adam([model.beta, model.gamma, model.Sigma_inv], lr=1e-2)
    criterion = nn.KLDivLoss(reduction="batchmean")
    
    # --- 4. 内存优化设置 ---
    # 使用更小的批处理大小以节省内存
    train_batch_size = 16  # 从 32 降到 16
    eval_batch_size = 32   # 评估时也可以用稍大一点的 batch size
    print(f"训练批处理大小: {train_batch_size}")
    print(f"评估批处理大小: {eval_batch_size}")
    
    # --- 5. 训练循环 ---
    model.train()
    print("开始训练...")
    for epoch in range(epochs):
        epoch_loss = 0
        num_batches = 0
        start_time = time.time()
        
        # 分批处理训练数据 (内存优化)
        for i in range(0, len(X_train_exemplars), train_batch_size):
            X_batch = X_train_exemplars[i:i+train_batch_size]
            y_batch = y_train_exemplars[i:i+train_batch_size]
            
            X_batch_tensor = torch.tensor(X_batch).float().to(device)
            y_batch_tensor = torch.tensor(y_batch).float().to(device)
            
            optimizer.zero_grad()
            preds = model(X_batch_tensor) # 模型内部会处理与 Exemplars 的距离计算
            loss = criterion(torch.log(preds + 1e-8), y_batch_tensor)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            # 可选：释放中间变量，尝试减少内存峰值 (PyTorch 通常会自动管理)
            # del X_batch_tensor, y_batch_tensor, preds, loss
        
        epoch_time = time.time() - start_time
        if epoch % 5 == 0 or epoch == epochs - 1: # 每5个epoch或最后一次打印
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
            print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s")
    
    print("训练完成!")

    # --- 6. 保存模型 ---
    # 注意：保存时可能需要特殊处理大的 Exemplar 数据
    os.makedirs("/content/gdrive/MyDrive/battleday_varimnist/results", exist_ok=True)
    model_save_path = f"/content/gdrive/MyDrive/battleday_varimnist/results/exemplar_{model_name}_full.pth"
    
    # 为了避免 pickle 大数组的问题，可以只保存模型状态和关键参数
    torch.save({
        'model_state_dict': model.state_dict(),
        'exemplar_count': len(X_train_exemplars),
        'feature_dim': feature_dim,
        # 'exemplars': X_train_exemplars, # 警告：这会创建一个非常大的文件，可能不建议直接保存
        # 'labels': train_labels_exemplars, # 同上
        'beta': model.beta.item(),
        'gamma': model.gamma.item(),
        'Sigma_inv': model.Sigma_inv.detach().cpu().numpy()
    }, model_save_path)
    print(f"模型元数据已保存到: {model_save_path}")
    print("注意：完整的 Exemplar 数据未直接保存在模型文件中，它来自于训练特征。")

    # --- 7. 快速评估模型 (使用内存优化) ---
    print("开始评估模型...")
    model.eval()
    all_preds = []
    all_targets = []
    
    # 释放训练时可能占用的缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    with torch.no_grad():
        # 分批处理验证数据 (内存优化)
        for i in range(0, len(X_val), eval_batch_size):
            X_val_batch = X_val[i:i+eval_batch_size]
            y_val_batch = y_val[i:i+eval_batch_size]
            
            X_val_tensor = torch.tensor(X_val_batch).float().to(device)
            preds_batch = model(X_val_tensor)
            
            all_preds.append(preds_batch.cpu().numpy())
            all_targets.append(y_val_batch)
            
            # 释放批次数据
            # del X_val_tensor, preds_batch
    
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    # --- 8. 计算最终指标 ---
    nll = compute_nll(all_preds, all_targets)
    spearman = compute_spearman_per_image(all_preds, all_targets) # 使用符合原文的 Spearman
    k = sum(p.numel() for p in model.parameters())
    aic = compute_aic(nll, k)
    
    print(f"验证集性能 ({model_name} + exemplar - Full Dataset):")
    print(f"  NLL: {nll:.4f}")
    print(f"  Spearman (per-image): {spearman:.4f}")
    print(f"  AIC: {aic:.4f}")
    print(f"  可学习参数数量: {k}") # 强调这是可学习参数
    # 计算总存储成本（非参数，但重要）
    total_storage_cost = X_train_exemplars.shape[0] * X_train_exemplars.shape[1] # Exemplars
    total_storage_cost += train_labels_exemplars.shape[0] # Labels
    print(f"  总存储成本 (Exemplars + Labels): {total_storage_cost}") 
    
    return {
        "nll": nll,
        "spearman": spearman,
        "aic": aic,
        "trainable_params": k,
        "total_storage": total_storage_cost,
        "num_exemplars": X_train_exemplars.shape[0]
    }

if __name__ == "__main__":
    train_exemplar_model('resnet18')
