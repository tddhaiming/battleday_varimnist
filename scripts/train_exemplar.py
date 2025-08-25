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

def compute_nll(preds, targets):
    """计算负对数似然"""
    preds = np.clip(preds, 1e-8, 1 - 1e-8)
    return -np.mean(np.sum(targets * np.log(preds), axis=1))

def compute_spearman(preds, targets):
    """计算 Spearman 相关系数"""
    from scipy.stats import spearmanr
    preds_max = np.argmax(preds, axis=1)
    targets_max = np.argmax(targets, axis=1)
    if len(preds_max) > 1 and len(np.unique(preds_max)) > 1 and len(np.unique(targets_max)) > 1:
        correlation, _ = spearmanr(preds_max, targets_max)
        return correlation
    return 0.0

def compute_aic(nll, k):
    """计算 AIC"""
    return 2 * k - 2 * (-nll)

def train_exemplar_model(model_name='resnet18', epochs=50):  # 减少 epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载特征
    X_train, y_train, X_val, y_val = get_train_val_features(model_name)
    if X_train is None:
        print("请先运行特征提取脚本!")
        return
    
    feature_dim = X_train.shape[1]
    original_train_size = X_train.shape[0]
    print(f"特征维度: {feature_dim}")
    print(f"原始训练样本数: {original_train_size}")
    
    # 采样减少 exemplar 数量（例如只用 5000 个样本）
    max_exemplars = min(5000, original_train_size)  # 限制在 5000 个以内
    if original_train_size > max_exemplars:
        indices = np.random.choice(original_train_size, max_exemplars, replace=False)
        X_train_sampled = X_train[indices]
        y_train_sampled = y_train[indices]
        train_labels_sampled = np.argmax(y_train_sampled, axis=1)
        print(f"采样 {max_exemplars} 个 exemplars 进行训练")
    else:
        X_train_sampled = X_train
        y_train_sampled = y_train
        train_labels_sampled = np.argmax(y_train, axis=1)
    
    # 初始化模型
    model = ExemplarModel(X_train_sampled, train_labels_sampled, feature_dim=feature_dim).to(device)
    optimizer = torch.optim.Adam([model.beta, model.gamma, model.Sigma_inv], lr=1e-2)
    criterion = nn.KLDivLoss(reduction="batchmean")
    
    # 使用适中的批处理大小
    batch_size = 32
    print(f"使用批处理大小: {batch_size}")
    
    # 训练
    model.train()
    print("开始训练...")
    for epoch in range(epochs):
        epoch_loss = 0
        num_batches = 0
        start_time = time.time()
        
        # 分批处理训练数据
        for i in range(0, len(X_train_sampled), batch_size):
            X_batch = X_train_sampled[i:i+batch_size]
            y_batch = y_train_sampled[i:i+batch_size]
            
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
        if epoch % 10 == 0:  # 每10个epoch打印一次
            avg_loss = epoch_loss / num_batches
            print(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s")
    
    print("训练完成!")
    
    # 保存模型
    os.makedirs("/content/gdrive/MyDrive/battleday_varimnist/results", exist_ok=True)
    torch.save({
        'state_dict': model.state_dict(),
        'exemplars': X_train_sampled,
        'labels': train_labels_sampled
    }, f"/content/gdrive/MyDrive/battleday_varimnist/results/exemplar_{model_name}.pth")
    print(f"样本模型训练完成并保存")
    
    # 快速评估模型
    print("开始快速评估模型...")
    model.eval()
    all_preds = []
    all_targets = []
    
    # 使用更大的批处理大小进行评估
    eval_batch_size = 64
    
    with torch.no_grad():
        for i in range(0, len(X_val), eval_batch_size):
            X_val_batch = X_val[i:i+eval_batch_size]
            y_val_batch = y_val[i:i+eval_batch_size]
            
            X_val_tensor = torch.tensor(X_val_batch).float().to(device)
            preds_batch = model(X_val_tensor)
            
            all_preds.append(preds_batch.cpu().numpy())
            all_targets.append(y_val_batch)
    
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    # 计算指标
    nll = compute_nll(all_preds, all_targets)
    spearman = compute_spearman(all_preds, all_targets)
    k = sum(p.numel() for p in model.parameters())
    aic = compute_aic(nll, k)
    
    print(f"验证集性能 ({model_name} + exemplar):")
    print(f"  NLL: {nll:.4f}")
    print(f"  Spearman: {spearman:.4f}")
    print(f"  AIC: {aic:.4f}")
    print(f"  参数数量: {k}")
    
    return {
        "nll": nll,
        "spearman": spearman,
        "aic": aic,
        "params": k
    }

if __name__ == "__main__":
    train_exemplar_model('resnet18')