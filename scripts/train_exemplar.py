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
from utils.metrics import evaluate_model

def train_exemplar_model(model_name='resnet18', epochs=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载特征
    X_train, y_train, X_val, y_val = get_train_val_features(model_name)
    if X_train is None:
        print("请先运行特征提取脚本!")
        return
    
    feature_dim = X_train.shape[1]
    print(f"特征维度: {feature_dim}")
    print(f"训练样本数: {X_train.shape[0]}")
    
    # 使用训练集作为 exemplar 集合
    train_labels = np.argmax(y_train, axis=1)  # 使用硬标签
    
    # 初始化模型
    model = ExemplarModel(X_train, train_labels, feature_dim=feature_dim).to(device)
    optimizer = torch.optim.Adam([model.beta, model.gamma, model.Sigma_inv], lr=1e-2)
    criterion = nn.KLDivLoss(reduction="batchmean")
    
    # 关键修改：使用非常小的批处理大小
    batch_size = 16  # 从默认的 64 降低到 16 或更小
    print(f"使用批处理大小: {batch_size}")
    
    # 训练
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        num_batches = 0
        
        # 分批处理训练数据
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]
            
            X_batch_tensor = torch.tensor(X_batch).float().to(device)
            y_batch_tensor = torch.tensor(y_batch).float().to(device)
            
            optimizer.zero_grad()
            preds = model(X_batch_tensor)
            loss = criterion(torch.log(preds + 1e-8), y_batch_tensor)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        if epoch % 20 == 0:
            avg_loss = epoch_loss / num_batches
            print(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")
    
    # 保存模型
    os.makedirs("/content/gdrive/MyDrive/battleday_varimnist/results", exist_ok=True)
    torch.save({
        'state_dict': model.state_dict(),
        'exemplars': X_train,
        'labels': train_labels
    }, f"/content/gdrive/MyDrive/battleday_varimnist/results/exemplar_{model_name}.pth")
    print(f"样本模型训练完成并保存")
    
    # 评估模型 (同样需要小批次评估)
    print("开始评估模型...")
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for i in range(0, len(X_val), batch_size):
            X_val_batch = X_val[i:i+batch_size]
            y_val_batch = y_val[i:i+batch_size]
            
            X_val_tensor = torch.tensor(X_val_batch).float().to(device)
            preds_batch = model(X_val_tensor)
            
            all_preds.append(preds_batch.cpu().numpy())
            all_targets.append(y_val_batch)
    
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    # 手动计算指标
    from utils.metrics import compute_nll, compute_spearman, compute_aic
    
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