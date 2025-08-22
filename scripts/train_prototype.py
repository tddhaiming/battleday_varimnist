# scripts/train_prototype.py
import torch
import torch.nn as nn
import numpy as np
import os
import sys

# 将项目根目录添加到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from models.prototype import PrototypeModel
from utils.data_utils import get_train_val_features
from utils.metrics import evaluate_model

def train_prototype_model(model_name='resnet18', mode="classic", epochs=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载特征
    X_train, y_train, X_val, y_val = get_train_val_features(model_name)
    if X_train is None:
        print("请先运行特征提取脚本!")
        return
    
    feature_dim = X_train.shape[1]
    print(f"特征维度: {feature_dim}")
    
    # 转换为 PyTorch 张量
    X_train = torch.tensor(X_train).float().to(device)
    y_train = torch.tensor(y_train).float().to(device)
    
    # 初始化模型
    model = PrototypeModel(mode=mode, feature_dim=feature_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.KLDivLoss(reduction="batchmean")
    
    # 训练
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        preds = model(X_train)
        loss = criterion(torch.log(preds + 1e-8), y_train)
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    # 保存模型
    os.makedirs("/content/gdrive/MyDrive/battleday_varimnist/results", exist_ok=True)
    torch.save(model.state_dict(), f"/content/gdrive/MyDrive/battleday_varimnist/results/prototype_{mode}_{model_name}.pth")
    print(f"原型模型 ({mode}) 训练完成并保存")
    
    # 评估模型
    metrics = evaluate_model(model, X_val, y_val, device)
    print(f"验证集性能 ({model_name} + {mode}):")
    print(f"  NLL: {metrics['nll']:.4f}")
    print(f"  Spearman: {metrics['spearman']:.4f}")
    print(f"  AIC: {metrics['aic']:.4f}")
    print(f"  参数数量: {metrics['params']}")
    
    return metrics

def compare_all_prototype_models(model_name='resnet18'):
    """比较所有原型模型"""
    results = {}
    modes = ['classic', 'linear', 'quadratic']
    
    for mode in modes:
        print(f"\n=== 训练原型模型 ({mode}) ===")
        metrics = train_prototype_model(model_name, mode)
        results[mode] = metrics
    
    # 打印比较结果
    print(f"\n=== {model_name} 原型模型比较结果 ===")
    print("模式\t\tNLL\t\tSpearman\tAIC")
    print("-" * 50)
    for mode, metrics in results.items():
        print(f"{mode}\t\t{metrics['nll']:.4f}\t\t{metrics['spearman']:.4f}\t\t{metrics['aic']:.4f}")

if __name__ == "__main__":
    compare_all_prototype_models('resnet18')