# scripts/train_exemplar.py
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
    
    # 使用训练集作为 exemplar 集合
    train_labels = np.argmax(y_train, axis=1)  # 使用硬标签
    
    # 初始化模型
    model = ExemplarModel(X_train, train_labels, feature_dim=feature_dim).to(device)
    optimizer = torch.optim.Adam([model.beta, model.gamma, model.Sigma_inv], lr=1e-2)
    criterion = nn.KLDivLoss(reduction="batchmean")
    
    # 转换训练数据
    X_train_tensor = torch.tensor(X_train).float().to(device)
    y_train_tensor = torch.tensor(y_train).float().to(device)
    
    # 训练
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        preds = model(X_train_tensor)
        loss = criterion(torch.log(preds + 1e-8), y_train_tensor)
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    # 保存模型
    os.makedirs("/content/gdrive/MyDrive/battleday_varimnist/results", exist_ok=True)
    torch.save({
        'state_dict': model.state_dict(),
        'exemplars': X_train,
        'labels': train_labels
    }, f"/content/gdrive/MyDrive/battleday_varimnist/results/exemplar_{model_name}.pth")
    print(f"样本模型训练完成并保存")
    
    # 评估模型
    metrics = evaluate_model(model, X_val, y_val, device)
    print(f"验证集性能 ({model_name} + exemplar):")
    print(f"  NLL: {metrics['nll']:.4f}")
    print(f"  Spearman: {metrics['spearman']:.4f}")
    print(f"  AIC: {metrics['aic']:.4f}")
    print(f"  参数数量: {metrics['params']}")
    
    return metrics

if __name__ == "__main__":
    train_exemplar_model('resnet18')