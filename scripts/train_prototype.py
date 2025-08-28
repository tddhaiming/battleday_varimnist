# # scripts/train_prototype.py
# import torch
# import torch.nn as nn
# import numpy as np
# import os
# import sys

# # 将项目根目录添加到 Python 路径
# project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# if project_root not in sys.path:
#     sys.path.append(project_root)

# from models.prototype import PrototypeModel
# from utils.data_utils import get_train_val_features
# from utils.metrics import evaluate_model

# def train_prototype_model(model_name='resnet18', mode="classic", epochs=100):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"使用设备: {device}")
    
#     # 加载特征
#     X_train, y_train, X_val, y_val = get_train_val_features(model_name)
#     if X_train is None:
#         print("请先运行特征提取脚本!")
#         return
    
#     feature_dim = X_train.shape[1]
#     print(f"特征维度: {feature_dim}")
    
#     # 转换为 PyTorch 张量
#     X_train = torch.tensor(X_train).float().to(device)
#     y_train = torch.tensor(y_train).float().to(device)
    
#     # 初始化模型
#     model = PrototypeModel(mode=mode, feature_dim=feature_dim).to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
#     criterion = nn.KLDivLoss(reduction="batchmean")
    
#     # 训练
#     model.train()
#     for epoch in range(epochs):
#         optimizer.zero_grad()
#         preds = model(X_train)
#         loss = criterion(torch.log(preds + 1e-8), y_train)
#         loss.backward()
#         optimizer.step()
        
#         if epoch % 20 == 0:
#             print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
#     # 保存模型
#     os.makedirs("/content/gdrive/MyDrive/battleday_varimnist/results", exist_ok=True)
#     torch.save(model.state_dict(), f"/content/gdrive/MyDrive/battleday_varimnist/results/prototype_{mode}_{model_name}.pth")
#     print(f"原型模型 ({mode}) 训练完成并保存")
    
#     # 评估模型
#     metrics = evaluate_model(model, X_val, y_val, device)
#     print(f"验证集性能 ({model_name} + {mode}):")
#     print(f"  Top-1 Accuracy: {metrics['top1_acc']:.2f}%")
#     print(f"  Expected Accuracy: {metrics['exp_acc']:.2f}%")
#     print(f"  NLL: {metrics['nll']:.4f}")
#     print(f"  Spearman: {metrics['spearman']:.4f}")
#     print(f"  AIC: {metrics['aic']:.4f}")
#     print(f"  参数数量: {metrics['params']}")
    
#     return metrics

# def compare_all_prototype_models(model_name='resnet18'):
#     """比较所有原型模型"""
#     results = {}
#     modes = ['classic', 'linear', 'quadratic']
    
#     for mode in modes:
#         print(f"\n=== 训练原型模型 ({mode}) ===")
#         metrics = train_prototype_model(model_name, mode)
#         results[mode] = metrics
    
#     # 打印比较结果
#     print(f"\n=== {model_name} 原型模型比较结果 ===")
#     print("模式\t\tNLL\t\tSpearman\tAIC")
#     print("-" * 50)
#     for mode, metrics in results.items():
#         print(f"{mode}\t\t{metrics['nll']:.4f}\t\t{metrics['spearman']:.4f}\t\t{metrics['aic']:.4f}")

# if __name__ == "__main__":
#     compare_all_prototype_models('resnet18')
"""
Training script for Prototype models (Classic / Linear / Quadratic).
Supports:
- AMP (mixed precision)
- Resume from checkpoint
- TensorBoard logging
- Early stopping (patience)
- Command-line args for common hyperparams

Example:
    python scripts/train_prototype.py --model linear --features data/features.pt --labels data/labels.pt --prototypes data/prototypes.pt
"""

import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import json

from models.prototype import ClassicPrototype, LinearPrototype, QuadraticPrototype
from utils.metrics import accuracy, topk_accuracy, nll_loss, aic_from_nll

# === 新代码开始 ===
def save_checkpoint(state, path):
    torch.save(state, path)

def save_results_to_json(args, final_metrics, best_val, epoch):
    # 确保目录存在
    results_dir = "/mnt/dataset0/thm/code/battleday_varimnist/results"
    os.makedirs(results_dir, exist_ok=True)
    
    # 准备结果数据
    results = {
        "model_type": args.model,
        "final_epoch": epoch,
        "best_val_nll": best_val,
        "final_metrics": {
            "val_nll": final_metrics['val_nll'],
            "val_accuracy": final_metrics['val_acc'],
            "val_top5_accuracy": final_metrics['val_top5']
        },
        "hyperparameters": {
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "patience": args.patience
        }
    }
    
    # 生成文件名
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"results_{args.model}_{timestamp}.json"
    filepath = os.path.join(results_dir, filename)
    
    # 保存JSON文件
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {filepath}")
# === 新代码结束 ===


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    features = torch.load(args.features)  # (N,D)
    labels = torch.load(args.labels)      # (N,)
    prototypes = torch.load(args.prototypes)  # (C, D)

    D = features.shape[1]
    C = prototypes.shape[0]

    # Split dataset into train and validation
    dataset = TensorDataset(features, labels)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create data loaders with optimized configuration
    train_loader = DataLoader(train_dataset, batch_size=512,
                              shuffle=True, num_workers=8,
                              pin_memory=True,
                              persistent_workers=True,
                              prefetch_factor=4)

    val_loader = DataLoader(val_dataset, batch_size=512,
                            num_workers=8, pin_memory=True,
                            persistent_workers=True,
                            prefetch_factor=4)

    # choose model type
    if args.model.lower() == "classic":
        model = ClassicPrototype(prototypes.to(device), feature_dim=D, num_classes=C).to(device)
        trainable_params = list(model.parameters())
    elif args.model.lower() == "linear":
        model = LinearPrototype(prototypes.to(device), feature_dim=D, num_classes=C).to(device)
        trainable_params = list(model.parameters())
    elif args.model.lower() == "quadratic":
        model = QuadraticPrototype(prototypes.to(device), feature_dim=D, num_classes=C).to(device)
        trainable_params = list(model.parameters())
    else:
        raise ValueError("Unknown model type")

    # Option: let gamma be learnable
    gamma_raw = nn.Parameter(torch.tensor(1.0).log(), requires_grad=True)
    gamma_raw = gamma_raw.to(device)
    # include gamma in optimizer
    optimizer = optim.Adam(trainable_params + [gamma_raw], lr=args.lr)
    scaler = GradScaler()
    writer = SummaryWriter(log_dir=args.logdir)

    start_epoch = 0
    best_val = float("inf")
    patience_counter = 0

    # resume logic
    if args.resume and os.path.exists(args.checkpoint):
        ck = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ck["model_state"])
        optimizer.load_state_dict(ck["optimizer_state"])
        scaler.load_state_dict(ck["scaler_state"])
        gamma_raw.data = ck.get("gamma_raw", gamma_raw.data)
        start_epoch = ck["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        start = time.time()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with autocast():
                gamma = torch.exp(gamma_raw)
                logits = model(x, gamma=gamma)
                loss = F.nll_loss(F.log_softmax(logits, dim=1), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item() * x.size(0)
        epoch_loss = epoch_loss / len(train_loader.dataset)
        elapsed = time.time() - start

        # validation
        model.eval()
        val_loss = 0.0
        all_logits = []
        all_labels = []
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                with autocast():
                    gamma = torch.exp(gamma_raw)
                    logits = model(x, gamma=gamma)
                    loss = F.nll_loss(F.log_softmax(logits, dim=1), y, reduction='sum')
                    val_loss += loss.item()
                    all_logits.append(logits)
                    all_labels.append(y)
            
            val_loss = val_loss / len(val_dataset)
            all_logits = torch.cat(all_logits, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            
            val_nll = F.nll_loss(F.log_softmax(all_logits, dim=1), all_labels, reduction='mean').item()
            val_acc = accuracy(all_logits, all_labels)
            val_top5 = topk_accuracy(all_logits, all_labels, k=5)

        print(f"[Epoch {epoch}] train_loss={epoch_loss:.4f} val_nll={val_nll:.4f} val_acc={val_acc:.4f} time={elapsed:.1f}s")

        # logging
        writer.add_scalar("Loss/train", epoch_loss, epoch)
        writer.add_scalar("NLL/val", val_nll, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)
        writer.add_scalar("Accuracy/val_top5", val_top5, epoch)
        writer.add_scalar("gamma", torch.exp(gamma_raw).item(), epoch)

        # checkpoint
        state = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "gamma_raw": gamma_raw.data
        }
        save_checkpoint(state, args.checkpoint)

        # early stopping based on val NLL
        if val_nll < best_val - args.early_stop_delta:
            best_val = val_nll
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("Early stopping triggered.")
                break

    writer.close()
    
    # 保存最终评估结果到JSON文件
    final_metrics = {
        'val_nll': val_nll,
        'val_acc': val_acc,
        'val_top5': val_top5
    }
    save_results_to_json(args, final_metrics, best_val, epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="linear", choices=["classic", "linear", "quadratic"])
    parser.add_argument("--features", type=str, default="data/features.pt")
    parser.add_argument("--labels", type=str, default="data/labels.pt")
    parser.add_argument("--prototypes", type=str, default="data/prototypes.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--logdir", type=str, default="./runs/prototype")
    parser.add_argument("--checkpoint", type=str, default="checkpoint_prototype.pth")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--early_stop_delta", type=float, default=1e-4)
    args = parser.parse_args()
    train(args)