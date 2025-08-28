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
import sys
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
try:
    from torch.amp import autocast, GradScaler
    USE_NEW_AMP = True
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
    USE_NEW_AMP = False
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import json

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)
    print(f"已将项目根目录添加到路径: {project_root}")

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

def convert_to_class_indices(labels):
    """将 one-hot 编码转换为类别索引，用于评估指标计算"""
    if labels.dim() == 2 and labels.shape[1] > 1:
        return torch.argmax(labels, dim=1)
    return labels

def kl_div_loss_with_onehot(logits, targets_onehot):
    """使用 KL 散度损失处理 one-hot 编码标签"""
    log_probs = F.log_softmax(logits, dim=1)
    # 对 one-hot 标签使用 KL 散度
    return F.kl_div(log_probs, targets_onehot, reduction='batchmean')
# === 新代码结束 ===


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    features = torch.load(args.features)  # (N,D)
    labels = torch.load(args.labels)      # (N,) 或 (N, C)
    prototypes = torch.load(args.prototypes)  # (C, D)

    print(f"特征形状: {features.shape}")
    print(f"标签形状: {labels.shape}")
    print(f"标签维度: {labels.dim()}")

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
    # 修复：创建 gamma_raw 参数并确保它是叶子张量
    gamma_raw = nn.Parameter(torch.tensor(1.0).log())  # 创建时就在 CPU 上
    # include gamma in optimizer - 在移动到设备之前添加到优化器
    optimizer = optim.Adam(trainable_params + [gamma_raw], lr=args.lr)
    # 然后将 gamma_raw 移动到设备
    gamma_raw = gamma_raw.to(device)
    
    # 处理不同版本的 AMP API
    if USE_NEW_AMP:
        scaler = GradScaler()
    else:
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
        # 修复：从检查点恢复 gamma_raw
        if "gamma_raw" in ck:
            # 创建新的参数并设置值
            gamma_raw.data = ck["gamma_raw"].to(device)
        start_epoch = ck["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        start = time.time()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            # 处理不同版本的 autocast API
            if USE_NEW_AMP:
                with autocast():
                    gamma = torch.exp(gamma_raw)
                    logits = model(x, gamma=gamma)
                    # 使用 KL 散度损失处理 one-hot 标签
                    loss = kl_div_loss_with_onehot(logits, y)
            else:
                with autocast():
                    gamma = torch.exp(gamma_raw)
                    logits = model(x, gamma=gamma)
                    # 使用 KL 散度损失处理 one-hot 标签
                    loss = kl_div_loss_with_onehot(logits, y)
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
                # 处理不同版本的 autocast API
                if USE_NEW_AMP:
                    with autocast():
                        gamma = torch.exp(gamma_raw)
                        logits = model(x, gamma=gamma)
                        loss = kl_div_loss_with_onehot(logits, y)
                else:
                    with autocast():
                        gamma = torch.exp(gamma_raw)
                        logits = model(x, gamma=gamma)
                        loss = kl_div_loss_with_onehot(logits, y)
                val_loss += loss.item() * x.size(0)
                all_logits.append(logits)
                all_labels.append(y)
            
            val_loss = val_loss / len(val_dataset)
            all_logits = torch.cat(all_logits, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            
            # 转换为类别索引用于评估指标
            labels_for_metrics = convert_to_class_indices(all_labels)
            val_nll = F.nll_loss(F.log_softmax(all_logits, dim=1), labels_for_metrics, reduction='mean').item()
            val_acc = accuracy(all_logits, labels_for_metrics)
            val_top5 = topk_accuracy(all_logits, labels_for_metrics, k=5)

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
            "gamma_raw": gamma_raw.data.cpu()  # 保存时移动到 CPU
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
    parser.add_argument("--features", type=str, default="/mnt/dataset0/thm/code/battleday_varimnist/data/features.pt")
    parser.add_argument("--labels", type=str, default="/mnt/dataset0/thm/code/battleday_varimnist/data/labels.pt")
    parser.add_argument("--prototypes", type=str, default="/mnt/dataset0/thm/code/battleday_varimnist/data/prototypes.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--logdir", type=str, default="/mnt/dataset0/thm/code/battleday_varimnist/runs/prototype")
    parser.add_argument("--checkpoint", type=str, default="/mnt/dataset0/thm/code/battleday_varimnist/checkpoint_prototype.pth")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--early_stop_delta", type=float, default=1e-4)
    args = parser.parse_args()
    train(args)