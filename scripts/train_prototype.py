# scripts/train_prototype.py
"""训练 Prototype模型（Classic/ Linear/ Quadratic）
特点（根据你的优化要求）：
-使用.pt features（features/train_features.pt, features/train_labels.pt, val_*）
- DataLoader:默认 batch_size=256, num_workers=8, pin_memory=True
-验证间隔：每 VAL_CHECK_EVERY轮计算一次 val loss；若连续 PATIENCE次无改 善则早停
-支持学习 gamma/ c/ c_c（论文中描述的参数）
-保留原始算法逻辑（计算 Mahalanobis距离并用 Luce-Shepard样式的 softchoice）
- 更新：使用 utils/metrics.py 中的新指标，并保存到 JSON
"""
import os
import argparse
import sys
# 获取当前脚本所在目录的父目录 (即 your_project_root)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 将项目根目录添加到 sys.path
sys.path.insert(0, project_root)
import json
import torch
from torch import nn
from utils.data_utils import make_dataloader
from models.prototype import ClassicPrototype, LinearPrototype, QuadraticPrototype
# --- 更新 import ---
from utils.metrics import accuracy, nll_loss, topk_accuracy, sba

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
VAL_CHECK_EVERY = 5
PATIENCE = 3


# --- 更新 evaluate 函数 ---
def evaluate(model, loader, device):
    model.eval()
    total_metrics = {
        'loss': 0.0,
        'acc': 0.0,
        'top5_acc': 0.0, # 新增 Top-5 Accuracy
        'sba': 0.0       # 新增 Second-Best Accuracy
    }
    count = 0
    with torch.no_grad():
        for feats, labels in loader:
            feats, labels = feats.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            logits = model(feats)
            
            loss = nll_loss(logits, labels)
            acc = accuracy(logits, labels)
            top5_acc = topk_accuracy(logits, labels, k=5) # 计算 Top-5
            sba_score = sba(logits, labels)               # 计算 SBA
            
            B = feats.size(0)
            total_metrics['loss'] += loss * B
            total_metrics['acc'] += acc * B
            total_metrics['top5_acc'] += top5_acc * B
            total_metrics['sba'] += sba_score * B
            count += B
            
    if count > 0:
        for key in total_metrics:
            total_metrics[key] /= count
    else:
        # Handle empty loader case
        total_metrics = {k: float('inf') if k == 'loss' else 0.0 for k in total_metrics}
        
    return total_metrics
# --- 结束 evaluate 函数更新 ---


def train(args):
    # 1. 构建 DataLoader
    train_loader = make_dataloader(
        args.train_feat, args.train_label,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True
    )
    val_loader = make_dataloader(
        args.val_feat, args.val_label,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True
    )

        # 2. 初始化模型 (修改点：根据模型类型传递不同参数)
    model_cls_map = {
        'classic': ClassicPrototype,
        'linear': LinearPrototype,
        'quadratic': QuadraticPrototype
    }
    if args.proto_type not in model_cls_map:
        raise ValueError(f"Unknown proto_type: {args.proto_type}")
    ModelClass = model_cls_map[args.proto_type]

    # --- 修改：根据 proto_type 构建模型参数字典 ---
    model_kwargs = {
        'feature_dim': args.feature_dim,
        'num_classes': args.num_classes,
        'learn_gamma': args.learn_gamma,
        'init_gamma': args.init_gamma,
    }
    # 为需要的模型类型添加特定参数
    if args.proto_type in ['linear']:
        model_kwargs['init_c'] = args.init_c
    elif args.proto_type in ['quadratic']:
        # 注意：QuadraticPrototype 使用 init_cc 而不是 init_c
        model_kwargs['init_cc'] = args.init_c

    # 使用 **kwargs 方式传递参数
    model = ModelClass(**model_kwargs).to(DEVICE)
    # --- 结束修改 ---

    # 3. 加载训练数据用于构建初始 prototypes
    train_feats = torch.load(args.train_feat).float()
    train_labels = torch.load(args.train_label).long()
    model.build_prototypes_from_features(train_feats, train_labels)
    print(f'Built initial prototypes from {train_feats.size(0)} exemplars.')

    # 4. 检查模型是否有可学习参数
    has_params = any(p.requires_grad for p in model.parameters())
    print(f'Model has {"learnable" if has_params else "no"} trainable parameters.')

    # 5. 如果有可学习参数，则设置优化器
    optimizer = None
    if has_params:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 6. 训练/评估循环
    os.makedirs('results', exist_ok=True)
    best_val_loss = float('inf')
    checks_no_improve = 0
    epoch = 0
    best_metrics = {}

    while True:
        epoch += 1
        if has_params and optimizer:
            model.train()
            train_total_loss = 0.0
            train_count = 0
            for feats, labels in train_loader:
                feats = feats.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)
                optimizer.zero_grad()
                logits = model(feats)
                loss = nn.CrossEntropyLoss()(logits, labels)
                loss.backward()
                optimizer.step()
                B = feats.size(0)
                train_total_loss += loss.item() * B
                train_count += B
            avg_train_loss = train_total_loss / train_count if train_count > 0 else float('inf')
            print(f'== Epoch {epoch} ==')
            print(f'Train loss: {avg_train_loss:.6f}')
        else:
            print(f'== Epoch {epoch} ==')
            print('Model has no trainable parameters; skipping parameter updates.')

        # validation every VAL_CHECK_EVERY epochs
        if epoch % VAL_CHECK_EVERY == 0:
            # --- 使用更新后的 evaluate 函数 ---
            val_metrics = evaluate(model, val_loader, DEVICE)
            val_loss = val_metrics['loss']
            print(f'Validation metrics at epoch {epoch}:')
            for k, v in val_metrics.items():
                print(f'  {k}: {v:.6f}')

            if val_loss + 1e-9 < best_val_loss:
                best_val_loss = val_loss
                checks_no_improve = 0
                # --- 更新最佳指标字典 ---
                best_metrics = {'epoch': epoch, **val_metrics}
                torch.save(model.state_dict(), f'results/prototype_{args.proto_type}_best.pth')
                print('Saved best model.')
            else:
                checks_no_improve += 1
                print(f'No improvement count: {checks_no_improve}/{args.patience_checks}')
                if checks_no_improve >= args.patience_checks:
                    print('Early stopping triggered.')
                    break

        if epoch >= args.epochs:
             print(f'Max epochs ({args.epochs}) reached.')
             break

    # --- 最终评估和保存指标到 JSON ---
    if best_metrics:
        model.load_state_dict(torch.load(f'results/prototype_{args.proto_type}_best.pth', map_location=DEVICE))
        print("Loaded best model weights for final evaluation.")

        final_metrics = evaluate(model, val_loader, DEVICE)
        print('Final validation metrics:')
        for k, v in final_metrics.items():
            print(f'  {k}: {v:.6f}')

        # 准备保存到 JSON 的所有信息
        results_dict = {
            'model_type': 'prototype',
            'proto_type': args.proto_type,
            'feature_dim': args.feature_dim,
            'num_classes': args.num_classes,
            'best_epoch': best_metrics['epoch'],
            'best_metrics': best_metrics,      # 包含所有最佳指标
            'final_metrics': final_metrics,    # 包含所有最终指标
            'hyperparameters': {
                'batch_size': args.batch_size,
                'num_workers': args.num_workers,
                'epochs': args.epochs,
                'lr': args.lr,
                'weight_decay': args.weight_decay,
                'val_check_every': VAL_CHECK_EVERY,
                'patience_checks': args.patience_checks,
                'learn_gamma': args.learn_gamma,
                'init_gamma': args.init_gamma,
                'init_c': args.init_c,
            }
        }

        json_filename = f'results/prototype_{args.proto_type}_results.json'
        with open(json_filename, 'w') as f:
            json.dump(results_dict, f, indent=4)
        print(f'Final results saved to {json_filename}')
    else:
        print("No best model was saved, skipping final evaluation and JSON save.")
    print('Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-feat', type=str, default='features/train_features.pt')
    parser.add_argument('--train-label', type=str, default='features/train_labels.pt')
    parser.add_argument('--val-feat', type=str, default='features/val_features.pt')
    parser.add_argument('--val-label', type=str, default='features/val_labels.pt')
    parser.add_argument('--feature-dim', type=int, required=True)
    parser.add_argument('--num-classes', type=int, default=10)
    parser.add_argument('--proto-type', type=str, required=True, choices=['classic', 'linear', 'quadratic'])

    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--val-check-every', type=int, default=5, help='(Ignored, kept for alignment)')
    parser.add_argument('--patience-checks', type=int, default=3)

    parser.add_argument('--learn-gamma', action='store_true', help='If set, gamma will be a learnable parameter.')
    parser.add_argument('--init-gamma', type=float, default=1.0, help='Initial value for gamma.')
    parser.add_argument('--init-c', type=float, default=1.0, help='Initial value for c (and c_c if applicable).')

    args = parser.parse_args()
    train(args)
