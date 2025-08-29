# scripts/train_exemplar.py
"""
训练 Exemplar 模型（no-attention / attention）
要点：
- 使用 .pt 特征与 DataLoader（batch default 256，num_workers 8，pin_memory True）
- 支持把 exemplar bank 一次性 cache 到 GPU（若显存足够，推荐）
- 使用 torch.cdist 或向量化加权距离；支持 chunk_size 分块以控制峰值内存
- Mixed precision + early stopping（每 val_check_every 轮验证，patience_check 次无改善则 early stop）
- 更新：使用 utils/metrics.py 中的新指标，并保存到 JSON
- 新增：定期保存检查点 (checkpoint) 到指定目录
- 新增：处理 Ctrl+C (KeyboardInterrupt) 并保存检查点
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
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from utils.data_utils import make_dataloader
from models.exemplar import ExemplarNoAttention, ExemplarAttention
from utils.metrics import accuracy, nll_loss, topk_accuracy, sba

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- 更新 evaluate 函数 ---
def evaluate(model, loader, device, chunk_size=None):
    """
    在给定的数据加载器上评估模型。
    返回包含 loss, acc, top5_acc, sba 的字典。
    """
    model.eval()
    total_metrics = {
        'loss': 0.0,
        'acc': 0.0,
        'top5_acc': 0.0,
        'sba': 0.0
    }
    count = 0
    with torch.no_grad():
        for feats, labels in loader:
            feats, labels = feats.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            logits = model(feats, chunk_size=chunk_size, use_gpu=True)
            
            loss = nll_loss(logits, labels)
            acc = accuracy(logits, labels)
            top5_acc = topk_accuracy(logits, labels, k=5)
            sba_score = sba(logits, labels)
            
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
        total_metrics = {k: float('inf') if k == 'loss' else 0.0 for k in total_metrics}
        
    return total_metrics
# --- 结束 evaluate 函数更新 ---

# --- 新增：保存检查点函数 ---
def save_checkpoint(state, filename):
    """保存训练状态到文件"""
    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")
# --- 结束新增 ---

# --- 新增：加载检查点函数 (可选，用于恢复训练) ---
def load_checkpoint(filename, model, optimizer, scaler):
    """从文件加载训练状态"""
    if os.path.isfile(filename):
        print(f"Loading checkpoint '{filename}'")
        checkpoint = torch.load(filename, map_location=DEVICE)
        model.load_state_dict(checkpoint['state_dict'])
        if optimizer and 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if scaler and 'scaler' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = checkpoint.get('epoch', 0) + 1 # 从下一个 epoch 开始
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        checks_no_improve = checkpoint.get('checks_no_improve', 0)
        print(f"Checkpoint loaded. Resuming from epoch {start_epoch}")
        return start_epoch, best_val_loss, checks_no_improve
    else:
        print(f"No checkpoint found at '{filename}'")
        return 0, float('inf'), 0
# --- 结束新增 ---

def train(args):
    # 1. Load exemplar bank (exemplar features + labels)
    exemplar_feats = torch.load(args.exemplar_feat).float()
    exemplar_labels = torch.load(args.exemplar_label).long()
    
    train_loader = make_dataloader(
        args.train_feat, args.train_label,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = make_dataloader(
        args.val_feat, args.val_label,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # 2. Choose and initialize model
    if args.exemplar_type == 'no-attention':
        model = ExemplarNoAttention(args.feature_dim, args.num_classes, exemplar_feats, exemplar_labels)
    elif args.exemplar_type == 'attention':
        model = ExemplarAttention(args.feature_dim, args.num_classes, exemplar_feats, exemplar_labels)
    else:
        raise ValueError(f'Unknown exemplar_type: {args.exemplar_type}')

    model.to(DEVICE)

    # 3. Try caching exemplar bank on GPU (recommended on A100)
    if args.cache_exemplar_to_gpu:
        try:
            model.cache_exemplars_to_device(DEVICE)
            print('Exemplar bank cached to device', DEVICE)
        except Exception as e:
            print('Failed to cache exemplar to device:', e)

    # 4. Setup optimizer only if model has trainable params
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay) if params else None
    scaler = GradScaler() if params else None

    # 5. Setup for training loop
    # --- 新增：设置检查点目录 ---
    checkpoint_dir = '/mnt/dataset0/thm/code/battleday_varimnist/results/checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    # --- 结束新增 ---
    
    best_val_loss = float('inf')
    checks_no_improve = 0
    os.makedirs('results', exist_ok=True)
    best_metrics = {}
    
    # --- 新增：初始化起始 epoch ---
    start_epoch = 0
    # 如果需要从检查点恢复，可以在这里调用 load_checkpoint
    # 例如: start_epoch, best_val_loss, checks_no_improve = load_checkpoint('path/to/checkpoint.pth', model, optimizer, scaler)
    # --- 结束新增 ---

    # --- 修改：将主训练循环放入 try-except 块 ---
    try:
        # for epoch in range(1, args.epochs + 1): # 原循环
        for epoch in range(start_epoch, args.epochs): # 修改为从 start_epoch 开始
            print(f'== Epoch {epoch + 1} ==') # epoch 从 0 开始，打印时 +1
            
            if params and optimizer and scaler:
                model.train()
                running_loss = 0.0
                for feats, labels in train_loader:
                    feats = feats.to(DEVICE, non_blocking=True)
                    labels = labels.to(DEVICE)
                    optimizer.zero_grad()
                    with autocast():
                        logits = model(feats, chunk_size=args.chunk_size, use_gpu=True)
                        loss = nn.CrossEntropyLoss()(logits, labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    running_loss += loss.item() * feats.size(0)
                train_loss = running_loss / len(train_loader.dataset)
                print('Train loss:', train_loss)
            else:
                print('Model has no trainable params (only inference/likelihood fitting) - skipping parameter updates')

            # validation every val_check_every epochs
            if (epoch + 1) % args.val_check_every == 0: # epoch 从 0 开始
                val_metrics = evaluate(model, val_loader, DEVICE, chunk_size=args.chunk_size)
                val_loss = val_metrics['loss']
                print(f'Validation metrics at epoch {epoch + 1}:')
                for k, v in val_metrics.items():
                    print(f'  {k}: {v:.6f}')

                if val_loss + 1e-9 < best_val_loss:
                    best_val_loss = val_loss
                    checks_no_improve = 0
                    best_metrics = {'epoch': epoch + 1, **val_metrics}
                    torch.save(model.state_dict(), f'results/exemplar_{args.exemplar_type}_best.pth')
                    print('Saved best model.')
                else:
                    checks_no_improve += 1
                    print(f'No improvement count: {checks_no_improve}/{args.patience_checks}')
                    if checks_no_improve >= args.patience_checks:
                        print('Early stopping triggered.')
                        break

            # --- 新增：定期保存检查点 ---
            # 例如，每隔 10 个 epoch 保存一次，或者在最后几个 epoch 保存
            if (epoch + 1) % 10 == 0 or epoch >= args.epochs - 3: # 每10个epoch或最后3个epoch保存
                 checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pth')
                 save_checkpoint({
                     'epoch': epoch,
                     'state_dict': model.state_dict(),
                     'optimizer': optimizer.state_dict() if optimizer else None,
                     'scaler': scaler.state_dict() if scaler else None,
                     'best_val_loss': best_val_loss,
                     'checks_no_improve': checks_no_improve,
                     'args': vars(args) # 保存命令行参数，方便恢复
                 }, checkpoint_path)
            # --- 结束新增 ---
            
        # --- 新增：如果循环正常结束，也保存一个最终检查点 ---
        final_checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_final.pth')
        save_checkpoint({
            'epoch': args.epochs - 1, # 或者是实际结束的 epoch
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict() if optimizer else None,
            'scaler': scaler.state_dict() if scaler else None,
            'best_val_loss': best_val_loss,
            'checks_no_improve': checks_no_improve,
            'args': vars(args)
        }, final_checkpoint_path)
        print("Final checkpoint saved.")
        # --- 结束新增 ---

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received. Saving checkpoint...")
        # --- 新增：在 KeyboardInterrupt 时保存检查点 ---
        interrupt_checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_interrupted_epoch_{epoch + 1}.pth')
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict() if optimizer else None,
            'scaler': scaler.state_dict() if scaler else None,
            'best_val_loss': best_val_loss,
            'checks_no_improve': checks_no_improve,
            'args': vars(args)
        }, interrupt_checkpoint_path)
        print("Checkpoint saved. Exiting...")
        # 重新抛出异常，让程序正常退出
        raise
        # --- 结束新增 ---
    # --- 结束修改 ---

    # --- 新增：最终评估和保存指标到 JSON (移到 try 块外面，确保即使中断也能（尝试）执行) ---
    # 注意：如果在 early stopping 或 KeyboardInterrupt 时中断，这部分可能不会执行
    # 或者需要在中断处理中也调用它。这里为了简化，放在 finally-like 的位置（但不是 finally 块）
    # 一个更健壮的方法是在中断处理和正常结束时都调用一个专门的“保存最终结果”函数。
    # 这里我们保留原逻辑，但要注意它依赖于 best_metrics 是否在中断前被设置。
    if best_metrics: # 确保有最佳模型被保存过
        try:
            model.load_state_dict(torch.load(f'results/exemplar_{args.exemplar_type}_best.pth', map_location=DEVICE))
            print("Loaded best model weights for final evaluation.")

            final_metrics = evaluate(model, val_loader, DEVICE, chunk_size=args.chunk_size)
            print('Final validation metrics:')
            for k, v in final_metrics.items():
                print(f'  {k}: {v:.6f}')

            results_dict = {
                'model_type': 'exemplar',
                'exemplar_type': args.exemplar_type,
                'feature_dim': args.feature_dim,
                'num_classes': args.num_classes,
                'best_epoch': best_metrics['epoch'],
                'best_metrics': best_metrics,
                'final_metrics': final_metrics,
                'hyperparameters': {
                    'batch_size': args.batch_size,
                    'num_workers': args.num_workers,
                    'epochs': args.epochs,
                    'lr': args.lr,
                    'weight_decay': args.weight_decay,
                    'chunk_size': args.chunk_size,
                    'val_check_every': args.val_check_every,
                    'patience_checks': args.patience_checks,
                    'cache_exemplar_to_gpu': args.cache_exemplar_to_gpu,
                }
            }

            json_filename = f'results/exemplar_{args.exemplar_type}_results.json'
            with open(json_filename, 'w') as f:
                json.dump(results_dict, f, indent=4)
            print(f'Final results saved to {json_filename}')
        except Exception as e:
            print(f"Error during final evaluation or saving JSON: {e}")
    else:
        print("No best model was saved, skipping final evaluation and JSON save.")
    print('Training finished (or interrupted).')
    # --- 结束新增 ---


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-feat', default='features/train_features.pt')
    parser.add_argument('--train-label', default='features/train_labels.pt')
    parser.add_argument('--val-feat', default='features/val_features.pt')
    parser.add_argument('--val-label', default='features/val_labels.pt')
    parser.add_argument('--exemplar-feat', default='features/exemplar_features.pt')
    parser.add_argument('--exemplar-label', default='features/exemplar_labels.pt')
    parser.add_argument('--feature-dim', type=int, required=True)
    parser.add_argument('--num-classes', type=int, default=10)
    parser.add_argument('--exemplar-type', choices=['no-attention','attention'], required=True)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--val-check-every', type=int, default=5)
    parser.add_argument('--patience-checks', type=int, default=3)
    parser.add_argument('--chunk-size', type=int, default=None, help='若 exemplar 大，可用此参数分块计算距离（例如 16384）')
    parser.add_argument('--cache-exemplar-to-gpu', action='store_true', dest='cache_exemplar_to_gpu', help='尝试将 exemplar bank 常驻 GPU')
    # --- 新增：添加恢复训练的检查点路径参数 (可选) ---
    # parser.add_argument('--resume', type=str, default='', help='path to latest checkpoint (default: none)')
    # --- 结束新增 ---
    args = parser.parse_args()
    
    # --- 新增：处理恢复训练的逻辑 (可选) ---
    # if args.resume:
    #     if os.path.isfile(args.resume):
    #         print(f"=> loading checkpoint '{args.resume}'")
    #         checkpoint = torch.load(args.resume)
    #         # ... (加载逻辑，见 load_checkpoint 函数)
    #     else:
    #         print(f"=> no checkpoint found at '{args.resume}'")
    # --- 结束新增 ---
    
    train(args)