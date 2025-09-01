# scripts/train_exemplar.py
""" 训练 Exemplar模型（no-attention/ attention）
要点：
- 使用.pt特征与 DataLoader（batch default 256，num_workers 8，pin_memory True）
- 支持把 exemplar bank 一次性 cache 到 GPU（若显存足够，推荐）
- 使用 torch.cdist 或向量化加权距离；支持 chunk_size 分块以控制峰值内存
- Mixed precision + early stopping（每 val_check_every 轮验证，patience_check 次无改善则 early stop）
- 更新：使用 utils/metrics.py 中的新指标，并保存到 JSON
- 新增：定期保存检查点(checkpoint)到指定目录
- 新增：处理 Ctrl+C(KeyboardInterrupt)并保存检查点
- 优化：为 ExemplarAttention 模型设置不同的默认学习率和权重衰减
"""
import os
import argparse
import sys
# 获取当前脚本所在目录的父目录(即 your_project_root)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 将项目根目录添加到 sys.path
sys.path.insert(0, project_root)

import json
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from utils.data_utils import make_dataloader
from models.exemplar import ExemplarNoAttention, ExemplarAttention
from utils.metrics import accuracy, nll_loss, topk_accuracy, sba # 修正导入: nll_loss -> nll

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

#---新增：加载检查点函数(可选，用于恢复训练)---
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
        start_epoch = checkpoint.get('epoch', 0) + 1 #从下一个 epoch开始
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        checks_no_improve = checkpoint.get('checks_no_improve', 0)
        print(f"Checkpoint loaded. Resuming from epoch {start_epoch}")
        return start_epoch, best_val_loss, checks_no_improve
    else:
        print(f"No checkpoint found at '{filename}'")
        return 0, float('inf'), 0
#---结束新增---

# --- 更新 evaluate 函数 ---
def evaluate(model, loader, device, chunk_size=None):
    """ 在给定的数据加载器上评估模型。
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

            # --- 修改点：使用修正后的 nll 函数 ---
            # utils/metrics.py 中的 nll 实际上调用了 F.cross_entropy
            loss = nll_loss(logits, labels)
            # --- 结束修改点 ---
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

def train(args):
    # 1. Load exemplar bank(exemplar features + labels)
    exemplar_feats = torch.load(args.exemplar_feat).float()
    exemplar_labels = torch.load(args.exemplar_label).long()

    # --- 修改点：根据新的数据划分加载训练和验证数据 ---
    # 假设 data_utils.py 或 preprocess_data.py 生成的 index 文件结构已更新
    # 这里直接使用 make_dataloader 加载 train 和 val 的 .pt 文件
    # 注意：你的 extract_features.py 可能也需要更新，以根据新的划分生成 val_features.pt
    # 目前假设 val 数据也是从 train set 中划分出来的，或者你有单独的 val .pt 文件
    # 如果 val .pt 文件不存在，你可能需要修改 extract_features.py 来处理 'test' split
    # 为了兼容性，我们暂时保持加载 val_features.pt 的逻辑，但实际内容应来自原 CSV 的 'test' 部分
    train_loader = make_dataloader(
        args.train_feat, args.train_label,
        batch_size=args.batch_size, shuffle=True, # 训练时 shuffle
        num_workers=args.num_workers, pin_memory=True
    )
    val_loader = make_dataloader(
        args.val_feat, args.val_label, # 确保这些文件对应 CSV 中的 'test' 集
        batch_size=args.batch_size, shuffle=False, # 验证时不 shuffle
        num_workers=args.num_workers, pin_memory=True
    )
    # --- 结束修改点 ---


    # 2. Choose and initialize model
    if args.exemplar_type == 'no-attention':
        model = ExemplarNoAttention(args.feature_dim, args.num_classes, exemplar_feats, exemplar_labels)
    elif args.exemplar_type == 'attention':
        model = ExemplarAttention(args.feature_dim, args.num_classes, exemplar_feats, exemplar_labels)
    else:
        raise ValueError(f'Unknown exemplar_type: {args.exemplar_type}')

    model.to(DEVICE)

    # 3. Try caching exemplar bank on GPU(recommended on A100)
    if args.cache_exemplar_to_gpu:
        try:
            model.cache_exemplars_to_device(DEVICE)
            print('Exemplar bank cached to device', DEVICE)
        except Exception as e:
            print('Failed to cache exemplar to device:', e)

    # 4. Setup optimizer only if model has trainable params
    params = [p for p in model.parameters() if p.requires_grad]

    # --- 修改点：为不同的 Exemplar 模型类型设置不同的默认优化器参数 ---
    # 核心优化点：ExemplarAttention 的 w 参数可能需要不同的学习率或正则化
    if args.exemplar_type == 'attention':
        # 如果命令行没有指定，则使用为 Attention 模型优化的默认值
        lr = args.lr if args.lr != 1e-3 else 5e-4 # 默认学习率降低
        weight_decay = args.weight_decay if args.weight_decay != 0.0 else 1e-5 # 默认增加轻微的 L2 正则化
        print(f"Using optimized hyperparameters for ExemplarAttention: lr={lr}, weight_decay={weight_decay}")
    else:
        lr = args.lr
        weight_decay = args.weight_decay
    # --- 结束修改点 ---

    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay) if params else None
    scaler = torch.amp.GradScaler() if params else None

    # 5. Setup for training loop
    checkpoint_dir = 'results/checkpoints' # 简化路径
    os.makedirs(checkpoint_dir, exist_ok=True)

    best_val_loss = float('inf')
    checks_no_improve = 0
    os.makedirs('results', exist_ok=True)
    best_metrics = {}

    start_epoch = 0
    resume_checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint_interrupted_epoch_11.pth')
    if os.path.isfile(resume_checkpoint_path): # 检查文件是否存在
        print(f"Attempting to resume training from checkpoint: {resume_checkpoint_path}")
        start_epoch, best_val_loss, checks_no_improve = load_checkpoint(
            resume_checkpoint_path, model, optimizer, scaler
        )
    else:
        print(f"No checkpoint found at '{resume_checkpoint_path}', starting from scratch.")
        start_epoch = 0
        best_val_loss = float('inf')
        checks_no_improve = 0
        
    # --- 修改：将主训练循环放入 try-except 块 ---
    try:
        for epoch in range(start_epoch, args.epochs): # 修改为从 start_epoch 开始
            print(f'== Epoch {epoch + 1} ==') # epoch 从 0 开始，打印时 +1

            if params and optimizer and scaler:
                model.train()
                running_loss = 0.0
                num_batches = 0
                for feats, labels in train_loader:
                    feats = feats.to(DEVICE, non_blocking=True)
                    labels = labels.to(DEVICE)
                    optimizer.zero_grad()
                    with torch.cuda.amp.autocast():
                        logits = model(feats, chunk_size=args.chunk_size, use_gpu=True)
                        # --- 修改点：明确使用 CrossEntropyLoss ---
                        # 这与 utils/metrics.py 中的 nll 函数行为一致
                        loss_fn = nn.CrossEntropyLoss()
                        loss = loss_fn(logits, labels)
                        # --- 结束修改点 ---
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    running_loss += loss.item() * feats.size(0)
                    num_batches += 1

                if num_batches > 0:
                     train_loss = running_loss / len(train_loader.dataset) # 或者 / num_batches / batch_size
                     print('Train loss:', train_loss)
                else:
                     print('No training batches processed.')
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
            if (epoch + 1) % 10 == 0 or epoch >= args.epochs - 3: # 每 10 个 epoch 或最后 3 个 epoch 保存
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
    # --- 结束修改 ---

    # --- 新增：最终评估和保存指标到 JSON ---
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
                    'lr': lr, # 保存实际使用的 lr
                    'weight_decay': weight_decay, # 保存实际使用的 weight_decay
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
    parser.add_argument('--train-feat', default='features/train_features.pt') # 应该是 CSV 中 is_train=1 的特征
    parser.add_argument('--train-label', default='features/train_labels.pt')
    parser.add_argument('--val-feat', default='features/test_features.pt') # 应该是 CSV 中 is_train=0 的特征 (即 test set)
    parser.add_argument('--val-label', default='features/test_labels.pt')
    parser.add_argument('--exemplar-feat', default='features/exemplar_features.pt') # 通常使用 train set 作为 exemplar bank
    parser.add_argument('--exemplar-label', default='features/exemplar_labels.pt')
    parser.add_argument('--feature-dim', type=int, required=True)
    parser.add_argument('--num-classes', type=int, default=10)
    parser.add_argument('--exemplar-type', choices=['no-attention', 'attention'], required=True)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=80)
    # --- 修改点：调整默认学习率和权重衰减 ---
    parser.add_argument('--lr', type=float, default=1e-3) # 现在作为基础默认值，Attention 会覆盖
    parser.add_argument('--weight-decay', type=float, default=0.0) # 现在作为基础默认值，Attention 会覆盖
    # --- 结束修改点 ---
    parser.add_argument('--val-check-every', type=int, default=5)
    parser.add_argument('--patience-checks', type=int, default=3)
    parser.add_argument('--chunk-size', type=int, default=None, help='若 exemplar 大，可用此参数分块计算距离（例如 16384）')
    parser.add_argument('--cache-exemplar-to-gpu', action='store_true', dest='cache_exemplar_to_gpu', help='尝试将 exemplar bank 常驻 GPU')
    args = parser.parse_args()

    train(args)
