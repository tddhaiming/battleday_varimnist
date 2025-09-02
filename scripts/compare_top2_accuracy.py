# scripts/compare_top2_accuracy.py
"""
比较不同模型在验证集上的 Top-2 Accuracy。
假设模型检查点已保存在 results/ 目录下，命名为 {model_type}_{sub_type}_best.pth
假设验证集数据为 features/test_features.pt 和 features/test_labels.pt
"""
import os
import sys

# 获取当前脚本所在目录的父目录(即 your_project_root)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 将项目根目录添加到 sys.path
sys.path.insert(0, project_root)
import argparse
import json
import torch
from utils.data_utils import make_dataloader
# 假设 models 目录在项目根目录下，且已在 sys.path 中（由 train_*.py 脚本处理）
# 如果直接运行此脚本，可能需要调整路径或确保 models 可导入
from models.prototype import ClassicPrototype, LinearPrototype, QuadraticPrototype
from models.exemplar import ExemplarNoAttention, ExemplarAttention
from utils.metrics import topk_accuracy # 确保这个函数可以计算 topk

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 定义模型配置和初始化参数
MODEL_CONFIGS = {
    # Prototype Models
    'prototype_classic': {
        'class': ClassicPrototype,
        'init_kwargs': {
            'feature_dim': None, # 需要从 args 或特征文件推断
            'num_classes': None, # 需要从 args 或特征文件推断
            'learn_gamma': True, # 通常训练时会学 gamma
            'init_gamma': 1.0,
        },
        'checkpoint_path': 'results/prototype_classic_best.pth'
    },
    'prototype_linear': {
        'class': LinearPrototype,
        'init_kwargs': {
            'feature_dim': None,
            'num_classes': None,
            'learn_gamma': True,
            'init_gamma': 1.0,
            'init_c': 1.0, # 默认初始值，实际可能需要从训练时的 args 或 checkpoint 恢复
        },
        'checkpoint_path': 'results/prototype_linear_best.pth'
    },
    'prototype_quadratic': {
        'class': QuadraticPrototype,
        'init_kwargs': {
            'feature_dim': None,
            'num_classes': None,
            'learn_gamma': True,
            'init_gamma': 1.0,
            'init_cc': 1.0, # 默认初始值
        },
        'checkpoint_path': 'results/prototype_quadratic_best.pth'
    },
    # Exemplar Models
    # 注意：Exemplar 模型需要 exemplar bank 来初始化
    'exemplar_no_attention': {
        'class': ExemplarNoAttention,
        'init_kwargs': {
            'feature_dim': None,
            'num_classes': None,
            # exemplar_feats 和 exemplar_labels 需要额外加载
        },
        'checkpoint_path': 'results/exemplar_no_attention_best.pth',
        'needs_exemplar_bank': True
    },
    'exemplar_attention': {
        'class': ExemplarAttention,
        'init_kwargs': {
            'feature_dim': None,
            'num_classes': None,
            # exemplar_feats 和 exemplar_labels 需要额外加载
        },
        'checkpoint_path': 'results/exemplar_attention_best.pth',
        'needs_exemplar_bank': True
    },
}

def load_features_info(feat_path, label_path):
    """加载特征和标签以获取维度和类别数"""
    feats = torch.load(feat_path)
    labels = torch.load(label_path)
    feature_dim = feats.shape[1]
    num_classes = len(torch.unique(labels))
    # 确保标签是连续的 0 到 num_classes-1
    assert labels.min() >= 0 and labels.max() < num_classes, "Labels should be in range [0, num_classes)"
    print(f"Loaded features: shape {feats.shape}, labels: shape {labels.shape}")
    print(f"Inferred feature_dim: {feature_dim}, num_classes: {num_classes}")
    return feats, labels, feature_dim, num_classes

def evaluate_top2(model, loader, device, model_name, chunk_size=None):
    """计算模型在数据加载器上的 Top-2 Accuracy"""
    model.eval()
    correct_top2 = 0.0
    total = 0
    with torch.no_grad():
        for feats, labels in loader:
            feats = feats.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # 调用模型的 forward 方法
            if 'exemplar' in model_name.lower():
                logits = model(feats, chunk_size=chunk_size, use_gpu=True)
            else:
                logits = model(feats)
            
            # 计算 Top-2 Accuracy
            top2_acc = topk_accuracy(logits, labels, k=2)
            
            B = feats.size(0)
            correct_top2 += top2_acc * B
            total += B
            
    if total > 0:
        final_top2_acc = correct_top2 / total
    else:
        final_top2_acc = 0.0
        
    print(f"{model_name} - Top-2 Accuracy: {final_top2_acc:.6f}")
    return final_top2_acc

def main(args):
    # 1. 加载验证集数据信息
    print("Loading validation set features and labels...")
    try:
        val_feats, val_labels, feature_dim, num_classes = load_features_info(args.val_feat, args.val_label)
    except Exception as e:
        print(f"Error loading features/labels: {e}")
        return

    # 2. 创建 DataLoader
    print("Creating DataLoader...")
    val_loader = make_dataloader(
        args.val_feat, args.val_label,
        batch_size=args.batch_size,
        shuffle=False, # 验证时不打乱
        num_workers=args.num_workers,
        pin_memory=True
    )

    # 3. 加载 Exemplar Bank (如果需要)
    exemplar_feats, exemplar_labels = None, None
    if any(config.get('needs_exemplar_bank', False) for config in MODEL_CONFIGS.values()):
        print("Loading exemplar bank for Exemplar models...")
        try:
            exemplar_feats = torch.load(args.exemplar_feat).float()
            exemplar_labels = torch.load(args.exemplar_label).long()
            print(f"Exemplar bank loaded: feats {exemplar_feats.shape}, labels {exemplar_labels.shape}")
        except Exception as e:
            print(f"Error loading exemplar bank: {e}")
            return # 如果 Exemplar 模型需要它，加载失败则无法继续

    # 4. 遍历模型配置并评估
    results = {}
    for model_key, config in MODEL_CONFIGS.items():
        print(f"\n--- Evaluating {model_key} ---")
        checkpoint_path = config['checkpoint_path']
        
        if not os.path.exists(checkpoint_path):
            print(f"  Checkpoint not found at {checkpoint_path}, skipping.")
            results[model_key] = None
            continue

        try:
            # 准备模型初始化参数
            init_kwargs = config['init_kwargs'].copy()
            init_kwargs['feature_dim'] = feature_dim
            init_kwargs['num_classes'] = num_classes

            # 如果是 Exemplar 模型，需要传入 exemplar bank
            if config.get('needs_exemplar_bank', False):
                if exemplar_feats is None or exemplar_labels is None:
                    print(f"  Exemplar bank not loaded, skipping {model_key}.")
                    results[model_key] = None
                    continue
                model = config['class'](
                    exemplar_feats=exemplar_feats,
                    exemplar_labels=exemplar_labels,
                    **init_kwargs
                )
            else:
                model = config['class'](**init_kwargs)
            
            model.to(DEVICE)
            
            # 加载模型权重
            print(f"  Loading checkpoint from {checkpoint_path}")
            # 使用 map_location 确保在正确的设备上加载
            model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
            print(f"  Checkpoint loaded successfully.")
            
            # 评估 Top-2 Accuracy
            top2_acc = evaluate_top2(model, val_loader, DEVICE, model_key, chunk_size=args.chunk_size)
            results[model_key] = top2_acc
            
        except Exception as e:
            print(f"  Error evaluating {model_key}: {e}")
            results[model_key] = None

    # 5. 打印汇总结果
    print("\n====================")
    print("Top-2 Accuracy Summary")
    print("====================")
    for model_name, acc in results.items():
        if acc is not None:
            print(f"{model_name:<30}: {acc:.6f}")
        else:
            print(f"{model_name:<30}: Not evaluated (missing checkpoint or error)")

    # 6. (可选) 保存结果到 JSON 文件
    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"\nResults saved to {args.output_json}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compare Top-2 Accuracy of Trained Models")
    parser.add_argument('--val-feat', type=str, default='features/test_features.pt', help='Path to validation features .pt file')
    parser.add_argument('--val-label', type=str, default='features/test_labels.pt', help='Path to validation labels .pt file')
    parser.add_argument('--exemplar-feat', type=str, default='features/exemplar_features.pt', help='Path to exemplar features .pt file (for Exemplar models)')
    parser.add_argument('--exemplar-label', type=str, default='features/exemplar_labels.pt', help='Path to exemplar labels .pt file (for Exemplar models)')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size for evaluation')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers for DataLoader')
    parser.add_argument('--chunk-size', type=int, default=None, help='Chunk size for Exemplar models (if needed for memory)')
    parser.add_argument('--output-json', type=str, default='results/top2_accuracy_comparison.json', help='Path to save results as JSON')
    
    args = parser.parse_args()
    main(args)