import sys
import os
import numpy as np
import torch
import json

# 添加项目根目录到路径
sys.path.append("/mnt/dataset0/thm/code/battleday_varimnist")

def build_prototypes_and_exemplars(features_npy, softlabels_npy, splits_json, out_prefix="data/", num_classes=10):
    """
    生成原型和示例并保存为 .pt 文件，修复索引类型错误
    """
    # 1. 加载特征（.npy 格式）
    if not os.path.exists(features_npy):
        raise FileNotFoundError(f"特征文件不存在: {features_npy}")
    feats = np.load(features_npy)  # 形状 (N, 512)
    print(f"成功加载特征，形状: {feats.shape}")

    # 2. 加载软标签（.npy 格式，确认是数组而非 npz 对象）
    if not os.path.exists(softlabels_npy):
        raise FileNotFoundError(f"软标签文件不存在: {softlabels_npy}")
    softlabels = np.load(softlabels_npy)
    # 确保软标签是二维数组 (N, 10)
    if softlabels.ndim != 2 or softlabels.shape[1] != num_classes:
        raise ValueError(f"软标签形状错误，预期 (N, {num_classes})，实际 {softlabels.shape}")
    print(f"成功加载软标签，形状: {softlabels.shape}")

    # 3. 加载数据划分（关键修复：确保索引是整数类型）
    if not os.path.exists(splits_json):
        raise FileNotFoundError(f"划分文件不存在: {splits_json}")
    with open(splits_json, "r") as f:
        splits = json.load(f)
    # 将 JSON 中的列表转换为整数数组（避免字符串索引）
    train_idx = np.array(splits["train"], dtype=int)  # 转为整数数组
    print(f"训练集索引数量: {len(train_idx)}，索引类型: {train_idx.dtype}")

    # 4. 验证索引有效性（确保索引不超出特征数组范围）
    if len(train_idx) == 0:
        raise ValueError("训练集索引为空，请检查 splits.json")
    if train_idx.max() >= len(feats):
        raise IndexError(f"训练集索引超出特征范围（最大索引 {train_idx.max()}，特征长度 {len(feats)}）")

    # 5. 提取训练集数据
    train_feats = feats[train_idx]  # 使用整数数组索引
    train_softlabels = softlabels[train_idx]
    # 转换为一维类别标签（取概率最高的类别）
    train_labels = train_softlabels.argmax(axis=1)  # 形状 (N_train,)

    # 6. 计算每个类别的原型
    prototypes = []
    for c in range(num_classes):
        mask = (train_labels == c)  # 布尔掩码，形状 (N_train,)
        if mask.sum() == 0:
            # 若该类别无样本，用零向量填充
            prototypes.append(torch.zeros(train_feats.shape[1], dtype=torch.float32))
        else:
            # 计算该类别特征的均值
            class_feats = torch.tensor(train_feats[mask], dtype=torch.float32)
            prototypes.append(class_feats.mean(dim=0))
    prototypes_tensor = torch.stack(prototypes, dim=0)  # 形状 (10, 512)

    # 7. 保存为 .pt 文件
    os.makedirs(out_prefix, exist_ok=True)
    torch.save(prototypes_tensor, os.path.join(out_prefix, "prototypes.pt"))
    torch.save(torch.tensor(train_feats, dtype=torch.float32), 
               os.path.join(out_prefix, "exemplars_features.pt"))
    torch.save(torch.tensor(train_labels, dtype=torch.long), 
               os.path.join(out_prefix, "exemplars_labels.pt"))

    print(f"\n已保存 .pt 文件到 {out_prefix}")
    print(f"原型形状: {prototypes_tensor.shape}")
    print(f"示例特征形状: {train_feats.shape}")
    print(f"示例标签形状: {train_labels.shape}")

if __name__ == "__main__":
    # 配置文件路径
    base_path = "/mnt/dataset0/thm/code/battleday_varimnist"
    features_path = f"{base_path}/features/resnet18_features.npy"
    softlabels_path = f"{base_path}/data/processed/softlabels.npy"
    splits_path = f"{base_path}/data/splits.json"
    output_path = f"{base_path}/data/"

    # 检查文件存在性
    print("文件存在性检查:")
    print(f"特征文件: {os.path.exists(features_path)}")
    print(f"软标签文件: {os.path.exists(softlabels_path)}")
    print(f"划分文件: {os.path.exists(splits_path)}")

    # 执行构建函数
    try:
        build_prototypes_and_exemplars(
            features_npy=features_path,
            softlabels_npy=softlabels_path,
            splits_json=splits_path,
            out_prefix=output_path
        )
    except Exception as e:
        print(f"执行出错: {str(e)}")