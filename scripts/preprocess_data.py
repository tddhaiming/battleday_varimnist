""" preprocess_data.py
功能：读取宽格式 CSV（df_DigitRecog.csv）及图像目录，生成 index列表并进行 train/val/test划分，保存为 data/processed/dataset_index.pt
使用场景：在 Colab上运行（数据挂载在 Google Drive）
输出：data/processed/dataset_index.pt，文件内容为字典：{
'items':[{'path': str,'label': int,'id': str,'split':'train'|'val'|'test'},...],
'num_classes': C }
注意：此脚本专为处理 df_DigitRecog.csv 的宽格式而设计。
""" #新代码
import os
import argparse
import csv
from PIL import Image
import random
import torch

# --- 新增常量 ---
# 定义需要从 CSV 路径中移除的前缀
PATH_PREFIX_TO_REMOVE = "round2_dataset/"
# --- 结束新增 ---

# 注意：infer_csv_columns 函数在此脚本中不再使用，但保留以维持结构
def infer_csv_columns(sample_row):
    keys=[k.lower() for k in sample_row.keys()]
    img_candidates=[k for k in sample_row.keys() if'file' in k.lower() or'img' in k.lower() or'path' in k.lower()]
    label_candidates=[k for k in sample_row.keys() if'label' in k.lower() or'class' in k.lower() or'digit' in k.lower()]
    return img_candidates[0] if img_candidates else list(sample_row.keys())[0], label_candidates[0] if label_candidates else list(sample_row.keys())[-1]

def stratified_split(items, train_ratio=0.7, val_ratio=0.15, seed=42): # items: list of dicts with'label'
    random.seed(seed)
    by_label={}
    for it in items:
        by_label.setdefault(it['label'],[]).append(it)
    train, val, test=[],[],[]
    for lab, group in by_label.items():
        random.shuffle(group)
        n= len(group)
        n_train= int(n* train_ratio)
        n_val= int(n* val_ratio)
        train.extend(group[:n_train])
        val.extend(group[n_train:n_train+n_val])
        test.extend(group[n_train+n_val:])
    #标注 split
    for it in train:
        it['split']='train'
    for it in val:
        it['split']='val'
    for it in test:
        it['split']='test'
    return train+ val+ test

# --- 新增函数：解析宽格式 CSV ---
def parse_wide_format_csv(csv_path):
    """解析 df_DigitRecog.csv 的宽格式"""
    items = []
    expected_group_size = 7 # 每组7列

    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for line_number, row in enumerate(reader, 1): # 从1开始计数行号
            num_cols = len(row)
            if num_cols % expected_group_size != 0:
                 print(f"警告: 第 {line_number} 行列数 ({num_cols}) 不是 {expected_group_size} 的整数倍，可能数据不完整，跳过该行。")
                 continue

            num_groups = num_cols // expected_group_size
            for group_index in range(num_groups):
                start_col = group_index * expected_group_size
                # ID (0), Path (1), Label (2), Col4 (3), Col5 (4), Col6 (5), Col7 (6)
                # 我们只需要 Path (索引1) 和 Label (索引2)
                path_raw = row[start_col + 1].strip()
                label_raw = row[start_col + 2].strip()

                if not path_raw or not label_raw:
                    # print(f"信息: 第 {line_number} 行, 第 {group_index+1} 组路径或标签为空，跳过。")
                    continue

                try:
                    label = int(label_raw)
                except ValueError:
                    print(f"警告: 第 {line_number} 行, 第 {group_index+1} 组标签 '{label_raw}' 无法转换为整数，跳过。")
                    continue

                # --- 处理路径前缀 ---
                if path_raw.startswith(PATH_PREFIX_TO_REMOVE):
                    path_relative = path_raw[len(PATH_PREFIX_TO_REMOVE):]
                else:
                    path_relative = path_raw
                # --- 结束路径前缀处理 ---

                item_id = os.path.basename(path_raw) # 使用原始路径的文件名作为ID
                items.append({
                    'path': path_relative, # 存储处理后的相对路径
                    'label': label,
                    'id': item_id
                })

    print(f"从宽格式 CSV 中解析出 {len(items)} 个项目。")
    return items
# --- 结束新增函数 ---

def main(csv_path, img_root, out_path, train_ratio=0.7, val_ratio=0.15): # 移除了 img_col, label_col 参数
    # --- 修改点1：使用新的解析函数 ---
    raw_items = parse_wide_format_csv(csv_path)
    # --- 结束修改点1 ---

    if not raw_items:
         raise RuntimeError("未能从 CSV 文件中解析出任何有效数据项。")

    # --- 修改点2：验证并构建完整路径 ---
    print("开始验证文件路径...")
    items = []
    for it in raw_items:
        rel_path = it['path']
        # 构建完整路径
        full_path = os.path.join(img_root, rel_path)

        # 检查文件是否存在
        if os.path.exists(full_path):
            it['path'] = full_path # 更新 'path' 为完整路径
            items.append(it)
        else:
             # 如果直接拼接失败，尝试在 img_root 下递归查找（作为后备方案，但会慢）
             # 注意：这可能会很慢，如果 CSV 路径是正确的，应该不需要这一步。
             # 但为了健壮性，可以保留。
             found_path = None
             try:
                 for root, _, files in os.walk(img_root):
                     if os.path.basename(rel_path) in files:
                         found_path = os.path.join(root, os.path.basename(rel_path))
                         break
             except KeyboardInterrupt:
                 print("\n用户中断了文件查找过程。")
                 raise
             if found_path and os.path.exists(found_path):
                 print(f"信息: 通过递归查找找到文件: {rel_path} -> {found_path}")
                 it['path'] = found_path
                 items.append(it)
             else:
                 print(f"警告: 文件不存在: {full_path} (原始相对路径: {rel_path})")

    print(f"验证后剩余 {len(items)} 个有效样本项。")
    if not items:
        raise RuntimeError("验证后没有找到任何有效的图像文件。请检查 img-root 路径和 CSV 中的路径是否匹配。")
    # --- 结束修改点2 ---

    # 3. 进行分层划分
    items = stratified_split(items, train_ratio=train_ratio, val_ratio=val_ratio)

    # 4. 统计类别
    classes = sorted(list({it['label'] for it in items}))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    for it in items:
        it['label'] = class_to_idx[it['label']] # 将标签映射为从0开始的索引

    # 5. 保存结果 (此部分未修改，保持原样)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    meta = {'items': items, 'num_classes': len(classes)}
    torch.save(meta, out_path)
    print('Saved dataset index to', out_path)
    print(f"最终数据集信息: 总样本数={len(items)}, 类别数={len(classes)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True, help='宽格式 CSV 文件路径')
    parser.add_argument('--img-root', type=str, required=True, help='包含图像的根目录路径')
    parser.add_argument('--out', type=str, default='data/processed/dataset_index.pt', help='输出索引文件路径')
    # --- 修改点3：移除不再需要的参数 ---
    # parser.add_argument('--img-col', type=str, default=None)
    # parser.add_argument('--label-col', type=str, default=None)
    # --- 结束修改点3 ---
    parser.add_argument('--train-ratio', type=float, default=0.7)
    parser.add_argument('--val-ratio', type=float, default=0.15)
    args = parser.parse_args()

    # --- 修改点4：调用 main 时移除 img_col, label_col ---
    main(args.csv, args.img_root, args.out, train_ratio=args.train_ratio, val_ratio=args.val_ratio)
    # --- 结束修改点4 ---
