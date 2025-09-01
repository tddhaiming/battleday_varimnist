# preprocess_data.py
""" preprocess_data.py 功能：读取长格式 CSV（df_DigitRecog.csv）及图像目录，
生成 index 列表并根据 varMNIST 的 is_train 字段做划分，保存为 data/processed/dataset_index.pt
使用场景：在 Colab 上运行（数据挂载在 Google Drive）
输出：data/processed/dataset_index.pt，文件内容为字典：
{
    'items': [{'path': str, 'label': int, 'id': str, 'split': 'train' | 'test'}, ...],
    'num_classes': C,
    'subjects': list of unique subject_ids # 新增：记录所有被试ID
}
注意：此脚本专为处理 df_DigitRecog.csv 的长格式而设计，并使用 is_train 字段。
"""
# 新代码 (适配长格式 CSV)
import os
import argparse
import pandas as pd # 使用 pandas 处理 CSV 更方便
import torch

# --- 新增常量 ---
# 定义需要从 CSV 路径中移除的前缀
PATH_PREFIX_TO_REMOVE = "round2_dataset/"
# --- 结束新增 ---

def parse_long_format_csv_with_is_train(csv_path):
    """解析 df_DigitRecog.csv 的长格式，并使用 is_train 字段进行划分"""
    print(f"开始读取 CSV 文件: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
        print(f"CSV 文件读取成功，共有 {len(df)} 行数据。")
    except Exception as e:
        raise RuntimeError(f"读取 CSV 文件失败: {e}")

    required_columns = ['stimulus', 'response', 'is_train']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"CSV 文件缺少必要的列: {missing_columns}")

    items = []
    subjects = set() # 用于收集唯一的 subject_id

    for index, row in df.iterrows():
        try:
            path_raw = str(row['stimulus']).strip()
            label_raw = str(row['response']).strip()
            is_train_raw = str(row['is_train']).strip().lower()

            if not path_raw or path_raw.lower() == 'nan':
                print(f"警告: 第 {index} 行的 stimulus 路径为空或无效，跳过。")
                continue
            if not label_raw or label_raw.lower() == 'nan':
                print(f"警告: 第 {index} 行的 response 标签为空或无效，跳过。")
                continue

            label = int(float(label_raw)) # 处理可能的浮点数字符串

            # 处理 is_train
            if is_train_raw in ['1', 'true', 'yes']:
                split = 'train'
            elif is_train_raw in ['0', 'false', 'no']:
                split = 'test'
            else:
                print(f"警告: 第 {index} 行的 is_train 值 '{is_train_raw}' 无法识别，跳过。")
                continue

            # --- 处理路径前缀 ---
            if path_raw.startswith(PATH_PREFIX_TO_REMOVE):
                path_relative = path_raw[len(PATH_PREFIX_TO_REMOVE):]
            else:
                path_relative = path_raw
            # --- 结束路径前缀处理 ---

            item_id = os.path.basename(path_raw) # 使用原始路径的文件名作为 ID

            item = {
                'path': path_relative,  # 存储处理后的相对路径
                'label': label,
                'id': item_id,
                'split': split,
            }
            # 如果有 subject_id 字段，也加入
            if 'subject_id' in row:
                subject_id_raw = str(row['subject_id']).strip()
                if subject_id_raw and subject_id_raw.lower() != 'nan':
                    item['subject_id'] = subject_id_raw
                    subjects.add(subject_id_raw) # 收集 subject_id

            items.append(item)

        except (ValueError, KeyError) as e:
            print(f"警告: 解析第 {index} 行时出错 ({e})，跳过该行。")
            continue
        except Exception as e:
            print(f"警告: 解析第 {index} 行时发生未知错误 ({e})，跳过该行。")
            continue

    print(f"从长格式 CSV 中解析出 {len(items)} 个有效项目。")
    sorted_subjects = sorted(list(subjects)) if subjects else []
    return items, sorted_subjects


def main(csv_path, img_root, out_path):
    # --- 修改点 1：使用新的解析函数处理长格式 CSV ---
    raw_items, subjects = parse_long_format_csv_with_is_train(csv_path)
    # --- 结束修改点 1 ---

    if not raw_items:
        raise RuntimeError("未能从 CSV 文件中解析出任何有效数据项。")

    # --- 修改点 2：验证并构建完整路径 ---
    print("开始验证文件路径...")
    items = []
    for it in raw_items:
        rel_path = it['path']
        # 构建完整路径
        full_path = os.path.join(img_root, rel_path)

        # 检查文件是否存在
        if os.path.exists(full_path):
            it['path'] = full_path  # 更新 'path' 为完整路径
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
    # --- 结束修改点 2 ---

    # 3. 不再进行分层划分，直接使用 CSV 中的 split 信息
    #    items 列表中的每个元素已经包含了 'split': 'train' 或 'test'

    # 4. 统计类别 (只统计 train 和 test 中的类别，理论上应该一致)
    all_labels = [it['label'] for it in items]
    classes = sorted(list(set(all_labels)))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    for it in items:
        it['label'] = class_to_idx[it['label']]  # 将标签映射为从 0 开始的索引

    # 5. 保存结果
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    meta = {
        'items': items,
        'num_classes': len(classes),
        'subjects': subjects # 新增：保存被试 ID
    }
    torch.save(meta, out_path)
    print('Saved dataset index to', out_path)

    train_count = sum(1 for it in items if it['split'] == 'train')
    test_count = sum(1 for it in items if it['split'] == 'test')
    print(f"最终数据集信息: 总样本数={len(items)}, 训练集={train_count}, 测试集={test_count}, 类别数={len(classes)}, 被试数={len(subjects)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True, help='长格式 CSV 文件路径 (df_DigitRecog.csv)')
    parser.add_argument('--img-root', type=str, required=True, help='包含图像的根目录路径')
    parser.add_argument('--out', type=str, default='data/processed/dataset_index.pt', help='输出索引文件路径')
    args = parser.parse_args()

    main(args.csv, args.img_root, args.out)
