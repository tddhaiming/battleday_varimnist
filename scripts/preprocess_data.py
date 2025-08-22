# scripts/preprocess_data.py
import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import json

def find_image_file(base_dir, stimulus_path):
    """尝试多种可能的路径来找到图像文件"""
    # 原始路径
    possible_paths = [
        os.path.join(base_dir, stimulus_path),
        os.path.join(base_dir, stimulus_path.replace('dataset_PNG/', '')),
        os.path.join(base_dir, stimulus_path.replace('round2_dataset/', '')),
        os.path.join(base_dir, stimulus_path.split('/')[-1]),  # 只取文件名
    ]
    
    # 也尝试在不同的子目录中查找
    subdirs = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'controversial', 'uncertainty']
    for subdir in subdirs:
        possible_paths.append(os.path.join(base_dir, subdir, stimulus_path.split('/')[-1]))
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None

def preprocess_varimnist():
    """预处理 variMNIST 数据"""
    # 加载 CSV
    csv_path = '/content/gdrive/MyDrive/battleday_varimnist/variMNIST/varMNIST/df_DigitRecog.csv'
    print(f"尝试加载 CSV: {csv_path}")
    
    if not os.path.exists(csv_path):
        print(f"错误: 找不到 CSV 文件 {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    
    print("CSV 列名:", df.columns.tolist())
    print("数据形状:", df.shape)
    
    # 检查基础图像目录
    base_image_dir = '/content/gdrive/MyDrive/battleday_varimnist/variMNIST/varMNIST/dataset_PNG'
    print(f"基础图像目录存在: {os.path.exists(base_image_dir)}")
    
    if os.path.exists(base_image_dir):
        print("目录内容:", os.listdir(base_image_dir)[:10])  # 显示前10个
    
    # 收集所有图像路径和创建软标签
    image_paths = []
    softlabels = []
    found_files = 0
    missing_files = 0
    
    # 遍历所有行 (只处理前1000个作为测试)
    print("开始处理数据...")
    sample_size = min(1000, len(df))  # 先处理小样本测试
    for idx in range(sample_size):
        row_data = df.iloc[idx]
        
        # 获取图像路径
        stimulus_path = row_data['stimulus']
        
        # 尝试找到文件
        full_path = find_image_file(base_image_dir, stimulus_path)
        
        if full_path and os.path.exists(full_path):
            image_paths.append(full_path)
            found_files += 1
            
            # 创建软标签
            response_counts = []
            for i in range(10):
                col_name = f'response_{i}'
                if col_name in row_data:
                    response_counts.append(row_data[col_name])
                else:
                    response_counts.append(0)
            
            total_responses = sum(response_counts)
            
            if total_responses > 0:
                soft_label = np.array(response_counts) / total_responses
            else:
                # 尝试找到真实标签列
                true_label = 0
                if 'true_label' in row_data:
                    true_label = int(row_data['true_label'])
                elif 'label' in row_data:
                    true_label = int(row_data['label'])
                soft_label = np.zeros(10)
                soft_label[true_label] = 1.0
            
            softlabels.append(soft_label)
        else:
            missing_files += 1
            if missing_files <= 10:  # 只显示前10个缺失文件
                print(f"警告: 文件不存在: {stimulus_path}")
    
    print(f"测试样本 - 找到文件: {found_files}, 缺失文件: {missing_files}")
    
    if found_files == 0:
        print("错误: 没有找到任何有效的图像文件!")
        print("尝试列出数据目录结构...")
        
        # 尝试列出目录结构来诊断问题
        if os.path.exists(base_image_dir):
            for root, dirs, files in os.walk(base_image_dir):
                level = root.replace(base_image_dir, '').count(os.sep)
                indent = ' ' * 2 * level
                print(f'{indent}{os.path.basename(root)}/')
                subindent = ' ' * 2 * (level + 1)
                for file in files[:5]:  # 只显示前5个文件
                    print(f'{subindent}{file}')
                if len(files) > 5:
                    print(f'{subindent}... and {len(files)-5} more files')
        return
    
    # 读取图像
    images = []
    valid_indices = []
    
    print("开始读取图像...")
    for idx, path in enumerate(tqdm(image_paths, desc="读取图像")):
        try:
            img = Image.open(path).convert('L')
            img = img.resize((28, 28))
            images.append(np.array(img))
            valid_indices.append(idx)
        except Exception as e:
            print(f"无法读取图像 {path}: {e}")
    
    if len(images) == 0:
        print("错误: 没有成功读取任何图像!")
        return
    
    # 过滤有效的标签
    valid_softlabels = [softlabels[i] for i in valid_indices]
    
    images = np.array(images)
    softlabels = np.array(valid_softlabels)
    
    print(f"最终图像形状: {images.shape}")
    print(f"最终标签形状: {softlabels.shape}")
    
    # 检查数据是否有效
    if images.size == 0:
        print("错误: 图像数据为空!")
        return
    
    # 保存预处理数据
    output_dir = "/content/gdrive/MyDrive/battleday_varimnist/data/processed"
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "images.npy"), images)
    np.save(os.path.join(output_dir, "softlabels.npy"), softlabels)
    
    print(f"数据已保存到: {output_dir}")
    
    # 创建数据划分
    n_samples = len(images)
    indices = np.random.RandomState(42).permutation(n_samples)
    train_split = int(0.8 * n_samples)
    val_split = int(0.9 * n_samples)
    
    splits = {
        "train": indices[:train_split].tolist(),
        "val": indices[train_split:val_split].tolist(),
        "test": indices[val_split:].tolist()
    }
    
    splits_file = "/content/gdrive/MyDrive/battleday_varimnist/data/splits.json"
    with open(splits_file, "w") as f:
        json.dump(splits, f)
    
    print("数据预处理完成!")
    print(f"训练集: {len(splits['train'])} 样本")
    print(f"验证集: {len(splits['val'])} 样本")
    print(f"测试集: {len(splits['test'])} 样本")

def diagnose_data_structure():
    """诊断数据结构"""
    print("=== 诊断数据结构 ===")
    
    # 检查 CSV
    csv_path = '/content/gdrive/MyDrive/battleday_varimnist/variMNIST/varMNIST/df_DigitRecog.csv'
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"CSV 文件形状: {df.shape}")
        print("前几行:")
        print(df.head())
        print("列名:", df.columns.tolist())
    else:
        print("找不到 CSV 文件")
    
    # 检查图像目录
    img_dir = '/content/gdrive/MyDrive/battleday_varimnist/variMNIST/varMNIST/dataset_PNG'
    if os.path.exists(img_dir):
        print(f"\n图像目录 {img_dir} 存在")
        print("目录内容:")
        for item in os.listdir(img_dir)[:10]:
            print(f"  {item}")
    else:
        print(f"\n图像目录 {img_dir} 不存在")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "diagnose":
        diagnose_data_structure()
    else:
        preprocess_varimnist()