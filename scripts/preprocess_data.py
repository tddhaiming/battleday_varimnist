# scripts/preprocess_data.py
import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import json

def find_image_file(base_dir, stimulus_path):
    """尝试多种可能的路径来找到图像文件"""
    # 清理路径
    clean_path = stimulus_path.replace('dataset_PNG/', '')
    clean_path = clean_path.replace('round2_dataset/', '')
    
    # 可能的路径组合
    possible_paths = [
        os.path.join(base_dir, stimulus_path),
        os.path.join(base_dir, clean_path),
        os.path.join(base_dir, 'uncertainty', os.path.basename(clean_path)),
        os.path.join(base_dir, 'controversial', os.path.basename(clean_path)),
        os.path.join(base_dir, 'uncertainty', os.path.basename(stimulus_path)),
        os.path.join(base_dir, 'controversial', os.path.basename(stimulus_path)),
    ]
    
    # 也尝试直接使用文件名在各个子目录中查找
    filename = os.path.basename(stimulus_path)
    for subdir in ['uncertainty', 'controversial']:
        possible_paths.append(os.path.join(base_dir, subdir, filename))
    
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
        subdirs = [d for d in os.listdir(base_image_dir) if os.path.isdir(os.path.join(base_image_dir, d))]
        print(f"子目录: {subdirs}")
    
    # 先测试前100个样本，看看能否找到文件
    print("测试前100个样本的路径查找...")
    test_count = 0
    found_count = 0
    
    for idx in range(min(100, len(df))):
        row_data = df.iloc[idx]
        stimulus_path = row_data['stimulus']
        
        full_path = find_image_file(base_image_dir, stimulus_path)
        
        if full_path and os.path.exists(full_path):
            found_count += 1
            if test_count < 5:  # 显示前5个找到的文件
                print(f"找到文件: {stimulus_path} -> {full_path}")
        else:
            if test_count < 5:  # 显示前5个未找到的文件
                print(f"未找到: {stimulus_path}")
        
        test_count += 1
        if test_count >= 10:  # 只测试前10个
            break
    
    print(f"测试结果: {found_count}/{test_count} 文件找到")
    
    if found_count == 0:
        print("警告: 测试样本中没有找到任何文件，继续处理所有数据...")
    
    # 收集所有图像路径和创建软标签
    image_paths = []
    softlabels = []
    found_files = 0
    missing_files = 0
    
    # 遍历所有行
    print("开始处理数据...")
    for idx in tqdm(range(len(df)), desc="处理数据"):
        row_data = df.iloc[idx]
        
        # 获取图像路径
        stimulus_path = row_data['stimulus']
        
        # 尝试找到文件
        full_path = find_image_file(base_image_dir, stimulus_path)
        
        if full_path and os.path.exists(full_path):
            image_paths.append(full_path)
            found_files += 1
            
            # 创建软标签 - 基于 response 列
            response = row_data['response']
            
            # 创建 one-hot 编码作为软标签
            soft_label = np.zeros(10)
            if 0 <= response <= 9:
                soft_label[response] = 1.0
            else:
                # 如果 response 不在 0-9 范围内，使用默认值
                soft_label[0] = 1.0
            
            softlabels.append(soft_label)
        else:
            missing_files += 1
            if missing_files <= 10:  # 只显示前10个缺失文件
                print(f"警告: 文件不存在: {stimulus_path}")
    
    print(f"处理完成 - 找到文件: {found_files}, 缺失文件: {missing_files}")
    
    if found_files == 0:
        print("错误: 没有找到任何有效的图像文件!")
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
        
        # 检查 stimulus 列的一些示例值
        print("\nstimulus 列示例值:")
        for i in range(min(10, len(df))):
            print(f"  {df.iloc[i]['stimulus']}")
        
        # 检查 response 列的值范围
        if 'response' in df.columns:
            print(f"\nresponse 列值范围: [{df['response'].min()}, {df['response'].max()}]")
            print(f"response 列唯一值: {sorted(df['response'].unique())[:20]}")
    else:
        print("找不到 CSV 文件")
    
    # 检查图像目录结构
    img_dir = '/content/gdrive/MyDrive/battleday_varimnist/variMNIST/varMNIST/dataset_PNG'
    print(f"\n图像目录存在: {os.path.exists(img_dir)}")

    if os.path.exists(img_dir):
        subdirs = [d for d in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, d))]
        print(f"子目录: {subdirs}")
        
        for subdir in subdirs:
            subdir_path = os.path.join(img_dir, subdir)
            if os.path.exists(subdir_path):
                files = [f for f in os.listdir(subdir_path) if f.endswith('.png')]
                print(f"  {subdir}: {len(files)} PNG 文件")
                if len(files) > 0:
                    print(f"    示例文件: {files[:5]}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "diagnose":
        diagnose_data_structure()
    else:
        preprocess_varimnist()