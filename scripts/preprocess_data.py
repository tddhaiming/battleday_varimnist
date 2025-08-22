# scripts/preprocess_data.py
import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import json

def preprocess_varimnist():
    """预处理 variMNIST 数据"""
    # 加载 CSV (更新路径)
    csv_path = '/content/gdrive/MyDrive/battleday_varimnist/variMNIST/varMNIST/df_DigitRecog.csv'
    df = pd.read_csv(csv_path)
    
    print("CSV 列名:", df.columns.tolist())
    print("数据形状:", df.shape)
    
    # 收集所有图像路径和创建软标签
    image_paths = []
    softlabels = []
    
    # 遍历所有行
    for idx, row in enumerate(tqdm(df.iterrows(), total=len(df), desc="处理数据")):
        _, row_data = row
        # 获取图像路径
        stimulus_path = row_data['stimulus']
        if 'dataset_PNG/' in stimulus_path:
            stimulus_path = stimulus_path.replace('dataset_PNG/', '')
        
        # 更新路径
        full_path = os.path.join('/content/gdrive/MyDrive/battleday_varimnist/variMNIST/varMNIST/dataset_PNG', stimulus_path)
        
        # 检查文件是否存在
        if os.path.exists(full_path):
            image_paths.append(full_path)
            
            # 创建软标签
            response_counts = [row_data.get(f'response_{i}', 0) for i in range(10)]
            total_responses = sum(response_counts)
            
            if total_responses > 0:
                soft_label = np.array(response_counts) / total_responses
            else:
                true_label = int(row_data.get('true_label', 0))
                soft_label = np.zeros(10)
                soft_label[true_label] = 1.0
            
            softlabels.append(soft_label)
        else:
            print(f"文件不存在: {full_path}")
    
    print(f"成功处理 {len(image_paths)} 张图像")
    
    # 读取图像
    images = []
    valid_indices = []
    
    for idx, path in enumerate(tqdm(image_paths, desc="读取图像")):
        try:
            img = Image.open(path).convert('L')
            img = img.resize((28, 28))
            images.append(np.array(img))
            valid_indices.append(idx)
        except Exception as e:
            print(f"无法读取图像 {path}: {e}")
    
    # 过滤有效的标签
    valid_softlabels = [softlabels[i] for i in valid_indices]
    
    images = np.array(images)
    softlabels = np.array(valid_softlabels)
    
    print(f"最终图像形状: {images.shape}")
    print(f"最终标签形状: {softlabels.shape}")
    
    # 保存预处理数据
    os.makedirs("/content/gdrive/MyDrive/battleday_varimnist/data/processed", exist_ok=True)
    np.save("/content/gdrive/MyDrive/battleday_varimnist/data/processed/images.npy", images)
    np.save("/content/gdrive/MyDrive/battleday_varimnist/data/processed/softlabels.npy", softlabels)
    
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
    
    with open("/content/gdrive/MyDrive/battleday_varimnist/data/splits.json", "w") as f:
        json.dump(splits, f)
    
    print("数据预处理完成!")

if __name__ == "__main__":
    preprocess_varimnist()