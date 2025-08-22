# scripts/extract_features.py
import torch
import torch.nn as nn
import numpy as np
import os
from torchvision.models import resnet18, vgg11
from torchvision import transforms
from tqdm import tqdm

def get_pretrained_feature_extractor(model_name='resnet18'):
    """获取预训练特征提取器"""
    if model_name == 'resnet18':
        # 使用 ResNet18
        model = resnet18(pretrained=True)
        # 修改第一层以适应灰度图像
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # 移除最后的分类层
        model.fc = nn.Identity()
        feature_dim = 512
    elif model_name == 'vgg11':
        # 使用 VGG11
        model = vgg11(pretrained=True)
        # 修改第一层
        model.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        # 移除分类器的最后一层
        model.classifier[-1] = nn.Identity()
        feature_dim = 4096
    else:
        raise ValueError(f"不支持的模型: {model_name}")
    
    return model, feature_dim

def extract_features_with_pretrained(model_name='resnet18'):
    """使用预训练模型提取特征"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 获取预训练模型
    model, feature_dim = get_pretrained_feature_extractor(model_name)
    model = model.to(device)
    
    # 冻结所有参数
    for param in model.parameters():
        param.requires_grad = False
    
    model.eval()
    
    # 加载 variMNIST 数据
    try:
        images = np.load("/content/gdrive/MyDrive/battleday_varimnist/data/processed/images.npy")
        softlabels = np.load("/content/gdrive/MyDrive/battleday_varimnist/data/processed/softlabels.npy")
    except:
        print("请先运行数据预处理脚本!")
        return
    
    print(f"加载图像数据: {images.shape}")
    
    # 预处理图像
    images_tensor = torch.tensor(images).float() / 255.0
    if images_tensor.dim() == 3:
        images_tensor = images_tensor.unsqueeze(1)  # 添加通道维度
    
    # 标准化 (使用 ImageNet 的均值和标准差，但适应灰度图)
    normalize = transforms.Normalize(mean=[0.485], std=[0.229])  # 近似值
    images_tensor = normalize(images_tensor)
    
    # 提取特征
    features = []
    batch_size = 32  # 减小 batch size 以适应 GPU 内存
    
    with torch.no_grad():
        for i in tqdm(range(0, len(images_tensor), batch_size), desc="提取特征"):
            batch = images_tensor[i:i+batch_size].to(device)
            batch_features = model(batch)
            features.append(batch_features.cpu().numpy())
    
    features = np.vstack(features)
    
    # 保存特征
    os.makedirs("/content/gdrive/MyDrive/battleday_varimnist/features", exist_ok=True)
    np.save(f"/content/gdrive/MyDrive/battleday_varimnist/features/{model_name}_features.npy", features)
    np.save("/content/gdrive/MyDrive/battleday_varimnist/data/processed/softlabels.npy", softlabels)  # 确保标签也被保存
    
    print(f"特征提取完成!")
    print(f"特征形状: {features.shape}")
    print(f"标签形状: {softlabels.shape}")
    print(f"使用的模型: {model_name}")

if __name__ == "__main__":
    # 使用 ResNet18 提取特征
    extract_features_with_pretrained('resnet18')