""" extract_features.py
功能：读取 data/processed/dataset_index.pt，使用 ResNet18 特征提取器把图像转为 vector features，并保存为 features/*.pt
输出文件：
 - features/train_features.pt (Tensor float32 (N_train, D))
 - features/train_labels.pt (LongTensor (N_train,))
 - features/val_features.pt
 - features/val_labels.pt
 - features/exemplar_features.pt -> 默认等于 train_features (用于 ExemplarModel)
 - features/exemplar_labels.pt
注意：请确保 FEATURE_DIM 与模型初始化时一致。
""" #新代码
import os
import argparse
from PIL import Image
import torch
from torch import nn
from torchvision import transforms, models # 新增 models 导入
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# 注意：SimpleCNNFeatureExtractor 已被移除

class ImageListDataset(Dataset):
    def __init__(self, items, transform=None):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        # 原始图像是灰度图 (L)
        img = Image.open(it['path']).convert('L')
        if self.transform:
            img = self.transform(img)
        label = int(it['label'])
        return img, label, it['id']

def extract(meta_path, out_dir='features', feature_dim=512, batch_size=256, num_workers=8, device='cuda'):
    os.makedirs(out_dir, exist_ok=True)
    meta = torch.load(meta_path)
    items = meta['items']

    # 拆分
    splits = {'train': [], 'val': [], 'test': []}
    for it in items:
        splits[it['split']].append(it)

    # --- 只使用 ResNet18 ---
    # 加载预训练的 ResNet18 模型
    model = models.resnet18(pretrained=True)
    # 替换最后的全连接层以输出指定的 feature_dim
    # ResNet 的 fc 层输入维度通常是 512 (对于 resnet18/34)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, feature_dim) # 替换为输出 feature_dim 的线性层

    # 对于 ResNet，输入需要是 3 通道 RGB 图像
    # 因此，transform 需要先将灰度图转为 RGB
    transform = transforms.Compose([
        transforms.Resize((224, 224)), # ResNet 通常使用 224x224 输入
        transforms.Grayscale(num_output_channels=3), # 将单通道转为三通道
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # ImageNet 标准化
    ])
    print(f"Loaded pretrained ResNet18, modified final layer to output {feature_dim} dims.")
    # --- 结束 ResNet18 配置 ---

    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()

    for split_name in ['train', 'val', 'test']:
        split_items = splits[split_name]
        if len(split_items) == 0:
            continue
        ds = ImageListDataset(split_items, transform=transform) # 使用 ResNet 对应的 transform
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

        all_feats = []
        all_labels = []
        all_ids = []
        with torch.no_grad():
            for imgs, labels, ids in tqdm(loader, desc=f'extract {split_name}'):
                imgs = imgs.to(device, non_blocking=True)
                feats = model(imgs)
                feats = feats.cpu()
                all_feats.append(feats)
                all_labels.append(labels)
                all_ids.extend(ids)

        feats = torch.cat(all_feats, dim=0)
        labs = torch.cat(all_labels, dim=0).long()

        torch.save(feats, os.path.join(out_dir, f'{split_name}_features.pt'))
        torch.save(labs, os.path.join(out_dir, f'{split_name}_labels.pt'))
        torch.save(all_ids, os.path.join(out_dir, f'{split_name}_ids.pt'))
        print(f'saved {split_name} features: {feats.shape} -> {out_dir}/{split_name}_features.pt')

    # exemplar bank 默认使用 train features
    if os.path.exists(os.path.join(out_dir, 'train_features.pt')):
        torch.save(torch.load(os.path.join(out_dir, 'train_features.pt')), os.path.join(out_dir, 'exemplar_features.pt'))
        torch.save(torch.load(os.path.join(out_dir, 'train_labels.pt')), os.path.join(out_dir, 'exemplar_labels.pt'))
        print('saved exemplar bank from train set')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--meta', type=str, default='data/processed/dataset_index.pt')
    parser.add_argument('--out', type=str, default='features')
    parser.add_argument('--feature-dim', type=int, default=512)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--num-workers', type=int, default=8)
    # --- 移除了 --model-type 参数 ---
    args = parser.parse_args()
    # --- 调用 extract 时不再传递 model_type ---
    extract(args.meta, out_dir=args.out, feature_dim=args.feature_dim, batch_size=args.batch_size,
            num_workers=args.num_workers) # 移除了 model_type 参数
