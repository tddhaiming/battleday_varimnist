# train_exemplar.py
import sys
import os
import json
import time
import numpy as np
import torch
import torch.nn as nn

# 把项目根目录加入 Python 路径
project_root = "/mnt/dataset0/thm/code/battleday_varimnist"
sys.path.append(project_root)

from models.exemplar import ExemplarModel
from utils.metrics import (
    compute_top1_accuracy,
    compute_expected_accuracy,
    compute_nll,
    compute_spearman_per_image,
    compute_aic
)

# --------------------------------------------------
# 手动加载数据（绕过 get_train_val_features）
# --------------------------------------------------
base_path = "/mnt/dataset0/thm/code/battleday_varimnist"

features  = np.load(f"{base_path}/features/resnet18_features.npy")      # (116715, 512)
softlabels = np.load(f"{base_path}/data/processed/softlabels.npy")      # (116715, 10)

with open(f"{base_path}/data/splits.json", "r") as f:
    splits = json.load(f)

train_idx = splits["train"]
val_idx   = splits["val"]

X_train = torch.tensor(features[train_idx]).float()
y_train = torch.tensor(softlabels[train_idx]).float()
X_val   = torch.tensor(features[val_idx]).float()
y_val   = torch.tensor(softlabels[val_idx]).float()

print(f"Train samples: {X_train.shape[0]}, features: {X_train.shape[1]}")
print(f"Val   samples: {X_val.shape[0]},   features: {X_val.shape[1]}")

# --------------------------------------------------
# 训练配置
# --------------------------------------------------
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs      = 80
train_bs    = 16
eval_bs     = 32

model = ExemplarModel(
    exemplars=X_train.numpy(),
    labels=y_train.argmax(dim=1).numpy(),
    feature_dim=X_train.shape[1]
).to(device)

optimizer = torch.optim.Adam([model.beta, model.gamma, model.Sigma_inv], lr=1e-2)
criterion = nn.KLDivLoss(reduction="batchmean")

# --------------------------------------------------
# 训练循环
# --------------------------------------------------
model.train()
for epoch in range(epochs):
    epoch_loss = 0.0
    num_batches = 0
    start_time = time.time()

    for i in range(0, len(X_train), train_bs):
        xb = X_train[i:i+train_bs].to(device)
        yb = y_train[i:i+train_bs].to(device)

        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(torch.log(preds + 1e-8), yb)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        num_batches += 1

    if epoch % 5 == 0 or epoch == epochs - 1:
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1}/{epochs}  |  loss={avg_loss:.4f}  |  "
              f"time={time.time()-start_time:.2f}s")

print("训练完成！")

# --------------------------------------------------
# 保存模型
# --------------------------------------------------
os.makedirs(f"{base_path}/results", exist_ok=True)
torch.save({
    'model_state_dict': model.state_dict(),
    'exemplar_count': len(X_train),
    'feature_dim': X_train.shape[1],
    'beta': model.beta.item(),
    'gamma': model.gamma.item(),
    'Sigma_inv': model.Sigma_inv.detach().cpu().numpy()
}, f"{base_path}/results/exemplar_resnet18_full.pth")

# --------------------------------------------------
# 评估
# --------------------------------------------------
model.eval()
all_preds, all_targets = [], []

with torch.no_grad():
    for i in range(0, len(X_val), eval_bs):
        xb = X_val[i:i+eval_bs].to(device)
        pb = model(xb)
        all_preds.append(pb.cpu().numpy())
        all_targets.append(y_val[i:i+eval_bs].numpy())

preds    = np.vstack(all_preds)
targets  = np.vstack(all_targets)

top1_acc = compute_top1_accuracy(preds, targets)
exp_acc  = compute_expected_accuracy(preds, targets)
nll      = compute_nll(preds, targets)
spearman = compute_spearman_per_image(preds, targets)
k        = sum(p.numel() for p in model.parameters())
aic      = compute_aic(nll, k)

print("\n验证集性能 (Exemplar):")
print(f"  Top-1 Accuracy : {top1_acc:.2f}%")
print(f"  Expected Acc   : {exp_acc:.2f}%")
print(f"  NLL            : {nll:.4f}")
print(f"  Spearman       : {spearman:.4f}")
print(f"  AIC            : {aic:.4f}")
print(f"  Params         : {k}")
print(f"  Storage        : {len(X_train)*512 + len(X_train)}")

# --------------------------------------------------
# 保存结果到 JSON
# --------------------------------------------------
results_file = f"{base_path}/results/model_comparison_results.json"
result_dict = {
    "top1_accuracy": top1_acc,
    "expected_accuracy": exp_acc,
    "nll": nll,
    "spearman": spearman,
    "aic": aic,
    "trainable_params": k,
    "total_storage": len(X_train)*512 + len(X_train),
    "num_exemplars": len(X_train)
}

if os.path.exists(results_file):
    with open(results_file, "r") as f:
        all_results = json.load(f)
else:
    all_results = {"prototype": {}, "exemplar": {}}

all_results["exemplar"]["resnet18_full"] = result_dict
with open(results_file, "w") as f:
    json.dump(all_results, f, indent=2)

print("评估结果已写入", results_file)