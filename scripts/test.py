# scripts/preprocess_data.py
"""
Preprocess script:
- Convert existing features.npy / labels.npy into .pt (torch tensors)
- Compute prototypes (per-class means)
- Save exemplars (features + labels) as .pt for exemplar model usage.

Usage:
    python scripts/preprocess_data.py --feature_npy features.npy --label_npy labels.npy --out_dir data/ --num_classes 10
"""

import os
import argparse
import numpy as np
import torch


# === 新代码开始 ===
def npy_to_pt(feature_npy, label_npy, out_prefix="data/"):
    feats = np.load(feature_npy)  # (N, D)
    labs = np.load(label_npy)     # (N,)
    os.makedirs(out_prefix, exist_ok=True)
    torch.save(torch.tensor(feats, dtype=torch.float32), os.path.join(out_prefix, "features.pt"))
    torch.save(torch.tensor(labs, dtype=torch.long), os.path.join(out_prefix, "labels.pt"))
    print("Saved .pt files to", out_prefix)


def build_prototypes_and_exemplars(features_pt, labels_pt, out_prefix="data/", num_classes=10):
    feats = torch.load(features_pt)  # (N, D)
    labs = torch.load(labels_pt)     # (N,)
    prototypes = []
    for c in range(num_classes):
        mask = (labs == c)
        if mask.sum() == 0:
            prototypes.append(torch.zeros(feats.shape[1], dtype=torch.float32))
        else:
            prototypes.append(feats[mask].mean(dim=0))
    prototypes = torch.stack(prototypes, dim=0)  # (C, D)
    os.makedirs(out_prefix, exist_ok=True)
    torch.save(prototypes, os.path.join(out_prefix, "prototypes.pt"))
    # Save exemplars as-is (features + labels)
    torch.save(feats, os.path.join(out_prefix, "exemplars_features.pt"))
    torch.save(labs, os.path.join(out_prefix, "exemplars_labels.pt"))
    print("Saved prototypes and exemplars to", out_prefix)
# === 新代码结束 ===


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_npy", type=str, default="/mnt/dataset0/thm/code/battleday_varimnist/resnet18_features.npy")
    parser.add_argument("--label_npy", type=str, default="/mnt/dataset0/thm/code/battleday_varimnist/softlabels.npy")
    parser.add_argument("--out_dir", type=str, default="data/")
    parser.add_argument("--num_classes", type=int, default=10)
    args = parser.parse_args()

    npy_to_pt(args.feature_npy, args.label_npy, out_prefix=args.out_dir)
    build_prototypes_and_exemplars(os.path.join(args.out_dir, "features.pt"),
                                   os.path.join(args.out_dir, "labels.pt"),
                                   out_prefix=args.out_dir,
                                   num_classes=args.num_classes)
