import sys
import os
sys.path.append("/mnt/dataset0/thm/code/battleday_varimnist")

import numpy as np

# 手动检查每个文件是否存在
base_path = "/mnt/dataset0/thm/code/battleday_varimnist"

features_path = f"{base_path}/features/resnet18_features.npy"
softlabels_path = f"{base_path}/data/processed/softlabels.npy"
splits_path = f"{base_path}/data/splits.json"

print("features exists:", os.path.exists(features_path))
print("softlabels exists:", os.path.exists(softlabels_path))
print("splits exists:", os.path.exists(splits_path))

# 如果文件都存在，手动加载试试
if os.path.exists(features_path):
    features = np.load(features_path)
    print("Features loaded, shape:", features.shape)
else:
    print("❌ Features file missing!")

if os.path.exists(softlabels_path):
    softlabels = np.load(softlabels_path)
    print("Softlabels loaded, shape:", softlabels.shape)
else:
    print("❌ Softlabels file missing!")

if os.path.exists(splits_path):
    import json
    with open(splits_path, "r") as f:
        splits = json.load(f)
    print("Splits loaded, keys:", list(splits.keys()))
else:
    print("❌ Splits file missing!")