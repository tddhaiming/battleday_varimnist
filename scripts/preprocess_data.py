""" preprocess_data.py
功能：读取 CSV（df_DigitRecog.csv）及图像目录，生成 index列表并进行 train/val/test划分，保存为 data/processed/dataset_index.pt
使用场景：在 Colab上运行（数据挂载在 Google Drive）
输出：data/processed/dataset_index.pt，文件内容为字典：{
'items':[{'path': str,'label': int,'id': str,'split':'train'|'val'|'test'},...],
'num_classes': C }
注意：脚本尝试自动识别 CSV中的路径/label列。若 CSV结构特别，请用--img-col/--label-col指定。
""" #新代码
import os
import argparse
import csv
from PIL import Image
import random
import torch

def infer_csv_columns(sample_row): #尝试自动识别可能的列名
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

def main(csv_path, img_root, out_path, img_col=None, label_col=None, train_ratio=0.7, val_ratio=0.15):
    items=[]
    with open(csv_path,'r', newline='', encoding='utf-8') as f:
        reader= csv.DictReader(f)
        rows= list(reader)
        if not rows:
            raise RuntimeError('CSV empty')
    if img_col is None or label_col is None:
        img_col_infer, label_col_infer= infer_csv_columns(rows[0])
        if img_col is None:
            img_col= img_col_infer
        if label_col is None:
            label_col= label_col_infer
    for r in rows:
        raw_path= r.get(img_col,'').strip()
        label_raw= r.get(label_col, None)
        if raw_path=='' or label_raw is None:
            continue
        #若 CSV中存的是相对路径或文件名，尝试在 img_root下找到 full_path
        full_path= raw_path
        if not os.path.isabs(full_path):
            cand= os.path.join(img_root, raw_path)
            if os.path.exists(cand):
                full_path= cand
            else:
                #有时候 CSV只给文件名，尝试在 subdirs
                found= None
                for root,_, files in os.walk(img_root):
                    if os.path.basename(raw_path) in files:
                        found= os.path.join(root, os.path.basename(raw_path))
                        break
                if found:
                    full_path= found
        if not os.path.exists(full_path):
            #跳过不可达文件，但记录
            print(f"警告:文件不存在，跳过:{full_path}")
            continue
        try:
            lab= int(label_raw)
        except Exception:
            lab= label_raw
        items.append({'path': full_path,'label': lab,'id': os.path.basename(full_path)})
    # stratified split
    items= stratified_split(items, train_ratio=train_ratio, val_ratio=val_ratio)
    #统计 classes
    classes= sorted(list({it['label'] for it in items}))
    class_to_idx={c:i for i,c in enumerate(classes)}
    for it in items:
        it['label']= class_to_idx[it['label']]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    meta={'items': items,'num_classes': len(classes)}
    torch.save(meta, out_path)
    print('Saved dataset index to', out_path)

if __name__=='__main__':
    parser= argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True)
    parser.add_argument('--img-root', type=str, required=True)
    parser.add_argument('--out', type=str, default='data/processed/dataset_index.pt')
    parser.add_argument('--img-col', type=str, default=None)
    parser.add_argument('--label-col', type=str, default=None)
    parser.add_argument('--train-ratio', type=float, default=0.7)
    parser.add_argument('--val-ratio', type=float, default=0.15)
    args= parser.parse_args()
    main(args.csv, args.img_root, args.out, img_col=args.img_col, label_col=args.label_col, train_ratio=args.train_ratio, val_ratio=args.val_ratio)
