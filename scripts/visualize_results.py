# plot_all_metrics.py
"""
读取指定目录下的 JSON 结果文件，并为每个关键指标（包括 AIC）生成单独的柱状图。
只使用最终评估结果 (final_metrics)。
调整模型顺序: classic, linear, quadratic, no-attention, attention
图表标题只显示 'Cross-Entropy' 而不是 'NLL/Cross-Entropy'
"""

import os
import json
from unittest import result
import matplotlib.pyplot as plt
import numpy as np

# --- 定义模型显示顺序 ---
MODEL_ORDER = ['classic', 'linear', 'quadratic', 'no-attention', 'no_attention', 'attention']
# --- 结束定义 ---

def load_results(results_dir):
    """加载所有 JSON 结果文件"""
    json_files = [f for f in os.listdir(results_dir) if f.endswith('_results.json')]
    data = []
    for filename in json_files:
        filepath = os.path.join(results_dir, filename)
        try:
            with open(filepath, 'r') as f:
                json_data = json.load(f)
            
            # 从文件名和内容中提取模型信息
            # 假设文件名格式为: exemplar_{type}_results.json 或 prototype_{type}_results.json
            parts = filename.replace('_results.json', '').split('_')
            if len(parts) >= 2:
                model_category = parts[0].capitalize() # 'prototype' -> 'Prototype'
                # 处理文件名中的连字符
                model_type_raw = '_'.join(parts[1:]).replace('-', '_')
                # 为了排序，将 'no_attention' 统统为 'no-attention' 的形式（不带下划线）
                model_type_for_sort = model_type_raw.replace('_', '-')
            else:
                # 如果文件名格式不符合预期，则从 JSON 内容中尝试获取
                model_category = json_data.get('model_type', 'Unknown').capitalize()
                model_type_raw = json_data.get(f"{json_data.get('model_type', 'unknown')}_type", 'Unknown_Type')
                model_type_for_sort = model_type_raw
            
            model_name = f"{model_category}_{model_type_raw}"
            
            # 加载最终和最佳指标
            final_metrics = json_data.get('final_metrics', {})
            # best_metrics = json_data.get('best_metrics', {}) # 不再需要
            
            # --- 计算 AIC ---
            # AIC = 2k - 2ln(L) = 2k + 2 * NLL
            # 假设 NLL 就是 final_loss (交叉熵)
            # k_params 需要从模型或配置中获取，这里假设一个简单的估算方法
            # 或者，如果 JSON 中有保存，可以直接读取。
            # 这里我们使用一个简化的假设：k_params 与 feature_dim * num_classes 或 feature_dim 相关
            # 更精确的方法是让训练脚本在保存 JSON 时计算并保存 k_params。
            # 为了演示，我们假设 k_params 存在于 hyperparameters 中，或者使用一个默认估算。
            hyperparams = json_data.get('hyperparameters', {})
            # 尝试从 hyperparameters 获取 k_params (如果之前保存过)
            k_params = hyperparams.get('k_params', None)
            
            final_loss = final_metrics.get('loss', float('inf'))
            
            # 如果 JSON 中没有直接提供 k_params，这里需要一个估算逻辑
            # 由于模型结构未知，我们暂时无法精确计算。
            # 一种替代方法是，如果训练脚本计算了 AIC 并保存在 final_metrics 中，直接读取。
            # 否则，可以假设所有模型参数量相同（不现实）或省略 AIC 计算。
            # 为简单起见，我们假设 JSON 中已经包含了计算好的 AIC (如果有的话)。
            # 如果没有，我们标记为 None 或 NaN。
            final_aic = final_metrics.get('aic', np.nan) # 期望 JSON 中有 'aic' 键
            
            # --- 结束 AIC 计算 ---
            
            data.append({
                'name': model_name,
                'category': model_category,
                'type': model_type_raw,
                'type_for_sort': model_type_for_sort, # 用于排序的类型
                'final_acc': final_metrics.get('acc', 0),
                'final_top5_acc': final_metrics.get('top5_acc', 0),
                'final_sba': final_metrics.get('sba', 0),
                'final_loss': final_loss,
                'final_aic': final_aic, # 添加 AIC
                # 'best_acc': best_metrics.get('acc', 0), # 移除 best 指标
                # ... 其他 best 指标 ...
            })
            print(f"Loaded: {model_name}")
        except Exception as e:
            print(f"Warning: Could not load {filename}. Error: {e}")
    
    # --- 根据预定义顺序排序 ---
    def get_model_sort_key(item):
        category = item['category'].lower()
        mtype = item['type_for_sort'].lower()
        # 确定大类顺序：Prototype (0), Exemplar (1)
        category_order = 0 if category == 'prototype' else 1 if category == 'exemplar' else 2
        # 确定小类顺序
        try:
            type_order = MODEL_ORDER.index(mtype)
        except ValueError:
            # 如果类型不在预定义列表中，放在最后
            type_order = len(MODEL_ORDER)
        return (category_order, type_order)

    data.sort(key=get_model_sort_key)
    # --- 结束排序 ---
    
    return data

def plot_single_bar_chart(data, metric_key, metric_name, output_dir):
    """绘制单个指标的柱状图 (仅使用 final_ 指标)"""
    if not data:
        print(f"No data to plot for {metric_name}.")
        return

    # 确定使用最终指标
    value_key = f"final_{metric_key}"
    
    model_names = [d['name'] for d in data]
    values = [d[value_key] for d in data]

    # 处理 inf, nan, 或 aic 的 nan
    valid_indices = [i for i, v in enumerate(values) 
                     if not (isinstance(v, float) and (np.isnan(v) or np.isinf(v)))]
    if not valid_indices:
        print(f"All {metric_name} values are invalid (NaN or Inf). Skipping plot.")
        return
    model_names = [model_names[i] for i in valid_indices]
    values = [values[i] for i in valid_indices]

    if not model_names:
        print(f"No valid data points for {metric_name}.")
        return

    plt.figure(figsize=(max(10, len(model_names) * 0.8), 6))
    # 使用循环颜色映射，确保颜色多样
    colors = plt.cm.Set3(np.linspace(0, 1, len(model_names))) 
    bars = plt.bar(model_names, values, color=colors)

    plt.xlabel('Model')
    plt.ylabel(metric_name)
    # --- 修改标题，移除 NLL，为 AIC 特殊处理 ---
    display_metric_name = "Loss (Cross-Entropy)" if metric_key == "loss" else metric_name
    # 如果是 AIC，标题可以更明确
    if metric_key == "aic":
        display_metric_name = "AIC (Lower is Better)"
    plt.title(f'Comparison of Model Final {display_metric_name}')
    # --- 结束标题修改 ---
    
    # 设置 y 轴范围
    if 'loss' in metric_key.lower():
        valid_values = [v for v in values if not (isinstance(v, float) and (np.isnan(v) or np.isinf(v)))]
        if valid_values:
            min_val = min(valid_values)
            plt.ylim(bottom=max(0, min_val - 0.1 * abs(min_val)) if min_val != 0 else 0) 
    elif 'aic' in metric_key.lower():
        # AIC 通常没有固定范围，不设置 ylim 或根据数据动态调整
        pass
    else:
        plt.ylim(0, 1.05) # Accuracy 类指标范围是 0 到 1

    # 在柱状图上添加数值标签
    for bar, val in zip(bars, values):
        height = bar.get_height()
        # 处理负值或极小值的情况
        if not (isinstance(height, float) and (np.isnan(height) or np.isinf(height))):
             # AIC 值可能较大，调整标签格式
             label = f'{height:.1f}' if 'aic' in metric_key.lower() else f'{height:.3f}'
             plt.text(bar.get_x() + bar.get_width()/2.0, height + (abs(height) * 0.01 if height >= 0 else -abs(height) * 0.03),
                     label, ha='center', va='bottom' if height >= 0 else 'top', fontsize=8)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # 保存图表
    safe_metric_key = metric_key.replace("/", "_").replace(" ", "_") # 防止文件名中有非法字符
    filename = f"final_{safe_metric_key}_comparison.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    print(f"Saved Final {display_metric_name} comparison chart to {filepath}") # 使用修改后的名称打印
    plt.close() # 关闭图形以释放内存

def main():
    """主函数"""
    # 1. 设置路径
    results_dir = '/mnt/dataset0/thm/code/battleday_varimnist/results/final' # 指定你的 JSON 文件目录
    output_dir = '/mnt/dataset0/thm/code/battleday_varimnist/results/final/charts_final_only_ce_aic' # 修改输出目录名
    os.makedirs(output_dir, exist_ok=True) # 确保输出目录存在

    # 2. 加载数据
    if not os.path.exists(results_dir):
        print(f"Error: Results directory '{results_dir}' not found.")
        return

    print("Loading results...")
    results_data = load_results(results_dir)
    


    # 3. 定义要绘制的指标 (只包含 final_ 指标，并添加 AIC)
    # 格式: (指标键名, 图表标题)
    metrics_to_plot = [
        ('acc', 'Accuracy'),
        ('top5_acc', 'Top-5 Accuracy'),
        ('sba', 'Second-Best Accuracy (SBA)'),
        ('loss', 'Loss (Cross-Entropy)'), # <--- 修改这里
        ('aic', 'AIC') # <--- 添加 AIC
    ]

    # 4. 为每个指标绘制图表 (仅使用 Final)
    print("\nPlotting charts for Final Metrics...")
    for metric_key, metric_name in metrics_to_plot:
        plot_single_bar_chart(results_data, metric_key, metric_name, output_dir)
        
    # 注意：移除了 Best Metrics 的绘图循环
        
    print("\nAll charts (Final metrics only) have been generated and saved.")

if __name__ == '__main__':
    main()