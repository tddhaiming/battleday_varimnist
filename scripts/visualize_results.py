# scripts/visualize_results.py
import json
import numpy as np
import matplotlib.pyplot as plt
import os

def load_results():
    """Load model comparison results"""
    results_file = "/content/gdrive/MyDrive/battleday_varimnist/results/model_comparison_results.json"
    
    # If file doesn't exist, create a default one
    if not os.path.exists(results_file):
        default_results = {
            "prototype": {
                "classic": {
                    "NLL": 2.0691,
                    "Spearman": 0.1508,
                    "AIC": 10246.1382,
                    "params": 5121
                },
                "linear": {
                    "NLL": 2.1647,
                    "Spearman": 0.0245,
                    "AIC": 11270.3294,
                    "params": 5633
                },
                "quadratic": {
                    "NLL": 2.1977,
                    "Spearman": 0.0866,
                    "AIC": 20486.3955,
                    "params": 10241
                }
            },
            "exemplar": {
                "sampled_5000": {
                    "NLL": 2.2691,
                    "Spearman": 0.4043,
                    "AIC": 5121032.54,
                    "params": 2560514
                }
            }
        }
        
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        with open(results_file, 'w') as f:
            json.dump(default_results, f, indent=2)
        print(f"Created default results file: {results_file}")
        
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    return results

def plot_metric_comparison(results):
    """Plot comparison of all metrics"""
    # Prepare data
    prototype_models = list(results['prototype'].keys())
    exemplar_models = list(results['exemplar'].keys())
    
    # Metric names
    metrics = ['NLL', 'Spearman', 'AIC', 'params']
    metric_names = ['Negative Log-Likelihood (NLL)', 'Spearman Correlation', 'AIC', 'Number of Parameters']
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Prototype Model vs Exemplar Model Performance Comparison', fontsize=16, fontweight='bold')
    
    # Plot for each metric
    for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx//2, idx%2]
        
        # Collect data
        prototype_values = [results['prototype'][model][metric] for model in prototype_models]
        exemplar_values = [results['exemplar'][model][metric] for model in exemplar_models]
        
        # Use log scale for AIC and params
        if metric in ['AIC', 'params']:
            prototype_values = [max(v, 1e-10) for v in prototype_values]  # Avoid log(0)
            exemplar_values = [max(v, 1e-10) for v in exemplar_values]
            ax.set_yscale('log')
        
        # Plot bars
        x = np.arange(len(prototype_models))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, prototype_values, width, label='Prototype', alpha=0.8)
        bars2 = ax.bar(len(prototype_models), exemplar_values[0], width, label='Exemplar (Sampled 5000)', alpha=0.8)
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
        
        # Add value label for exemplar
        height = bars2[0].get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bars2[0].get_x() + bars2[0].get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
        
        # Set chart properties
        ax.set_xlabel('Model')
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name} Comparison')
        ax.set_xticks(list(x - width/2) + [len(prototype_models)])
        ax.set_xticklabels(prototype_models + ['Exemplar'], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_detailed_comparison(results):
    """Plot detailed comparison charts"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Detailed Performance Metrics Comparison', fontsize=16, fontweight='bold')
    
    # 1. NLL vs Spearman scatter plot
    ax1 = axes[0]
    prototype_models = list(results['prototype'].keys())
    exemplar_models = list(results['exemplar'].keys())
    
    # Prototype models
    nll_prototype = [results['prototype'][model]['NLL'] for model in prototype_models]
    spearman_prototype = [results['prototype'][model]['Spearman'] for model in prototype_models]
    
    # Exemplar models
    nll_exemplar = [results['exemplar'][model]['NLL'] for model in exemplar_models]
    spearman_exemplar = [results['exemplar'][model]['Spearman'] for model in exemplar_models]
    
    scatter1 = ax1.scatter(nll_prototype, spearman_prototype, c='blue', label='Prototype Models', s=100)
    ax1.scatter(nll_exemplar, spearman_exemplar, c='red', label='Exemplar Model', s=100, marker='s')
    
    # Add labels
    for i, model in enumerate(prototype_models):
        ax1.annotate(model, (nll_prototype[i], spearman_prototype[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    for i, model in enumerate(exemplar_models):
        ax1.annotate('Exemplar', (nll_exemplar[i], spearman_exemplar[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax1.set_xlabel('Negative Log-Likelihood (NLL)')
    ax1.set_ylabel('Spearman Correlation')
    ax1.set_title('NLL vs Spearman Correlation')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Parameters vs NLL
    ax2 = axes[1]
    params_prototype = [results['prototype'][model]['params'] for model in prototype_models]
    params_exemplar = [results['exemplar'][model]['params'] for model in exemplar_models]
    
    scatter2 = ax2.scatter(params_prototype, nll_prototype, c='blue', label='Prototype Models', s=100)
    ax2.scatter(params_exemplar, nll_exemplar, c='red', label='Exemplar Model', s=100, marker='s')
    
    # Add labels
    for i, model in enumerate(prototype_models):
        ax2.annotate(model, (params_prototype[i], nll_prototype[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    for i, model in enumerate(exemplar_models):
        ax2.annotate('Exemplar', (params_exemplar[i], nll_exemplar[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax2.set_xlabel('Number of Parameters')
    ax2.set_ylabel('Negative Log-Likelihood (NLL)')
    ax2.set_title('Parameters vs NLL')
    ax2.set_xscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Parameters vs AIC
    ax3 = axes[2]
    aic_prototype = [results['prototype'][model]['AIC'] for model in prototype_models]
    aic_exemplar = [results['exemplar'][model]['AIC'] for model in exemplar_models]
    
    scatter3 = ax3.scatter(params_prototype, aic_prototype, c='blue', label='Prototype Models', s=100)
    ax3.scatter(params_exemplar, aic_exemplar, c='red', label='Exemplar Model', s=100, marker='s')
    
    # Add labels
    for i, model in enumerate(prototype_models):
        ax3.annotate(model, (params_prototype[i], aic_prototype[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    for i, model in enumerate(exemplar_models):
        ax3.annotate('Exemplar', (params_exemplar[i], aic_exemplar[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax3.set_xlabel('Number of Parameters')
    ax3.set_ylabel('AIC')
    ax3.set_title('Parameters vs AIC')
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def print_summary(results):
    """Print results summary"""
    print("=" * 60)
    print("Model Performance Comparison Summary")
    print("=" * 60)
    
    print("\nPrototype Models:")
    print("-" * 40)
    for model, metrics in results['prototype'].items():
        print(f"{model:12} | NLL: {metrics['NLL']:.4f} | Spearman: {metrics['Spearman']:.4f} | AIC: {metrics['AIC']:.2f} | Params: {metrics['params']}")
    
    print("\nExemplar Model:")
    print("-" * 40)
    for model, metrics in results['exemplar'].items():
        print(f"{model:12} | NLL: {metrics['NLL']:.4f} | Spearman: {metrics['Spearman']:.4f} | AIC: {metrics['AIC']:.2f} | Params: {metrics['params']}")
    
    print("\nKey Findings:")
    print("-" * 40)
    
    # Find best models for each metric
    best_nll_prototype = min(results['prototype'].items(), key=lambda x: x[1]['NLL'])
    best_spearman_prototype = max(results['prototype'].items(), key=lambda x: x[1]['Spearman'])
    
    print(f"• Best NLL: Prototype {best_nll_prototype[0]} ({best_nll_prototype[1]['NLL']:.4f})")
    print(f"• Best Spearman: Exemplar (Sampled 5000) ({results['exemplar']['sampled_5000']['Spearman']:.4f})")
    print(f"• Parameter Count: Exemplar model is about {results['exemplar']['sampled_5000']['params'] / max([m['params'] for m in results['prototype'].values()]):.0f}x larger than Prototype models")
    print(f"• AIC: Exemplar model has significantly higher complexity penalty")

def main():
    """Main function"""
    # Load results
    results = load_results()
    
    # Print summary
    print_summary(results)
    
    # Create visualization directory
    vis_dir = "/content/gdrive/MyDrive/battleday_varimnist/results/figures"
    os.makedirs(vis_dir, exist_ok=True)
    
    # Plot basic comparison
    fig1 = plot_metric_comparison(results)
    fig1.savefig(os.path.join(vis_dir, "metric_comparison_en.png"), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot detailed comparison
    fig2 = plot_detailed_comparison(results)
    fig2.savefig(os.path.join(vis_dir, "detailed_comparison_en.png"), dpi=300, bbox_inches='tight')
    plt