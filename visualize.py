"""
Visualization utilities for experimental results.
"""
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List


def load_results(output_dir: str) -> Dict:
    """Load all experimental results."""
    comparison_path = os.path.join(output_dir, 'comparison.json')
    with open(comparison_path, 'r') as f:
        comparison = json.load(f)

    method_metrics = {}
    for method in ['naive', 'ewc', 'ling_ewc', 'random_ewc']:
        method_path = os.path.join(output_dir, f'{method}_metrics.json')
        if os.path.exists(method_path):
            with open(method_path, 'r') as f:
                method_metrics[method] = json.load(f)

    return {
        'comparison': comparison,
        'method_metrics': method_metrics
    }


def plot_forgetting_comparison(results: Dict, output_path: str):
    """Plot forgetting comparison across methods."""
    comparison = results['comparison']

    methods = []
    forgetting_values = []

    for method in ['naive', 'ewc', 'ling_ewc', 'random_ewc']:
        if method in comparison:
            methods.append(method.replace('_', ' ').title())
            forgetting_values.append(comparison[method]['average_forgetting'])

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(methods, forgetting_values, color=['#e74c3c', '#3498db', '#2ecc71', '#95a5a6'])

    ax.set_ylabel('Average Forgetting', fontsize=12)
    ax.set_xlabel('Method', fontsize=12)
    ax.set_title('Catastrophic Forgetting Comparison', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, value in zip(bars, forgetting_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.4f}',
                ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved forgetting comparison to {output_path}")
    plt.close()


def plot_final_accuracy(results: Dict, output_path: str):
    """Plot final accuracy for each language and method."""
    comparison = results['comparison']

    methods = []
    bengali_acc = []
    hindi_acc = []

    for method in ['naive', 'ewc', 'ling_ewc', 'random_ewc']:
        if method in comparison:
            methods.append(method.replace('_', ' ').title())
            final_perf = comparison[method]['final_performance']
            bengali_acc.append(final_perf.get('bengali', 0))
            hindi_acc.append(final_perf.get('hindi', 0))

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, bengali_acc, width, label='Bengali', color='#3498db')
    bars2 = ax.bar(x + width/2, hindi_acc, width, label='Hindi', color='#e74c3c')

    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_xlabel('Method', fontsize=12)
    ax.set_title('Final Accuracy on Each Language', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1])

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved final accuracy comparison to {output_path}")
    plt.close()


def plot_learning_curves(results: Dict, output_path: str):
    """Plot learning curves showing accuracy over time."""
    method_metrics = results['method_metrics']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for method, metrics in method_metrics.items():
        history = metrics['history']

        # Bengali performance over time
        if 'bengali' in history:
            steps = [entry['step'] for entry in history['bengali']]
            accuracies = [entry['accuracy'] for entry in history['bengali']]
            axes[0].plot(steps, accuracies, marker='o', label=method.replace('_', ' ').title())

        # Hindi performance over time
        if 'hindi' in history:
            steps = [entry['step'] for entry in history['hindi']]
            accuracies = [entry['accuracy'] for entry in history['hindi']]
            axes[1].plot(steps, accuracies, marker='o', label=method.replace('_', ' ').title())

    axes[0].set_xlabel('Training Step', fontsize=11)
    axes[0].set_ylabel('Accuracy', fontsize=11)
    axes[0].set_title('Bengali Performance Over Time', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].set_xlabel('Training Step', fontsize=11)
    axes[1].set_ylabel('Accuracy', fontsize=11)
    axes[1].set_title('Hindi Performance Over Time', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved learning curves to {output_path}")
    plt.close()


def generate_visualizations(output_dir: str = './results'):
    """Generate all visualizations."""
    print("Loading results...")
    results = load_results(output_dir)

    print("Generating visualizations...")
    plot_forgetting_comparison(results, os.path.join(output_dir, 'forgetting_comparison.png'))
    plot_final_accuracy(results, os.path.join(output_dir, 'final_accuracy.png'))
    plot_learning_curves(results, os.path.join(output_dir, 'learning_curves.png'))

    print("\nVisualization complete! Check the results directory.")


if __name__ == "__main__":
    generate_visualizations()
