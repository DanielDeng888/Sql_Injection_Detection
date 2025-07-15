import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import os

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def plot_multiple_roc_curves(models_metrics, title='SQL Injection Detection ROC Comparison', save_dir="results/ablation_study"):
    plt.figure(figsize=(12, 10))

    colors = ['#FF5733', '#33A8FF', '#33FF57', '#FF33A8', '#A833FF', '#FFBD33']
    line_styles = ['-', '--', '-.', ':']

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.plot([0, 1], [0, 1], linestyle='--', color='#999999', linewidth=2, label='Random Guess')

    for idx, (model_name, metrics) in enumerate(models_metrics.items()):
        true_labels = metrics['true_labels']
        predicted_probs = metrics['malicious_probs']

        fpr, tpr, _ = roc_curve(true_labels, predicted_probs)
        roc_auc = auc(fpr, tpr)

        color = colors[idx % len(colors)]
        linestyle = line_styles[idx % len(line_styles)]

        plt.plot(fpr, tpr, color=color, linestyle=linestyle,
                 linewidth=3, label=f'{model_name} (AUC = {roc_auc:.3f})')

    plt.title(title, fontsize=18, fontweight='bold')
    plt.xlabel('False Positive Rate (FPR)', fontsize=14)
    plt.ylabel('True Positive Rate (TPR)', fontsize=14)
    plt.legend(loc='lower right', fontsize=12)
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'multiple_roc_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_multiple_pr_curves(models_metrics, title='SQL Injection Detection PR Comparison', save_dir="results/ablation_study"):
    plt.figure(figsize=(12, 10))

    colors = ['#FF5733', '#33A8FF', '#33FF57', '#FF33A8', '#A833FF', '#FFBD33']
    line_styles = ['-', '--', '-.', ':']

    plt.grid(True, linestyle='--', alpha=0.6)

    for idx, (model_name, metrics) in enumerate(models_metrics.items()):
        true_labels = metrics['true_labels']
        predicted_probs = metrics['malicious_probs']

        precision, recall, _ = precision_recall_curve(true_labels, predicted_probs)
        pr_auc = auc(recall, precision)
        f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
        max_f1_idx = np.argmax(f1_scores)
        max_f1 = f1_scores[max_f1_idx]
        max_f1_recall = recall[max_f1_idx]
        max_f1_precision = precision[max_f1_idx]

        color = colors[idx % len(colors)]
        linestyle = line_styles[idx % len(line_styles)]

        plt.plot(recall, precision, color=color, linestyle=linestyle,
                 linewidth=3, label=f'{model_name} (AUC={pr_auc:.3f}, F1={max_f1:.3f})')
        plt.scatter([max_f1_recall], [max_f1_precision], color=color, s=80, marker='o', edgecolors='k', zorder=5)
        plt.text(max_f1_recall, max_f1_precision+0.02, f'MaxF1={max_f1:.2f}', color=color, fontsize=11, ha='center')

    plt.title(title, fontsize=18, fontweight='bold')
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.legend(loc='lower left', fontsize=12)
    plt.xlim([-0.01, 1.01])
    plt.ylim([0.0, 1.05])

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'multiple_pr_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_ablation_results(ablation_results, save_dir="results/ablation_study"):
    # Accuracy comparison over training steps
    plt.figure(figsize=(12, 8))

    colors = ['#FF5733', '#33A8FF', '#33FF57', '#FF33A8', '#A833FF', '#FFBD33']
    line_styles = ['-', '--', '-.', ':']

    plt.grid(True, linestyle='--', alpha=0.6)

    for idx, (name, results) in enumerate(ablation_results.items()):
        eval_steps = results['results']['eval_steps']
        eval_accuracies = results['results']['eval_accuracies']

        if eval_steps and eval_accuracies:
            color = colors[idx % len(colors)]
            linestyle = line_styles[idx % len(line_styles)]

            plt.plot(eval_steps, eval_accuracies, 'o-',
                     linewidth=2.5, color=color, linestyle=linestyle,
                     markerfacecolor='white', markersize=7, markeredgewidth=1.5,
                     label=results['name'])

    plt.title('SQL Injection Detection - Ablation Study Accuracy Comparison', fontsize=18, fontweight='bold')
    plt.xlabel('Training Steps', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.legend(loc='lower right', fontsize=12)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'ablation_accuracies.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Final accuracy bar chart
    final_accuracies = {}
    for name, results in ablation_results.items():
        eval_accuracies = results['results']['eval_accuracies']
        if eval_accuracies:
            final_accuracies[results['name']] = eval_accuracies[-1]
        else:
            final_accuracies[results['name']] = 0

    sorted_results = sorted(final_accuracies.items(), key=lambda x: x[1], reverse=True)
    names = [x[0] for x in sorted_results]
    accs = [x[1] for x in sorted_results]

    plt.figure(figsize=(12, 8))
    bar_colors = ['#3366CC', '#DC3912', '#FF9900', '#109618', '#990099']
    bars = plt.bar(names, accs, color=bar_colors[:len(names)],
                   edgecolor='#333333', linewidth=1.5, alpha=0.8, width=0.6)

    for bar, acc in zip(bars, accs):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f'{acc:.3f}', ha='center', fontsize=12, fontweight='bold')

    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.title('SQL Injection Detection - Ablation Study Final Accuracy', fontsize=18, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=14)
    plt.ylim(0, max(accs) + 0.1)

    if len(names) > 4:
        plt.xticks(rotation=30, ha='right', fontsize=12)
    else:
        plt.xticks(fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'ablation_final_accuracies.png'), dpi=300, bbox_inches='tight')
    plt.close()

