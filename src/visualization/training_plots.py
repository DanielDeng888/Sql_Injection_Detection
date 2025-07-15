import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
import seaborn as sns
import os

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def plot_training_results(training_results, save_dir="results/visualizations"):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Evaluation accuracy curve
    ax1 = axes[0, 0]
    ax1.plot(training_results['eval_steps'], training_results['eval_accuracies'], 'o-',
             linewidth=2.5, color='#3366CC', markerfacecolor='white',
             markersize=8, markeredgewidth=2)
    ax1.set_title('Evaluation Accuracy', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Training Steps', fontsize=14)
    ax1.set_ylabel('Accuracy', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # 2. Evaluation reward curve
    ax2 = axes[0, 1]
    ax2.plot(training_results['eval_steps'], training_results['eval_scores'], 'o-',
             linewidth=2.5, color='#CC3366', markerfacecolor='white',
             markersize=8, markeredgewidth=2)
    ax2.set_title('Evaluation Reward', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Training Steps', fontsize=14)
    ax2.set_ylabel('Average Reward', fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.5)

    # 3. Training loss curve
    ax3 = axes[1, 0]
    losses = training_results['train_info']['losses']
    if len(losses) > 1000:
        indices = np.linspace(0, len(losses) - 1, 1000, dtype=int)
        losses = [losses[i] for i in indices]

    smoothed_losses = gaussian_filter1d(losses, sigma=5.0)
    ax3.plot(smoothed_losses, linewidth=2.5, color='#33CC66')
    ax3.set_title('Training Loss', fontsize=16, fontweight='bold')
    ax3.set_xlabel('Training Steps', fontsize=14)
    ax3.set_ylabel('Loss Value', fontsize=14)
    ax3.set_yscale('log')
    ax3.grid(True, linestyle='--', alpha=0.5)

    # 4. Epsilon decay curve
    ax4 = axes[1, 1]
    ax4.plot(training_results['train_info']['epsilons'], linewidth=2.5, color='#9966CC')
    ax4.set_title('Epsilon Decay', fontsize=16, fontweight='bold')
    ax4.set_xlabel('Training Steps', fontsize=14)
    ax4.set_ylabel('Epsilon Value', fontsize=14)
    ax4.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    fig.suptitle('SQL Injection Detection DQN Training Process', fontsize=20, fontweight='bold', y=1.02)

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'training_results.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_roc_curve(true_labels, predicted_probs, title='ROC Curve', save_dir="results/visualizations"):
    fpr, tpr, thresholds = roc_curve(true_labels, predicted_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='#FF5733', lw=3, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='#888888', lw=2, linestyle='--', label='Random Guess')

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.title(title, fontsize=18, fontweight='bold')
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.legend(loc='lower right', fontsize=12)
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()

    return roc_auc


def plot_pr_curve(true_labels, predicted_probs, title='Precision-Recall Curve', save_dir="results/visualizations"):
    precision, recall, thresholds = precision_recall_curve(true_labels, predicted_probs)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='#33A8FF', lw=3, label=f'PR Curve (AUC = {pr_auc:.3f})')

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.title(title, fontsize=18, fontweight='bold')
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.legend(loc='lower left', fontsize=12)
    plt.xlim([-0.01, 1.01])
    plt.ylim([0.0, 1.05])

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'pr_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()

    return pr_auc


def plot_confusion_matrix(true_labels, predictions, title='Confusion Matrix', save_dir="results/visualizations"):
    cm = confusion_matrix(true_labels, predictions)
    class_names = ['Normal', 'Malicious']

    plt.figure(figsize=(10, 8))
    cmap = sns.color_palette("Blues", as_cmap=True)

    ax = sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                     xticklabels=class_names, yticklabels=class_names,
                     annot_kws={"size": 16, "weight": "bold"},
                     linewidths=1, linecolor='white',
                     square=True, cbar_kws={"shrink": 0.8})

    plt.title(title, fontsize=18, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)

    accuracy = np.sum(np.diag(cm)) / np.sum(cm)
    plt.figtext(0.5, 0.01, f'Accuracy: {accuracy:.2%}',
                ha='center', fontsize=14, weight='bold')

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

    return cm