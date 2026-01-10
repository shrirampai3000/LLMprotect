"""
Visualization utilities for model evaluation.

Provides:
- Confusion matrix plots
- ROC and PR curves
- Attention heatmaps
- Training history plots
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple
from pathlib import Path


def plot_confusion_matrix(
    tp: int,
    fp: int,
    fn: int,
    tn: int,
    save_path: Optional[Path] = None,
    title: str = "Confusion Matrix"
) -> plt.Figure:
    """
    Plot confusion matrix.
    
    Args:
        tp, fp, fn, tn: Confusion matrix values
        save_path: Optional path to save figure
        title: Plot title
        
    Returns:
        matplotlib Figure
    """
    cm = np.array([[tn, fp], [fn, tp]])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Benign', 'Adversarial'],
        yticklabels=['Benign', 'Adversarial'],
        ax=ax,
        annot_kws={'size': 16}
    )
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    # Add percentages
    total = tp + fp + fn + tn
    for i in range(2):
        for j in range(2):
            value = cm[i, j]
            pct = value / total * 100
            ax.text(j + 0.5, i + 0.7, f'({pct:.1f}%)', 
                   ha='center', va='center', fontsize=10, color='gray')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc_score: float,
    save_path: Optional[Path] = None,
    title: str = "ROC Curve"
) -> plt.Figure:
    """
    Plot ROC curve.
    
    Args:
        fpr: False positive rates
        tpr: True positive rates
        auc_score: Area under curve
        save_path: Optional path to save figure
        title: Plot title
        
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC curve (AUC = {auc_score:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
            label='Random classifier')
    
    ax.fill_between(fpr, tpr, alpha=0.2, color='darkorange')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate (Detection Rate)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add target threshold line
    ax.axhline(y=0.99, color='green', linestyle=':', lw=1.5,
               label='Target Detection Rate (99%)')
    ax.axvline(x=0.05, color='red', linestyle=':', lw=1.5,
               label='Target FPR (5%)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_precision_recall_curve(
    precision: np.ndarray,
    recall: np.ndarray,
    ap_score: float,
    save_path: Optional[Path] = None,
    title: str = "Precision-Recall Curve"
) -> plt.Figure:
    """
    Plot Precision-Recall curve.
    
    Args:
        precision: Precision values
        recall: Recall values
        ap_score: Average precision score
        save_path: Optional path to save figure
        
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(recall, precision, color='darkorange', lw=2,
            label=f'PR curve (AP = {ap_score:.3f})')
    ax.fill_between(recall, precision, alpha=0.2, color='darkorange')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall (Detection Rate)', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_attention_heatmap(
    tokens: List[str],
    attention_weights: np.ndarray,
    save_path: Optional[Path] = None,
    title: str = "Attention Weights"
) -> plt.Figure:
    """
    Plot attention heatmap for explainability.
    
    Args:
        tokens: List of token strings
        attention_weights: 2D attention matrix or 1D token importance
        save_path: Optional path to save figure
        
    Returns:
        matplotlib Figure
    """
    if attention_weights.ndim == 1:
        # 1D token importance
        fig, ax = plt.subplots(figsize=(12, 3))
        
        x = np.arange(len(tokens))
        colors = plt.cm.RdYlGn_r(attention_weights / attention_weights.max())
        
        bars = ax.bar(x, attention_weights, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Attention Weight', fontsize=10)
        ax.set_title(title, fontsize=12)
        
    else:
        # 2D attention matrix
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(
            attention_weights,
            xticklabels=tokens,
            yticklabels=tokens,
            cmap='YlOrRd',
            ax=ax
        )
        
        ax.set_title(title, fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_training_history(
    train_history: List[Dict],
    val_history: List[Dict],
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot training history (loss and metrics over epochs).
    
    Args:
        train_history: List of training metrics per epoch
        val_history: List of validation metrics per epoch
        save_path: Optional path to save figure
        
    Returns:
        matplotlib Figure
    """
    epochs = range(1, len(train_history) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss
    ax = axes[0, 0]
    ax.plot(epochs, [h['loss'] for h in train_history], 'b-', label='Train')
    ax.plot(epochs, [h['loss'] for h in val_history], 'r-', label='Validation')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # F1 Score
    ax = axes[0, 1]
    ax.plot(epochs, [h['f1'] for h in train_history], 'b-', label='Train')
    ax.plot(epochs, [h['f1'] for h in val_history], 'r-', label='Validation')
    ax.axhline(y=0.97, color='green', linestyle='--', label='Target (0.97)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('F1 Score')
    ax.set_title('F1 Score')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Detection Rate
    ax = axes[1, 0]
    if 'detection_rate' in val_history[0]:
        ax.plot(epochs, [h.get('detection_rate', h.get('recall', 0)) for h in val_history], 
                'r-', label='Validation')
        ax.axhline(y=0.99, color='green', linestyle='--', label='Target (0.99)')
    else:
        ax.plot(epochs, [h['recall'] for h in train_history], 'b-', label='Train')
        ax.plot(epochs, [h['recall'] for h in val_history], 'r-', label='Validation')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Detection Rate')
    ax.set_title('Detection Rate (Recall)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # FPR (from validation)
    ax = axes[1, 1]
    if 'fpr' in val_history[0]:
        ax.plot(epochs, [h['fpr'] for h in val_history], 'r-', label='Validation FPR')
        ax.axhline(y=0.05, color='red', linestyle='--', label='Target (≤0.05)')
    else:
        ax.plot(epochs, [h['precision'] for h in train_history], 'b-', label='Train Precision')
        ax.plot(epochs, [h['precision'] for h in val_history], 'r-', label='Val Precision')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Rate')
    ax.set_title('False Positive Rate / Precision')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Training History', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_latency_histogram(
    latencies: List[float],
    save_path: Optional[Path] = None,
    title: str = "Inference Latency Distribution"
) -> plt.Figure:
    """
    Plot histogram of inference latencies.
    
    Args:
        latencies: List of latency values in milliseconds
        save_path: Optional path to save figure
        
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    latencies = np.array(latencies)
    
    # Histogram
    ax.hist(latencies, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    
    # Add statistics lines
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    
    ax.axvline(p50, color='green', linestyle='--', lw=2, label=f'P50: {p50:.1f}ms')
    ax.axvline(p95, color='orange', linestyle='--', lw=2, label=f'P95: {p95:.1f}ms')
    ax.axvline(p99, color='red', linestyle='--', lw=2, label=f'P99: {p99:.1f}ms')
    
    # Target line
    ax.axvline(50, color='purple', linestyle=':', lw=2, label='Target: 50ms')
    
    ax.set_xlabel('Latency (ms)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add stats text
    stats_text = f'Mean: {np.mean(latencies):.1f}ms\nStd: {np.std(latencies):.1f}ms'
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_baseline_comparison(
    comparison_results: Dict[str, Dict],
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot comparison of model vs. baselines.
    
    Args:
        comparison_results: Dict from BaselineComparison.run_baseline_comparison
        save_path: Optional path to save figure
        
    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    methods = list(comparison_results.keys())
    x = np.arange(len(methods))
    
    # Detection Rate
    ax = axes[0]
    values = [comparison_results[m]['detection_rate'] for m in methods]
    bars = ax.bar(x, values, color=['steelblue', 'lightblue', 'darkorange'])
    ax.axhline(0.99, color='green', linestyle='--', label='Target (99%)')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15)
    ax.set_ylabel('Detection Rate')
    ax.set_title('Detection Rate Comparison')
    ax.set_ylim(0, 1.1)
    ax.legend()
    
    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2%}', ha='center', va='bottom', fontsize=9)
    
    # F1 Score
    ax = axes[1]
    values = [comparison_results[m]['f1_score'] for m in methods]
    bars = ax.bar(x, values, color=['steelblue', 'lightblue', 'darkorange'])
    ax.axhline(0.97, color='green', linestyle='--', label='Target (97%)')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15)
    ax.set_ylabel('F1 Score')
    ax.set_title('F1 Score Comparison')
    ax.set_ylim(0, 1.1)
    ax.legend()
    
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2%}', ha='center', va='bottom', fontsize=9)
    
    # Latency
    ax = axes[2]
    values = [comparison_results[m]['mean_latency_ms'] for m in methods]
    bars = ax.bar(x, values, color=['steelblue', 'lightblue', 'darkorange'])
    ax.axhline(50, color='red', linestyle='--', label='Target (≤50ms)')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15)
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Latency Comparison')
    ax.legend()
    
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}ms', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('Baseline Comparison', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig
