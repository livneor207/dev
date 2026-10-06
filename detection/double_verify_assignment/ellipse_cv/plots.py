"""
Plotting utilities for training visualization.

- Loss curves (classification, regression, total)
- Metric curves (accuracy, AUC, F1, etc.)
- PR curve with threshold markers
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve


def plot_training_curves(
    train_metrics: List[Dict[str, float]],
    val_metrics: List[Dict[str, float]],
    save_path: Optional[str | Path] = None,
    figsize: tuple = (16, 12),
) -> plt.Figure:
    """
    Plot training curves for losses and metrics.
    
    Args:
        train_metrics: List of metric dicts per epoch
        val_metrics: List of metric dicts per epoch
        save_path: Optional path to save figure
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    epochs = range(1, len(train_metrics) + 1)
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle('Training Curves', fontsize=14)
    
    # Plot 1: Total loss
    ax = axes[0, 0]
    if 'avg_total_loss' in train_metrics[0]:
        train_loss = [m['avg_total_loss'] for m in train_metrics]
        val_loss = [m['avg_total_loss'] for m in val_metrics]
        ax.plot(epochs, train_loss, 'b-', label='Train')
        ax.plot(epochs, val_loss, 'r-', label='Val')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Total Loss')
        ax.set_title('Total Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 2: Classification loss
    ax = axes[0, 1]
    if 'avg_cls_loss' in train_metrics[0]:
        train_cls = [m['avg_cls_loss'] for m in train_metrics]
        val_cls = [m['avg_cls_loss'] for m in val_metrics]
        ax.plot(epochs, train_cls, 'b-', label='Train')
        ax.plot(epochs, val_cls, 'r-', label='Val')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Classification Loss')
        ax.set_title('Classification Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 3: Geometric losses (center, radii, angle)
    ax = axes[0, 2]
    geom_keys = [
        ('avg_center_loss', 'Center', '-'),
        ('avg_radii_loss', 'Radii', '--'),
        ('avg_angle_loss', 'Angle', ':'),
    ]
    plotted = False
    for key, label, style in geom_keys:
        if key in train_metrics[0]:
            ax.plot(epochs, [m[key] for m in train_metrics], f'b{style}', label=f'Train {label}')
            ax.plot(epochs, [m[key] for m in val_metrics], f'r{style}', label=f'Val {label}')
            plotted = True
    if not plotted and 'mse' in train_metrics[0]:
        ax.plot(epochs, [m['mse'] for m in train_metrics], 'b-', label='Train')
        ax.plot(epochs, [m['mse'] for m in val_metrics], 'r-', label='Val')
        ax.set_ylabel('MSE')
        ax.set_title('Regression MSE')
    else:
        ax.set_ylabel('Loss')
        ax.set_title('Geometric Losses')
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Accuracy
    ax = axes[1, 0]
    if 'accuracy' in train_metrics[0]:
        train_acc = [m['accuracy'] for m in train_metrics]
        val_acc = [m['accuracy'] for m in val_metrics]
        ax.plot(epochs, train_acc, 'b-', label='Train')
        ax.plot(epochs, val_acc, 'r-', label='Val')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 5: ROC-AUC and PR-AUC
    ax = axes[1, 1]
    if 'roc_auc' in train_metrics[0]:
        train_roc = [m['roc_auc'] for m in train_metrics]
        val_roc = [m['roc_auc'] for m in val_metrics]
        train_pr = [m['pr_auc'] for m in train_metrics]
        val_pr = [m['pr_auc'] for m in val_metrics]
        ax.plot(epochs, train_roc, 'b-', label='Train ROC-AUC')
        ax.plot(epochs, val_roc, 'r-', label='Val ROC-AUC')
        ax.plot(epochs, train_pr, 'b--', label='Train PR-AUC')
        ax.plot(epochs, val_pr, 'r--', label='Val PR-AUC')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('AUC')
        ax.set_title('AUC Metrics')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 6: F1 score
    ax = axes[1, 2]
    if 'f1' in train_metrics[0]:
        train_f1 = [m['f1'] for m in train_metrics]
        val_f1 = [m['f1'] for m in val_metrics]
        ax.plot(epochs, train_f1, 'b-', label='Train')
        ax.plot(epochs, val_f1, 'r-', label='Val')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('F1 Score')
        ax.set_title('F1 Score')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")
    
    return fig


def plot_pr_curve(
    probs: np.ndarray,
    targets: np.ndarray,
    best_threshold: Optional[float] = None,
    save_path: Optional[str | Path] = None,
    figsize: tuple = (8, 6),
) -> plt.Figure:
    """
    Plot precision-recall curve with optional threshold marker.
    
    Args:
        probs: Predicted probabilities
        targets: Binary targets
        best_threshold: Optional threshold to mark on curve
        save_path: Optional path to save figure
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    precision, recall, thresholds = precision_recall_curve(targets, probs)
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(recall, precision, 'b-', linewidth=2)
    
    # Mark best threshold if provided
    if best_threshold is not None:
        # Find closest threshold
        idx = np.argmin(np.abs(thresholds - best_threshold))
        ax.scatter(recall[idx], precision[idx], c='red', s=100, zorder=5,
                   label=f'Threshold={best_threshold:.3f}')
        ax.legend()
    
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved PR curve to {save_path}")
    
    return fig


def plot_regression_errors(
    pred_geom: np.ndarray,
    target_geom: np.ndarray,
    mask: np.ndarray,
    save_path: Optional[str | Path] = None,
    figsize: tuple = (16, 4),
) -> plt.Figure:
    """
    Plot regression error distributions per component.
    
    Args:
        pred_geom: [N, 5] predicted geometry
        target_geom: [N, 5] target geometry
        mask: [N] ellipse mask
        save_path: Optional path to save figure
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    mask = mask.astype(bool)
    pred = pred_geom[mask]
    target = target_geom[mask]
    
    errors = pred - target
    component_names = ['Center X', 'Center Y', 'Angle', 'Axis 1', 'Axis 2']
    
    fig, axes = plt.subplots(1, 5, figsize=figsize)
    fig.suptitle('Regression Error Distributions (Normalized Space)', fontsize=12)
    
    for i, (ax, name) in enumerate(zip(axes, component_names)):
        err = errors[:, i]
        if i == 2:  # Angle - use circular error
            err = np.abs(err)
            err = np.minimum(err, 1.0 - err)
        
        ax.hist(err, bins=30, alpha=0.7, edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', alpha=0.7)
        ax.set_xlabel('Error')
        ax.set_ylabel('Count')
        ax.set_title(f'{name}\nMAE={np.abs(err).mean():.4f}')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved regression errors to {save_path}")
    
    return fig


if __name__ == '__main__':
    # Test plots
    import tempfile
    
    # Generate fake metrics
    n_epochs = 10
    train_metrics = []
    val_metrics = []
    
    for i in range(n_epochs):
        train_metrics.append({
            'avg_total_loss': 1.0 - i * 0.08 + np.random.rand() * 0.05,
            'avg_cls_loss': 0.5 - i * 0.04 + np.random.rand() * 0.02,
            'mse': 0.3 - i * 0.02 + np.random.rand() * 0.02,
            'accuracy': 0.5 + i * 0.04 + np.random.rand() * 0.02,
            'roc_auc': 0.6 + i * 0.03 + np.random.rand() * 0.02,
            'pr_auc': 0.7 + i * 0.02 + np.random.rand() * 0.02,
            'f1': 0.5 + i * 0.04 + np.random.rand() * 0.02,
        })
        val_metrics.append({
            'avg_total_loss': 1.1 - i * 0.07 + np.random.rand() * 0.05,
            'avg_cls_loss': 0.55 - i * 0.035 + np.random.rand() * 0.02,
            'mse': 0.35 - i * 0.015 + np.random.rand() * 0.02,
            'accuracy': 0.48 + i * 0.035 + np.random.rand() * 0.02,
            'roc_auc': 0.58 + i * 0.025 + np.random.rand() * 0.02,
            'pr_auc': 0.68 + i * 0.018 + np.random.rand() * 0.02,
            'f1': 0.48 + i * 0.035 + np.random.rand() * 0.02,
        })
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test training curves
        fig1 = plot_training_curves(
            train_metrics, val_metrics,
            save_path=f'{tmpdir}/curves.png'
        )
        plt.close(fig1)
        
        # Test PR curve
        probs = np.random.rand(100)
        targets = (np.random.rand(100) > 0.3).astype(float)
        fig2 = plot_pr_curve(
            probs, targets,
            best_threshold=0.5,
            save_path=f'{tmpdir}/pr_curve.png'
        )
        plt.close(fig2)
        
        # Test regression errors
        pred = np.random.rand(50, 5)
        target = np.random.rand(50, 5)
        mask = np.ones(50)
        fig3 = plot_regression_errors(
            pred, target, mask,
            save_path=f'{tmpdir}/reg_errors.png'
        )
        plt.close(fig3)
        
        print("All plots generated successfully!")
