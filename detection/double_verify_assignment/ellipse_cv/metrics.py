"""
Metrics for ellipse detection evaluation.

Classification metrics:
- Accuracy, ROC-AUC, PR-AUC, F1, Precision, Recall
- F-beta threshold sweep for optimal threshold selection

Regression metrics:
- MSE, MAE, Huber (on ellipse samples only)
"""
from __future__ import annotations

from typing import Dict, Tuple, Optional
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)


def compute_classification_metrics(
    probs: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute classification metrics.
    
    Args:
        probs: Predicted probabilities [N]
        targets: Binary targets [N]
        threshold: Classification threshold
    
    Returns:
        Dict with accuracy, roc_auc, pr_auc, f1, precision, recall
    """
    preds = (probs >= threshold).astype(int)
    
    metrics = {
        'accuracy': accuracy_score(targets, preds),
        'f1': f1_score(targets, preds, zero_division=0),
        'precision': precision_score(targets, preds, zero_division=0),
        'recall': recall_score(targets, preds, zero_division=0),
    }
    
    # AUC metrics need probabilities
    if len(np.unique(targets)) > 1:
        metrics['roc_auc'] = roc_auc_score(targets, probs)
        metrics['pr_auc'] = average_precision_score(targets, probs)
    else:
        metrics['roc_auc'] = 0.0
        metrics['pr_auc'] = 0.0
    
    return metrics


def find_best_fbeta_threshold(
    probs: np.ndarray,
    targets: np.ndarray,
    beta: float = 2.0,
    n_thresholds: int = 100,
) -> Tuple[float, float]:
    """
    Find threshold that maximizes F-beta score.
    
    F-beta = (1 + beta^2) * (precision * recall) / (beta^2 * precision + recall)
    
    Args:
        probs: Predicted probabilities [N]
        targets: Binary targets [N]
        beta: Beta value (>1 weights recall more, <1 weights precision more)
        n_thresholds: Number of thresholds to try
    
    Returns:
        best_threshold, best_fbeta
    """
    precision, recall, thresholds = precision_recall_curve(targets, probs)
    
    # Compute F-beta for each threshold
    # Note: precision and recall arrays are one longer than thresholds
    beta_sq = beta ** 2
    fbeta_scores = []
    
    for i in range(len(thresholds)):
        p, r = precision[i], recall[i]
        if p + r == 0:
            fbeta = 0.0
        else:
            fbeta = (1 + beta_sq) * (p * r) / (beta_sq * p + r)
        fbeta_scores.append(fbeta)
    
    if not fbeta_scores:
        return 0.5, 0.0
    
    best_idx = np.argmax(fbeta_scores)
    best_threshold = thresholds[best_idx]
    best_fbeta = fbeta_scores[best_idx]
    
    return float(best_threshold), float(best_fbeta)


def compute_regression_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute regression metrics on ellipse samples.
    
    Args:
        pred: Predicted geometry [N, 5] or [N]
        target: Target geometry [N, 5] or [N]
        mask: Optional mask for ellipse samples
    
    Returns:
        Dict with mse, mae, huber (delta=0.1)
    """
    if mask is not None:
        mask = mask.astype(bool)
        if not mask.any():
            return {'mse': 0.0, 'mae': 0.0, 'huber': 0.0}
        pred = pred[mask]
        target = target[mask]
    
    if len(pred) == 0:
        return {'mse': 0.0, 'mae': 0.0, 'huber': 0.0}
    
    # MSE
    mse = np.mean((pred - target) ** 2)
    
    # MAE
    mae = np.mean(np.abs(pred - target))
    
    # Huber (pseudo-Huber for simplicity)
    delta = 0.1
    diff = np.abs(pred - target)
    huber = np.mean(np.where(diff <= delta, 0.5 * diff ** 2, delta * (diff - 0.5 * delta)))
    
    return {
        'mse': float(mse),
        'mae': float(mae),
        'huber': float(huber),
    }


def compute_geometry_metrics_per_component(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> Dict[str, float]:
    """
    Compute regression metrics per geometry component.
    
    Args:
        pred: [N, 6] predicted geometry (cx, cy, a, b, cos(2θ), sin(2θ))
        target: [N, 6] target geometry (cx, cy, a, b, cos(2θ), sin(2θ))
        mask: [N] ellipse mask
    
    Returns:
        Dict with per-component metrics
    """
    mask = mask.astype(bool)
    if not mask.any():
        return {
            'center_mse': 0.0, 'center_mae': 0.0,
            'axes_mse': 0.0, 'axes_mae': 0.0,
            'angle_cos_sim': 0.0, 'angle_mae_deg': 0.0,
        }
    
    pred = pred[mask]
    target = target[mask]
    
    # Center (cx, cy) - indices 0, 1
    center_pred = pred[:, :2]
    center_target = target[:, :2]
    center_mse = np.mean((center_pred - center_target) ** 2)
    center_mae = np.mean(np.abs(center_pred - center_target))
    
    # Axes (a, b) - indices 2, 3
    axes_pred = pred[:, 2:4]
    axes_target = target[:, 2:4]
    axes_mse = np.mean((axes_pred - axes_target) ** 2)
    axes_mae = np.mean(np.abs(axes_pred - axes_target))
    
    # Angle (cos(2θ), sin(2θ)) - indices 4, 5
    pred_angle = pred[:, 4:6]  # [N, 2]
    target_angle = target[:, 4:6]  # [N, 2]
    
    # Cosine similarity
    cos_sim = (pred_angle * target_angle).sum(axis=1)  # [N]
    angle_cos_sim = np.mean(cos_sim)
    
    # Decode angles to degrees for interpretable MAE
    pred_theta = np.arctan2(pred_angle[:, 1], pred_angle[:, 0]) / 2
    target_theta = np.arctan2(target_angle[:, 1], target_angle[:, 0]) / 2
    pred_deg = np.degrees(pred_theta) % 180
    target_deg = np.degrees(target_theta) % 180
    
    # Circular angle error
    angle_err_deg = np.abs(pred_deg - target_deg)
    angle_err_deg = np.minimum(angle_err_deg, 180 - angle_err_deg)
    angle_mae_deg = np.mean(angle_err_deg)
    
    return {
        'center_mse': float(center_mse),
        'center_mae': float(center_mae),
        'axes_mse': float(axes_mse),
        'axes_mae': float(axes_mae),
        'angle_cos_sim': float(angle_cos_sim),
        'angle_mae_deg': float(angle_mae_deg),
    }


class MetricsAccumulator:
    """
    Accumulates predictions and targets for batch-wise metric computation.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all accumulators."""
        self.probs = []
        self.targets = []
        self.pred_geom = []
        self.target_geom = []
        self.ellipse_mask = []
        self.loss_components = []
    
    def update(
        self,
        logits: torch.Tensor,
        is_ellipse: torch.Tensor,
        pred_geom: torch.Tensor,
        target_geom: torch.Tensor,
        ellipse_mask: torch.Tensor,
        loss_components: Optional[Dict] = None,
    ):
        """
        Add batch predictions and targets.
        
        Args:
            logits: [B] raw logits
            is_ellipse: [B] binary targets
            pred_geom: [B, 6] predicted geometry (cx, cy, a, b, cos(2θ), sin(2θ))
            target_geom: [B, 6] target geometry (cx, cy, a, b, cos(2θ), sin(2θ))
            ellipse_mask: [B] ellipse mask
            loss_components: Optional dict of loss components
        """
        # Convert to numpy
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        self.probs.append(probs)
        self.targets.append(is_ellipse.detach().cpu().numpy())
        self.pred_geom.append(pred_geom.detach().cpu().numpy())
        self.target_geom.append(target_geom.detach().cpu().numpy())
        self.ellipse_mask.append(ellipse_mask.detach().cpu().numpy())
        
        if loss_components is not None:
            self.loss_components.append(loss_components)
    
    def compute(self, threshold: float = 0.5) -> Dict[str, float]:
        """
        Compute all metrics from accumulated data.
        
        Args:
            threshold: Classification threshold
        
        Returns:
            Dict with all metrics
        """
        # Concatenate
        probs = np.concatenate(self.probs)
        targets = np.concatenate(self.targets)
        pred_geom = np.concatenate(self.pred_geom)
        target_geom = np.concatenate(self.target_geom)
        mask = np.concatenate(self.ellipse_mask)
        
        # Classification metrics
        cls_metrics = compute_classification_metrics(probs, targets, threshold)
        
        # Regression metrics (on geometry only)
        reg_metrics = compute_regression_metrics(pred_geom, target_geom, mask)
        
        # Per-component metrics (center, axes, angle)
        comp_metrics = compute_geometry_metrics_per_component(pred_geom, target_geom, mask)
        
        # Average loss components
        avg_loss = {}
        if self.loss_components:
            for key in self.loss_components[0].keys():
                if key != 'n_ellipse':
                    values = [c[key] for c in self.loss_components]
                    avg_loss[f'avg_{key}'] = float(np.mean(values))
        
        # Combine all
        all_metrics = {
            **cls_metrics,
            **reg_metrics,
            **comp_metrics,
            **avg_loss,
        }
        
        return all_metrics
    
    def find_best_threshold(self, beta: float = 2.0) -> Tuple[float, float]:
        """
        Find best F-beta threshold from accumulated data.
        
        Returns:
            best_threshold, best_fbeta
        """
        probs = np.concatenate(self.probs)
        targets = np.concatenate(self.targets)
        return find_best_fbeta_threshold(probs, targets, beta=beta)


if __name__ == '__main__':
    # Test metrics
    np.random.seed(42)
    
    # Test classification metrics
    probs = np.array([0.9, 0.8, 0.7, 0.4, 0.3, 0.2, 0.1, 0.05])
    targets = np.array([1, 1, 1, 1, 1, 0, 0, 0])
    
    cls_metrics = compute_classification_metrics(probs, targets, threshold=0.5)
    print("Classification metrics:")
    for k, v in cls_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # Test F-beta threshold
    best_thresh, best_fbeta = find_best_fbeta_threshold(probs, targets, beta=2.0)
    print(f"\nBest F-beta threshold: {best_thresh:.4f}")
    print(f"Best F-beta score: {best_fbeta:.4f}")
    
    # Test regression metrics
    pred = np.random.rand(10, 5) * 0.1 + np.random.rand(10, 5)
    target = np.random.rand(10, 5)
    mask = np.array([1, 1, 1, 1, 1, 1, 1, 0, 0, 0])
    
    reg_metrics = compute_regression_metrics(pred, target, mask)
    print("\nRegression metrics:")
    for k, v in reg_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    comp_metrics = compute_geometry_metrics_per_component(pred, target, mask)
    print("\nPer-component metrics:")
    for k, v in comp_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # Test accumulator
    print("\nTesting MetricsAccumulator...")
    acc = MetricsAccumulator()
    
    for _ in range(3):
        logits = torch.randn(4)
        is_ell = torch.tensor([1, 1, 0, 1], dtype=torch.float32)
        pred_g = torch.rand(4, 5)  # 5-dim geometry
        target_g = torch.rand(4, 5)
        mask_t = torch.tensor([1, 1, 0, 1], dtype=torch.float32)
        
        acc.update(logits, is_ell, pred_g, target_g, mask_t)
    
    all_metrics = acc.compute()
    print("Accumulated metrics:")
    for k, v in all_metrics.items():
        print(f"  {k}: {v:.4f}")
