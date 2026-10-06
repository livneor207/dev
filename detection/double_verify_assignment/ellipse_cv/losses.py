"""
Loss functions for multi-task ellipse detection.

L = λ1 L_cls + y (λ2 L_center + λ3 L_radii + λ4 L_angle)

  L_cls    = BCEWithLogitsLoss
  L_center = SmoothL1((ĉx, ĉy), (cx, cy))
  L_radii  = SmoothL1((â, ˆb), (a, b))
  L_angle  = (1 - v̂ · v) + λh * Huber(v̂, v),
             v = (cos 2θ, sin 2θ)

Geometric terms are applied only when an ellipse is present (y = 1).
Center and radii targets are normalized by image size in the dataset.
"""
from __future__ import annotations

from typing import Literal, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced binary classification.
    
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    
    Args:
        alpha: Weighting factor for positive class (default from pos_rate)
        gamma: Focusing parameter (default 2)
        reduction: 'mean', 'sum', or 'none'
    """
    
    def __init__(
        self,
        alpha: float = 0.7,  # Default for ~70% positive rate
        gamma: float = 2.0,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            logits: [B] or [B, 1] raw logits
            targets: [B] binary targets {0, 1}
        
        Returns:
            Scalar loss if reduction != 'none', else [B] losses
        """
        logits = logits.view(-1)
        targets = targets.view(-1)
        
        # Compute BCE
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # Compute probabilities
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # Compute alpha weights
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Compute focal weight
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        
        # Apply focal weight
        focal_loss = focal_weight * bce
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


def circular_distance(pred: torch.Tensor, target: torch.Tensor, period: float = 1.0) -> torch.Tensor:
    """
    Compute circular distance for periodic values.
    
    For angles normalized to [0, 1] (representing [0, 180)):
        distance = min(|a - b|, 1 - |a - b|)
    
    Args:
        pred: Predicted values (normalized to [0, period])
        target: Target values (normalized to [0, period])
        period: Period of the circular variable
    
    Returns:
        Circular distance tensor
    """
    diff = torch.abs(pred - target)
    return torch.min(diff, period - diff)


def circular_mse(pred: torch.Tensor, target: torch.Tensor, period: float = 1.0) -> torch.Tensor:
    """Circular MSE loss."""
    return (circular_distance(pred, target, period) ** 2).mean()


def circular_mae(pred: torch.Tensor, target: torch.Tensor, period: float = 1.0) -> torch.Tensor:
    """Circular MAE loss."""
    return circular_distance(pred, target, period).mean()


class MultiTaskLoss(nn.Module):
    """
    Four-term ellipse loss:

        L = λ1 L_cls + y (λ2 L_center + λ3 L_radii + λ4 L_angle)

    Geometric terms are gated by y (ellipse present). Optional Kendall
    uncertainty weights replace the fixed λs when the model has auto_tune.
    """

    def __init__(
        self,
        cls_loss: Literal['bce', 'focal'] = 'bce',
        reg_loss: Literal['smooth_l1', 'mse', 'mae', 'huber'] = 'smooth_l1',
        pos_rate: float = 0.7,
        focal_gamma: float = 2.0,
        huber_delta: float = 1.0,
        smooth_l1_beta: float = 1.0,
        lambda_cls: float = 1.0,
        lambda_center: float = 1.0,
        lambda_radii: float = 1.0,
        lambda_angle: float = 1.0,
        angle_huber_weight: float = 0.0,
        w_cls: Optional[float] = None,
        w_center: Optional[float] = None,
        w_axes: Optional[float] = None,
        w_angle: Optional[float] = None,
    ):
        """
        Args:
            cls_loss: 'bce' (BCEWithLogitsLoss) or 'focal'
            reg_loss: SmoothL1 for center/radii; mse/mae/huber kept for ablations
            pos_rate: Positive class rate (focal alpha only)
            focal_gamma: Gamma for focal loss
            huber_delta: Delta when reg_loss='huber'
            smooth_l1_beta: Beta for SmoothL1Loss
            lambda_cls: λ1 classification weight
            lambda_center: λ2 center weight
            lambda_radii: λ3 radii weight
            lambda_angle: λ4 angle weight
            angle_huber_weight: Additive Huber-loss weight for the angle vector.
                Zero retains the cosine-only angle objective.
            w_*: Aliases for the lambda_* weights
        """
        super().__init__()

        self.cls_loss_type = cls_loss
        self.reg_loss_type = reg_loss

        if cls_loss == 'focal':
            self.cls_criterion = FocalLoss(alpha=pos_rate, gamma=focal_gamma)
        else:
            self.cls_criterion = nn.BCEWithLogitsLoss()

        if reg_loss == 'smooth_l1':
            self.reg_criterion = nn.SmoothL1Loss(reduction='mean', beta=smooth_l1_beta)
        elif reg_loss == 'huber':
            self.reg_criterion = nn.HuberLoss(reduction='mean', delta=huber_delta)
        elif reg_loss == 'mae':
            self.reg_criterion = nn.L1Loss(reduction='mean')
        else:
            self.reg_criterion = nn.MSELoss(reduction='mean')

        self.lambda_cls = w_cls if w_cls is not None else lambda_cls
        self.lambda_center = w_center if w_center is not None else lambda_center
        self.lambda_radii = w_axes if w_axes is not None else lambda_radii
        self.lambda_angle = w_angle if w_angle is not None else lambda_angle
        self.angle_huber_weight = angle_huber_weight
        self.angle_huber_criterion = nn.HuberLoss(
            reduction='mean', delta=huber_delta,
        )
        self.w_cls = self.lambda_cls
        self.w_center = self.lambda_center
        self.w_axes = self.lambda_radii
        self.w_angle = self.lambda_angle

    def _geometric_losses(
        self,
        pred: torch.Tensor,
        geometry: torch.Tensor,
        y: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Regression and hybrid angle losses, gated by y=1."""
        mask = y.view(-1).bool()
        n_ellipse = int(mask.sum().item())
        zero = pred.new_zeros(())

        if n_ellipse == 0:
            return zero, zero, zero, zero, zero, n_ellipse

        pred_ell = pred[mask]
        target_ell = geometry[mask]

        # MPS SmoothL1 requires contiguous views after boolean indexing/slicing.
        pred_center = pred_ell[:, 1:3].contiguous()
        target_center = target_ell[:, :2].contiguous()
        pred_radii = pred_ell[:, 3:5].contiguous()
        target_radii = target_ell[:, 2:4].contiguous()
        center_loss = self.reg_criterion(pred_center, target_center)
        radii_loss = self.reg_criterion(pred_radii, target_radii)
        # For unit vectors, mean squared component error equals 1 - dot product.
        pred_angle = pred_ell[:, 5:7].contiguous()
        target_angle = target_ell[:, 4:6].contiguous()
        angle_cos_loss = (1.0 - (pred_angle * target_angle).sum(dim=-1)).mean()
        angle_huber_loss = self.angle_huber_criterion(pred_angle, target_angle)
        # Preserve historical behavior for all existing configurations. The
        # hybrid objective explicitly uses cosine + Huber when requested.
        angle_loss = (
            angle_cos_loss + self.angle_huber_weight * angle_huber_loss
            if self.angle_huber_weight > 0
            else self.reg_criterion(pred_angle, target_angle)
        )
        return (
            center_loss, radii_loss, angle_loss,
            angle_cos_loss, angle_huber_loss, n_ellipse,
        )

    def forward(
        self,
        pred: torch.Tensor,
        is_ellipse: torch.Tensor,
        geometry: torch.Tensor,
        ellipse_mask: torch.Tensor,
        model: Optional[nn.Module] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute multi-task loss.

        Args:
            pred: [B, 7] [ellipse_logit, cx, cy, a, b, u, v]
            is_ellipse: [B] binary targets y
            geometry: [B, 6] [cx, cy, a, b, cos(2θ), sin(2θ)]
            ellipse_mask: [B] 1 if ellipse else 0 (same as y when labeled)
            model: Model for Kendall uncertainty weights (optional)

        Returns:
            total_loss, dict of component losses
        """
        y = is_ellipse.view(-1).to(dtype=pred.dtype)
        logits = pred[:, 0]
        cls_loss = self.cls_criterion(logits, y)

        gate = ellipse_mask.view(-1).to(dtype=pred.dtype)
        if gate.numel() != y.numel():
            gate = y
        (
            center_loss, radii_loss, angle_loss,
            angle_cos_loss, angle_huber_loss, n_ellipse,
        ) = self._geometric_losses(
            pred, geometry, gate,
        )

        lambda_cls = self.lambda_cls
        lambda_center = self.lambda_center
        lambda_radii = self.lambda_radii
        lambda_angle = self.lambda_angle
        reg_term = pred.new_zeros(())

        if model is not None and getattr(model, 'auto_tune', False):
            weights = model.get_uncertainty_weights()
            lambda_cls = weights[0]
            lambda_center = weights[1]
            lambda_radii = weights[2]
            lambda_angle = weights[3]
            reg_term = model.get_regularization_term()

        fixed_geom = (
            self.lambda_center * center_loss
            + self.lambda_radii * radii_loss
            + self.lambda_angle * angle_loss
        )
        fixed_total_loss = self.lambda_cls * cls_loss + fixed_geom
        optimized_geom = (
            lambda_center * center_loss
            + lambda_radii * radii_loss
            + lambda_angle * angle_loss
        )
        total_loss = lambda_cls * cls_loss + optimized_geom + reg_term

        components = {
            'cls_loss': cls_loss.item(),
            'center_loss': center_loss.item(),
            'radii_loss': radii_loss.item(),
            'axes_loss': radii_loss.item(),
            'angle_loss': angle_loss.item(),
            'angle_cos_loss': angle_cos_loss.item(),
            'angle_huber_loss': angle_huber_loss.item(),
            # Keep the monitored metric comparable across fixed and adaptive
            # weighting. optimized_loss is the objective used for backprop.
            'total_loss': fixed_total_loss.item(),
            'optimized_loss': total_loss.item(),
            'n_ellipse': n_ellipse,
        }
        return total_loss, components


def build_loss(
    cls_loss: str = 'bce',
    reg_loss: str = 'smooth_l1',
    pos_rate: float = 0.7,
    **kwargs,
) -> MultiTaskLoss:
    """Build the four-term ellipse loss."""
    return MultiTaskLoss(
        cls_loss=cls_loss,
        reg_loss=reg_loss,
        pos_rate=pos_rate,
        **kwargs,
    )


if __name__ == '__main__':
    torch.manual_seed(42)

    focal = FocalLoss(alpha=0.7, gamma=2.0)
    logits = torch.randn(8)
    targets = torch.tensor([1, 1, 1, 1, 1, 0, 0, 0], dtype=torch.float32)
    loss = focal(logits, targets)
    print(f"Focal loss: {loss.item():.4f}")

    pred_c = torch.tensor([0.0, 0.9, 0.1])
    target_c = torch.tensor([0.1, 0.1, 0.9])
    dist = circular_distance(pred_c, target_c)
    print(f"Circular distances: {dist}")

    multi_loss = build_loss()
    pred = torch.randn(4, 7)
    pred[:, 1:5] = torch.sigmoid(pred[:, 1:5])
    pred[:, 5:7] = F.normalize(pred[:, 5:7], p=2, dim=-1)
    is_ellipse = torch.tensor([1, 1, 0, 1], dtype=torch.float32)
    geometry = torch.rand(4, 6)
    geometry[:, 4:6] = F.normalize(geometry[:, 4:6], p=2, dim=-1)
    ellipse_mask = is_ellipse.clone()

    total, components = multi_loss(pred, is_ellipse, geometry, ellipse_mask)
    print(f"\nMulti-task loss components:")
    for k, v in components.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Geometric terms must be zero when no ellipse is present
    neg_mask = torch.zeros(4)
    total_neg, components_neg = multi_loss(pred, torch.zeros(4), geometry, neg_mask)
    assert components_neg['center_loss'] == 0.0
    assert components_neg['radii_loss'] == 0.0
    assert components_neg['angle_loss'] == 0.0
    print("Gating check passed (geometric losses are 0 when y=0).")
