"""
CNN and MLP backbones for ellipse detection.

Model output is [B, 7] tensor: [ellipse_logit, cx, cy, a, b, u, v]
  - ellipse_logit: linear (BCEWithLogitsLoss)
  - cx, cy, a, b: sigmoid, normalized by image size → [0, 1]
  - u, v = (cos 2θ, sin 2θ): linear + L2 normalize → unit circle

Backbones:
  - CNN: Conv2x2 with residual connections, BatchNorm -> ReLU -> Dropout
  - MLP: Linear -> BatchNorm1d -> ReLU -> Dropout blocks
"""
from __future__ import annotations

from typing import Optional, Literal
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Mish(nn.Module):
    """Mish activation: x * tanh(softplus(x))."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(F.softplus(x))


def make_activation(name: str = 'relu', inplace: bool = True) -> nn.Module:
    """Build activation module for backbone and heads."""
    if name == 'relu':
        return nn.ReLU(inplace=inplace)
    if name == 'mish':
        return Mish()
    raise ValueError(f"Unknown activation: {name!r}. Choose from: relu, mish")


class ResidualBlock(nn.Module):
    """Residual block with configurable kernel size, BatchNorm, activation, Dropout."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout_rate: float = 0.2,
        activation: str = 'relu',
    ):
        super().__init__()
        padding = kernel_size // 2  # Same padding
        act = make_activation(activation)
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = act
        self.dropout = nn.Dropout2d(dropout_rate)
        
        # Skip connection (1x1 conv if channels change)
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.skip_bn = nn.Identity() if in_channels == out_channels else nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        identity = self.skip(x)
        identity = self.skip_bn(identity)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = out + identity
        out = self.act(out)
        return out


def create_cnn_backbone(
    in_channels: int = 1,
    base_filters: int = 32,
    depth: int = 3,
    dropout_rate: float = 0.2,
    fine_grained: bool = False,
    kernel_size: Optional[int] = None,
    activation: str = 'relu',
) -> tuple[nn.Module, int]:
    """
    Create CNN backbone with decreasing kernel sizes (7->5->3) and residual connections.
    
    Architecture:
        ResidualBlock(kernel=7) -> MaxPool -> ResidualBlock(kernel=5) -> MaxPool -> ...
    
    Args:
        in_channels: Number of input channels (1 for grayscale, 3 for RGB)
        base_filters: Number of filters in first conv layer
        depth: Number of residual blocks
        dropout_rate: Dropout probability
        fine_grained: Use more blocks with smaller filter growth
    
    Returns:
        backbone module, output feature dimension
    """
    layers = []
    current_channels = in_channels
    current_size = 50
    
    # Use one requested kernel throughout, or the legacy 7→5→3 schedule.
    kernel_sizes = (
        [kernel_size] * depth
        if kernel_size is not None
        else [7, 5, 3, 3, 3][:depth]
    )
    
    if fine_grained:
        # Fine-grained schedule at base_filters=32 is [16, 32, 48, 64, 96].
        # Scale it for width experiments while preserving the existing default.
        filter_sizes = [
            max(1, round(base_filters * multiplier))
            for multiplier in (0.5, 1.0, 1.5, 2.0, 3.0)
        ][:depth]
    else:
        # Standard: faster filter growth
        filter_sizes = [base_filters * (2 ** min(i, 2)) for i in range(depth)]
    
    for i, (out_channels, kernel) in enumerate(zip(filter_sizes, kernel_sizes)):
        layers.append(ResidualBlock(
            current_channels, out_channels,
            kernel_size=kernel,
            dropout_rate=dropout_rate,
            activation=activation,
        ))
        layers.append(nn.MaxPool2d(2, 2))
        current_channels = out_channels
        current_size = current_size // 2
    
    layers.append(nn.Flatten())
    backbone = nn.Sequential(*layers)
    
    output_dim = current_channels * (current_size ** 2)
    return backbone, output_dim


def create_mlp_backbone(
    in_channels: int = 1,
    img_size: int = 50,
    hidden_ratio: float = 0.5,
    depth: int = 3,
    dropout_rate: float = 0.3,
    activation: str = 'relu',
) -> tuple[nn.Module, int]:
    """
    Create MLP backbone.
    
    Architecture per block:
        Linear -> BatchNorm1d -> ReLU -> Dropout
    
    Widths shrink linearly toward hidden_ratio * input_size.
    
    Args:
        in_channels: Number of input channels
        img_size: Input image size (assumed square)
        hidden_ratio: Final hidden layer is this ratio of input size
        depth: Number of hidden layers
        dropout_rate: Dropout probability
    
    Returns:
        backbone module, output feature dimension
    """
    input_dim = in_channels * img_size * img_size
    target_dim = int(input_dim * hidden_ratio)
    
    # Compute layer sizes shrinking linearly
    hidden_sizes = []
    for i in range(depth):
        size = int(input_dim - (input_dim - target_dim) * (i + 1) / depth)
        hidden_sizes.append(size)
    
    act = make_activation(activation)
    layers = [nn.Flatten()]
    current_dim = input_dim
    
    for out_dim in hidden_sizes:
        layers.extend([
            nn.Linear(current_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            act,
            nn.Dropout(dropout_rate),
        ])
        current_dim = out_dim
    
    backbone = nn.Sequential(*layers)
    return backbone, current_dim


def create_head(
    in_features: int,
    out_features: int,
    hidden_dim: Optional[int] = None,
    dropout_rate: float = 0.2,
    depth: int = 2,
    activation: str = 'relu',
) -> nn.Module:
    """
    Create a head (classification or regression) with BatchNorm.
    
    Args:
        in_features: Input dimension
        out_features: Output dimension
        hidden_dim: Optional hidden layer dimension
        dropout_rate: Dropout probability
        depth: Number of hidden layers (default 2)
    
    Returns:
        Head module
    """
    if hidden_dim is None:
        return nn.Linear(in_features, out_features)
    
    act = make_activation(activation)
    layers = []
    current_dim = in_features
    
    # Hidden layers with BatchNorm -> activation -> Dropout
    for i in range(depth):
        out_dim = hidden_dim if i < depth - 1 else hidden_dim // 2
        layers.extend([
            nn.Linear(current_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            act,
            nn.Dropout(dropout_rate),
        ])
        current_dim = out_dim
    
    # Output layer
    layers.append(nn.Linear(current_dim, out_features))
    
    return nn.Sequential(*layers)


class EllipseDetector(nn.Module):
    """
    Multi-task ellipse detector.
    
    Output: [B, 7] tensor = [logit, cx, cy, a, b, cos(2θ), sin(2θ)]
    
    The logit is raw (no sigmoid) for BCE/focal loss.
    Geometry predictions:
      - cx, cy: center normalized to [0, 1]
      - a, b: semi-axes normalized to [0, ~0.5]
      - cos(2θ), sin(2θ): angle as unit circle → normalized to unit length
    
    Optional auto_tune: Kendall-style uncertainty weighting.
    When enabled, learns log-variance parameters for each task group:
        L_total = sum_i (exp(-s_i) * L_i + s_i)
    """
    
    def __init__(
        self,
        backbone: Literal['cnn', 'mlp'] = 'cnn',
        in_channels: int = 1,
        base_filters: int = 32,
        hidden_ratio: float = 0.5,
        depth: int = 3,
        dropout_rate: float = 0.2,
        head_hidden_dim: Optional[int] = 128,
        auto_tune: bool = False,
        fine_grained: bool = False,
        kernel_size: Optional[int] = None,
        activation: str = 'relu',
    ):
        """
        Args:
            backbone: 'cnn' or 'mlp'
            in_channels: Input channels (1 for grayscale, 3 for RGB)
            base_filters: Base filters for CNN backbone
            hidden_ratio: Hidden ratio for MLP backbone
            depth: Number of blocks in backbone
            dropout_rate: Dropout probability
            head_hidden_dim: Hidden dimension in heads (None for linear)
            auto_tune: Enable Kendall-style uncertainty weighting
            fine_grained: Use fine-grained CNN (smaller filters, less pooling)
        """
        super().__init__()
        
        self.backbone_type = backbone
        self.auto_tune = auto_tune
        self.is_multi_task = True
        self.num_tasks = 4
        self.activation = activation
        
        # Create backbone
        if backbone == 'cnn':
            self.backbone, feat_dim = create_cnn_backbone(
                in_channels=in_channels,
                base_filters=base_filters,
                depth=depth,
                dropout_rate=dropout_rate,
                fine_grained=fine_grained,
                kernel_size=kernel_size,
                activation=activation,
            )
        else:
            self.backbone, feat_dim = create_mlp_backbone(
                in_channels=in_channels,
                hidden_ratio=hidden_ratio,
                depth=depth,
                dropout_rate=dropout_rate,
                activation=activation,
            )
        
        head = lambda out_dim: create_head(
            in_features=feat_dim,
            out_features=out_dim,
            hidden_dim=head_hidden_dim,
            dropout_rate=dropout_rate,
            depth=2,
            activation=activation,
        )
        # Four named heads → 7 outputs: [logit, cx, cy, a, b, u, v]
        self.cls_head = head(1)
        self.center_head = head(2)
        self.radii_head = head(2)
        self.angle_head = head(2)
        
        # Kendall-style uncertainty weights: [cls, center, radii, angle]
        if auto_tune:
            self.log_vars = nn.Parameter(torch.zeros(self.num_tasks))
        
        # Store config for checkpoint
        self.config = {
            'backbone': backbone,
            'in_channels': in_channels,
            'base_filters': base_filters,
            'hidden_ratio': hidden_ratio,
            'depth': depth,
            'dropout_rate': dropout_rate,
            'head_hidden_dim': head_hidden_dim,
            'auto_tune': auto_tune,
            'fine_grained': fine_grained,
            'kernel_size': kernel_size,
            'activation': activation,
        }
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            image: [B, C, H, W] input tensor
        
        Returns:
            [B, 7] tensor: [logit, cx, cy, a, b, u, v]
            - logit: ellipse classification (raw, use BCEWithLogitsLoss)
            - cx, cy: center position, sigmoid activated, normalized by image size
            - a, b: radii, sigmoid activated, normalized by image size
            - u, v: angle as unit vector (cos2θ, sin2θ), L2 normalized
        """
        features = self.backbone(image)
        
        logit = self.cls_head(features)  # [B, 1] linear (BCEWithLogitsLoss)
        center = torch.sigmoid(self.center_head(features))  # [B, 2] cx, cy in [0, 1]
        radii = torch.sigmoid(self.radii_head(features))  # [B, 2] a, b in [0, 1]
        angle = F.normalize(self.angle_head(features), p=2, dim=-1)  # [B, 2] unit (u, v)
        
        # [B, 7] = [ellipse_logit, cx, cy, a, b, u, v]
        return torch.cat([logit, center, radii, angle], dim=-1)
    
    def get_uncertainty_weights(self) -> Optional[torch.Tensor]:
        """
        Get precision weights from log variances.
        
        Returns:
            exp(-log_var) for each task group, or None if auto_tune is False
        """
        if not self.auto_tune:
            return None
        return torch.exp(-self.log_vars)
    
    def get_regularization_term(self) -> torch.Tensor:
        """
        Get regularization term for Kendall loss.
        
        Returns:
            sum of log_vars (encourages not making all variances large)
        """
        if not self.auto_tune:
            return torch.tensor(0.0)
        return self.log_vars.sum()


def build_model(
    backbone: str = 'cnn',
    grayscale: bool = True,
    in_channels: Optional[int] = None,
    base_filters: int = 32,
    hidden_ratio: float = 0.5,
    depth: int = 3,
    dropout_rate: float = 0.2,
    head_hidden_dim: int = 128,
    auto_tune: bool = False,
    fine_grained: bool = False,
    kernel_size: Optional[int] = None,
    activation: str = 'relu',
) -> EllipseDetector:
    """
    Build ellipse detector model.
    
    Args:
        backbone: 'cnn' or 'mlp'
        grayscale: If True, in_channels=1 unless overridden
        in_channels: Explicit input channel count (e.g. 4 for RGB+edge)
        base_filters: Base filters for CNN
        hidden_ratio: Hidden ratio for MLP
        depth: Backbone depth
        dropout_rate: Dropout probability
        head_hidden_dim: Hidden dim in heads
        auto_tune: Enable uncertainty weighting
        fine_grained: Use fine-grained CNN (smaller filters, less pooling)
    
    Returns:
        EllipseDetector model
    """
    if in_channels is None:
        in_channels = 1 if grayscale else 3
    
    model = EllipseDetector(
        backbone=backbone,
        in_channels=in_channels,
        fine_grained=fine_grained,
        base_filters=base_filters,
        hidden_ratio=hidden_ratio,
        depth=depth,
        dropout_rate=dropout_rate,
        head_hidden_dim=head_hidden_dim,
        auto_tune=auto_tune,
        kernel_size=kernel_size,
        activation=activation,
    )
    
    return model


def freeze_all(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_all(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        param.requires_grad = True
        print(name)


def freeze_backbone(model: nn.Module) -> None:
    """Keep heads trainable; freeze the backbone by name prefix."""
    freeze_all(model)
    for attr in ('cls_head', 'center_head', 'radii_head', 'angle_head'):
        if hasattr(model, attr):
            for name, param in getattr(model, attr).named_parameters():
                param.requires_grad = True
                print(f"{attr}.{name}")


def print_model_summary(model: nn.Module, input_size: tuple = (1, 1, 50, 50)):
    """Print model summary using torchinfo."""
    try:
        from torchinfo import summary
        print("\n" + "=" * 60)
        print("MODEL SUMMARY")
        print("=" * 60)
        summary(model, input_size=input_size, col_names=["input_size", "output_size", "num_params"])
        print("=" * 60 + "\n")
    except ImportError:
        print("torchinfo not available, skipping summary")
        # Fallback: count parameters
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total:,}")
        print(f"Trainable parameters: {trainable:,}")


if __name__ == '__main__':
    # Test both backbones
    print("Testing CNN backbone (grayscale):")
    model_cnn = build_model(backbone='cnn', grayscale=True, depth=3)
    print_model_summary(model_cnn, input_size=(1, 1, 50, 50))
    
    # Test forward
    x = torch.randn(4, 1, 50, 50)
    out = model_cnn(x)
    print(f"CNN output shape: {out.shape}")  # [4, 7]
    print(f"  angle L2 norms: {out[:, 5:7].norm(dim=-1)}")
    
    print("\nTesting MLP backbone (grayscale):")
    model_mlp = build_model(backbone='mlp', grayscale=True, depth=3)
    print_model_summary(model_mlp, input_size=(1, 1, 50, 50))
    
    out = model_mlp(x)
    print(f"MLP output shape: {out.shape}")  # [4, 7]
    
    print("\nTesting auto_tune mode:")
    model_auto = build_model(backbone='cnn', grayscale=True, auto_tune=True)
    weights = model_auto.get_uncertainty_weights()
    print(f"Uncertainty weights: {weights}")
    reg = model_auto.get_regularization_term()
    print(f"Regularization term: {reg}")
