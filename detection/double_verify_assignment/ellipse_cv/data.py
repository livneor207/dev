"""
CSV parsing, stratified split, and EllipseDataset for training and inference.

Sample format from CSV:
    images/train/0001.jpg True, 34, 32, 25, 16, 17

Dataset returns dict for model(**inputs) style:
    - image: [C, 50, 50] tensor
    - is_ellipse: float {0, 1}
    - geometry: [cx/W, cy/H, a/S, b/S, cos(2θ), sin(2θ)] - 6 values
    - ellipse_mask: 1 if ellipse else 0

Angle encoding: cos(2θ), sin(2θ)
  - Handles 180° symmetry naturally (cos(2θ) has period 180°)
  - No discontinuity at boundaries
  - Values in [-1, 1] range
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert an RGB float array in [0, 1] to normalized CIELAB channels."""
    linear_rgb = np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4,
    )
    xyz = linear_rgb @ np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=np.float32,
    ).T
    xyz /= np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
    delta = 6.0 / 29.0
    xyz_root = np.where(
        xyz > delta ** 3,
        np.cbrt(xyz),
        xyz / (3 * delta ** 2) + 4.0 / 29.0,
    )
    l_star = 116.0 * xyz_root[..., 1] - 16.0
    a_star = 500.0 * (xyz_root[..., 0] - xyz_root[..., 1])
    b_star = 200.0 * (xyz_root[..., 1] - xyz_root[..., 2])
    return np.stack([l_star / 100.0, a_star / 128.0, b_star / 128.0], axis=-1)


def _conv2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Reflect-padded 2D convolution for small kernels."""
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(img, ((ph, ph), (pw, pw)), mode='reflect')
    out = np.zeros_like(img, dtype=np.float32)
    for i in range(kh):
        for j in range(kw):
            out += kernel[i, j] * padded[i:i + img.shape[0], j:j + img.shape[1]]
    return out


def sobel_edge_magnitude(gray: np.ndarray) -> np.ndarray:
    """Sobel gradient magnitude from a grayscale image in [0, 1]."""
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    gx = _conv2d(gray.astype(np.float32), kx)
    gy = _conv2d(gray.astype(np.float32), ky)
    return np.sqrt(gx * gx + gy * gy)


def input_channel_count(grayscale: bool, color_space: str = 'rgb') -> int:
    """Number of model input channels for the selected preprocessing."""
    if grayscale:
        return 1
    if color_space == 'rgb_edge':
        return 4
    return 3


def image_to_array(img: Image.Image, color_space: str = 'rgb') -> np.ndarray:
    """Convert an RGB PIL image to a channel-last float array for the model."""
    if color_space == 'hsv':
        return np.asarray(img.convert('HSV'), dtype=np.float32) / 255.0

    rgb = np.asarray(img.convert('RGB'), dtype=np.float32) / 255.0
    if color_space == 'lab':
        return _rgb_to_lab(rgb)
    if color_space == 'rgb_edge':
        gray = (
            0.299 * rgb[..., 0]
            + 0.587 * rgb[..., 1]
            + 0.114 * rgb[..., 2]
        )
        edge = sobel_edge_magnitude(gray)
        return np.concatenate([rgb, edge[..., np.newaxis]], axis=-1)
    if color_space == 'rgb':
        return rgb
    raise ValueError(
        f"Unknown color space: {color_space!r}. Choose from: rgb, rgb_edge, lab, hsv"
    )


def parse_csv(csv_path: str | Path) -> pd.DataFrame:
    """
    Parse the awkward CSV format into a clean DataFrame.
    
    Input format: "images/train/0001.jpg True, 34, 32, 25, 16, 17"
    Output columns: path, is_ellipse, center_x, center_y, angle, axis_1, axis_2
    """
    rows = []
    with open(csv_path, 'r') as f:
        lines = f.readlines()
    
    # Skip header line
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        # Split on first space: path rest
        parts = line.split(' ', 1)
        if len(parts) != 2:
            continue
        path = parts[0]
        rest = parts[1]
        # rest = "True, 34, 32, 25, 16, 17"
        values = [v.strip() for v in rest.split(',')]
        is_ellipse = values[0] == 'True'
        center_x = int(values[1])
        center_y = int(values[2])
        angle = int(values[3])
        axis_1 = int(values[4])
        axis_2 = int(values[5])
        rows.append({
            'path': path,
            'is_ellipse': is_ellipse,
            'center_x': center_x,
            'center_y': center_y,
            'angle': angle,
            'axis_1': axis_1,
            'axis_2': axis_2,
        })
    
    df = pd.DataFrame(rows)
    print(f"Parsed {len(df)} samples from {csv_path}")
    print(f"  Ellipse: {df['is_ellipse'].sum()} ({df['is_ellipse'].mean()*100:.1f}%)")
    print(f"  Non-ellipse: {(~df['is_ellipse']).sum()} ({(~df['is_ellipse']).mean()*100:.1f}%)")
    return df


def stratified_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split DataFrame stratified by is_ellipse."""
    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df['is_ellipse'],
        random_state=random_state,
    )
    print(f"Train: {len(train_df)} samples, Val: {len(val_df)} samples")
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def compute_normalization_stats(
    df: pd.DataFrame,
    base_dir: str | Path,
    grayscale: bool = True,
    color_space: str = 'rgb',
) -> Tuple[float | List[float], float | List[float]]:
    """
    Compute streaming mean/std from training images after scaling uint8 to [0, 1].

    Returns scalar statistics for grayscale and per-channel lists otherwise.
    """
    channels = input_channel_count(grayscale, color_space)
    pixel_sum = np.zeros(channels, dtype=np.float64)
    pixel_sq_sum = np.zeros(channels, dtype=np.float64)
    pixel_count = 0

    for path in df['path'].values:
        full_path = os.path.join(base_dir, path)
        img = Image.open(full_path)
        if grayscale:
            arr = np.asarray(img.convert('L'), dtype=np.float64) / 255.0
        else:
            arr = image_to_array(img, color_space=color_space).astype(np.float64)
        if grayscale:
            arr = arr[..., np.newaxis]  # [H, W, 1]
        elif arr.ndim == 2:
            arr = arr[..., np.newaxis]
        flat = arr.reshape(-1, channels)
        pixel_sum += flat.sum(axis=0)
        pixel_sq_sum += np.square(flat).sum(axis=0)
        pixel_count += flat.shape[0]

    mean = pixel_sum / pixel_count
    variance = np.maximum(pixel_sq_sum / pixel_count - np.square(mean), 0.0)
    std = np.sqrt(variance)

    if grayscale:
        mean_out: float | List[float] = float(mean[0])
        std_out: float | List[float] = float(std[0])
    else:
        mean_out = mean.tolist()
        std_out = std.tolist()
    print(
        f"Normalization stats (grayscale={grayscale}, color_space={color_space}): "
        f"mean={mean_out}, std={std_out}"
    )
    return mean_out, std_out


class EllipseDataset(Dataset):
    """
    Dataset for ellipse detection with multi-task learning.
    
    Supports:
    - Grayscale (default) or RGB input
    - Optional train-only photometric augmentation (color jitter, grayscale, blur)
    - Train-only normalization stats
    - Unlabeled test mode (when df has no label columns)
    
    Returns dict:
        image: [C, 50, 50] tensor
        is_ellipse: float {0, 1}
        geometry: [6] tensor (cx/W, cy/H, a/S, b/S, cos(2θ), sin(2θ))
        ellipse_mask: float {0, 1}
        path: str (for debugging)
    
    Geometry encoding:
        - cx, cy: center divided by image size → [0, 1]
        - a, b: semi-axes divided by image size → [0, ~0.5]
        - cos(2θ), sin(2θ): angle as unit circle → [-1, 1]
    """
    
    # Constants for normalization
    IMG_SIZE = 50
    GEOM_DIM = 6  # cx, cy, a, b, cos(2θ), sin(2θ)
    
    def __init__(
        self,
        df: pd.DataFrame,
        base_dir: str | Path,
        train: bool = True,
        grayscale: bool = True,
        color_space: str = 'rgb',
        normalize_mean: Optional[float | List[float]] = None,
        normalize_std: Optional[float | List[float]] = None,
        blur_prob: float = 0.0,
        blur_radius: float = 1.0,
        brightness_jitter: float = 0.0,
        contrast_jitter: float = 0.0,
        saturation_jitter: float = 0.0,
        grayscale_aug_prob: float = 0.0,
    ):
        """
        Args:
            df: DataFrame with columns path, is_ellipse, center_x, center_y, angle, axis_1, axis_2
                For unlabeled data, only 'path' column is required.
            base_dir: Base directory for image paths
            train: Whether this is training data (enables augmentation)
            grayscale: Convert to grayscale (default True for smaller model)
            color_space: RGB, RGB+edge, CIELAB, or HSV input when grayscale is False
            normalize_mean: Mean for normalization (computed from train if None)
            normalize_std: Std for normalization (computed from train if None)
            blur_prob: Probability of applying Gaussian blur (train only)
            blur_radius: Radius for Gaussian blur
            brightness_jitter: Maximum relative brightness adjustment for RGB train images
            contrast_jitter: Maximum relative contrast adjustment for RGB train images
            saturation_jitter: Maximum relative saturation adjustment for RGB train images
            grayscale_aug_prob: Probability of converting an RGB train image to grayscale
                and replicating it across all three channels
        """
        self.df = df.reset_index(drop=True)
        self.base_dir = Path(base_dir)
        self.train = train
        self.grayscale = grayscale
        self.color_space = color_space
        self.blur_prob = blur_prob if train else 0.0
        self.blur_radius = blur_radius
        self.brightness_jitter = brightness_jitter if train and not grayscale else 0.0
        self.contrast_jitter = contrast_jitter if train and not grayscale else 0.0
        self.saturation_jitter = saturation_jitter if train and not grayscale else 0.0
        self.grayscale_aug_prob = grayscale_aug_prob if train and not grayscale else 0.0
        
        # Check if labeled data
        self.is_labeled = 'is_ellipse' in df.columns
        
        # Normalization stats
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        
        # Precompute arrays for efficiency
        self.paths = df['path'].values
        if self.is_labeled:
            self.is_ellipse = df['is_ellipse'].values.astype(np.float32)
            # Canonicalize equivalent ellipse labels:
            # (axis_1, axis_2, θ) == (axis_2, axis_1, θ + 90°).
            # Always make axis_1 the major radius so one image has one angle target.
            angles = df['angle'].values.astype(np.float32).copy()
            axis_1 = df['axis_1'].values.astype(np.float32).copy()
            axis_2 = df['axis_2'].values.astype(np.float32).copy()
            swap = (axis_1 < axis_2) & self.is_ellipse.astype(bool)
            axis_1[swap], axis_2[swap] = axis_2[swap], axis_1[swap].copy()
            angles[swap] += 90.0
            self.angles = (angles % 180).astype(np.float32)
            self.center_x = df['center_x'].values.astype(np.float32)
            self.center_y = df['center_y'].values.astype(np.float32)
            self.axis_1 = axis_1
            self.axis_2 = axis_2
        
        # Metadata
        self.dataset_size = len(df)
        
        # Positive class rate for focal loss alpha
        if self.is_labeled:
            self.pos_rate = float(df['is_ellipse'].mean())
        else:
            self.pos_rate = None
    
    def __len__(self) -> int:
        return self.dataset_size
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Load image
        path = self.paths[idx]
        full_path = self.base_dir / path
        img = Image.open(full_path)
        
        # Photometric augmentations operate in RGB; color conversion happens after.
        if self.grayscale:
            img = img.convert('L')
        else:
            img = img.convert('RGB')
        
        # Train-only photometric augmentation; it never changes spatial geometry.
        if self.train and self.brightness_jitter > 0:
            factor = np.random.uniform(1 - self.brightness_jitter, 1 + self.brightness_jitter)
            img = ImageEnhance.Brightness(img).enhance(factor)
        if self.train and self.contrast_jitter > 0:
            factor = np.random.uniform(1 - self.contrast_jitter, 1 + self.contrast_jitter)
            img = ImageEnhance.Contrast(img).enhance(factor)
        if self.train and self.saturation_jitter > 0:
            factor = np.random.uniform(1 - self.saturation_jitter, 1 + self.saturation_jitter)
            img = ImageEnhance.Color(img).enhance(factor)
        if self.train and self.grayscale_aug_prob > 0 and np.random.random() < self.grayscale_aug_prob:
            img = img.convert('L').convert('RGB')

        # Optional blur augmentation.
        if self.train and self.blur_prob > 0 and np.random.random() < self.blur_prob:
            img = img.filter(ImageFilter.GaussianBlur(radius=self.blur_radius))
        
        # Convert to the requested input color space and channel-first layout.
        if self.grayscale:
            arr = np.array(img, dtype=np.float32) / 255.0
            arr = arr[np.newaxis, :, :]  # [1, H, W]
        else:
            arr = image_to_array(img, color_space=self.color_space)
            arr = arr.transpose(2, 0, 1)  # [C, H, W]
        
        # Standardize with training-split statistics; RGB uses one value/channel.
        if self.normalize_mean is not None and self.normalize_std is not None:
            mean = np.asarray(self.normalize_mean, dtype=np.float32)
            std = np.asarray(self.normalize_std, dtype=np.float32)
            if mean.ndim == 1:
                mean = mean[:, np.newaxis, np.newaxis]
                std = std[:, np.newaxis, np.newaxis]
            arr = (arr - mean) / (std + 1e-8)
        
        image = torch.from_numpy(arr)
        
        # Labels
        if self.is_labeled:
            is_ellipse = torch.tensor(self.is_ellipse[idx], dtype=torch.float32)
            ellipse_mask = is_ellipse.clone()
            
            # Normalize geometry: [cx/W, cy/H, a/S, b/S, cos(2θ), sin(2θ)]
            cx = self.center_x[idx] / self.IMG_SIZE
            cy = self.center_y[idx] / self.IMG_SIZE
            a = self.axis_1[idx] / self.IMG_SIZE
            b = self.axis_2[idx] / self.IMG_SIZE
            
            # Angle encoding: cos(2θ), sin(2θ) for continuous representation
            # angle is already in [0, 180) from preprocessing
            theta_rad = np.deg2rad(self.angles[idx])
            cos_2theta = np.cos(2 * theta_rad)
            sin_2theta = np.sin(2 * theta_rad)
            
            geometry = torch.tensor([cx, cy, a, b, cos_2theta, sin_2theta], dtype=torch.float32)
        else:
            # Unlabeled mode - placeholders
            is_ellipse = torch.tensor(0.0, dtype=torch.float32)
            ellipse_mask = torch.tensor(0.0, dtype=torch.float32)
            geometry = torch.zeros(self.GEOM_DIM, dtype=torch.float32)
        
        return {
            'image': image,
            'is_ellipse': is_ellipse,
            'geometry': geometry,
            'ellipse_mask': ellipse_mask,
            'path': path,
        }
    
    @staticmethod
    def inverse_transform_geometry(
        pred_geom: torch.Tensor,
        img_size: int = 50,
    ) -> torch.Tensor:
        """
        Convert model output back to pixel/degree space.
        
        Args:
            pred_geom: [B, 6] tensor with (cx/W, cy/H, a/S, b/S, cos(2θ), sin(2θ))
            img_size: Image size for denormalization
        
        Returns:
            [B, 5] tensor with (cx, cy, angle, axis_1, axis_2) in pixel/degree space
        """
        if pred_geom.dim() == 1:
            pred_geom = pred_geom.unsqueeze(0)
        
        B = pred_geom.shape[0]
        result = torch.zeros(B, 5, dtype=pred_geom.dtype, device=pred_geom.device)
        
        # Denormalize position and axes
        result[:, 0] = pred_geom[:, 0] * img_size  # cx
        result[:, 1] = pred_geom[:, 1] * img_size  # cy
        result[:, 3] = pred_geom[:, 2] * img_size  # axis_1 (a)
        result[:, 4] = pred_geom[:, 3] * img_size  # axis_2 (b)
        
        # Decode angle from cos(2θ), sin(2θ)
        cos_2theta = pred_geom[:, 4]
        sin_2theta = pred_geom[:, 5]
        # theta = 0.5 * atan2(sin2theta, cos2theta), then wrap to [0, π)
        theta_rad = 0.5 * torch.atan2(sin_2theta, cos_2theta)
        theta_rad = torch.remainder(theta_rad, torch.pi)  # Ensure [0, π)
        result[:, 2] = torch.rad2deg(theta_rad)  # Convert to degrees [0, 180)
        
        return result.round()


def create_dataloaders(
    csv_path: str | Path,
    base_dir: str | Path,
    batch_size: int = 32,
    grayscale: bool = True,
    color_space: str = 'rgb',
    blur_prob: float = 0.0,
    blur_radius: float = 1.0,
    brightness_jitter: float = 0.0,
    contrast_jitter: float = 0.0,
    saturation_jitter: float = 0.0,
    grayscale_aug_prob: float = 0.0,
    num_workers: int = 0,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
    """
    Create train and validation dataloaders from CSV.
    
    Returns:
        train_loader, val_loader, metadata dict
    """
    # Parse and split
    df = parse_csv(csv_path)
    train_df, val_df = stratified_split(df, test_size=test_size, random_state=random_state)
    
    # Compute normalization stats from training data only
    mean, std = compute_normalization_stats(
        train_df, base_dir, grayscale=grayscale, color_space=color_space,
    )
    
    # Create datasets
    train_dataset = EllipseDataset(
        df=train_df,
        base_dir=base_dir,
        train=True,
        grayscale=grayscale,
        color_space=color_space,
        normalize_mean=mean,
        normalize_std=std,
        blur_prob=blur_prob,
        blur_radius=blur_radius,
        brightness_jitter=brightness_jitter,
        contrast_jitter=contrast_jitter,
        saturation_jitter=saturation_jitter,
        grayscale_aug_prob=grayscale_aug_prob,
    )
    
    val_dataset = EllipseDataset(
        df=val_df,
        base_dir=base_dir,
        train=False,
        grayscale=grayscale,
        color_space=color_space,
        normalize_mean=mean,
        normalize_std=std,
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    metadata = {
        'normalize_mean': mean,
        'normalize_std': std,
        'pixel_scaling': 'uint8_div_255',
        'normalization': 'train_channel_mean_std',
        'grayscale': grayscale,
        'color_space': color_space,
        'canonical_axes': True,
        'pos_rate': train_dataset.pos_rate,
        'train_size': len(train_dataset),
        'val_size': len(val_dataset),
    }
    
    return train_loader, val_loader, metadata


def create_inference_loader(
    image_dir: str | Path,
    csv_path: Optional[str | Path] = None,
    batch_size: int = 32,
    grayscale: bool = True,
    color_space: str = 'rgb',
    normalize_mean: Optional[float | List[float]] = None,
    normalize_std: Optional[float | List[float]] = None,
    num_workers: int = 0,
) -> Tuple[DataLoader, bool]:
    """
    Create dataloader for inference.
    
    If csv_path is provided, loads labeled data. Otherwise, loads images from directory.
    
    Returns:
        loader, is_labeled
    """
    if csv_path is not None:
        # Labeled inference
        df = parse_csv(csv_path)
        base_dir = Path(image_dir).parent  # Assume CSV paths are relative to parent
    else:
        # Unlabeled inference - list images in directory
        image_dir = Path(image_dir)
        image_files = sorted(list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png')))
        
        # Create DataFrame with just paths
        # Paths should be relative like "images/test/0000.jpg"
        df = pd.DataFrame({
            'path': [str(f.relative_to(image_dir.parent.parent)) for f in image_files]
        })
        base_dir = image_dir.parent.parent
        print(f"Found {len(df)} images for inference")
    
    dataset = EllipseDataset(
        df=df,
        base_dir=base_dir,
        train=False,
        grayscale=grayscale,
        color_space=color_space,
        normalize_mean=normalize_mean,
        normalize_std=normalize_std,
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return loader, dataset.is_labeled


if __name__ == '__main__':
    # Quick test
    import sys
    base_dir = Path(__file__).parent.parent
    csv_path = base_dir / 'train_data.csv'
    
    if csv_path.exists():
        train_loader, val_loader, meta = create_dataloaders(
            csv_path=csv_path,
            base_dir=base_dir,
            batch_size=4,
            grayscale=True,
        )
        
        # Test one batch
        batch = next(iter(train_loader))
        print(f"\nBatch shapes:")
        print(f"  image: {batch['image'].shape}")
        print(f"  is_ellipse: {batch['is_ellipse'].shape}")
        print(f"  geometry: {batch['geometry'].shape}")
        print(f"  ellipse_mask: {batch['ellipse_mask'].shape}")
        print(f"\nMetadata: {meta}")
    else:
        print(f"CSV not found: {csv_path}")
