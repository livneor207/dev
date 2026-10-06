"""
Inference module for ellipse detector.

Loads best model, applies threshold, and predicts on dataset.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Dict, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import EllipseDataset, create_inference_loader
from .model import EllipseDetector, build_model
from .metrics import compute_classification_metrics, compute_regression_metrics


def load_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> Tuple[nn.Module, Dict]:
    """
    Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
    
    Returns:
        model, config dict
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    config = checkpoint.get('config', {})
    
    # Build model from config
    in_channels = config.get('in_channels')
    model = build_model(
        backbone=config.get('backbone', 'cnn'),
        grayscale=config.get('in_channels', 1) == 1 if in_channels is None else in_channels == 1,
        in_channels=in_channels,
        base_filters=config.get('base_filters', 32),
        kernel_size=config.get('kernel_size'),
        hidden_ratio=config.get('hidden_ratio', 0.5),
        depth=config.get('depth', 3),
        dropout_rate=config.get('dropout_rate', 0.2),
        head_hidden_dim=config.get('head_hidden_dim', 128),
        auto_tune=config.get('auto_tune', False),
        fine_grained=config.get('fine_grained', False),
        activation=config.get('activation', 'relu'),
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, config


def load_threshold(
    run_dir: str | Path,
    default: float = 0.5,
) -> float:
    """
    Load best threshold from training run.
    
    Args:
        run_dir: Directory containing best_threshold.json
        default: Default threshold if file not found
    
    Returns:
        Threshold value
    """
    threshold_path = Path(run_dir) / 'best_threshold.json'
    if threshold_path.exists():
        with open(threshold_path) as f:
            data = json.load(f)
        return data.get('threshold', default)
    return default


def predict(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Run inference on data.
    
    Args:
        model: Model to use
        data_loader: DataLoader
        device: Device
        threshold: Classification threshold
    
    Returns:
        probs, preds (binary), geometry (denormalized), paths
    """
    model.eval()
    
    all_probs = []
    all_geom = []
    all_paths = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Inference'):
            images = batch['image'].to(device)
            paths = batch['path']
            
            # Forward
            out = model(images)
            
            # Classification
            probs = torch.sigmoid(out[:, 0]).cpu().numpy()
            
            # Geometry (denormalize)
            # Model output: [logit, cx, cy, a, b, cos(2θ), sin(2θ)]
            geom_norm = out[:, 1:7].cpu()  # [B, 6]
            geom = EllipseDataset.inverse_transform_geometry(geom_norm)
            
            all_probs.append(probs)
            all_geom.append(geom.numpy())
            all_paths.extend(paths)
    
    probs = np.concatenate(all_probs)
    preds = (probs >= threshold).astype(int)
    geom = np.concatenate(all_geom)
    
    # Zero out geometry for non-ellipse predictions
    geom[preds == 0] = 0
    
    return probs, preds, geom, all_paths


def create_predictions_df(
    paths: list,
    probs: np.ndarray,
    preds: np.ndarray,
    geom: np.ndarray,
) -> pd.DataFrame:
    """
    Create predictions DataFrame.
    
    Args:
        paths: Image paths
        probs: Probabilities
        preds: Binary predictions
        geom: Geometry predictions [N, 5]
    
    Returns:
        DataFrame with predictions
    """
    df = pd.DataFrame({
        'path': paths,
        'prob': probs,
        'is_ellipse': preds.astype(bool),
        'center_x': geom[:, 0].astype(int),
        'center_y': geom[:, 1].astype(int),
        'angle': geom[:, 2].astype(int),
        'axis_1': geom[:, 3].astype(int),
        'axis_2': geom[:, 4].astype(int),
    })
    return df


def run_inference(
    checkpoint_path: str | Path,
    image_dir: str | Path,
    output_path: str | Path,
    csv_path: Optional[str | Path] = None,
    threshold: Optional[float] = None,
    batch_size: int = 32,
    device: Optional[torch.device] = None,
) -> Tuple[pd.DataFrame, Optional[Dict]]:
    """
    Run full inference pipeline.
    
    Args:
        checkpoint_path: Path to model checkpoint
        image_dir: Directory with images
        output_path: Path to save predictions CSV
        csv_path: Optional path to labeled CSV for metrics
        threshold: Classification threshold (loaded from run_dir if None)
        batch_size: Batch size
        device: Device (auto-detected if None)
    
    Returns:
        predictions_df, metrics (if labeled data)
    """
    # Setup
    checkpoint_path = Path(checkpoint_path)
    run_dir = checkpoint_path.parent
    
    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device('mps')
        elif torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    
    print(f"Device: {device}")
    
    # Load model
    print(f"Loading model from {checkpoint_path}")
    model, config = load_checkpoint(checkpoint_path, device)
    in_channels = config.get('in_channels')
    grayscale = in_channels == 1 if in_channels is not None else config.get('in_channels', 1) == 1
    
    # Load threshold
    if threshold is None:
        threshold = load_threshold(run_dir)
    print(f"Using threshold: {threshold:.4f}")
    
    # Load training config for normalization stats
    config_path = run_dir / 'config.json'
    if config_path.exists():
        with open(config_path) as f:
            train_config = json.load(f)
        normalize_mean = train_config.get('normalize_mean')
        normalize_std = train_config.get('normalize_std')
        color_space = train_config.get('color_space', 'rgb')
    else:
        normalize_mean = None
        normalize_std = None
        color_space = 'rgb'
    
    # Create data loader
    print(f"Loading data from {image_dir}")
    data_loader, is_labeled = create_inference_loader(
        image_dir=image_dir,
        csv_path=csv_path,
        batch_size=batch_size,
        grayscale=grayscale,
        color_space=color_space,
        normalize_mean=normalize_mean,
        normalize_std=normalize_std,
    )
    
    # Predict
    probs, preds, geom, paths = predict(model, data_loader, device, threshold)
    
    # Create DataFrame
    df = create_predictions_df(paths, probs, preds, geom)
    
    # Save predictions
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path}")
    
    # Compute metrics if labeled
    metrics = None
    if is_labeled and csv_path is not None:
        # Load true labels
        from .data import parse_csv
        true_df = parse_csv(csv_path)
        
        # Match by path
        df = df.set_index('path')
        true_df = true_df.set_index('path')
        
        # Get matched data
        common_paths = df.index.intersection(true_df.index)
        
        if len(common_paths) > 0:
            probs_matched = df.loc[common_paths, 'prob'].values
            targets_matched = true_df.loc[common_paths, 'is_ellipse'].values.astype(float)
            
            # Classification metrics
            cls_metrics = compute_classification_metrics(probs_matched, targets_matched, threshold)
            
            # Regression metrics (on ellipse samples) - in pixel/degree space
            mask = targets_matched.astype(bool)
            if mask.sum() > 0:
                pred_geom = df.loc[common_paths, ['center_x', 'center_y', 'angle', 'axis_1', 'axis_2']].values.astype(float)
                true_geom = true_df.loc[common_paths, ['center_x', 'center_y', 'angle', 'axis_1', 'axis_2']].values.astype(float)
                
                # Compute overall regression metrics in pixel/degree space
                geom_metrics = compute_regression_metrics(pred_geom, true_geom, mask)
                
                # Add per-component metrics manually (in original space for interpretability)
                pred_masked = pred_geom[mask]
                true_masked = true_geom[mask]
                
                # Center error (in pixels)
                center_err = np.sqrt((pred_masked[:, 0] - true_masked[:, 0])**2 + 
                                     (pred_masked[:, 1] - true_masked[:, 1])**2)
                geom_metrics['center_rmse_px'] = float(np.sqrt(np.mean(center_err**2)))
                
                # Axes error (in pixels)
                axes_err = np.abs(pred_masked[:, 3:5] - true_masked[:, 3:5])
                geom_metrics['axes_mae_px'] = float(np.mean(axes_err))
                
                # Angle error (in degrees, accounting for 180° periodicity)
                angle_diff = np.abs(pred_masked[:, 2] - true_masked[:, 2])
                angle_diff = np.minimum(angle_diff, 180 - angle_diff)
                geom_metrics['angle_mae_deg'] = float(np.mean(angle_diff))
            else:
                geom_metrics = {}
            
            metrics = {**cls_metrics, **geom_metrics}
            
            print("\nMetrics:")
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}")
        
        df = df.reset_index()
    
    return df, metrics


if __name__ == '__main__':
    # Quick test
    print("Inference module loaded successfully")
