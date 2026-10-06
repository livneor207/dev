"""
Training loop for ellipse detector.

Follows helper-dispatch style:
- One forward function for train and eval
- Checkpoint on improvement
- Patience-based early stopping
- Per-epoch report persistence
"""
from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .losses import MultiTaskLoss
from .metrics import MetricsAccumulator
from .optim import get_lr
from .visualize import generate_highest_error_grid


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """Get best available device."""
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def forward_all_dataset(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: MultiTaskLoss,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    desc: str = 'Training',
) -> Tuple[MetricsAccumulator, float]:
    """
    Forward pass over entire dataset.
    
    If optimizer is provided, performs training (with gradients).
    Otherwise, performs evaluation (no gradients).
    
    Args:
        model: Model to train/evaluate
        data_loader: DataLoader
        criterion: Loss function
        device: Device to use
        optimizer: Optimizer (None for eval mode)
        desc: Description for progress bar
    
    Returns:
        metrics_accumulator, average_loss
    """
    is_training = optimizer is not None
    model.train() if is_training else model.eval()
    
    accumulator = MetricsAccumulator()
    total_loss = 0.0
    n_samples = 0
    
    context = torch.enable_grad() if is_training else torch.no_grad()
    
    with context:
        pbar = tqdm(data_loader, desc=desc, leave=False)
        for batch in pbar:
            # Move to device
            images = batch['image'].to(device)
            is_ellipse = batch['is_ellipse'].to(device)
            geometry = batch['geometry'].to(device)
            ellipse_mask = batch['ellipse_mask'].to(device)
            
            # Forward
            pred = model(images)
            
            # Loss
            loss, loss_components = criterion(
                pred, is_ellipse, geometry, ellipse_mask, model
            )
            
            # Backward (training only)
            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Accumulate
            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            n_samples += batch_size
            
            # Update accumulator
            # pred format: [logit, cx, cy, a, b, cos(2θ), sin(2θ)]
            # geometry format: [cx, cy, a, b, cos(2θ), sin(2θ)]
            accumulator.update(
                logits=pred[:, 0],
                is_ellipse=is_ellipse,
                pred_geom=pred[:, 1:7],  # cx, cy, a, b, cos, sin
                target_geom=geometry,  # cx, cy, a, b, cos, sin
                ellipse_mask=ellipse_mask,
                loss_components=loss_components,
            )
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'cls': f'{loss_components["cls_loss"]:.4f}',
                'ctr': f'{loss_components["center_loss"]:.4f}',
                'rad': f'{loss_components["radii_loss"]:.4f}',
                'ang': f'{loss_components["angle_loss"]:.4f}',
            })
    
    avg_loss = total_loss / max(n_samples, 1)
    return accumulator, avg_loss


def initial_best_result(max_opt: bool) -> float:
    """Get initial best value based on optimization direction."""
    return float('-inf') if max_opt else float('inf')


def is_improvement(current: float, best: float, max_opt: bool) -> bool:
    """Check if current value is an improvement over best."""
    if max_opt:
        return current > best
    return current < best


def update_epoch_result(val_loss: float, val_metric: float, max_opt: bool) -> float:
    """Get the value to monitor based on optimization direction."""
    return val_metric if max_opt else val_loss


def checkpoint_model(model: nn.Module, path: Path):
    """Save model checkpoint to CPU."""
    # Create CPU copy
    model_cpu = copy.deepcopy(model).cpu()
    
    # Save state dict and config
    checkpoint = {
        'model_state_dict': model_cpu.state_dict(),
        'config': getattr(model, 'config', {}),
    }
    
    torch.save(checkpoint, path)
    logger.info(f"Saved checkpoint to {path}")


def load_model(model: nn.Module, path: Path, device: torch.device) -> nn.Module:
    """Load model checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model.to(device)


def scheduler_step(
    scheduler: Optional[Any],
    metric: float,
):
    """Step the scheduler if it exists."""
    if scheduler is None:
        return
    # Epoch-based schedulers (e.g. cosine) ignore validation metrics.
    if scheduler.__class__.__name__ == 'ReduceLROnPlateau':
        scheduler.step(metric)
    else:
        scheduler.step()


def print_epoch_results(
    epoch: int,
    n_epochs: int,
    train_metrics: Dict,
    val_metrics: Dict,
    lr: float,
):
    """Print epoch results summary."""
    logger.info(
        f"Epoch {epoch+1}/{n_epochs} | "
        f"LR: {lr:.6f} | "
        f"Train Loss: {train_metrics.get('avg_total_loss', 0):.4f} | "
        f"Val Loss: {val_metrics.get('avg_total_loss', 0):.4f} | "
        f"Train Acc: {train_metrics.get('accuracy', 0):.4f} | "
        f"Val Acc: {val_metrics.get('accuracy', 0):.4f} | "
        f"Val F1: {val_metrics.get('f1', 0):.4f} | "
        f"Val AUC: {val_metrics.get('roc_auc', 0):.4f}"
    )


def write_training_report(
    results: List[Dict],
    path: Path,
) -> pd.DataFrame:
    """Write training results to CSV."""
    df = pd.DataFrame(results)
    df.to_csv(path, index=False)
    return df


def train_loop(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: MultiTaskLoss,
    num_epochs: int = 50,
    device: Optional[torch.device] = None,
    scheduler: Optional[Any] = None,
    output_dir: str | Path = './runs/default',
    max_opt: bool = False,
    fbeta: float = 2.0,
) -> Tuple[nn.Module, pd.DataFrame, Dict]:
    """
    Main training loop.
    
    Args:
        model: Model to train
        optimizer: Optimizer
        train_loader: Training dataloader
        val_loader: Validation dataloader
        criterion: Loss function
        num_epochs: Number of epochs
        device: Device (auto-detected if None)
        scheduler: Learning rate scheduler
        output_dir: Directory for outputs
        max_opt: If True, maximize val metric; else minimize val loss
        fbeta: Beta for F-beta threshold optimization
    
    Returns:
        best_model, results_df, metadata
    """
    # Setup
    device = device or get_device()
    model = model.to(device)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Training on device: {device}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Optimization direction: {'maximize' if max_opt else 'minimize'}")
    
    # Paths
    best_path = output_dir / 'best.pt'
    last_path = output_dir / 'last.pt'
    metrics_path = output_dir / 'metrics.csv'
    threshold_path = output_dir / 'best_threshold.json'
    
    # Initialize tracking
    results_list = []
    patience = 0
    max_patience = max(num_epochs // 4, 5)
    best_score = initial_best_result(max_opt)
    best_threshold = 0.5
    
    # Store all val predictions for final threshold sweep
    all_val_probs = []
    all_val_targets = []
    
    for epoch in range(num_epochs):
        # Training phase
        train_acc, train_loss = forward_all_dataset(
            model, train_loader, criterion, device,
            optimizer=optimizer,
            desc=f'Epoch {epoch+1}/{num_epochs} [Train]',
        )
        train_metrics = train_acc.compute()
        
        # Validation phase
        val_acc, val_loss = forward_all_dataset(
            model, val_loader, criterion, device,
            optimizer=None,
            desc=f'Epoch {epoch+1}/{num_epochs} [Val]',
        )
        val_metrics = val_acc.compute()
        
        # Store val predictions for final threshold sweep
        all_val_probs = np.concatenate(val_acc.probs)
        all_val_targets = np.concatenate(val_acc.targets)
        
        # Determine monitored value
        if max_opt:
            current_score = val_metrics.get('pr_auc', 0)
        else:
            current_score = val_metrics.get('avg_total_loss', val_loss)
        
        # Log current LR
        current_lr = get_lr(optimizer)
        
        # Print epoch summary
        print_epoch_results(epoch, num_epochs, train_metrics, val_metrics, current_lr)
        
        # Accumulate results
        epoch_result = {
            'epoch': epoch + 1,
            'lr': current_lr,
            **{f'train_{k}': v for k, v in train_metrics.items()},
            **{f'val_{k}': v for k, v in val_metrics.items()},
        }
        results_list.append(epoch_result)
        
        # Write report (every epoch)
        write_training_report(results_list, metrics_path)
        
        # Scheduler step
        scheduler_step(scheduler, val_metrics.get('avg_total_loss', val_loss))
        
        # Check for improvement
        if is_improvement(current_score, best_score, max_opt):
            logger.info(f"Improvement: {best_score:.4f} -> {current_score:.4f}")
            best_score = current_score
            patience = 0
            
            # Save best checkpoint
            checkpoint_model(model, best_path)
            
            # Update best threshold
            thresh, fbeta_score = val_acc.find_best_threshold(beta=fbeta)
            best_threshold = thresh
            logger.info(f"Best F-beta threshold: {best_threshold:.4f} (F{fbeta}={fbeta_score:.4f})")
        else:
            patience += 1
            logger.info(f"No improvement. Patience: {patience}/{max_patience}")
        
        # Save last checkpoint
        checkpoint_model(model, last_path)
        
        # Early stopping
        if patience > max_patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Final threshold sweep on all validation data
    from .metrics import find_best_fbeta_threshold
    best_threshold, best_fbeta = find_best_fbeta_threshold(
        all_val_probs, all_val_targets, beta=fbeta
    )
    logger.info(f"Final best threshold: {best_threshold:.4f} (F{fbeta}={best_fbeta:.4f})")
    
    # Save threshold
    with open(threshold_path, 'w') as f:
        json.dump({
            'threshold': best_threshold,
            'fbeta': fbeta,
            'fbeta_score': best_fbeta,
        }, f, indent=2)
    
    # Load best model
    model = load_model(model, best_path, device)
    
    # Highest-error examples: 4 rows (angle, center, axis-x, axis-y)
    error_grid_path = output_dir / 'highest_errors.png'
    try:
        generate_highest_error_grid(
            model=model,
            data_loader=val_loader,
            device=device,
            save_path=error_grid_path,
            title='Highest per-component errors (validation ellipses)',
        )
        logger.info(f"Highest-error grid saved to {error_grid_path}")
    except Exception as exc:
        logger.warning(f"Could not generate highest-error grid: {exc}")
    
    # Final results
    results_df = pd.DataFrame(results_list)
    
    # Metadata
    metadata = {
        'best_threshold': best_threshold,
        'best_fbeta': best_fbeta,
        'fbeta': fbeta,
        'best_epoch': int(results_df['epoch'][results_df[f'val_avg_total_loss'].idxmin()]) if not max_opt else int(results_df['epoch'][results_df['val_pr_auc'].idxmax()]),
        'final_epoch': epoch + 1,
    }
    
    logger.info(f"Training complete. Best model saved to {best_path}")
    
    return model, results_df, metadata


if __name__ == '__main__':
    # Quick test with dummy data
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    
    # Create dummy model
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(50*50, 7)
            self.config = {}
        
        def forward(self, x):
            x = x.view(x.size(0), -1)
            return self.fc(x)
    
    model = DummyModel()
    
    # Create dummy data
    n_train = 100
    n_val = 20
    
    class DummyDataset:
        def __init__(self, n):
            self.n = n
        def __len__(self):
            return self.n
        def __getitem__(self, idx):
            return {
                'image': torch.randn(1, 50, 50),
                'is_ellipse': torch.tensor(float(idx % 2)),
                'geometry': torch.rand(6),
                'ellipse_mask': torch.tensor(float(idx % 2)),
                'path': f'dummy_{idx}.jpg',
            }
    
    train_loader = DataLoader(DummyDataset(n_train), batch_size=8)
    val_loader = DataLoader(DummyDataset(n_val), batch_size=8)
    
    # Create loss
    criterion = MultiTaskLoss(cls_loss='bce', reg_loss='smooth_l1', pos_rate=0.5)
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Train for a few epochs
    print("Testing training loop...")
    model, results_df, metadata = train_loop(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        num_epochs=3,
        output_dir='./runs/test',
    )
    
    print(f"\nResults shape: {results_df.shape}")
    print(f"Metadata: {metadata}")
    print("\nTraining loop test passed!")
