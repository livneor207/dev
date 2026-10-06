"""
Visualization utilities for ellipse detection results.

Generates example images with predicted and ground-truth ellipses overlaid.

Geometry format: [cx, cy, a, b, cos(2θ), sin(2θ)]
  - For display, angle = atan2(sin, cos) / 2
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import torch
from torch.utils.data import DataLoader


def geom6_to_pixel(geom: np.ndarray, img_size: int = 50) -> np.ndarray:
    """Convert [..., 6] (cx, cy, a, b, cos2θ, sin2θ) to [..., 5] pixel/degree."""
    geom = np.asarray(geom, dtype=np.float64)
    squeeze = geom.ndim == 1
    if squeeze:
        geom = geom[np.newaxis, :]
    out = np.zeros((geom.shape[0], 5), dtype=np.float64)
    out[:, 0] = geom[:, 0] * img_size
    out[:, 1] = geom[:, 1] * img_size
    out[:, 3] = geom[:, 2] * img_size
    out[:, 4] = geom[:, 3] * img_size
    out[:, 2] = [
        decode_angle_from_trig(c, s) for c, s in zip(geom[:, 4], geom[:, 5])
    ]
    return out[0] if squeeze else out


def circular_angle_error_deg(pred_deg: np.ndarray, true_deg: np.ndarray) -> np.ndarray:
    """Smallest angle difference on [0, 180)."""
    diff = np.abs(pred_deg - true_deg)
    return np.minimum(diff, 180.0 - diff)


def denormalize_image(
    img: np.ndarray,
    normalize_mean: Optional[float | List[float]] = None,
    normalize_std: Optional[float | List[float]] = None,
) -> np.ndarray:
    """CHW tensor image -> display HWC/HW in [0, 1]."""
    img = np.asarray(img, dtype=np.float32)
    if normalize_mean is not None and normalize_std is not None:
        mean = np.asarray(normalize_mean, dtype=np.float32)
        std = np.asarray(normalize_std, dtype=np.float32)
        if mean.ndim == 1:
            mean = mean[:, np.newaxis, np.newaxis]
            std = std[:, np.newaxis, np.newaxis]
        img = img * std + mean
    img = np.clip(img, 0, 1)
    if img.ndim == 3:
        if img.shape[0] == 1:
            return img[0]
        if img.shape[0] >= 3:
            # rgb_edge and other multi-channel inputs: show RGB for overlays
            return img[:3].transpose(1, 2, 0)
    return img


def decode_angle_from_trig(cos_2theta: float, sin_2theta: float) -> float:
    """
    Decode angle from cos(2θ), sin(2θ) encoding.
    
    Args:
        cos_2theta: cos(2θ) value
        sin_2theta: sin(2θ) value
    
    Returns:
        Angle in degrees in [0, 180) range
    """
    # theta = 0.5 * atan2(sin2theta, cos2theta), then wrap to [0, π)
    theta_rad = 0.5 * np.arctan2(sin_2theta, cos_2theta)
    theta_rad = theta_rad % np.pi  # Ensure [0, π)
    return np.degrees(theta_rad)  # [0, 180)


def draw_ellipse_on_ax(
    ax,
    cx: float, cy: float, 
    angle: float, 
    axis_1: float, axis_2: float,
    color: str = 'green',
    linestyle: str = '-',
    linewidth: float = 2,
    label: str = '',
):
    """
    Draw ellipse on matplotlib axis.
    
    Args:
        ax: Matplotlib axis
        cx, cy: Center coordinates (pixel space)
        angle: Rotation angle in degrees
        axis_1, axis_2: Semi-axes lengths
        color: Ellipse color
        linestyle: Line style
        linewidth: Line width
        label: Label for legend
    """
    # matplotlib Ellipse uses width/height (full axes, not semi-axes)
    # and angle is counter-clockwise from horizontal
    ellipse = mpatches.Ellipse(
        (cx, cy),
        width=2 * axis_1,
        height=2 * axis_2,
        angle=angle,
        fill=False,
        edgecolor=color,
        linestyle=linestyle,
        linewidth=linewidth,
        label=label,
    )
    ax.add_patch(ellipse)


def visualize_predictions(
    images: List[np.ndarray],
    pred_labels: List[bool],
    pred_geom: List[Tuple[float, ...]],
    true_labels: Optional[List[bool]] = None,
    true_geom: Optional[List[Tuple[float, ...]]] = None,
    probs: Optional[List[float]] = None,
    paths: Optional[List[str]] = None,
    title: str = 'Predictions',
    save_path: Optional[str | Path] = None,
    max_examples: int = 12,
    ncols: int = 4,
    figsize_per_img: float = 3,
) -> plt.Figure:
    """
    Visualize predictions with ellipses overlaid.
    
    Args:
        images: List of images (H, W) or (H, W, C)
        pred_labels: Predicted is_ellipse labels
        pred_geom: Predicted geometry [(cx, cy, angle, a1, a2), ...]
        true_labels: Ground truth labels (optional)
        true_geom: Ground truth geometry (optional)
        probs: Prediction probabilities (optional)
        paths: Image paths for titles (optional)
        title: Overall figure title
        save_path: Path to save figure
        max_examples: Maximum number of examples to show
        ncols: Number of columns
        figsize_per_img: Size per image
    
    Returns:
        Matplotlib figure
    """
    n = min(len(images), max_examples)
    nrows = (n + ncols - 1) // ncols
    
    fig, axes = plt.subplots(
        nrows, ncols, 
        figsize=(ncols * figsize_per_img, nrows * figsize_per_img + 0.5)
    )
    fig.suptitle(title, fontsize=14)
    
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)
    
    for idx in range(n):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        
        # Show image
        img = images[idx]
        if img.ndim == 2:
            ax.imshow(img, cmap='gray')
        else:
            ax.imshow(img)
        
        # Draw ground truth ellipse (if available)
        if true_labels is not None and true_geom is not None:
            if true_labels[idx]:
                cx, cy, angle, a1, a2 = true_geom[idx]
                draw_ellipse_on_ax(
                    ax, cx, cy, angle, a1, a2,
                    color='green', linestyle='-', linewidth=2,
                    label='Ground Truth'
                )
        
        # Draw predicted ellipse
        if pred_labels[idx]:
            cx, cy, angle, a1, a2 = pred_geom[idx]
            draw_ellipse_on_ax(
                ax, cx, cy, angle, a1, a2,
                color='red', linestyle='--', linewidth=2,
                label='Prediction'
            )
        
        # Title
        title_parts = []
        if paths is not None:
            title_parts.append(Path(paths[idx]).name)
        
        pred_str = "Ellipse" if pred_labels[idx] else "No Ellipse"
        if probs is not None:
            pred_str += f" ({probs[idx]:.2f})"
        title_parts.append(f"Pred: {pred_str}")
        
        if true_labels is not None:
            true_str = "Ellipse" if true_labels[idx] else "No Ellipse"
            title_parts.append(f"True: {true_str}")
            
            # Mark correct/incorrect
            correct = pred_labels[idx] == true_labels[idx]
            color = 'green' if correct else 'red'
            ax.set_title('\n'.join(title_parts), fontsize=8, color=color)
        else:
            ax.set_title('\n'.join(title_parts), fontsize=8)
        
        ax.axis('off')
    
    # Hide unused axes
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].axis('off')
    
    # Legend
    handles = [
        mpatches.Patch(color='green', label='Ground Truth'),
        mpatches.Patch(color='red', label='Prediction'),
    ]
    fig.legend(handles=handles, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    return fig


def generate_example_predictions(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
    normalize_mean: Optional[float | List[float]] = None,
    normalize_std: Optional[float | List[float]] = None,
    max_examples: int = 12,
    title: str = 'Predictions',
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Generate example predictions from a data loader.
    
    Args:
        model: Trained model
        data_loader: DataLoader
        device: Device
        threshold: Classification threshold
        normalize_mean: Mean used for normalization (to denormalize for display)
        normalize_std: Std used for normalization
        max_examples: Maximum examples to show
        title: Figure title
        save_path: Path to save figure
    
    Returns:
        Matplotlib figure
    """
    model.eval()
    
    images = []
    pred_labels = []
    pred_geom = []
    true_labels = []
    true_geom = []
    probs_list = []
    paths = []
    
    with torch.no_grad():
        for batch in data_loader:
            batch_images = batch['image'].to(device)
            batch_paths = batch['path']
            
            # Forward
            out = model(batch_images)
            batch_probs = torch.sigmoid(out[:, 0]).cpu().numpy()
            batch_preds = batch_probs >= threshold
            
            # Denormalize geometry predictions
            # Model output: [logit, cx, cy, a, b, cos(2θ), sin(2θ)]
            geom_norm = out[:, 1:].cpu().numpy()  # [B, 6]
            # Convert to [cx, cy, angle, a, b] for display
            geom_denorm = np.zeros((len(geom_norm), 5))
            geom_denorm[:, 0] = geom_norm[:, 0] * 50  # cx
            geom_denorm[:, 1] = geom_norm[:, 1] * 50  # cy
            geom_denorm[:, 3] = geom_norm[:, 2] * 50  # a (axis_1)
            geom_denorm[:, 4] = geom_norm[:, 3] * 50  # b (axis_2)
            # Decode angle from cos/sin
            for j in range(len(geom_norm)):
                geom_denorm[j, 2] = decode_angle_from_trig(geom_norm[j, 4], geom_norm[j, 5])
            
            # Process images for display
            batch_img_np = batch_images.cpu().numpy()
            
            # Get ground truth if available
            has_labels = batch['ellipse_mask'].sum() > 0 or (1 - batch['is_ellipse']).sum() > 0
            
            for i in range(len(batch_images)):
                if len(images) >= max_examples:
                    break
                
                img = denormalize_image(
                    batch_img_np[i], normalize_mean, normalize_std
                )
                
                images.append(img)
                pred_labels.append(bool(batch_preds[i]))
                pred_geom.append(tuple(geom_denorm[i]))
                probs_list.append(float(batch_probs[i]))
                paths.append(batch_paths[i])
                
                # Ground truth
                if has_labels:
                    true_labels.append(bool(batch['is_ellipse'][i].item()))
                    # Denormalize ground truth geometry
                    # Format: [cx, cy, a, b, cos(2θ), sin(2θ)]
                    gt = batch['geometry'][i].numpy()
                    gt_denorm = np.array([
                        gt[0] * 50,  # cx
                        gt[1] * 50,  # cy
                        decode_angle_from_trig(gt[4], gt[5]),  # angle from cos/sin
                        gt[2] * 50,  # a (axis_1)
                        gt[3] * 50,  # b (axis_2)
                    ])
                    true_geom.append(tuple(gt_denorm))
            
            if len(images) >= max_examples:
                break
    
    # Generate visualization
    return visualize_predictions(
        images=images,
        pred_labels=pred_labels,
        pred_geom=pred_geom,
        true_labels=true_labels if true_labels else None,
        true_geom=true_geom if true_geom else None,
        probs=probs_list,
        paths=paths,
        title=title,
        save_path=save_path,
        max_examples=max_examples,
    )


ERROR_ROWS = (
    ('angle_err', 'Row 1 — highest angle error', 'Δθ={:.1f}°'),
    ('center_err', 'Row 2 — highest center error', 'Δc={:.1f}px'),
    ('axis_x_err', 'Row 3 — highest axis-x error', 'Δa={:.1f}px'),
    ('axis_y_err', 'Row 4 — highest axis-y error', 'Δb={:.1f}px'),
)


def collect_ellipse_error_records(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    normalize_mean: Optional[float | List[float]] = None,
    normalize_std: Optional[float | List[float]] = None,
    img_size: int = 50,
) -> List[Dict[str, Any]]:
    """Collect per-sample geometry errors for true ellipses (y=1)."""
    model.eval()
    records: List[Dict[str, Any]] = []

    with torch.no_grad():
        for batch in data_loader:
            images = batch['image'].to(device)
            out = model(images)
            pred_px = geom6_to_pixel(out[:, 1:7].cpu().numpy(), img_size)
            true_px = geom6_to_pixel(batch['geometry'].cpu().numpy(), img_size)
            mask = batch['ellipse_mask'].cpu().numpy().astype(bool)
            probs = torch.sigmoid(out[:, 0]).cpu().numpy()
            img_np = batch['image'].cpu().numpy()
            paths = batch['path']

            angle_err = circular_angle_error_deg(pred_px[:, 2], true_px[:, 2])
            center_err = np.hypot(pred_px[:, 0] - true_px[:, 0], pred_px[:, 1] - true_px[:, 1])
            axis_x_err = np.abs(pred_px[:, 3] - true_px[:, 3])
            axis_y_err = np.abs(pred_px[:, 4] - true_px[:, 4])

            for i in range(len(images)):
                if not mask[i]:
                    continue
                records.append({
                    'image': denormalize_image(img_np[i], normalize_mean, normalize_std),
                    'path': paths[i],
                    'prob': float(probs[i]),
                    'pred': pred_px[i],
                    'true': true_px[i],
                    'angle_err': float(angle_err[i]),
                    'center_err': float(center_err[i]),
                    'axis_x_err': float(axis_x_err[i]),
                    'axis_y_err': float(axis_y_err[i]),
                })
    return records


def generate_highest_error_grid(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    normalize_mean: Optional[float | List[float]] = None,
    normalize_std: Optional[float | List[float]] = None,
    n_cols: int = 5,
    img_size: int = 50,
    title: str = 'Highest per-component errors (validation ellipses)',
    save_path: Optional[str | Path] = None,
) -> Optional[plt.Figure]:
    """
    4-row grid of the worst validation ellipses:

      row 1: highest angle error
      row 2: highest center error
      row 3: highest axis-x (a) error
      row 4: highest axis-y (b) error
    """
    dataset = getattr(data_loader, 'dataset', None)
    if normalize_mean is None and dataset is not None:
        normalize_mean = getattr(dataset, 'normalize_mean', None)
    if normalize_std is None and dataset is not None:
        normalize_std = getattr(dataset, 'normalize_std', None)
    if dataset is not None:
        img_size = int(getattr(dataset, 'IMG_SIZE', img_size))

    records = collect_ellipse_error_records(
        model, data_loader, device,
        normalize_mean=normalize_mean,
        normalize_std=normalize_std,
        img_size=img_size,
    )
    if not records:
        print("No ellipse samples available for highest-error grid.")
        return None

    n_cols = min(n_cols, len(records))
    fig, axes = plt.subplots(
        4, n_cols,
        figsize=(n_cols * 3.0, 13.0),
    )
    if n_cols == 1:
        axes = axes.reshape(4, 1)
    fig.suptitle(title, fontsize=14)

    summary: Dict[str, List[Dict[str, float | str]]] = {}

    for row, (key, row_title, fmt) in enumerate(ERROR_ROWS):
        ranked = sorted(records, key=lambda r: r[key], reverse=True)
        picked = ranked[:n_cols]
        summary[key] = [
            {
                'path': rec['path'],
                'error': rec[key],
                'prob': rec['prob'],
            }
            for rec in picked
        ]
        if picked:
            print(f"  {row_title}: {Path(picked[0]['path']).name} "
                  f"({fmt.format(picked[0][key])})")

        for col in range(n_cols):
            ax = axes[row, col]
            ax.axis('off')
            if col >= len(picked):
                continue
            rec = picked[col]
            img = rec['image']
            if img.ndim == 2:
                ax.imshow(img, cmap='gray')
            else:
                ax.imshow(img)
            tcx, tcy, tangle, ta, tb = rec['true']
            pcx, pcy, pangle, pa, pb = rec['pred']
            draw_ellipse_on_ax(ax, tcx, tcy, tangle, ta, tb, color='green', linestyle='-', linewidth=2)
            draw_ellipse_on_ax(ax, pcx, pcy, pangle, pa, pb, color='red', linestyle='--', linewidth=2)
            name = Path(rec['path']).name
            ax.set_title(f"{name}\n{fmt.format(rec[key])}", fontsize=8)
            if col == 0:
                ax.set_ylabel(row_title, fontsize=9)
                ax.axis('on')
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)

    handles = [
        mpatches.Patch(color='green', label='Ground truth'),
        mpatches.Patch(color='red', label='Prediction'),
    ]
    fig.legend(handles=handles, loc='upper right', fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        json_path = save_path.with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Saved highest-error grid to {save_path}")
        print(f"Saved highest-error list to {json_path}")
        plt.close(fig)

    return fig


def visualize_dataset_samples(
    data_loader: DataLoader,
    normalize_mean: Optional[float | List[float]] = None,
    normalize_std: Optional[float | List[float]] = None,
    max_examples: int = 12,
    title: str = 'Dataset Samples',
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Visualize samples from a dataset with ground truth ellipses.
    
    Args:
        data_loader: DataLoader
        normalize_mean: Mean for denormalization
        normalize_std: Std for denormalization
        max_examples: Maximum examples
        title: Figure title
        save_path: Path to save
    
    Returns:
        Matplotlib figure
    """
    images = []
    labels = []
    geom_list = []
    paths = []
    
    for batch in data_loader:
        batch_img_np = batch['image'].numpy()
        
        for i in range(len(batch['image'])):
            if len(images) >= max_examples:
                break
            
            img = denormalize_image(
                batch_img_np[i], normalize_mean, normalize_std
            )
            
            images.append(img)
            labels.append(bool(batch['is_ellipse'][i].item()))
            
            # Denormalize geometry
            # Format: [cx, cy, a, b, cos(2θ), sin(2θ)]
            gt = batch['geometry'][i].numpy()
            gt_denorm = np.array([
                gt[0] * 50,  # cx
                gt[1] * 50,  # cy
                decode_angle_from_trig(gt[4], gt[5]),  # angle from cos/sin
                gt[2] * 50,  # a
                gt[3] * 50,  # b
            ])
            geom_list.append(tuple(gt_denorm))
            paths.append(batch['path'][i])
        
        if len(images) >= max_examples:
            break
    
    return visualize_predictions(
        images=images,
        pred_labels=labels,  # Use true labels as "predictions" for display
        pred_geom=geom_list,
        true_labels=None,
        true_geom=None,
        probs=None,
        paths=paths,
        title=title,
        save_path=save_path,
        max_examples=max_examples,
    )


if __name__ == '__main__':
    # Test with synthetic data
    import tempfile
    
    # Create dummy images and predictions
    np.random.seed(42)
    n = 8
    images = [np.random.rand(50, 50) * 0.5 + 0.5 for _ in range(n)]
    pred_labels = [True, True, False, True, True, False, True, True]
    pred_geom = [(25, 25, 45, 10, 15) for _ in range(n)]
    true_labels = [True, True, True, True, False, False, True, True]
    true_geom = [(25, 25, 40, 12, 14) for _ in range(n)]
    probs = [0.9, 0.8, 0.3, 0.95, 0.85, 0.2, 0.7, 0.6]
    paths = [f'image_{i}.jpg' for i in range(n)]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        fig = visualize_predictions(
            images=images,
            pred_labels=pred_labels,
            pred_geom=pred_geom,
            true_labels=true_labels,
            true_geom=true_geom,
            probs=probs,
            paths=paths,
            title='Test Visualization',
            save_path=f'{tmpdir}/test_viz.png',
        )
        plt.close(fig)
        print("Visualization test passed!")

        class _Tiny(torch.nn.Module):
            def forward(self, x):
                b = x.size(0)
                out = torch.zeros(b, 7)
                out[:, 0] = 2.0
                out[:, 1:5] = 0.5
                out[:, 5] = 1.0
                return out

        class _Ds:
            def __init__(self, n=12):
                self.n = n
                self.normalize_mean = 0.0
                self.normalize_std = 1.0
                self.IMG_SIZE = 50
            def __len__(self):
                return self.n
            def __getitem__(self, idx):
                y = 1.0 if idx % 2 == 0 else 0.0
                geom = torch.tensor([0.4, 0.4, 0.2, 0.3, 1.0, 0.0])
                geom = geom + 0.02 * idx
                return {
                    'image': torch.rand(1, 50, 50),
                    'is_ellipse': torch.tensor(y),
                    'geometry': geom.float(),
                    'ellipse_mask': torch.tensor(y),
                    'path': f'img_{idx}.jpg',
                }

        loader = DataLoader(_Ds(), batch_size=4)
        grid = generate_highest_error_grid(
            _Tiny(), loader, device=torch.device('cpu'),
            n_cols=3, save_path=f'{tmpdir}/highest_errors.png',
        )
        assert Path(f'{tmpdir}/highest_errors.png').exists()
        assert Path(f'{tmpdir}/highest_errors.json').exists()
        print("Highest-error grid test passed!")
