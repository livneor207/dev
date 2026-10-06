#!/usr/bin/env python3
"""
Training script for the 7-output ellipse detector.

Output: [ellipse_logit, cx, cy, a, b, u, v]
Loss:   L = λ1 BCE + y (λ2 SmoothL1_center + λ3 SmoothL1_radii + λ4 (1 - v̂·v))

Usage:
    python scripts/train.py --csv train_data.csv --epochs 50 --backbone cnn


See --help for all options.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use('Agg')
import torch

from ellipse_cv.data import create_dataloaders, input_channel_count
from ellipse_cv.model import build_model, print_model_summary
from ellipse_cv.losses import build_loss
from ellipse_cv.optim import build_optimizer, build_scheduler
from ellipse_cv.trainer import train_loop, get_device
from ellipse_cv.plots import plot_training_curves
from ellipse_cv.visualize import generate_example_predictions
from ellipse_cv.baseline import (
    load_best_baseline,
    print_baseline,
    maybe_update_best_baseline,
    metrics_from_training,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Train ellipse detector')
    
    # Data
    parser.add_argument('--csv', type=str, default='train_data.csv',
                        help='Path to training CSV')
    parser.add_argument('--base-dir', type=str, default='.',
                        help='Base directory for images')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='Number of data loader workers')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Validation split ratio')
    
    # Model
    parser.add_argument('--backbone', type=str, default='cnn',
                        choices=['cnn', 'mlp'],
                        help='Backbone architecture')
    parser.add_argument('--grayscale', action='store_true', default=True,
                        help='Use grayscale input (default)')
    parser.add_argument('--rgb', action='store_true',
                        help='Use RGB input instead of grayscale')
    parser.add_argument('--color-space', type=str, default='rgb',
                        choices=['rgb', 'rgb_edge', 'lab', 'hsv'],
                        help='Input color space when --rgb is selected')
    parser.add_argument('--depth', type=int, default=3,
                        help='Backbone depth')
    parser.add_argument('--base-filters', type=int, default=32,
                        help='Base filters for CNN')
    parser.add_argument('--kernel-size', type=int, default=None,
                        choices=[3, 5, 7],
                        help='Use this kernel size in every CNN block')
    parser.add_argument('--hidden-ratio', type=float, default=0.5,
                        help='Hidden ratio for MLP')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate')
    parser.add_argument('--head-hidden-dim', type=int, default=128,
                        help='Hidden dimension in heads')
    parser.add_argument('--auto-tune', action='store_true',
                        help='Enable Kendall uncertainty weighting')
    parser.add_argument('--fine-grained', action='store_true',
                        help='Use fine-grained CNN (smaller filters, less pooling for better angle)')
    parser.add_argument('--activation', type=str, default='relu',
                        choices=['relu', 'mish'],
                        help='Activation function in backbone and heads')
    
    # Loss
    parser.add_argument('--cls-loss', type=str, default='bce',
                        choices=['bce', 'focal'],
                        help='Classification loss (default: BCEWithLogitsLoss)')
    parser.add_argument('--reg-loss', type=str, default='smooth_l1',
                        choices=['smooth_l1', 'huber', 'mse', 'mae'],
                        help='Center/radii regression loss (default: SmoothL1)')
    parser.add_argument('--focal-gamma', type=float, default=2.0,
                        help='Focal loss gamma')
    parser.add_argument('--smooth-l1-beta', type=float, default=1.0,
                        help='SmoothL1 beta (Huber delta if --reg-loss huber)')
    parser.add_argument('--lambda-cls', type=float, default=1.0,
                        help='λ1 weight for classification loss')
    parser.add_argument('--lambda-center', type=float, default=1.0,
                        help='λ2 weight for center SmoothL1')
    parser.add_argument('--lambda-radii', type=float, default=1.0,
                        help='λ3 weight for radii SmoothL1')
    parser.add_argument('--lambda-angle', type=float, default=1.0,
                        help='λ4 weight for angle loss 1 - v̂·v')
    parser.add_argument('--angle-huber-weight', type=float, default=0.0,
                        help='Additive Huber-loss weight for the angle vector')
    
    # Optimizer
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['adamw', 'lion', 'sgd'],
                        help='Optimizer')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                        help='Weight decay')
    
    # Scheduler
    parser.add_argument('--scheduler', type=str, default='plateau',
                        choices=['plateau', 'cosine'],
                        help='LR scheduler: ReduceLROnPlateau or CosineAnnealingLR')
    parser.add_argument('--scheduler-patience', type=int, default=5,
                        help='Scheduler patience')
    parser.add_argument('--scheduler-factor', type=float, default=0.5,
                        help='Scheduler factor')
    parser.add_argument('--min-lr', type=float, default=1e-6,
                        help='Minimum learning rate')
    
    # Training
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--max-opt', action='store_true',
                        help='Maximize metric instead of minimize loss')
    parser.add_argument('--fbeta', type=float, default=2.0,
                        help='Beta for F-beta threshold')
    
    # Augmentation
    parser.add_argument('--blur-prob', type=float, default=0.0,
                        help='Probability of blur augmentation')
    parser.add_argument('--blur-radius', type=float, default=1.0,
                        help='Blur radius')
    parser.add_argument('--brightness-jitter', type=float, default=0.0,
                        help='Maximum relative RGB brightness jitter')
    parser.add_argument('--contrast-jitter', type=float, default=0.0,
                        help='Maximum relative RGB contrast jitter')
    parser.add_argument('--saturation-jitter', type=float, default=0.0,
                        help='Maximum relative RGB saturation jitter')
    parser.add_argument('--grayscale-aug-prob', type=float, default=0.0,
                        help='Probability of RGB-to-grayscale photometric augmentation')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='./runs/default',
                        help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    
    # Determine grayscale
    grayscale = not args.rgb
    
    print("=" * 60)
    print("ELLIPSE DETECTOR TRAINING")
    print("=" * 60)
    print(f"Backbone: {args.backbone}")
    print(f"Input: {'Grayscale' if grayscale else 'RGB'}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Optimizer: {args.optimizer}, LR: {args.lr}")
    print(f"Losses: L = λ1 L_cls + y (λ2 L_center + λ3 L_radii + λ4 L_angle)")
    cls_name = 'BCEWithLogits' if args.cls_loss == 'bce' else 'Focal'
    print(
        f"  L_cls={cls_name}, L_center/L_radii={args.reg_loss}, "
        f"L_angle=(1-v̂·v)+{args.angle_huber_weight}*Huber(v̂,v)"
    )
    print(f"  λ = [{args.lambda_cls}, {args.lambda_center}, {args.lambda_radii}, {args.lambda_angle}]")
    print("=" * 60)
    
    print_baseline(load_best_baseline(args.output_dir))
    if Path(args.output_dir).name == 'best_baseline':
        raise SystemExit(
            "Refusing to train into runs/best_baseline/. "
            "Use --output-dir runs/<experiment_name> so the winner is not overwritten."
        )
    
    # Create dataloaders
    print("\nLoading data...")
    train_loader, val_loader, data_meta = create_dataloaders(
        csv_path=args.csv,
        base_dir=args.base_dir,
        batch_size=args.batch_size,
        grayscale=grayscale,
        color_space=args.color_space,
        blur_prob=args.blur_prob,
        blur_radius=args.blur_radius,
        brightness_jitter=args.brightness_jitter,
        contrast_jitter=args.contrast_jitter,
        saturation_jitter=args.saturation_jitter,
        grayscale_aug_prob=args.grayscale_aug_prob,
        num_workers=args.num_workers,
        test_size=args.test_size,
        random_state=args.seed,
    )
    
    # Build model
    print("\nBuilding model...")
    in_channels = input_channel_count(grayscale, args.color_space)
    model = build_model(
        backbone=args.backbone,
        grayscale=grayscale,
        in_channels=in_channels,
        base_filters=args.base_filters,
        kernel_size=args.kernel_size,
        hidden_ratio=args.hidden_ratio,
        depth=args.depth,
        dropout_rate=args.dropout,
        head_hidden_dim=args.head_hidden_dim,
        auto_tune=args.auto_tune,
        fine_grained=args.fine_grained,
        activation=args.activation,
    )
    
    # Print model summary
    print_model_summary(model, input_size=(1, in_channels, 50, 50))
    
    # Build loss
    criterion = build_loss(
        cls_loss=args.cls_loss,
        reg_loss=args.reg_loss,
        pos_rate=data_meta['pos_rate'],
        focal_gamma=args.focal_gamma,
        smooth_l1_beta=args.smooth_l1_beta,
        huber_delta=args.smooth_l1_beta,
        lambda_cls=args.lambda_cls,
        lambda_center=args.lambda_center,
        lambda_radii=args.lambda_radii,
        lambda_angle=args.lambda_angle,
        angle_huber_weight=args.angle_huber_weight,
    )
    
    # Build optimizer
    optimizer = build_optimizer(
        model,
        optimizer_name=args.optimizer,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    # Build scheduler
    scheduler = build_scheduler(
        optimizer,
        scheduler_name=args.scheduler,
        mode='min' if not args.max_opt else 'max',
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
        min_lr=args.min_lr,
        num_epochs=args.epochs,
    )
    
    # Get device
    device = get_device()
    
    # Save config
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config = {
        **vars(args),
        **data_meta,
        'device': str(device),
    }
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Train
    print("\nStarting training...")
    model, results_df, train_meta = train_loop(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        num_epochs=args.epochs,
        device=device,
        scheduler=scheduler,
        output_dir=args.output_dir,
        max_opt=args.max_opt,
        fbeta=args.fbeta,
    )
    
    # Plot training curves
    print("\nGenerating plots...")
    train_metrics = results_df[[c for c in results_df.columns if c.startswith('train_')]].to_dict('records')
    val_metrics = results_df[[c for c in results_df.columns if c.startswith('val_')]].to_dict('records')
    
    # Rename columns for plotting
    train_metrics = [{k.replace('train_', ''): v for k, v in m.items()} for m in train_metrics]
    val_metrics = [{k.replace('val_', ''): v for k, v in m.items()} for m in val_metrics]
    
    plot_training_curves(
        train_metrics, val_metrics,
        save_path=output_dir / 'curves.png',
    )
    
    # Generate example visualizations
    print("\nGenerating example visualizations...")
    import matplotlib.pyplot as plt
    
    # Training examples
    generate_example_predictions(
        model=model,
        data_loader=train_loader,
        device=device,
        threshold=train_meta['best_threshold'],
        normalize_mean=data_meta['normalize_mean'],
        normalize_std=data_meta['normalize_std'],
        max_examples=12,
        title='Training Set Predictions',
        save_path=output_dir / 'examples_train.png',
    )
    plt.close('all')
    
    # Validation examples
    generate_example_predictions(
        model=model,
        data_loader=val_loader,
        device=device,
        threshold=train_meta['best_threshold'],
        normalize_mean=data_meta['normalize_mean'],
        normalize_std=data_meta['normalize_std'],
        max_examples=12,
        title='Validation Set Predictions',
        save_path=output_dir / 'examples_val.png',
    )
    plt.close('all')
    
    run_metrics = metrics_from_training(results_df, train_meta, max_opt=args.max_opt)
    maybe_update_best_baseline(
        run_dir=output_dir,
        metrics=run_metrics,
        config=config,
        max_opt=args.max_opt,
    )
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best model saved to: {output_dir / 'best.pt'}")
    print(f"Best threshold: {train_meta['best_threshold']:.4f}")
    print(f"Best F-beta: {train_meta['best_fbeta']:.4f}")
    print(f"Metrics saved to: {output_dir / 'metrics.csv'}")
    print(f"Training curves: {output_dir / 'curves.png'}")
    print(f"Train examples: {output_dir / 'examples_train.png'}")
    print(f"Val examples: {output_dir / 'examples_val.png'}")
    print(f"Highest-error grid: {output_dir / 'highest_errors.png'}")
    print(f"Baseline registry: {Path('runs') / 'best_baseline.json'}")
    print("=" * 60)


if __name__ == '__main__':
    main()
