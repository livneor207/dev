#!/usr/bin/env python3
"""
Visualization script for ellipse detector.

Generates example predictions with ellipse overlays for train, val, and test sets.

Usage:
    python scripts/visualize.py --ckpt runs/default/best.pt --csv train_data.csv --test-dir images/test

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
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import torch

from ellipse_cv.data import create_dataloaders, create_inference_loader
from ellipse_cv.inference import load_checkpoint, load_threshold
from ellipse_cv.visualize import generate_example_predictions


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize ellipse detector predictions')
    
    parser.add_argument('--ckpt', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--csv', type=str, default='train_data.csv',
                        help='Path to training CSV (for train/val split)')
    parser.add_argument('--base-dir', type=str, default='.',
                        help='Base directory for images')
    parser.add_argument('--test-dir', type=str, default=None,
                        help='Optional test directory for unlabeled predictions')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: same as checkpoint)')
    parser.add_argument('--max-examples', type=int, default=12,
                        help='Maximum examples per visualization')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Classification threshold (default: from training)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup
    ckpt_path = Path(args.ckpt)
    run_dir = ckpt_path.parent
    output_dir = Path(args.output_dir) if args.output_dir else run_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    print("=" * 60)
    print("ELLIPSE DETECTOR VISUALIZATION")
    print("=" * 60)
    print(f"Checkpoint: {ckpt_path}")
    print(f"Device: {device}")
    print("=" * 60)
    
    # Load model
    print("\nLoading model...")
    model, config = load_checkpoint(ckpt_path, device)
    grayscale = config.get('in_channels', 1) == 1
    
    # Load threshold
    threshold = args.threshold if args.threshold else load_threshold(run_dir)
    print(f"Using threshold: {threshold:.4f}")
    
    # Load training config for normalization stats
    config_path = run_dir / 'config.json'
    if config_path.exists():
        with open(config_path) as f:
            train_config = json.load(f)
        normalize_mean = train_config.get('normalize_mean')
        normalize_std = train_config.get('normalize_std')
    else:
        normalize_mean = None
        normalize_std = None
    
    # Create train/val dataloaders
    print("\nLoading data...")
    train_loader, val_loader, data_meta = create_dataloaders(
        csv_path=args.csv,
        base_dir=args.base_dir,
        batch_size=args.batch_size,
        grayscale=grayscale,
    )
    
    # Use loaded normalization stats if available
    if normalize_mean is None:
        normalize_mean = data_meta['normalize_mean']
        normalize_std = data_meta['normalize_std']
    
    # Generate train visualizations
    print("\nGenerating training set examples...")
    generate_example_predictions(
        model=model,
        data_loader=train_loader,
        device=device,
        threshold=threshold,
        normalize_mean=normalize_mean,
        normalize_std=normalize_std,
        max_examples=args.max_examples,
        title='Training Set Predictions',
        save_path=output_dir / 'viz_train.png',
    )
    plt.close('all')
    
    # Generate validation visualizations
    print("Generating validation set examples...")
    generate_example_predictions(
        model=model,
        data_loader=val_loader,
        device=device,
        threshold=threshold,
        normalize_mean=normalize_mean,
        normalize_std=normalize_std,
        max_examples=args.max_examples,
        title='Validation Set Predictions',
        save_path=output_dir / 'viz_val.png',
    )
    plt.close('all')
    
    # Generate test visualizations if test directory provided
    if args.test_dir:
        print("Generating test set examples...")
        test_loader, is_labeled = create_inference_loader(
            image_dir=args.test_dir,
            batch_size=args.batch_size,
            grayscale=grayscale,
            normalize_mean=normalize_mean,
            normalize_std=normalize_std,
        )
        
        generate_example_predictions(
            model=model,
            data_loader=test_loader,
            device=device,
            threshold=threshold,
            normalize_mean=normalize_mean,
            normalize_std=normalize_std,
            max_examples=args.max_examples,
            title='Test Set Predictions (Unlabeled)',
            save_path=output_dir / 'viz_test.png',
        )
        plt.close('all')
    
    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE")
    print("=" * 60)
    print(f"Train examples: {output_dir / 'viz_train.png'}")
    print(f"Val examples: {output_dir / 'viz_val.png'}")
    if args.test_dir:
        print(f"Test examples: {output_dir / 'viz_test.png'}")
    print("=" * 60)


if __name__ == '__main__':
    main()
