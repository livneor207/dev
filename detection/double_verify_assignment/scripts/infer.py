#!/usr/bin/env python3
"""
Inference script for ellipse detector.

Usage:
    python scripts/infer.py --ckpt runs/default/best.pt --data images/test --out predictions.csv

See --help for all options.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from ellipse_cv.inference import run_inference


def parse_args():
    parser = argparse.ArgumentParser(description='Run inference with ellipse detector')
    
    parser.add_argument('--ckpt', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to image directory or CSV file')
    parser.add_argument('--out', type=str, default='predictions.csv',
                        help='Output path for predictions')
    parser.add_argument('--csv', type=str, default=None,
                        help='Optional labeled CSV for metrics')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Classification threshold (default: from training)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (auto-detected if not specified)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("ELLIPSE DETECTOR INFERENCE")
    print("=" * 60)
    print(f"Checkpoint: {args.ckpt}")
    print(f"Data: {args.data}")
    print(f"Output: {args.out}")
    print("=" * 60)
    
    # Parse device
    device = None
    if args.device:
        device = torch.device(args.device)
    
    # Run inference
    predictions_df, metrics = run_inference(
        checkpoint_path=args.ckpt,
        image_dir=args.data,
        output_path=args.out,
        csv_path=args.csv,
        threshold=args.threshold,
        batch_size=args.batch_size,
        device=device,
    )
    
    print("\n" + "=" * 60)
    print("INFERENCE COMPLETE")
    print("=" * 60)
    print(f"Predictions saved to: {args.out}")
    print(f"Total samples: {len(predictions_df)}")
    print(f"Predicted ellipses: {predictions_df['is_ellipse'].sum()}")
    print(f"Predicted non-ellipses: {(~predictions_df['is_ellipse']).sum()}")
    print("=" * 60)


if __name__ == '__main__':
    main()
