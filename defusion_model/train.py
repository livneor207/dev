"""Train a class-conditional DDPM on cat and dog photos.

Datasets:
    oxford-pets   https://www.robots.ox.ac.uk/~vgg/data/pets/ (37 breeds, no credentials)
    dogs-vs-cats  https://www.kaggle.com/c/dogs-vs-cats (2 classes, needs kaggle.json)

Examples:
    # two-breed overfit check that breed conditioning works at all
    python train.py --breeds samoyed,pug --epochs 60 --image-size 64

    # full 37-breed run
    python train.py --epochs 25 --image-size 64
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from defusion_model.data import create_dataset
from diffusion.device import load_compatible_weights, resolve_device
from diffusion.freeze import freeze_unet_layers
from diffusion.schedule import GaussianDiffusion
from diffusion.unet import UNet
from train.loop import train_loop


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a diffusion model on Dogs vs Cats')
    parser.add_argument('--data-dir', default='./data', help='Directory holding the image dataset')
    parser.add_argument('--dataset', default='oxford-pets', choices=['oxford-pets', 'dogs-vs-cats'])
    parser.add_argument('--label-mode', default='breed', choices=['breed', 'species'],
                        help='oxford-pets only: condition on 37 breeds or on cat/dog')
    parser.add_argument('--breeds', default=None,
                        help='oxford-pets only: comma separated breeds to keep, e.g. samoyed,pug')
    parser.add_argument('--output-dir', default='./outputs')
    parser.add_argument('--image-size', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--timesteps', type=int, default=1000)
    parser.add_argument('--schedule', default='cosine', choices=['cosine', 'linear'])
    parser.add_argument('--model-channels', type=int, default=64)
    parser.add_argument('--attention-resolutions', default='auto',
                        help="Feature-map sizes that get self attention. 'auto' uses image_size//4 "
                             "(the DDPM default); 'wide' adds image_size//2, which is much slower")
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default=None)
    parser.add_argument('--max-samples', type=int, default=None, help='Optional cap for a smoke run')
    parser.add_argument('--val-fraction', type=float, default=0.1)
    parser.add_argument('--ema-decay', type=float, default=0.9999)
    parser.add_argument('--freeze-policy', default='none', choices=['none', 'all', 'encoder', 'decoder', 'time'])
    parser.add_argument('--no-download', action='store_true')
    parser.add_argument('--unconditional', action='store_true', help='Train without class labels')
    parser.add_argument('--label-dropout', type=float, default=0.1,
                        help='Probability of replacing a label with the null token, enabling CFG')
    parser.add_argument('--cfg-scale', type=float, default=3.0, help='Guidance scale for epoch previews')
    parser.add_argument('--sample-classes', default=None,
                        help='Comma separated class names to preview each epoch')
    parser.add_argument('--sample-every', type=int, default=1)
    parser.add_argument('--sample-steps', type=int, default=50)
    parser.add_argument('--patience', type=int, default=0,
                        help='Early-stopping patience in epochs; 0 disables it (default). '
                             'Validation noise-MSE tracks sample quality poorly, so stopping on '
                             'it tends to discard a still-improving model')
    parser.add_argument('--save-last-every', type=int, default=25,
                        help='Write the latest weights to model_last.pth every N epochs')
    parser.add_argument('--no-augment', action='store_true',
                        help='Disable random crop/flip/jitter. Use for a deliberate memorisation '
                             'check: augmentation is what stops a small set being overfit')
    parser.add_argument('--init-from', default=None,
                        help='Checkpoint to initialise weights from. Tensors whose shape no longer '
                             'matches (e.g. class_embed when the class count changed) keep their '
                             'fresh init; everything else is copied')
    parser.add_argument('--repeat', type=int, default=1,
                        help='Present the train split N times per epoch, so a tiny dataset stops '
                             'paying per-epoch validation overhead every few steps')
    return parser.parse_args()


def split_names(value):
    if not value:
        return None
    return [item.strip() for item in value.split(',') if item.strip()]


def resolve_attention_resolutions(spec, image_size):
    """Attention at image_size//2 roughly triples step time, so it is opt-in."""
    if spec == 'auto':
        sizes = (image_size // 4,)
    elif spec == 'wide':
        sizes = (image_size // 4, image_size // 2)
    else:
        sizes = tuple(int(item) for item in spec.split(',') if item.strip())
    return tuple(size for size in sizes if size >= 8)


def create_loaders(args):
    common = dict(
        data_dir=args.data_dir,
        image_size=args.image_size,
        val_fraction=args.val_fraction,
        seed=args.seed,
        download=not args.no_download,
        max_samples=args.max_samples,
        label_mode=args.label_mode,
        breeds=split_names(args.breeds),
        augment=not args.no_augment,
    )
    train_dataset = create_dataset(args.dataset, train=True, repeat=args.repeat, **common)
    val_dataset = create_dataset(args.dataset, train=False, **common)
    # persistent_workers matters a lot for small datasets: without it every epoch
    # re-spawns the worker processes, which on macOS costs seconds per epoch and
    # dwarfs the actual compute when an epoch is only a handful of steps.
    loader_kwargs = dict(
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=len(train_dataset) >= args.batch_size,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    logging.info(
        '%s split train=%s val=%s classes=%s image_dir=%s',
        args.dataset, len(train_dataset), len(val_dataset), train_dataset.num_classes,
        train_dataset.image_dir,
    )
    logging.info('class names: %s', ', '.join(train_dataset.class_names))
    logging.info('augment=%s repeat=%s steps_per_epoch=%s',
                 not args.no_augment, args.repeat, len(train_loader))
    return train_loader, val_loader


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    set_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader = create_loaders(args)
    num_classes = 0 if args.unconditional else train_loader.dataset.num_classes
    attention_resolutions = resolve_attention_resolutions(args.attention_resolutions, args.image_size)
    model = UNet(
        in_channels=3,
        model_channels=args.model_channels,
        out_channels=3,
        channel_multipliers=(1, 2, 4),
        num_res_blocks=2,
        attention_resolutions=attention_resolutions,
        dropout=0.1,
        num_classes=num_classes,
        image_size=args.image_size,
    )
    if args.init_from:
        load_compatible_weights(model, args.init_from, use_ema=True)
    freeze_unet_layers(model, policy=args.freeze_policy)
    model = model.to(device)
    diffusion = GaussianDiffusion(timesteps=args.timesteps, schedule=args.schedule).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    noise_loss = torch.nn.MSELoss()
    logging.info('device=%s param_count=%s', device, sum(p.numel() for p in model.parameters()))
    train_loop(
        model,
        optimizer,
        train_loader,
        val_loader,
        diffusion,
        noise_loss=noise_loss,
        num_epochs=args.epochs,
        device=device,
        scheduler=scheduler,
        model_path=str(output_dir / 'model.pth'),
        training_summary_df_path=str(output_dir / 'training_summary.csv'),
        sample_dir=str(output_dir / 'samples'),
        max_opt=False,
        ema_decay=args.ema_decay,
        label_dropout=args.label_dropout,
        cfg_scale=args.cfg_scale,
        sample_classes=split_names(args.sample_classes),
        sample_every=args.sample_every,
        sample_steps=args.sample_steps,
        patience_epochs=args.patience,
        save_last_every=args.save_last_every,
    )


if __name__ == '__main__':
    main()
