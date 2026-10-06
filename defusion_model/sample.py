"""Generate cat/dog images from a trained diffusion checkpoint.

Examples:
    python sample.py --checkpoint outputs/model.pth --class-name samoyed
    python sample.py --checkpoint outputs/model.pth --class-name all --num-samples 4
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from torch.nn import functional as F
from torchvision.utils import save_image

from diffusion.device import load_checkpoint, load_model, resolve_device
from diffusion.schedule import GaussianDiffusion
from diffusion.unet import UNet


def parse_args():
    parser = argparse.ArgumentParser(description='Sample images from a cat/dog diffusion model')
    parser.add_argument('--checkpoint', default='./outputs/model.pth')
    parser.add_argument('--output-dir', default='./outputs/generated')
    parser.add_argument('--num-samples', type=int, default=8)
    parser.add_argument('--class-name', default='all',
                        help="Class name(s) to generate, comma separated; 'all' or 'unconditional'")
    parser.add_argument('--sample-steps', type=int, default=100)
    parser.add_argument('--cfg-scale', type=float, default=3.0,
                        help='Classifier-free guidance scale; 1.0 disables guidance')
    parser.add_argument('--device', default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save-individual', action='store_true',
                        help='Also write every sample as its own png')
    parser.add_argument('--upscale', type=int, default=1,
                        help='Nearest-neighbour upscale factor for saved pngs, so 32px output is '
                             'actually viewable. Adds no detail, it only enlarges pixels')
    return parser.parse_args()


def resolve_class_ids(class_name, class_names, num_classes):
    """Map a --class-name request onto (label index, output stem) pairs."""
    if class_name == 'unconditional' or num_classes == 0:
        return [(None, 'unconditional')]
    lookup = {name.lower(): index for index, name in enumerate(class_names)}
    if class_name == 'all':
        return [(index, name) for index, name in enumerate(class_names)]
    requested = [item.strip().lower() for item in class_name.split(',') if item.strip()]
    unknown = [name for name in requested if name not in lookup]
    if unknown:
        raise SystemExit(
            f'Unknown class name(s) {unknown}.\nThis checkpoint knows: {", ".join(class_names)}'
        )
    return [(lookup[name], name) for name in requested]


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    payload = load_checkpoint(args.checkpoint, device='cpu')
    model_config = payload.get('model_config') or {}
    diffusion_config = payload.get('diffusion_config') or {}
    class_names = payload.get('class_names') or []
    logging.info('checkpoint epoch=%s val_loss=%s classes=%s',
                 payload.get('epoch'), payload.get('val_loss'), len(class_names))
    model = UNet(**model_config)
    model = load_model(model, args.checkpoint, device=device, use_ema=True)
    model.eval()
    diffusion = GaussianDiffusion(**diffusion_config).to(device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_size = model.image_size
    for class_id, stem in resolve_class_ids(args.class_name, class_names, model.num_classes):
        if class_id is None:
            class_label = None
        else:
            class_label = torch.full((args.num_samples,), class_id, device=device, dtype=torch.long)
        images = diffusion.ddim_sample_loop(
            model,
            shape=(args.num_samples, 3, image_size, image_size),
            class_label=class_label,
            device=device,
            sample_steps=args.sample_steps,
            cfg_scale=args.cfg_scale,
        )
        images = ((images.clamp(-1, 1) + 1.0) * 0.5)
        if args.upscale > 1:
            # nearest keeps the real 32x32 pixels visible rather than inventing detail
            images = F.interpolate(images, scale_factor=args.upscale, mode='nearest')
        save_path = output_dir / f'{stem}.png'
        save_image(images, save_path, nrow=min(args.num_samples, 8))
        logging.info('wrote %s (%s samples)', save_path, args.num_samples)
        if args.save_individual:
            for index, image in enumerate(images):
                save_image(image, output_dir / f'{stem}_{index:02d}.png')


if __name__ == '__main__':
    main()
