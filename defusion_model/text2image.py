"""Generate images from a text prompt with a pretrained latent diffusion model.

The from-scratch UNet in this repo is class-conditional, so it can only be asked
for labels it was trained on. Breed prompts like "australian shepherd" need a
text-conditioned model, which is what this script provides for comparison.

Examples:
    python text2image.py --prompt "a photo of an australian shepherd dog"
    python text2image.py --prompt "a photo of a tabby cat" --model stabilityai/sd-turbo
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from diffusion.device import resolve_device


# The turbo models are distilled for 1-4 steps with no guidance, which is what
# makes them usable on a laptop GPU. The tradeoff is realism: with guidance
# pinned at 0 they ignore negative prompts and tend to look illustrated.
TURBO_MODELS = ('stabilityai/sd-turbo', 'stabilityai/sdxl-turbo')

# SDXL and its finetunes were trained at 1024; asking for 512 costs a lot of detail.
SDXL_MODELS = ('stabilityai/stable-diffusion-xl-base-1.0', 'SG161222/RealVisXL_V4.0',
               'stabilityai/sdxl-turbo')

# Steers away from the illustrated / rendered look that shows up without it.
DEFAULT_NEGATIVE = (
    'illustration, painting, drawing, sketch, cartoon, anime, cgi, 3d render, digital art, '
    'plastic, toy, doll, oversaturated, overexposed, blurry, out of focus, low quality, '
    'jpeg artifacts, watermark, text, deformed, extra limbs, mutated, bad anatomy'
)


def parse_args():
    parser = argparse.ArgumentParser(description='Text to image with a pretrained diffusion model')
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--negative-prompt', default=None,
                        help='Defaults to a photorealism negative prompt; pass "" to disable')
    parser.add_argument('--model', default='SG161222/RealVisXL_V4.0')
    parser.add_argument('--output-dir', default='./outputs/text2image')
    parser.add_argument('--name', default=None, help='Output filename stem; defaults to a slug of the prompt')
    parser.add_argument('--num-images', type=int, default=4)
    parser.add_argument('--steps', type=int, default=None, help='Defaults to 4 for turbo models, 35 otherwise')
    parser.add_argument('--guidance-scale', type=float, default=None,
                        help='Defaults to 0.0 for turbo models, 7.0 otherwise')
    parser.add_argument('--size', type=int, default=None,
                        help='Defaults to 1024 for SDXL models, 512 otherwise')
    parser.add_argument('--variant', default='fp16', help="Weight variant to load, e.g. fp16; '' for default")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default=None)
    return parser.parse_args()


def slugify(text, max_length=60):
    kept = [char.lower() if char.isalnum() else '_' for char in text]
    slug = ''.join(kept).strip('_')
    while '__' in slug:
        slug = slug.replace('__', '_')
    return slug[:max_length] or 'sample'


def defaults_for(model, steps, guidance_scale, size, negative_prompt):
    is_turbo = model in TURBO_MODELS
    if steps is None:
        steps = 4 if is_turbo else 35
    if guidance_scale is None:
        guidance_scale = 0.0 if is_turbo else 7.0
    if size is None:
        size = 1024 if model in SDXL_MODELS and not is_turbo else 512
    if negative_prompt is None:
        # A negative prompt only does anything when guidance is actually applied.
        negative_prompt = DEFAULT_NEGATIVE if guidance_scale > 1.0 else None
    return steps, guidance_scale, size, (negative_prompt or None)


def build_pipeline(model, device, variant='fp16'):
    from diffusers import AutoPipelineForText2Image

    # float16 is the only practical dtype for these on 'mps'/'cuda'; cpu needs float32.
    dtype = torch.float32 if device.type == 'cpu' else torch.float16
    logging.info('loading %s (dtype=%s, first run downloads weights)', model, dtype)
    kwargs = dict(torch_dtype=dtype, use_safetensors=True)
    if variant and dtype == torch.float16:
        kwargs['variant'] = variant
    try:
        pipeline = AutoPipelineForText2Image.from_pretrained(model, **kwargs)
    except Exception as exc:
        if 'variant' not in kwargs:
            raise
        logging.warning('no %s variant for %s (%s), falling back to default weights', variant, model, exc)
        kwargs.pop('variant')
        pipeline = AutoPipelineForText2Image.from_pretrained(model, **kwargs)
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=None)
    # VAE slicing keeps peak memory down when decoding 1024px latents.
    if hasattr(pipeline, 'enable_vae_slicing'):
        pipeline.enable_vae_slicing()
    return pipeline


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    device = resolve_device(args.device)
    steps, guidance_scale, size, negative_prompt = defaults_for(
        args.model, args.steps, args.guidance_scale, args.size, args.negative_prompt)
    pipeline = build_pipeline(args.model, device, variant=args.variant)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.name or slugify(args.prompt)
    generator = torch.Generator(device='cpu').manual_seed(args.seed)
    logging.info('prompt=%r steps=%s guidance=%s size=%s n=%s',
                 args.prompt, steps, guidance_scale, size, args.num_images)
    if negative_prompt:
        logging.info('negative_prompt=%r', negative_prompt[:80] + '...')
    written = []
    for index in range(args.num_images):
        result = pipeline(
            prompt=args.prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            height=size,
            width=size,
            generator=generator,
        )
        save_path = output_dir / f'{stem}_{index:02d}.png'
        result.images[0].save(save_path)
        written.append(save_path)
        logging.info('wrote %s', save_path)
    logging.info('done, %s image(s) in %s', len(written), output_dir)


if __name__ == '__main__':
    main()
