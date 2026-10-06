"""Run SDXL with a trained LoRA adapter.

Loads the frozen base model, applies the LoRA weights produced by
``train_lora.py``, and writes one image per prompt.

Examples:
    python sdxl_lora/inference.py --lora-path ./outputs/lora_my_dog \
        --prompt "a photo of a sks dog running on a beach"

    # weigh the adapter down if it is overpowering the prompt
    python sdxl_lora/inference.py --lora-path ./outputs/lora_my_dog \
        --prompt "a sks dog wearing sunglasses" --lora-scale 0.7 --num-images 4

    # side by side: writes outputs/compare/base_model/ and outputs/compare/lora_model/
    # with matching filenames, same prompt and same seed in each
    python sdxl_lora/inference.py --lora-path ./outputs/lora_my_dog \
        --prompt "a photo of a sks dog" --compare-base
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from diffusers import AutoencoderKL, DPMSolverMultistepScheduler, StableDiffusionXLPipeline

DEFAULT_BASE_MODEL = 'stabilityai/stable-diffusion-xl-base-1.0'
FP16_SAFE_VAE = 'madebyollin/sdxl-vae-fp16-fix'
DEFAULT_NEGATIVE = (
    'illustration, painting, drawing, cartoon, anime, cgi, 3d render, plastic, toy, '
    'oversaturated, blurry, out of focus, low quality, jpeg artifacts, watermark, text, '
    'deformed, extra limbs, bad anatomy'
)


def parse_args():
    parser = argparse.ArgumentParser(description='SDXL + LoRA inference')
    parser.add_argument('--prompt', required=True, help='Prompt, or several separated by "|"')
    parser.add_argument('--negative-prompt', default=DEFAULT_NEGATIVE, help='Pass "" to disable')
    parser.add_argument('--lora-path', default=None,
                        help='Directory or .safetensors file holding the trained LoRA. '
                             'Omit to run the untouched base model')
    parser.add_argument('--lora-scale', type=float, default=1.0,
                        help='Adapter strength. Below 1.0 blends back toward the base model')
    parser.add_argument('--lora-scales', default=None,
                        help='Comma separated strengths to sweep, e.g. 0,0.2,0.4,0.6,0.8,1. Each '
                             'lands in its own <output-dir>/scale_<v>/ folder, sharing one model '
                             'load. Overrides --lora-scale')
    parser.add_argument('--base-model', default=DEFAULT_BASE_MODEL,
                        help='Hub id, or a path to a single .safetensors checkpoint')
    parser.add_argument('--vae-path', default=None)
    parser.add_argument('--output-dir', default='./outputs/compare',
                        help='Parent folder. Images land in <output-dir>/<base-dir-name> and '
                             '<output-dir>/<lora-dir-name>')
    parser.add_argument('--base-dir-name', default='base_model',
                        help='Subfolder for images from the untouched base model')
    parser.add_argument('--lora-dir-name', default='lora_model',
                        help='Subfolder for images from the LoRA fine-tuned model')
    parser.add_argument('--num-images', type=int, default=1, help='Images per prompt')
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--guidance-scale', type=float, default=7.0)
    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default=None)
    parser.add_argument('--dtype', default='auto', choices=['auto', 'fp16', 'bf16', 'fp32'])
    parser.add_argument('--compare-base', action='store_true',
                        help='Also render each prompt with the adapter disabled')
    return parser.parse_args()


def resolve_device(name=None):
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def resolve_dtype(choice, device):
    if choice == 'fp16':
        return torch.float16
    if choice == 'bf16':
        return torch.bfloat16
    if choice == 'fp32':
        return torch.float32
    return torch.float32 if device.type == 'cpu' else torch.float16


def slugify(text, max_length=48):
    slug = ''.join(c.lower() if c.isalnum() else '_' for c in text).strip('_')
    while '__' in slug:
        slug = slug.replace('__', '_')
    return slug[:max_length] or 'sample'


def build_pipeline(args, device, dtype):
    if args.base_model.endswith('.safetensors') and Path(args.base_model).exists():
        logging.info('loading single-file checkpoint %s', args.base_model)
        pipeline = StableDiffusionXLPipeline.from_single_file(args.base_model, torch_dtype=dtype)
    else:
        logging.info('loading %s', args.base_model)
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            args.base_model, torch_dtype=dtype, use_safetensors=True,
            variant='fp16' if dtype == torch.float16 else None)

    # The stock SDXL VAE overflows in fp16; swap in the rescaled weights.
    vae_path = args.vae_path or (FP16_SAFE_VAE if dtype == torch.float16 else None)
    if vae_path:
        pipeline.vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=dtype)

    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
        pipeline.scheduler.config, use_karras_sigmas=True)
    pipeline = pipeline.to(device)
    if hasattr(pipeline, 'enable_vae_slicing'):
        pipeline.enable_vae_slicing()
    return pipeline


LORA_WEIGHT_NAME = 'pytorch_lora_weights.safetensors'


def load_lora(pipeline, lora_path):
    """Attach the adapter, accepting either a directory or a .safetensors file.

    A resume checkpoint written by accelerate also contains ``model.safetensors``
    holding the whole training state. Left to itself ``load_lora_weights`` picks
    that up and dies on the non-LoRA parameter names, so name the adapter file
    explicitly whenever it is present.
    """
    path = Path(lora_path)
    if path.is_dir():
        adapter = path / LORA_WEIGHT_NAME
        if adapter.exists():
            pipeline.load_lora_weights(str(path), weight_name=LORA_WEIGHT_NAME)
        else:
            pipeline.load_lora_weights(str(path))
    elif path.is_file():
        pipeline.load_lora_weights(str(path.parent), weight_name=path.name)
    else:
        raise SystemExit(f'LoRA not found at {lora_path}')
    logging.info('loaded LoRA from %s', lora_path)


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    logging.info('device=%s dtype=%s', device, dtype)

    pipeline = build_pipeline(args, device, dtype)
    if args.lora_path:
        load_lora(pipeline, args.lora_path)

    parent = Path(args.output_dir)
    base_dir = parent / args.base_dir_name
    lora_dir = parent / args.lora_dir_name
    prompts = [p.strip() for p in args.prompt.split('|') if p.strip()]

    # Which passes to render. Filenames match across folders so the same prompt and
    # seed can be compared position for position.
    passes = []
    if args.lora_path and args.lora_scales:
        # A strength sweep: scale 0.0 is the untouched base model, so the sweep
        # contains its own control.
        for value in [v.strip() for v in args.lora_scales.split(',') if v.strip()]:
            scale = float(value)
            passes.append((f'scale_{scale:g}', parent / f'scale_{scale:g}', scale))
    elif args.lora_path:
        passes.append(('lora', lora_dir, args.lora_scale))
        if args.compare_base:
            # scale 0.0 mutes the adapter: same weights, same seed, no LoRA contribution.
            passes.append(('base', base_dir, 0.0))
    else:
        passes.append(('base', base_dir, None))
    for _, directory, _ in passes:
        directory.mkdir(parents=True, exist_ok=True)

    for prompt in prompts:
        stem = slugify(prompt)
        for index in range(args.num_images):
            seed = args.seed + index
            for label, directory, scale in passes:
                # Re-seed per pass so both renders start from identical noise.
                generator = torch.Generator(device='cpu').manual_seed(seed)
                kwargs = {} if scale is None else {'cross_attention_kwargs': {'scale': scale}}
                image = pipeline(
                    prompt=prompt,
                    negative_prompt=args.negative_prompt or None,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance_scale,
                    height=args.size,
                    width=args.size,
                    generator=generator,
                    **kwargs,
                ).images[0]
                path = directory / f'{stem}_{index:02d}.png'
                image.save(path)
                logging.info('[%s] wrote %s (seed %s, lora_scale %s)', label, path, seed,
                             'n/a' if scale is None else scale)

    for label, directory, _ in passes:
        logging.info('%s images -> %s', label, directory)


if __name__ == '__main__':
    main()
