"""Render one full-resolution photo per action for a given breed.

Writes a separate image per action -- ``beagle_jumping.png``, ``beagle_paw.png``
and so on -- each at the model's native resolution. Pass ``--sheet`` if you also
want them composed into a single contact sheet.

Actions need text conditioning, so this uses a pretrained text-to-image model
rather than the class-conditional UNet in this repo, which only knows the breed
labels it was trained on and has no notion of "jumping".

Examples:
    python breed_sheet.py --breed beagle
    python breed_sheet.py --breed corgi --actions paw,jumping,tongue
    python breed_sheet.py --breed persian --steps 25
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

from diffusion.device import resolve_device
from text2image import build_pipeline, defaults_for, slugify


# Each action supplies the clause describing the pose, plus the lens that suits it.
ACTIONS = {
    'portrait': ('close up head portrait looking straight at the camera', '85mm f/1.8 lens'),
    'paw': ("sitting on grass and giving its paw into a person's open hand, shaking hands",
            '85mm f/1.8 lens'),
    'tongue': ('panting happily with its tongue hanging out, close up head portrait',
               '85mm f/1.8 lens'),
    'jumping': ('jumping high in mid air catching a frisbee in a park, all four paws off the '
                'ground, motion frozen', '200mm telephoto lens'),
    'laying': ('lying down on a wooden floor, relaxed, head resting on its front paws',
               '50mm lens'),
    'running': ('running fast across a green field towards the camera, motion frozen',
                '200mm telephoto lens'),
}
DEFAULT_ACTIONS = ('portrait', 'paw', 'tongue', 'jumping', 'laying', 'running')

CAPTIONS = {
    'portrait': 'portrait',
    'paw': 'giving paw',
    'tongue': 'tongue out',
    'jumping': 'jumping',
    'laying': 'laying down',
    'running': 'running',
}

# The twelve Oxford-IIIT Pet cat breeds, so "persian" is not described as a dog.
CAT_BREEDS = {
    'abyssinian', 'bengal', 'birman', 'bombay', 'british shorthair', 'egyptian mau',
    'maine coon', 'persian', 'ragdoll', 'russian blue', 'siamese', 'sphynx',
}

FONT_CANDIDATES = (
    '/System/Library/Fonts/Supplemental/Arial Bold.ttf',
    '/System/Library/Fonts/Helvetica.ttc',
    '/Library/Fonts/Arial.ttf',
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='One full resolution photo per action for a breed')
    parser.add_argument('--breed', required=True, help='Breed name, e.g. beagle or corgi')
    parser.add_argument('--actions', default=','.join(DEFAULT_ACTIONS),
                        help=f'Comma separated, any of: {", ".join(ACTIONS)}')
    parser.add_argument('--animal', default=None, choices=['dog', 'cat'],
                        help='Defaults to cat for known cat breeds, dog otherwise')
    parser.add_argument('--model', default='SG161222/RealVisXL_V4.0')
    parser.add_argument('--output-dir', default='./outputs/breeds')
    parser.add_argument('--tile', type=int, default=512, help='Tile size in the sheet')
    parser.add_argument('--size', type=int, default=None, help='Generation size, default 1024')
    parser.add_argument('--steps', type=int, default=None)
    parser.add_argument('--guidance-scale', type=float, default=None)
    parser.add_argument('--columns', type=int, default=None, help='Defaults to 3, or 2 for <= 4 actions')
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--device', default=None)
    parser.add_argument('--sheet', action='store_true',
                        help='Also compose the actions into a single contact sheet')
    return parser.parse_args()


def resolve_animal(breed, animal):
    if animal:
        return animal
    return 'cat' if breed.strip().lower() in CAT_BREEDS else 'dog'


def build_prompt(breed, animal, action):
    clause, lens = ACTIONS[action]
    return (f'RAW photo of a {breed} {animal} {clause}, detailed fur texture, {lens}, '
            f'natural daylight, shallow depth of field, sharp focus on the eyes, '
            f'professional pet photography, highly detailed, photorealistic')


def load_font(size):
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def compose_sheet(tiles, breed, tile_size, columns):
    """Lay the rendered actions out under a title bar, each with its own caption."""
    pad, caption_h, title_h = 12, 40, 76
    rows = math.ceil(len(tiles) / columns)
    width = pad + columns * (tile_size + pad)
    height = title_h + rows * (tile_size + caption_h + pad) + pad
    sheet = Image.new('RGB', (width, height), (250, 249, 246))
    draw = ImageDraw.Draw(sheet)
    title_font, caption_font = load_font(30), load_font(19)

    draw.text((pad + 2, 24), breed.upper(), fill=(20, 19, 15), font=title_font)
    draw.line([(pad, title_h - 10), (width - pad, title_h - 10)], fill=(214, 210, 200), width=1)

    for index, (action, image) in enumerate(tiles):
        col, row = index % columns, index // columns
        x = pad + col * (tile_size + pad)
        y = title_h + row * (tile_size + caption_h + pad)
        sheet.paste(image.resize((tile_size, tile_size), Image.LANCZOS), (x, y))
        draw.text((x + 2, y + tile_size + 11), CAPTIONS.get(action, action),
                  fill=(82, 81, 78), font=caption_font)
    return sheet


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    actions = [item.strip().lower() for item in args.actions.split(',') if item.strip()]
    unknown = [action for action in actions if action not in ACTIONS]
    if unknown:
        raise SystemExit(f'Unknown action(s) {unknown}. Available: {", ".join(ACTIONS)}')

    animal = resolve_animal(args.breed, args.animal)
    device = resolve_device(args.device)
    steps, guidance_scale, size, negative_prompt = defaults_for(
        args.model, args.steps, args.guidance_scale, args.size, None)
    columns = args.columns or (2 if len(actions) <= 4 else 3)

    pipeline = build_pipeline(args.model, device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = slugify(args.breed)

    tiles = []
    for index, action in enumerate(actions):
        prompt = build_prompt(args.breed, animal, action)
        logging.info('[%s/%s] %s: %s', index + 1, len(actions), action, prompt[:88] + '...')
        generator = torch.Generator(device='cpu').manual_seed(args.seed + index)
        image = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            height=size,
            width=size,
            generator=generator,
        ).images[0]
        tiles.append((action, image))
        image_path = output_dir / f'{stem}_{action}.png'
        image.save(image_path)
        logging.info('wrote %s (%sx%s)', image_path, image.width, image.height)

    if args.sheet:
        sheet = compose_sheet(tiles, args.breed, args.tile, columns)
        sheet_path = output_dir / f'{stem}_sheet.png'
        sheet.save(sheet_path)
        logging.info('wrote %s (%sx%s, %s actions)', sheet_path, sheet.width, sheet.height, len(tiles))
    logging.info('done: %s full resolution image(s) in %s', len(tiles), output_dir)


if __name__ == '__main__':
    main()
