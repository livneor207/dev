"""Compose the three-way comparison sheet: real data vs base model vs LoRA.

Each row is one breed and holds, left to right:

    3 real training images | base SDXL output | LoRA fine-tuned output

The generated pair share a prompt and a seed, so anything that differs between
those two columns is the adapter's doing. The real images on the left are the
distribution the LoRA was pulled towards, which is what makes a "did it get the
breed right, or just the snapshot style?" judgement possible at a glance.

Example:
    python compare_sheet.py \
        --breeds Egyptian_Mau,Birman,Bombay,Siamese,Persian \
        --compare-dir outputs/compare_final \
        --out outputs/compare_final/sheet_cats.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

DATA_DIR = Path('data/lora_pets37')
FONT_CANDIDATES = (
    '/System/Library/Fonts/Supplemental/Arial Bold.ttf',
    '/System/Library/Fonts/Helvetica.ttc',
)


def parse_args():
    parser = argparse.ArgumentParser(description='Three-way comparison sheet')
    parser.add_argument('--breeds', required=True, help='Comma separated breed folder names')
    parser.add_argument('--compare-dir', required=True,
                        help='Parent holding base_model/ and lora_model/')
    parser.add_argument('--data-dir', default=str(DATA_DIR))
    parser.add_argument('--base-dir-name', default='base_model')
    parser.add_argument('--lora-dir-name', default='lora_model')
    parser.add_argument('--out', required=True)
    parser.add_argument('--tile', type=int, default=250)
    parser.add_argument('--real-count', type=int, default=3)
    parser.add_argument('--seed-index', type=int, default=0,
                        help='Which generated sample index to show (matches _NN in filenames)')
    parser.add_argument('--title', default='Real data vs base SDXL vs LoRA fine-tune')
    return parser.parse_args()


def load_font(size, bold=True):
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def square(image, size):
    """Centre-crop to a square then resize, so nothing is distorted."""
    edge = min(image.size)
    left, top = (image.width - edge) // 2, (image.height - edge) // 2
    return image.crop((left, top, left + edge, top + edge)).resize((size, size), Image.LANCZOS)


def prompt_stem(breed, species):
    label = breed.replace('_', ' ').lower()
    text = f'a photo of a {label} {species}'
    slug = ''.join(c.lower() if c.isalnum() else '_' for c in text).strip('_')
    while '__' in slug:
        slug = slug.replace('__', '_')
    return slug[:48]


def find_generated(directory, breed, species, index):
    """Locate the render for this breed, tolerating prompt-slug drift."""
    stem = prompt_stem(breed, species)
    exact = Path(directory) / f'{stem}_{index:02d}.png'
    if exact.exists():
        return exact
    token = breed.replace('_', '').lower()
    for candidate in sorted(Path(directory).glob(f'*_{index:02d}.png')):
        if token in candidate.stem.replace('_', '').lower():
            return candidate
    return None


def main():
    args = parse_args()
    breeds = [b.strip() for b in args.breeds.split(',') if b.strip()]
    data_dir = Path(args.data_dir)
    base_dir = Path(args.compare_dir) / args.base_dir_name
    lora_dir = Path(args.compare_dir) / args.lora_dir_name

    tile, pad, label_w = args.tile, 8, 165
    header_h, title_h = 30, 44
    gap = 26                                   # visual break between real and generated
    columns = args.real_count + 2
    width = label_w + args.real_count * (tile + pad) + gap + 2 * (tile + pad) + pad
    height = title_h + header_h + len(breeds) * (tile + pad) + pad

    sheet = Image.new('RGB', (width, height), (250, 249, 246))
    draw = ImageDraw.Draw(sheet)
    title_font, head_font, row_font = load_font(21), load_font(15), load_font(15)

    draw.text((pad + 4, 13), args.title, fill=(20, 19, 15), font=title_font)

    real_x = label_w + pad
    base_x = real_x + args.real_count * (tile + pad) + gap
    lora_x = base_x + tile + pad
    y_head = title_h + 6
    draw.text((real_x, y_head), f'REAL - Oxford-IIIT Pet (training data)', fill=(82, 81, 78), font=head_font)
    draw.text((base_x, y_head), 'BASE SDXL', fill=(42, 120, 214), font=head_font)
    draw.text((lora_x, y_head), 'LoRA fine-tuned', fill=(235, 104, 52), font=head_font)

    # divider between the real-data block and the generated block
    divider_x = base_x - gap // 2
    draw.line([(divider_x, title_h + header_h - 4), (divider_x, height - pad)],
              fill=(214, 210, 200), width=1)

    missing = []
    for row, breed in enumerate(breeds):
        species = 'cat' if breed[:1].isupper() else 'dog'
        y = title_h + header_h + row * (tile + pad)
        draw.text((10, y + tile // 2 - 8), breed.replace('_', ' '), fill=(20, 19, 15), font=row_font)

        for col, path in enumerate(sorted(data_dir.glob(f'{breed}_*.jpg'))[:args.real_count]):
            sheet.paste(square(Image.open(path).convert('RGB'), tile), (real_x + col * (tile + pad), y))

        for x, directory, tag in ((base_x, base_dir, 'base'), (lora_x, lora_dir, 'lora')):
            found = find_generated(directory, breed, species, args.seed_index)
            if found is None:
                missing.append(f'{breed}/{tag}')
                draw.rectangle([x, y, x + tile, y + tile], fill=(238, 236, 230))
                draw.text((x + 12, y + tile // 2), 'not generated', fill=(150, 148, 140), font=row_font)
            else:
                sheet.paste(square(Image.open(found).convert('RGB'), tile), (x, y))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    print(f'wrote {out} ({sheet.width}x{sheet.height}) for {len(breeds)} breeds')
    if missing:
        print('missing renders:', ', '.join(missing))


if __name__ == '__main__':
    main()
