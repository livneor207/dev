"""Build one comparison sheet per pair of breeds: real data vs base vs LoRA.

Two breeds per file, so the 37 breeds land in 19 sheets. Each row shows two real
training photos, then the base model and the LoRA rendering the same prompt at
the same seed -- so any difference between the last two columns is the adapter.

    python build_all_sheets.py --compare-dir ~/lora_runs/all37 --out-dir ~/lora_runs/sheets
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONT_CANDIDATES = (
    '/System/Library/Fonts/Supplemental/Arial Bold.ttf',
    '/System/Library/Fonts/Helvetica.ttc',
)


def parse_args():
    parser = argparse.ArgumentParser(description='Per-breed-pair comparison sheets')
    parser.add_argument('--compare-dir', required=True, help='Holds base_model/ and the LoRA folder')
    parser.add_argument('--lora-dir-name', default='v2_best_step125')
    parser.add_argument('--base-dir-name', default='base_model')
    parser.add_argument('--data-dir', default='data/lora_pets37')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--per-sheet', type=int, default=2, help='Breeds per sheet file')
    parser.add_argument('--tile', type=int, default=290)
    parser.add_argument('--real-count', type=int, default=2)
    return parser.parse_args()


def load_font(size):
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def square(image, size):
    edge = min(image.size)
    left, top = (image.width - edge) // 2, (image.height - edge) // 2
    return image.crop((left, top, left + edge, top + edge)).resize((size, size), Image.LANCZOS)


def prompt_stem(breed):
    species = 'cat' if breed[:1].isupper() else 'dog'
    text = f'a photo of a {breed.replace("_", " ").lower()} {species}'
    slug = ''.join(c.lower() if c.isalnum() else '_' for c in text).strip('_')
    while '__' in slug:
        slug = slug.replace('__', '_')
    return slug[:48]


def build_sheet(breeds, args, base_dir, lora_dir, data_dir):
    tile, pad, hdr, title, lbl = args.tile, 8, 28, 46, 150
    columns = args.real_count + 2
    width = lbl + pad + columns * (tile + pad)
    height = title + hdr + len(breeds) * (tile + pad)
    sheet = Image.new('RGB', (width, height), (250, 249, 246))
    draw = ImageDraw.Draw(sheet)
    title_font, head_font, row_font = load_font(19), load_font(13), load_font(14)

    draw.text((pad + 2, 14), ' / '.join(b.replace('_', ' ') for b in breeds)
              + '   -   real data vs base SDXL vs LoRA', fill=(20, 19, 15), font=title_font)

    heads = [f'REAL #{i + 1}' for i in range(args.real_count)] + ['BASE SDXL', 'LoRA v2']
    colors = [(82, 81, 78)] * args.real_count + [(42, 120, 214), (27, 175, 122)]
    for index, (head, color) in enumerate(zip(heads, colors)):
        draw.text((lbl + pad + index * (tile + pad) + 2, title + 5), head, fill=color, font=head_font)

    divider = lbl + pad + args.real_count * (tile + pad) - pad // 2
    draw.line([(divider, title + hdr - 5), (divider, height - pad)], fill=(206, 202, 192), width=1)

    missing = []
    for row, breed in enumerate(breeds):
        y = title + hdr + row * (tile + pad)
        draw.text((8, y + tile // 2 - 7), breed.replace('_', ' '), fill=(20, 19, 15), font=row_font)
        reals = sorted(data_dir.glob(f'{breed}_*.jpg'))[:args.real_count]
        for column, path in enumerate(reals):
            sheet.paste(square(Image.open(path).convert('RGB'), tile),
                        (lbl + pad + column * (tile + pad), y))
        stem = prompt_stem(breed)
        for offset, directory in enumerate((base_dir, lora_dir)):
            x = lbl + pad + (args.real_count + offset) * (tile + pad)
            path = directory / f'{stem}_00.png'
            if path.exists():
                sheet.paste(square(Image.open(path).convert('RGB'), tile), (x, y))
            else:
                missing.append(f'{breed}:{directory.name}')
                draw.rectangle([x, y, x + tile, y + tile], fill=(238, 236, 230))
                draw.text((x + 12, y + tile // 2), 'not generated', fill=(150, 148, 140), font=row_font)
    return sheet, missing


def main():
    args = parse_args()
    compare_dir = Path(os.path.expanduser(args.compare_dir))
    out_dir = Path(os.path.expanduser(args.out_dir))
    data_dir = Path(os.path.expanduser(args.data_dir))
    base_dir = compare_dir / args.base_dir_name
    lora_dir = compare_dir / args.lora_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Breed is the filename stem minus the trailing index, e.g. Egyptian_Mau_012.
    breeds = sorted({p.stem.rsplit('_', 1)[0] for p in data_dir.glob('*.jpg')})
    # cats first (capitalised in this dataset), then dogs, so sheets group sensibly
    breeds.sort(key=lambda b: (not b[:1].isupper(), b.lower()))
    groups = [breeds[i:i + args.per_sheet] for i in range(0, len(breeds), args.per_sheet)]
    print(f'{len(breeds)} breeds -> {len(groups)} sheets ({args.per_sheet} per sheet)')

    all_missing = []
    for index, group in enumerate(groups, start=1):
        sheet, missing = build_sheet(group, args, base_dir, lora_dir, data_dir)
        name = f'{index:02d}_' + '_'.join(group).lower() + '.png'
        sheet.save(out_dir / name)
        all_missing += missing
        print(f'  {name}  ({sheet.width}x{sheet.height})')
    if all_missing:
        print(f'\nmissing renders ({len(all_missing)}): {", ".join(all_missing[:8])}'
              + (' ...' if len(all_missing) > 8 else ''))
    print(f'\nwrote {len(groups)} sheets to {out_dir}')


if __name__ == '__main__':
    main()
