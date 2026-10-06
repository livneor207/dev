"""Download Oxford-IIIT Pet and build the captioned dataset the LoRA trains on.

One command takes you from nothing to a training-ready folder:

    python prepare_data.py

It downloads the archive (~792 MB, no credentials needed), extracts it, drops
unreadable or too-small files, and writes one ``.txt`` caption per image.

Captions carry a **style token**:

    "a snapshot photo of a beagle dog"

That token matters. Every image in this dataset shares the same amateur-snapshot
character, so with a caption template like ``"a photo of a <breed> dog"`` the only
varying token is the breed and the style binds to the shared phrase -- the model
then applies snapshot framing whenever you ask for any of these breeds. Giving
the style its own word lets you leave it out at inference and keep just the breed
knowledge. Measured effect: peak breed score 0.4189 with the token vs 0.3320
without (see README).
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

IMAGES_URL = 'https://thor.robots.ox.ac.uk/~vgg/data/pets/images.tar.gz'
ARCHIVE_BYTES = 791_918_971          # for the progress readout
MIN_SHORT_EDGE = 256                 # below this the crop to 512 is mush


def parse_args():
    parser = argparse.ArgumentParser(
        description='Download Oxford-IIIT Pet and write captioned training data')
    parser.add_argument('--raw-dir', default='data/oxford_pets',
                        help='Where the archive is downloaded and extracted')
    parser.add_argument('--out-dir', default='data/lora_pets37_styled',
                        help='Captioned dataset written here, ready for train_lora.py')
    parser.add_argument('--style-token', default='snapshot',
                        help='Word carrying the dataset style. Pass "" to disable, which '
                             'binds the style to the breed instead and is measurably worse')
    parser.add_argument('--min-short-edge', type=int, default=MIN_SHORT_EDGE,
                        help='Skip images whose short edge is below this')
    parser.add_argument('--force', action='store_true',
                        help='Rebuild the caption folder even if it already exists')
    parser.add_argument('--keep-archive', action='store_true',
                        help='Keep images.tar.gz after extracting (default: keep)')
    return parser.parse_args()


def _progress(count, block_size, total_size):
    total = total_size if total_size > 0 else ARCHIVE_BYTES
    done = min(count * block_size, total)
    percent = 100.0 * done / total
    sys.stdout.write(f'\r  downloading {percent:5.1f}%  ({done/1e6:.0f}/{total/1e6:.0f} MB)')
    sys.stdout.flush()


def download_and_extract(raw_dir):
    """Fetch and unpack the dataset; both steps are skipped if already done."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    images_dir = raw_dir / 'images'
    if images_dir.is_dir() and len(list(images_dir.glob('*.jpg'))) > 7000:
        logger.info('Images already extracted at %s', images_dir)
        return images_dir

    archive = raw_dir / 'images.tar.gz'
    # A tiny file means a previous download was interrupted; refetch rather than
    # failing later inside tarfile with a confusing error.
    if not archive.exists() or archive.stat().st_size < 1_000_000:
        logger.info('Downloading %s', IMAGES_URL)
        urllib.request.urlretrieve(IMAGES_URL, archive, reporthook=_progress)
        print()
    else:
        logger.info('Archive already present: %s', archive)

    logger.info('Extracting %s', archive)
    with tarfile.open(archive) as tar:
        tar.extractall(raw_dir, filter='data')      # filter= is required on 3.12+
    return images_dir


def caption_for(breed, style_token):
    """``Egyptian_Mau`` -> ``a snapshot photo of a egyptian mau cat``.

    The dataset capitalises cat breeds and lowercases dog breeds, which is the
    only species label it ships with.
    """
    species = 'cat' if breed[:1].isupper() else 'dog'
    label = breed.replace('_', ' ').lower()
    style = f'{style_token} ' if style_token else ''
    return f'a {style}photo of a {label} {species}'


def build_captioned(images_dir, out_dir, style_token, min_short_edge, force):
    out_dir = Path(out_dir)
    if out_dir.exists():
        if not force and any(out_dir.glob('*.txt')):
            logger.info('Captioned dataset already at %s (use --force to rebuild)', out_dir)
            return out_dir
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    kept, skipped, breeds = 0, 0, {}
    for path in sorted(Path(images_dir).glob('*.jpg')):
        breed = path.stem.rsplit('_', 1)[0]
        try:
            # verify() then reopen: verify() consumes the file object, so the
            # image has to be loaded again before it can be converted.
            Image.open(path).verify()               # catches truncated files
            image = Image.open(path).convert('RGB')  # a few are PNGs named .jpg
            if min(image.size) < min_short_edge:
                skipped += 1
                continue
        except Exception:
            skipped += 1
            continue
        # Re-index per breed so filenames stay contiguous even when files are
        # skipped; downstream tools parse the breed from the filename stem.
        index = breeds.get(breed, 0)
        target = out_dir / f'{breed}_{index:03d}.jpg'
        image.save(target, 'JPEG', quality=95)
        target.with_suffix('.txt').write_text(caption_for(breed, style_token))
        breeds[breed] = index + 1
        kept += 1

    cats = sum(1 for b in breeds if b[:1].isupper())
    logger.info('Wrote %s images across %s breeds (%s cat, %s dog); skipped %s',
                kept, len(breeds), cats, len(breeds) - cats, skipped)
    example = next(iter(sorted(breeds)))
    logger.info('Example caption: "%s"', caption_for(example, style_token))
    return out_dir


def main():
    args = parse_args()
    images_dir = download_and_extract(args.raw_dir)
    out_dir = build_captioned(images_dir, args.out_dir, args.style_token,
                              args.min_short_edge, args.force)
    print(f'\nReady: {out_dir}')
    print(f'Next:  poetry run train-lora --data-dir {out_dir} --output-dir runs/v1')


if __name__ == '__main__':
    main()
