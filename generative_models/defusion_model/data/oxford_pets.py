"""Download and locate the Oxford-IIIT Pet dataset.

Dataset page: https://www.robots.ox.ac.uk/~vgg/data/pets/

Images are named ``<breed>_<id>.jpg``. Cat breeds are capitalised
(``Abyssinian_1.jpg``), dog breeds are lowercase (``american_bulldog_1.jpg``),
which is how the dataset encodes species. 37 breeds, ~200 images each,
no credentials required.
"""

from __future__ import annotations

import logging
import tarfile
from pathlib import Path

from PIL import Image


IMAGES_URL = 'https://thor.robots.ox.ac.uk/~vgg/data/pets/images.tar.gz'
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png'}
VALID_CACHE_NAME = '.valid_images.txt'


def parse_breed(path):
    """``american_bulldog_12.jpg`` -> ``american_bulldog``."""
    return Path(path).stem.rsplit('_', 1)[0]


def parse_species(path):
    """Cat breeds are capitalised in this dataset, dog breeds are not."""
    return 'cat' if Path(path).name[:1].isupper() else 'dog'


def is_pet_image(path):
    return path.suffix.lower() in IMAGE_SUFFIXES and not path.name.startswith('.') and '_' in path.stem


def _verify_readable(path):
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.convert('RGB').load()
    except Exception as exc:
        logging.debug('dropping unreadable image %s (%s)', path, exc)
        return False
    return True


def list_pet_images(image_dir):
    """Return readable pet images, caching the (slow) validation pass on disk.

    A handful of files in this dataset are corrupt or are PNGs with a .jpg
    suffix, so every path is opened once and the survivors are cached.
    """
    image_dir = Path(image_dir)
    if not image_dir.exists():
        return []
    candidates = sorted(path for path in image_dir.rglob('*') if path.is_file() and is_pet_image(path))
    if not candidates:
        return []
    cache_path = image_dir / VALID_CACHE_NAME
    if cache_path.exists():
        cached = [image_dir / name for name in cache_path.read_text().split('\n') if name]
        if cached and all(path.exists() for path in cached):
            return cached
    logging.info('Validating %s Oxford Pet images (one time, results cached)', len(candidates))
    valid = [path for path in candidates if _verify_readable(path)]
    dropped = len(candidates) - len(valid)
    if dropped:
        logging.info('Dropped %s unreadable images', dropped)
    cache_path.write_text('\n'.join(str(path.relative_to(image_dir)) for path in valid))
    return valid


def find_pet_image_dir(data_dir):
    data_dir = Path(data_dir)
    if not data_dir.exists():
        return None
    candidates = [data_dir / 'oxford_pets' / 'images', data_dir / 'images', data_dir]
    recursive = [path for path in data_dir.rglob('*') if path.is_dir()]
    for candidate in candidates + recursive:
        if candidate.is_dir() and any(path.is_file() and is_pet_image(path) for path in candidate.iterdir()):
            return candidate
    return None


def _download_images(dest_dir):
    import urllib.request

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    archive_path = dest_dir / 'images.tar.gz'
    if not archive_path.exists() or archive_path.stat().st_size < 1024:
        logging.info('Downloading Oxford-IIIT Pet images (~792MB) from %s', IMAGES_URL)
        urllib.request.urlretrieve(IMAGES_URL, archive_path)
    logging.info('Extracting %s', archive_path)
    with tarfile.open(archive_path) as archive:
        archive.extractall(dest_dir, filter='data')
    return archive_path


def prepare_oxford_pets(data_dir='./data', download=True):
    """Return the directory holding Oxford Pet images, downloading if needed."""
    found = find_pet_image_dir(data_dir)
    if found is not None:
        logging.info('Found Oxford Pet images in %s', found)
        return found
    if not download:
        raise FileNotFoundError(
            f'Oxford Pet images were not found under {data_dir}. '
            f'Download {IMAGES_URL} and extract it so files look like '
            f'{Path(data_dir) / "oxford_pets" / "images" / "Abyssinian_1.jpg"}'
        )
    _download_images(Path(data_dir) / 'oxford_pets')
    found = find_pet_image_dir(data_dir)
    if found is None:
        raise FileNotFoundError(f'Download finished but no pet images were found under {data_dir}')
    return found
