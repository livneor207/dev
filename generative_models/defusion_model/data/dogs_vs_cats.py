"""Download and locate the Kaggle Dogs vs Cats competition dataset.

Competition page: https://www.kaggle.com/c/dogs-vs-cats

Training files are named ``cat.{id}.jpg`` / ``dog.{id}.jpg``. You must accept
the competition rules and have Kaggle API credentials in ``~/.kaggle/kaggle.json``.
"""

from __future__ import annotations

import logging
import subprocess
import zipfile
from pathlib import Path


COMPETITION = 'dogs-vs-cats'
TRAIN_ZIP_NAMES = ('train.zip', 'train.zip.zip')
ARCHIVE_NAMES = ('dogs-vs-cats.zip', 'dogs-vs-cats.zip.zip')


def is_labeled_train_image(path):
    name = path.name.lower()
    return path.suffix.lower() in {'.jpg', '.jpeg', '.png'} and (
        name.startswith('cat.') or name.startswith('dog.')
    )


def list_train_images(image_dir):
    image_dir = Path(image_dir)
    return sorted(path for path in image_dir.rglob('*') if is_labeled_train_image(path))


def find_train_image_dir(data_dir):
    data_dir = Path(data_dir)
    if not data_dir.exists():
        return None
    candidates = [data_dir / 'train', data_dir / 'dogs-vs-cats' / 'train', data_dir]
    recursive = [path for path in data_dir.rglob('*') if path.is_dir()]
    for candidate in candidates + recursive:
        if list_train_images(candidate):
            return candidate
    return None


def _extract_zip(zip_path, dest):
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    logging.info('Extracting %s -> %s', zip_path, dest)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(dest)


def _extract_nested_archives(data_dir):
    data_dir = Path(data_dir)
    changed = True
    while changed:
        changed = False
        for zip_path in sorted(data_dir.rglob('*.zip')):
            marker = zip_path.with_suffix(zip_path.suffix + '.extracted')
            if marker.exists():
                continue
            _extract_zip(zip_path, zip_path.parent)
            marker.touch()
            changed = True


def _download_with_kagglehub(data_dir):
    import kagglehub

    logging.info('Downloading %s with kagglehub', COMPETITION)
    downloaded = Path(kagglehub.competition_download(COMPETITION))
    if downloaded.is_dir():
        return downloaded
    return data_dir


def _download_with_kaggle_cli(data_dir):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    archive_path = data_dir / 'dogs-vs-cats.zip'
    if not archive_path.exists():
        logging.info('Downloading %s with the Kaggle CLI', COMPETITION)
        subprocess.run(
            ['kaggle', 'competitions', 'download', '-c', COMPETITION, '-p', str(data_dir)],
            check=True,
        )
    return data_dir


def prepare_dogs_vs_cats(data_dir='./data', download=True):
    """Return the directory that contains labeled train images, downloading if needed."""
    found = find_train_image_dir(data_dir)
    if found is not None:
        logging.info('Found Dogs vs Cats images in %s', found)
        return found
    if not download:
        raise FileNotFoundError(_missing_data_message(data_dir))
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    errors = []
    try:
        downloaded = _download_with_kagglehub(data_dir)
        _extract_nested_archives(downloaded)
        _extract_nested_archives(data_dir)
        found = find_train_image_dir(downloaded) or find_train_image_dir(data_dir)
        if found is not None:
            return found
        errors.append(f'kagglehub finished but no train images were found under {downloaded}')
    except Exception as exc:
        errors.append(f'kagglehub failed: {exc}')
    try:
        _download_with_kaggle_cli(data_dir)
        _extract_nested_archives(data_dir)
        found = find_train_image_dir(data_dir)
        if found is not None:
            return found
        errors.append(f'Kaggle CLI finished but no train images were found under {data_dir}')
    except Exception as exc:
        errors.append(f'Kaggle CLI failed: {exc}')
    raise FileNotFoundError(_missing_data_message(data_dir) + '\n\n' + '\n'.join(errors))


def _missing_data_message(data_dir):
    return (
        f'Dogs vs Cats train images were not found under {data_dir}.\n'
        'Accept the competition rules at https://www.kaggle.com/c/dogs-vs-cats\n'
        'then either:\n'
        '  1. Place kaggle.json in ~/.kaggle/ and rerun with --download, or\n'
        '  2. Download train.zip yourself and extract it so files look like:\n'
        f'     {Path(data_dir) / "train" / "cat.0.jpg"}\n'
        f'     {Path(data_dir) / "train" / "dog.0.jpg"}'
    )
