"""Image/caption dataset for SDXL LoRA fine-tuning.

Expects a flat folder of images with a sidecar caption per image::

    data/
      dog_001.jpg
      dog_001.txt      -> "a photo of a sks dog sitting on grass"
      dog_002.png
      dog_002.txt

Images without a sidecar fall back to ``--instance-prompt``, which is what you
want for single-subject training where every image shares one caption.

Beyond pixels and captions, the dataset returns SDXL's micro-conditioning
signals: the *original* image size and the crop offset actually used. SDXL was
trained with these as extra inputs so it could learn that low-resolution or
badly-cropped training images are not the target distribution. Feeding constants
here is a common cause of soft, off-centre LoRA output.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}


class ImageCaptionDataset(Dataset):
    """Images plus captions, normalised to the [-1, 1] range the SDXL VAE expects.

    Args:
        data_dir: Folder holding images and optional ``.txt`` captions.
        resolution: Square edge length to train at. SDXL's native size is 1024.
        instance_prompt: Caption for images that have no sidecar ``.txt``.
        caption_dropout: Probability of replacing a caption with ``""``. Training
            some steps unconditionally preserves the base model's response to
            classifier-free guidance; without it heavy LoRAs tend to ignore the
            negative prompt at inference time.
        center_crop: Centre crop instead of a random crop.
        random_flip: Horizontally flip at random, doubling effective data.
        repeats: Present the dataset this many times per epoch. Useful for small
            subject datasets where an epoch is otherwise only a few steps.
    """

    def __init__(
        self,
        data_dir,
        resolution=1024,
        instance_prompt=None,
        caption_dropout=0.0,
        center_crop=False,
        random_flip=True,
        repeats=1,
    ):
        self.data_dir = Path(data_dir)
        if not self.data_dir.is_dir():
            raise FileNotFoundError(f'Data directory not found: {self.data_dir}')

        self.image_paths = sorted(
            path for path in self.data_dir.rglob('*')
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not self.image_paths:
            raise FileNotFoundError(
                f'No images found under {self.data_dir}. '
                f'Supported suffixes: {", ".join(sorted(IMAGE_SUFFIXES))}'
            )

        self.resolution = resolution
        self.instance_prompt = instance_prompt
        self.caption_dropout = caption_dropout
        self.center_crop = center_crop
        self.random_flip = random_flip
        self.repeats = max(int(repeats), 1)

        captioned = sum(1 for path in self.image_paths if path.with_suffix('.txt').exists())
        if captioned < len(self.image_paths) and not instance_prompt:
            raise ValueError(
                f'{len(self.image_paths) - captioned} image(s) have no .txt caption and no '
                f'--instance-prompt was given. Provide one or add the missing caption files.'
            )
        logger.info(
            'Dataset: %s images (%s with captions), resolution %s, repeats %s',
            len(self.image_paths), captioned, resolution, self.repeats,
        )

        # Resize on the short edge so the crop below decides framing, not the resize.
        self.resize = transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR)
        self.crop = transforms.CenterCrop(resolution) if center_crop else transforms.RandomCrop(resolution)
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),   # [0, 1] -> [-1, 1]
        ])

    def __len__(self):
        return len(self.image_paths) * self.repeats

    def read_caption(self, image_path):
        caption_path = image_path.with_suffix('.txt')
        if caption_path.exists():
            caption = caption_path.read_text(encoding='utf-8').strip()
            if caption:
                return caption
        return self.instance_prompt or ''

    def __getitem__(self, index):
        image_path = self.image_paths[index % len(self.image_paths)]
        image = Image.open(image_path).convert('RGB')
        original_size = (image.height, image.width)

        image = self.resize(image)
        if self.center_crop:
            top = max(0, int(round((image.height - self.resolution) / 2.0)))
            left = max(0, int(round((image.width - self.resolution) / 2.0)))
            image = self.crop(image)
        else:
            top, left, height, width = self.crop.get_params(image, (self.resolution, self.resolution))
            image = transforms.functional.crop(image, top, left, height, width)

        if self.random_flip and random.random() < 0.5:
            image = transforms.functional.hflip(image)
            # The crop offset SDXL is told about is measured from the left edge,
            # so a horizontal flip has to mirror it too.
            left = max(0, image.width - left - self.resolution)

        caption = self.read_caption(image_path)
        if self.caption_dropout > 0.0 and random.random() < self.caption_dropout:
            caption = ''

        return {
            'pixel_values': self.to_tensor(image),
            'caption': caption,
            'original_size': torch.tensor(original_size, dtype=torch.long),
            'crop_top_left': torch.tensor((top, left), dtype=torch.long),
            'path': str(image_path),
        }


def collate_fn(examples):
    """Stack a batch, keeping captions as a list for the tokenizers."""
    return {
        'pixel_values': torch.stack([e['pixel_values'] for e in examples]).to(memory_format=torch.contiguous_format).float(),
        'captions': [e['caption'] for e in examples],
        'original_sizes': torch.stack([e['original_size'] for e in examples]),
        'crop_top_lefts': torch.stack([e['crop_top_left'] for e in examples]),
    }
