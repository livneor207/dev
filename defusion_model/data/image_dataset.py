from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2

from defusion_model.data.dogs_vs_cats import list_train_images, prepare_dogs_vs_cats
from defusion_model.data.oxford_pets import list_pet_images, parse_breed, parse_species, prepare_oxford_pets


CLASS_TO_INDEX = {'cat': 0, 'dog': 1}
INDEX_TO_CLASS = {0: 'cat', 1: 'dog'}


def parse_class_name(path):
    prefix = Path(path).name.split('.')[0].lower()
    if prefix not in CLASS_TO_INDEX:
        raise ValueError(f'Unsupported Dogs vs Cats filename: {path}')
    return prefix


def stratified_indices(labels, val_fraction, seed):
    rng = np.random.RandomState(seed)
    train_idx = []
    val_idx = []
    for cls in np.unique(labels):
        cls_idx = np.where(labels == cls)[0]
        rng.shuffle(cls_idx)
        n_val = max(1, int(round(len(cls_idx) * val_fraction)))
        n_val = min(n_val, len(cls_idx) - 1) if len(cls_idx) > 1 else 0
        val_idx.append(cls_idx[:n_val])
        train_idx.append(cls_idx[n_val:])
    train_idx = np.concatenate(train_idx) if train_idx else np.array([], dtype=np.int64)
    val_idx = np.concatenate(val_idx) if val_idx else np.array([], dtype=np.int64)
    return train_idx, val_idx


class PetImageDataset(Dataset):
    """Shared image/label plumbing for the cat-and-dog diffusion datasets.

    Subclasses only have to produce the path array, the integer label array and
    the ordered class names; everything below (splitting, augmentation, tensor
    conversion) is identical across datasets.
    """

    def __init__(self, paths, labels, class_names, image_size=64, train=True, debug=False,
                 val_fraction=0.1, seed=42, max_samples=None, augment=True):
        paths = np.array(paths)
        labels = np.asarray(labels, dtype=np.int64)
        if paths.size == 0:
            raise FileNotFoundError('No labeled images were found')
        unique, counts = np.unique(labels, return_counts=True)
        if unique.size < 2:
            raise ValueError(f'Expected at least two classes, found {dict(zip(unique.tolist(), counts.tolist()))}')
        # -----------
        train_idx, val_idx = stratified_indices(labels, val_fraction=val_fraction, seed=seed)
        chosen = train_idx if train else val_idx
        if chosen.size == 0:
            raise ValueError('Chosen split is empty; add more images or lower val_fraction')
        rng = np.random.RandomState(seed)
        if max_samples is not None and chosen.size > max_samples:
            chosen = rng.choice(chosen, size=max_samples, replace=False)
        self.path_array = paths[chosen]
        self.label_array = labels[chosen]
        # -----------
        # Augmentation effectively enlarges the dataset, which is the opposite of what a
        # memorisation check wants: with random crops and jitter the model never sees the
        # same image twice, so it cannot overfit a small set inside a short budget.
        if train and augment:
            self.geometric_transform = v2.Compose([
                v2.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                v2.RandomHorizontalFlip(),
            ])
            self.photometric = v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1)
        else:
            self.geometric_transform = v2.Compose([
                v2.Resize(image_size),
                v2.CenterCrop(image_size),
            ])
            self.photometric = v2.Identity()
        self.to_tensor = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ])
        # -----------
        self.class_names = list(class_names)
        self.label_encoder = {name: index for index, name in enumerate(self.class_names)}
        self.single_label = True
        self.is_prod = False
        self.dataset_size = int(self.path_array.size)
        self.num_classes = len(self.class_names)
        self.image_size = image_size
        self.train = train
        self.debug = debug
        self.augment = bool(train and augment)

    def inverse_normalize(self, tensor):
        """Model space [-1, 1] back to displayable [0, 1]."""
        return (tensor.clamp(-1, 1) + 1.0) * 0.5

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        path = Path(self.path_array[idx])
        image = Image.open(path).convert('RGB')
        image = self.geometric_transform(image)
        image_tensor = self.augment_with_retry(image)
        # ----------- [C, H, W] float32 in [-1, 1]
        samples_dict = {'image': image_tensor}
        class_label = torch.tensor(int(self.label_array[idx]), dtype=torch.long)
        if self.debug:
            samples_dict['debug_image'] = self.inverse_normalize(image_tensor)
        return samples_dict, class_label

    def to_model_tensor(self, pil_image):
        tensor = self.to_tensor(pil_image)
        return tensor * 2.0 - 1.0

    def augment_with_retry(self, pil_image, max_tries=4):
        """Retries photometric jitter while frames come out degenerate; falls back to geometric-only."""
        unaugmented = self.to_model_tensor(pil_image)
        if not self.augment:
            return unaugmented
        for _ in range(max_tries):
            candidate = self.to_model_tensor(self.photometric(pil_image))
            if float(candidate.std()) >= 0.05:
                return candidate
        return unaugmented


class DogsVsCatsDataset(PetImageDataset):
    """Kaggle Dogs vs Cats, labeled by species from the ``cat.`` / ``dog.`` filename prefix."""

    def __init__(self, data_dir='./data', image_size=64, train=True, debug=False,
                 val_fraction=0.1, seed=42, download=True, max_samples=None, augment=True):
        image_dir = prepare_dogs_vs_cats(data_dir, download=download)
        paths = np.array(list_train_images(image_dir))
        if paths.size == 0:
            raise FileNotFoundError(f'No labeled cat/dog images found in {image_dir}')
        labels = np.array([CLASS_TO_INDEX[parse_class_name(path)] for path in paths], dtype=np.int64)
        super().__init__(
            paths, labels, class_names=[INDEX_TO_CLASS[0], INDEX_TO_CLASS[1]],
            image_size=image_size, train=train, debug=debug,
            val_fraction=val_fraction, seed=seed, max_samples=max_samples, augment=augment,
        )
        self.image_dir = image_dir


class OxfordPetsDataset(PetImageDataset):
    """Oxford-IIIT Pet, labeled either by breed (37 classes) or species (2 classes)."""

    def __init__(self, data_dir='./data', image_size=64, train=True, debug=False,
                 val_fraction=0.1, seed=42, download=True, max_samples=None,
                 label_mode='breed', breeds=None, augment=True):
        if label_mode not in {'breed', 'species'}:
            raise ValueError(f'Unsupported label_mode: {label_mode}')
        image_dir = prepare_oxford_pets(data_dir, download=download)
        paths = np.array(list_pet_images(image_dir))
        if paths.size == 0:
            raise FileNotFoundError(f'No pet images found in {image_dir}')
        if breeds:
            wanted = {breed.lower() for breed in breeds}
            available = sorted({parse_breed(path).lower() for path in paths})
            unknown = wanted - set(available)
            if unknown:
                raise ValueError(f'Unknown breeds {sorted(unknown)}; available breeds are {available}')
            keep = np.array([parse_breed(path).lower() in wanted for path in paths])
            paths = paths[keep]
        if label_mode == 'breed':
            names = [parse_breed(path) for path in paths]
        else:
            names = [parse_species(path) for path in paths]
        class_names = sorted(set(names))
        encoder = {name: index for index, name in enumerate(class_names)}
        labels = np.array([encoder[name] for name in names], dtype=np.int64)
        super().__init__(
            paths, labels, class_names=class_names,
            image_size=image_size, train=train, debug=debug,
            val_fraction=val_fraction, seed=seed, max_samples=max_samples, augment=augment,
        )
        self.image_dir = image_dir
        self.label_mode = label_mode
        self.species_by_class = {name: parse_species(f'{name}_1.jpg') for name in self.class_names}


class RepeatDataset(Dataset):
    """Presents ``base`` ``times`` times over, so one epoch is that many passes.

    An epoch on a few hundred images is only a handful of optimizer steps, and the
    per-epoch validation and report writing then cost more than the training does.
    Repeating the data amortises that without changing what the model sees.
    """

    def __init__(self, base, times):
        if times < 1:
            raise ValueError(f'times must be >= 1, got {times}')
        self.base = base
        self.times = int(times)

    def __len__(self):
        return len(self.base) * self.times

    def __getitem__(self, idx):
        return self.base[idx % len(self.base)]

    def __getattr__(self, name):
        # proxy dataset metadata (class_names, image_size, single_label, ...)
        return getattr(self.base, name)


def create_dataset(dataset='oxford-pets', **kwargs):
    """Build a dataset by name; ``label_mode`` and ``breeds`` only apply to oxford-pets."""
    label_mode = kwargs.pop('label_mode', 'breed')
    breeds = kwargs.pop('breeds', None)
    repeat = kwargs.pop('repeat', 1)
    if dataset == 'dogs-vs-cats':
        built = DogsVsCatsDataset(**kwargs)
    elif dataset == 'oxford-pets':
        built = OxfordPetsDataset(label_mode=label_mode, breeds=breeds, **kwargs)
    else:
        raise ValueError(f'Unsupported dataset: {dataset}')
    return RepeatDataset(built, repeat) if repeat > 1 else built
