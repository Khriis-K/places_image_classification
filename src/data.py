"""Places365-Standard data pipeline: file lists, stratified split, transforms, DataLoaders.

Expected layout (produced by scripts/extract_data.sh):
    <root>/data_256/<letter>/<class>[/<subclass>]/*.jpg   training images
    <root>/val_256/Places365_val_*.jpg                    official validation images (our test set)
    <root>/filelists/categories_places365.txt             "/a/airfield 0"
    <root>/filelists/places365_train_standard.txt         "/a/airfield/00000001.jpg 0"
    <root>/filelists/places365_val.txt                    "Places365_val_00000001.jpg 0"
"""
from collections import defaultdict
from pathlib import Path
import random

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ImageNet statistics; scripts/explore.py measures Places365's own to compare
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def read_categories(filelist_dir):
    """Class names indexed by label, e.g. 'airfield', 'jacuzzi/indoor'."""
    names = {}
    for line in Path(filelist_dir, "categories_places365.txt").read_text().splitlines():
        path, label = line.rsplit(" ", 1)
        names[int(label)] = path.split("/", 2)[2]  # drop the "/a/" letter prefix
    return [names[i] for i in range(len(names))]


def read_file_list(path):
    """List of (relative image path, label) pairs."""
    samples = []
    for line in Path(path).read_text().splitlines():
        rel, label = line.rsplit(" ", 1)
        samples.append((rel.lstrip("/"), int(label)))
    return samples


def stratified_split(samples, val_fraction, seed=0):
    """Hold out val_fraction of every class so both splits keep the class balance."""
    by_class = defaultdict(list)
    for sample in samples:
        by_class[sample[1]].append(sample)

    rng = random.Random(seed)
    train, val = [], []
    for label in sorted(by_class):
        items = by_class[label]
        rng.shuffle(items)
        n_val = round(len(items) * val_fraction)
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    return train, val


class Places365(Dataset):
    def __init__(self, image_dir, samples, transform=None):
        self.image_dir = Path(image_dir)
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel, label = self.samples[idx]
        image = Image.open(self.image_dir / rel).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def build_transforms(train, image_size=224, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Train: random crop/flip/color jitter for augmentation. Eval: deterministic center crop."""
    normalize = [transforms.ToTensor(), transforms.Normalize(mean, std)]
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.35, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            *normalize,
        ])
    return transforms.Compose([
        transforms.Resize(int(image_size * 256 / 224)),
        transforms.CenterCrop(image_size),
        *normalize,
    ])


def build_datasets(root, val_fraction=0.05, seed=0, image_size=224, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Train/val come from a stratified split of the official training set; test is the official val set."""
    root = Path(root)
    filelists = root / "filelists"
    train_samples, val_samples = stratified_split(
        read_file_list(filelists / "places365_train_standard.txt"), val_fraction, seed
    )
    test_samples = read_file_list(filelists / "places365_val.txt")

    train_tf = build_transforms(True, image_size, mean, std)
    eval_tf = build_transforms(False, image_size, mean, std)
    return {
        "train": Places365(root / "data_256", train_samples, train_tf),
        "val": Places365(root / "data_256", val_samples, eval_tf),
        "test": Places365(root / "val_256", test_samples, eval_tf),
    }


def build_loaders(datasets, batch_size=256, num_workers=8, pin_memory=True):
    return {
        split: DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
            drop_last=(split == "train"),
        )
        for split, ds in datasets.items()
    }
