import hashlib
import io
import copy
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError
import torch
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split

from src.utils import logger


IMAGE_SIZE = (224, 224)
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def load_rgb_image(image_source) -> Image.Image:
    if isinstance(image_source, (str, Path)):
        with Image.open(image_source) as opened:
            image = ImageOps.exif_transpose(opened).copy()
    elif isinstance(image_source, bytes):
        with Image.open(io.BytesIO(image_source)) as opened:
            image = ImageOps.exif_transpose(opened).copy()
    else:
        raise ValueError("Unsupported image source type.")

    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, rgba).convert("RGB")
    else:
        image = image.convert("RGB")
    return image


def get_transforms(is_train: bool = True):
    # Logos contain lettering, so horizontal flips and aggressive crops are avoided.
    common = [
        transforms.Resize(IMAGE_SIZE, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ]
    if not is_train:
        return transforms.Compose(common)

    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE, antialias=True),
        transforms.RandomApply([
            transforms.RandomAffine(
                degrees=10,
                translate=(0.05, 0.05),
                scale=(0.90, 1.10),
                shear=3,
                fill=(255, 255, 255),
            )
        ], p=0.8),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.10, hue=0.02)
        ], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


class LogoDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = Path(root_dir)
        if not self.root_dir.is_dir():
            raise FileNotFoundError(f"Dataset directory does not exist: {self.root_dir}")
        self.transform = transform
        self.classes = sorted(
            directory.name
            for directory in self.root_dir.iterdir()
            if directory.is_dir() and not directory.name.startswith(".")
        )
        self.class_to_idx = {name: index for index, name in enumerate(self.classes)}
        self.samples = []
        seen_hashes = set()
        duplicate_count = 0

        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            for image_path in sorted(class_dir.rglob("*")):
                if not image_path.is_file() or image_path.suffix.lower() not in VALID_EXTENSIONS:
                    continue
                try:
                    digest = hashlib.sha256(image_path.read_bytes()).digest()
                except OSError as error:
                    logger.warning("Could not read image %s: %s", image_path, error)
                    continue
                if digest in seen_hashes:
                    duplicate_count += 1
                    continue
                seen_hashes.add(digest)
                self.samples.append((str(image_path), self.class_to_idx[class_name]))

        if duplicate_count:
            logger.info("Skipped %d byte-identical duplicate images.", duplicate_count)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        try:
            image = load_rgb_image(image_path)
        except (UnidentifiedImageError, OSError, ValueError) as error:
            raise ValueError(f"Invalid image in training data: {image_path}") from error

        if self.transform:
            image = self.transform(image)
        return image, label


def create_dataloaders(data_dir: str, batch_size: int = 16, val_split: float = 0.2, seed: int = 42):
    base_dataset = LogoDataset(data_dir)
    if not base_dataset.samples:
        raise RuntimeError(f"No images found in {data_dir}.")

    labels = [label for _, label in base_dataset.samples]
    counts = torch.bincount(torch.tensor(labels), minlength=len(base_dataset.classes))
    if torch.any(counts == 0):
        raise RuntimeError("Every class folder must contain at least one readable image.")

    # Keep singleton classes in training because they cannot contribute to both splits.
    singleton_indices = [i for i, label in enumerate(labels) if counts[label].item() == 1]
    splittable_indices = [i for i, label in enumerate(labels) if counts[label].item() > 1]
    train_indices = list(singleton_indices)
    val_indices = []

    if splittable_indices:
        split_train, split_val = train_test_split(
            splittable_indices,
            test_size=val_split,
            random_state=seed,
            stratify=[labels[i] for i in splittable_indices],
        )
        train_indices.extend(split_train)
        val_indices = split_val

    train_indices.sort()
    val_indices.sort()
    if not val_indices:
        raise RuntimeError("Not enough images to make a validation split.")

    # Share the same deduplicated, sorted sample list so split indices cannot drift.
    train_dataset = copy.copy(base_dataset)
    train_dataset.transform = get_transforms(is_train=True)
    eval_dataset = copy.copy(base_dataset)
    eval_dataset.transform = get_transforms(is_train=False)
    train_labels = [labels[i] for i in train_indices]
    train_counts = torch.bincount(torch.tensor(train_labels), minlength=len(base_dataset.classes)).float()
    sample_weights = torch.tensor([1.0 / max(train_counts[label].item(), 1.0) for label in train_labels])
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_indices), replacement=True)

    train_loader = DataLoader(
        Subset(train_dataset, train_indices), batch_size=batch_size, sampler=sampler,
        num_workers=0, pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        Subset(eval_dataset, val_indices), batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=torch.cuda.is_available(),
    )
    train_eval_loader = DataLoader(
        Subset(eval_dataset, train_indices), batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=torch.cuda.is_available(),
    )
    if singleton_indices:
        logger.warning(
            "%d classes have only one unique image and cannot be evaluated on held-out data.",
            len({labels[i] for i in singleton_indices}),
        )
    logger.info(
        "Dataset: %d unique images, %d classes, %d train, %d validation.",
        len(base_dataset), len(base_dataset.classes), len(train_indices), len(val_indices),
    )
    return train_loader, val_loader, train_eval_loader, base_dataset.classes


def preprocess_image_bytes(file_bytes: bytes) -> torch.Tensor:
    try:
        image = load_rgb_image(file_bytes)
    except (UnidentifiedImageError, ValueError, OSError) as error:
        raise ValueError("Invalid or corrupted image file.") from error
    return get_transforms(is_train=False)(image).unsqueeze(0)
