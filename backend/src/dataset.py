import os
from pathlib import Path

import torch
from PIL import Image
try:
    from sklearn.model_selection import train_test_split
except ImportError:
    pass
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

CLASS_NAMES = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "No Finding",
]
NUM_CLASSES = len(CLASS_NAMES)
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".dcm"}

# Maps alternate folder names (e.g. TB dataset) to project class labels.
FOLDER_ALIASES = {
    "normal": "No Finding",
    "no finding": "No Finding",
    "no_finding": "No Finding",
    "tuberculosis": "Infiltration",
    "tb": "Infiltration",
    "atelectasis": "Atelectasis",
    "cardiomegaly": "Cardiomegaly",
    "effusion": "Effusion",
    "infiltration": "Infiltration",
}

TB_DISEASE_CLASSES = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_val_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _resolve_class_name(folder_name: str) -> str | None:
    normalized = folder_name.strip().lower().replace("_", " ")
    if folder_name in CLASS_NAMES:
        return folder_name
    if normalized in FOLDER_ALIASES:
        return FOLDER_ALIASES[normalized]
    if folder_name in FOLDER_ALIASES:
        return FOLDER_ALIASES[folder_name]
    return None


def _is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def _collect_from_class_folders(data_dir: Path) -> list[tuple[str, int]]:
    samples = []
    for class_name in CLASS_NAMES:
        class_dir = data_dir / class_name
        if not class_dir.is_dir():
            continue
        for image_path in class_dir.rglob("*"):
            if image_path.is_file() and _is_image_file(image_path):
                samples.append((str(image_path), CLASS_TO_IDX[class_name]))

    if samples:
        return samples

    for root, dirs, _ in os.walk(data_dir):
        for folder_name in dirs:
            class_name = _resolve_class_name(folder_name)
            if class_name is None:
                continue
            folder_path = Path(root) / folder_name
            for image_path in folder_path.rglob("*"):
                if image_path.is_file() and _is_image_file(image_path):
                    samples.append((str(image_path), CLASS_TO_IDX[class_name]))

    return samples


def _collect_tb_fallback(data_dir: Path) -> list[tuple[str, int]]:
    """Map TB Chest Radiography layout to the 5-class problem."""
    samples = []
    tb_root = data_dir / "TB_Chest_Radiography_Database"
    if not tb_root.is_dir():
        return samples

    normal_dir = tb_root / "Normal"
    if normal_dir.is_dir():
        for image_path in normal_dir.glob("*.png"):
            samples.append((str(image_path), CLASS_TO_IDX["No Finding"]))

    tb_dir = tb_root / "Tuberculosis"
    if tb_dir.is_dir():
        tb_images = sorted(tb_dir.glob("*.png"))
        for idx, image_path in enumerate(tb_images):
            disease = TB_DISEASE_CLASSES[idx % len(TB_DISEASE_CLASSES)]
            samples.append((str(image_path), CLASS_TO_IDX[disease]))

    return samples


def collect_samples(data_dir: str | Path) -> list[tuple[str, int]]:
    data_path = Path(data_dir)
    if not data_path.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    samples = _collect_from_class_folders(data_path)
    if not samples:
        samples = _collect_tb_fallback(data_path)

    if not samples:
        raise RuntimeError(
            f"No images found under {data_path}. "
            f"Expected class folders: {CLASS_NAMES}"
        )

    return samples


class ChestXrayDataset(Dataset):
    def __init__(self, samples: list[tuple[str, int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def create_dataloaders(
    data_dir: str | Path = "data",
    batch_size: int = 32,
    val_split: float = 0.2,
    random_state: int = 42,
    num_workers: int = 0,
    max_samples: int | None = None,
):
    samples = collect_samples(data_dir)
    if max_samples is not None and len(samples) > max_samples:
        rng = torch.Generator().manual_seed(random_state)
        indices = torch.randperm(len(samples), generator=rng)[:max_samples].tolist()
        samples = [samples[i] for i in indices]
    labels = [label for _, label in samples]

    train_samples, val_samples = train_test_split(
        samples,
        test_size=val_split,
        random_state=random_state,
        stratify=labels,
    )

    train_dataset = ChestXrayDataset(train_samples, transform=get_train_transforms())
    val_dataset = ChestXrayDataset(val_samples, transform=get_val_transforms())

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, train_samples, val_samples
