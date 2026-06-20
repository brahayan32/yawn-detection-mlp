from pathlib import Path

import cv2
import numpy as np

from .config import CLASS_TO_LABEL, IMAGE_SIZE


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def preprocess_image(image_path: str | Path, image_size: tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    """Load, grayscale, smooth, resize, normalize and flatten one image."""
    image_path = Path(image_path)
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"No se pudo leer la imagen: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    resized = cv2.resize(blurred, image_size)
    normalized = resized.astype("float32") / 255.0
    return normalized.flatten()


def load_split(dataset_dir: str | Path, split: str, image_size: tuple[int, int] = IMAGE_SIZE):
    """Load one dataset split from folders named yawn and no_yawn."""
    split_dir = Path(dataset_dir) / split
    features, labels, paths = [], [], []

    for class_name, label in CLASS_TO_LABEL.items():
        class_dir = split_dir / class_name
        if not class_dir.exists():
            continue

        for image_path in sorted(class_dir.rglob("*")):
            if image_path.suffix.lower() not in VALID_EXTENSIONS:
                continue
            features.append(preprocess_image(image_path, image_size))
            labels.append(label)
            paths.append(str(image_path))

    if not features:
        return np.empty((0, image_size[0] * image_size[1]), dtype="float32"), np.array([], dtype="int32"), []

    return np.vstack(features).astype("float32"), np.array(labels, dtype="int32"), paths


def load_all_dataset(dataset_dir: str | Path, image_size: tuple[int, int] = IMAGE_SIZE):
    """Load train, validation and test splits."""
    return {
        split: load_split(dataset_dir, split, image_size)
        for split in ("train", "validation", "test")
    }
