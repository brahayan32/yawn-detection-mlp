from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "datasets"
MODELS_DIR = PROJECT_ROOT / "models"
METRICS_DIR = PROJECT_ROOT / "metrics"

IMAGE_SIZE = (64, 64)
CLASS_NAMES = ["no_yawn", "yawn"]
CLASS_TO_LABEL = {"no_yawn": 0, "yawn": 1}
LABEL_TO_CLASS = {0: "no_yawn", 1: "yawn"}

RANDOM_STATE = 42
BATCH_SIZE = 32
EPOCHS = 30
