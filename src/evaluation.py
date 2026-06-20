from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from .config import CLASS_NAMES


def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype("int32").reshape(-1)
    y_true = y_true.reshape(-1)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }


def save_history_plots(history, output_dir: str | Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for metric, filename in [("accuracy", "accuracy.png"), ("loss", "loss.png")]:
        plt.figure(figsize=(8, 5))
        plt.plot(history[metric], label=f"train_{metric}")
        val_metric = f"val_{metric}"
        if val_metric in history and history[val_metric]:
            plt.plot(history[val_metric], label=val_metric)
        plt.xlabel("Epocas")
        plt.ylabel(metric)
        plt.title(f"Evolucion de {metric}")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=150)
        plt.close()


def save_confusion_matrix(y_true, y_prob, output_path: str | Path, threshold: float = 0.5):
    y_pred = (y_prob >= threshold).astype("int32").reshape(-1)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Prediccion")
    plt.ylabel("Real")
    plt.title("Matriz de confusion")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_results_csv(metrics: dict, output_path: str | Path):
    pd.DataFrame([metrics]).to_csv(output_path, index=False)
