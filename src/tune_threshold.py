import numpy as np
import pandas as pd
import tensorflow as tf

from .config import DATASET_DIR, METRICS_DIR, MODELS_DIR
from .evaluation import compute_metrics
from .model import predict
from .preprocessing import load_split


def main():
    x_val, y_val, _ = load_split(DATASET_DIR, "validation")
    if len(x_val) == 0:
        raise RuntimeError("No hay imagenes en datasets/validation para calibrar el umbral.")

    model = tf.saved_model.load(str(MODELS_DIR / "final_model"))
    y_prob = predict(model, x_val)

    rows = []
    for threshold in np.arange(0.1, 0.91, 0.01):
        metrics = compute_metrics(y_val, y_prob, threshold=float(threshold))
        rows.append({"threshold": float(threshold), **metrics})

    results = pd.DataFrame(rows)
    best = results.sort_values(["f1_score", "accuracy", "recall"], ascending=False).iloc[0]

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(METRICS_DIR / "threshold_tuning.csv", index=False)
    (METRICS_DIR / "best_threshold.txt").write_text(f"{best['threshold']:.2f}\n", encoding="utf-8")

    print("Mejor umbral encontrado en validation:")
    print(best.to_string())
    print(f"\nGuardado en: {METRICS_DIR / 'best_threshold.txt'}")


if __name__ == "__main__":
    main()
