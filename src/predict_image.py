import argparse
from pathlib import Path

import tensorflow as tf

from .config import METRICS_DIR, MODELS_DIR
from .preprocessing import preprocess_image


def load_threshold(value: str) -> float:
    if value != "auto":
        return float(value)

    threshold_path = METRICS_DIR / "best_threshold.txt"
    if not threshold_path.exists():
        raise FileNotFoundError(
            "No existe metrics/best_threshold.txt. Ejecuta primero: python -m src.tune_threshold"
        )
    return float(threshold_path.read_text(encoding="utf-8").strip())


def main():
    parser = argparse.ArgumentParser(description="Predice yawn/no_yawn en una imagen nueva.")
    parser.add_argument("image_path", help="Ruta de la imagen a probar.")
    parser.add_argument("--model-dir", default=str(MODELS_DIR / "final_model"), help="Carpeta del modelo SavedModel.")
    parser.add_argument("--threshold", default="0.5", help="Umbral numerico o 'auto' para usar metrics/best_threshold.txt.")
    args = parser.parse_args()

    image_path = Path(args.image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"No existe la imagen: {image_path}")

    threshold = load_threshold(args.threshold)
    model = tf.saved_model.load(args.model_dir)
    x = preprocess_image(image_path).reshape(1, -1)
    probability = float(model(tf.convert_to_tensor(x, dtype=tf.float32)).numpy().reshape(-1)[0])

    label = "yawn" if probability >= threshold else "no_yawn"

    print(f"Imagen: {image_path}")
    print(f"Probabilidad de yawn: {probability:.4f}")
    print(f"Umbral usado: {threshold:.2f}")
    print(f"Prediccion: {label}")


if __name__ == "__main__":
    main()
