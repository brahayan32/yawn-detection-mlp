import argparse
import tempfile
from pathlib import Path

import cv2
import tensorflow as tf

from .config import MODELS_DIR
from .preprocessing import preprocess_image


DECISION_LIMIT = 0.5


def predict_frame(model, frame):
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        temp_path = Path(tmp.name)
    try:
        cv2.imwrite(str(temp_path), frame)
        x = preprocess_image(temp_path).reshape(1, -1)
        return float(model(tf.convert_to_tensor(x, dtype=tf.float32)).numpy().reshape(-1)[0])
    finally:
        temp_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Demo de deteccion de bostezo con webcam.")
    parser.add_argument("--camera", type=int, default=0, help="Indice de la camara.")
    parser.add_argument("--model-dir", default=str(MODELS_DIR / "final_model"), help="Carpeta del modelo SavedModel.")
    args = parser.parse_args()

    model = tf.saved_model.load(args.model_dir)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la camara.")

    print("Presiona q para salir.")
    frame_count = 0
    probability = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_count += 1
        if frame_count % 5 == 0:
            probability = predict_frame(model, frame)

        label = "YAWN" if probability >= DECISION_LIMIT else "NO YAWN"
        color = (0, 0, 255) if label == "YAWN" else (0, 180, 0)
        cv2.putText(frame, f"{label} ({probability:.2f})", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.imshow("Yawn Detection MLP", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
