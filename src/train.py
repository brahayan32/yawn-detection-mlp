import numpy as np
import tensorflow as tf

from .config import BATCH_SIZE, DATASET_DIR, EPOCHS, METRICS_DIR, MODELS_DIR, RANDOM_STATE
from .evaluation import compute_metrics, save_confusion_matrix, save_history_plots, save_results_csv
from .model import PureTensorFlowMLP, predict, save_model, train_mlp
from .preprocessing import load_all_dataset


def main():
    np.random.seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)
    data = load_all_dataset(DATASET_DIR)

    x_train, y_train, _ = data["train"]
    x_val, y_val, _ = data["validation"]
    x_test, y_test, _ = data["test"]

    if len(x_train) == 0:
        raise RuntimeError("No hay imagenes en datasets/train. Agrega el dataset antes de entrenar.")

    model = PureTensorFlowMLP(input_dim=x_train.shape[1], seed=RANDOM_STATE)

    history = train_mlp(
        model,
        x_train,
        y_train,
        x_val=x_val if len(x_val) else None,
        y_val=y_val if len(y_val) else None,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
    )

    save_history_plots(history, METRICS_DIR)
    save_model(model, MODELS_DIR / "best_model")

    if len(x_test):
        y_prob = predict(model, x_test)
        metrics = compute_metrics(y_test, y_prob)
        save_confusion_matrix(y_test, y_prob, METRICS_DIR / "confusion_matrix.png")
        save_results_csv(metrics, METRICS_DIR / "results.csv")

    save_model(model, MODELS_DIR / "final_model")


if __name__ == "__main__":
    main()
