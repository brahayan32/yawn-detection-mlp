import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold

from .config import BATCH_SIZE, DATASET_DIR, EPOCHS, L2_REGULARIZATION, METRICS_DIR, RANDOM_STATE
from .evaluation import compute_metrics
from .model import PureTensorFlowMLP, predict, train_mlp
from .preprocessing import load_paths, load_split


def main(n_splits: int = 5):
    np.random.seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)
    x_train, y_train, paths = load_split(DATASET_DIR, "train")

    if len(x_train) < n_splits:
        raise RuntimeError("No hay suficientes imagenes para aplicar validacion cruzada.")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    rows = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(x_train, y_train), start=1):
        x_fold_train, y_fold_train = load_paths(
            [paths[index] for index in train_idx],
            y_train[train_idx],
            augment=True,
        )
        model = PureTensorFlowMLP(input_dim=x_train.shape[1], seed=RANDOM_STATE + fold)
        train_mlp(
            model,
            x_fold_train,
            y_fold_train,
            x_val=x_train[val_idx],
            y_val=y_train[val_idx],
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            l2_strength=L2_REGULARIZATION,
        )
        y_prob = predict(model, x_train[val_idx])
        row = {"fold": fold, **compute_metrics(y_train[val_idx], y_prob)}
        rows.append(row)

    results = pd.DataFrame(rows)
    summary = results.drop(columns=["fold"]).agg(["mean", "std"])
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(METRICS_DIR / "cross_validation_results.csv", index=False)
    summary.to_csv(METRICS_DIR / "cross_validation_summary.csv")
    print(results)
    print(summary)


if __name__ == "__main__":
    main()
