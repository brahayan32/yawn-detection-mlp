import argparse
import time

import numpy as np
import tensorflow as tf

from .config import IMAGE_SIZE, RANDOM_STATE
from .model import PureTensorFlowMLP, AdamOptimizer, train_step


def list_devices():
    print("CPU disponibles:")
    for device in tf.config.list_physical_devices("CPU"):
        print(f"- {device.name}")

    print("\nGPU disponibles:")
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("- No se detecto GPU compatible con TensorFlow/CUDA.")
    for device in gpus:
        print(f"- {device.name}")


def run_device_benchmark(device_name: str, steps: int, batch_size: int):
    input_dim = IMAGE_SIZE[0] * IMAGE_SIZE[1]
    rng = np.random.default_rng(RANDOM_STATE)
    x = rng.random((batch_size, input_dim), dtype=np.float32)
    y = rng.integers(0, 2, size=(batch_size,), dtype=np.int32)

    with tf.device(device_name):
        model = PureTensorFlowMLP(input_dim=input_dim, seed=RANDOM_STATE)
        optimizer = AdamOptimizer(model.trainable_variables)
        x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
        y_tensor = tf.convert_to_tensor(y, dtype=tf.float32)

        train_step(model, optimizer, x_tensor, y_tensor)
        start = time.perf_counter()
        for _ in range(steps):
            loss, acc = train_step(model, optimizer, x_tensor, y_tensor)
        elapsed = time.perf_counter() - start

    print(
        f"{device_name} | pasos: {steps} | batch: {batch_size} | "
        f"tiempo: {elapsed:.4f}s | loss: {float(loss.numpy()):.4f} | "
        f"accuracy: {float(acc.numpy()):.4f}"
    )


def main():
    parser = argparse.ArgumentParser(description="Prueba tecnica de CPU/GPU con TensorFlow puro.")
    parser.add_argument("--device", choices=["cpu", "gpu", "both"], default="both")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    list_devices()
    print("\nPrueba de rendimiento con datos sinteticos:")

    if args.device in {"cpu", "both"}:
        run_device_benchmark("/CPU:0", args.steps, args.batch_size)

    if args.device in {"gpu", "both"}:
        if tf.config.list_physical_devices("GPU"):
            run_device_benchmark("/GPU:0", args.steps, args.batch_size)
        else:
            print("/GPU:0 | omitido porque TensorFlow no detecto GPU.")


if __name__ == "__main__":
    main()
