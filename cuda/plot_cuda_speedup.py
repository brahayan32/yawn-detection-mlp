from __future__ import annotations

import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "metrics" / "cuda_benchmark.csv"
PLOT_PATH = PROJECT_ROOT / "metrics" / "cuda_speedup.png"


def read_rows() -> list[dict[str, str]]:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"No existe {CSV_PATH}. Ejecuta primero el benchmark CUDA.")
    with CSV_PATH.open("r", encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def main() -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib no esta instalado. Instala las dependencias del proyecto.") from exc

    rows = read_rows()
    cuda_rows = [row for row in rows if row["version"] == "cuda"]
    if not cuda_rows:
        raise SystemExit("El CSV no contiene filas con version=cuda.")

    batch_sizes = [int(row["batch_size"]) for row in cuda_rows]
    speedups = [float(row["speedup"]) for row in cuda_rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(batch_sizes, speedups, marker="o", linewidth=2.5, color="#7c3aed", label="CUDA")
    ax.axhline(1.0, color="#555555", linestyle="--", linewidth=1.2, label="CPU base")
    ax.set_title("Speedup CUDA vs CPU - Forward pass MLP")
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Speedup")
    ax.set_xticks(batch_sizes)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()

    for x_value, y_value in zip(batch_sizes, speedups):
        ax.annotate(f"{y_value:.2f}x", (x_value, y_value), textcoords="offset points", xytext=(0, 8), ha="center")

    fig.tight_layout()
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_PATH, dpi=160)
    plt.close(fig)
    print(f"Grafica generada: {PLOT_PATH}")


if __name__ == "__main__":
    main()
