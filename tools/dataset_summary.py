from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "datasets"
METRICS_DIR = PROJECT_ROOT / "metrics"
CSV_PATH = METRICS_DIR / "dataset_summary.csv"
PLOT_PATH = METRICS_DIR / "dataset_distribution.png"

SPLITS = ("train", "validation", "test")
CLASSES = ("no_yawn", "yawn")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def count_images(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(1 for path in folder.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)


def collect_counts() -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        counts[split] = {}
        for class_name in CLASSES:
            counts[split][class_name] = count_images(DATASET_DIR / split / class_name)
    return counts


def write_csv(counts: dict[str, dict[str, int]]) -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    total_images = sum(sum(classes.values()) for classes in counts.values())

    with CSV_PATH.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "split",
                "class",
                "count",
                "percentage_in_split",
                "percentage_total",
            ],
        )
        writer.writeheader()

        for split, class_counts in counts.items():
            split_total = sum(class_counts.values())
            for class_name, count in class_counts.items():
                writer.writerow(
                    {
                        "split": split,
                        "class": class_name,
                        "count": count,
                        "percentage_in_split": round((count / split_total * 100) if split_total else 0, 2),
                        "percentage_total": round((count / total_images * 100) if total_images else 0, 2),
                    }
                )


def write_plot(counts: dict[str, dict[str, int]]) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    x_positions = range(len(SPLITS))
    width = 0.35
    no_yawn_counts = [counts[split]["no_yawn"] for split in SPLITS]
    yawn_counts = [counts[split]["yawn"] for split in SPLITS]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([x - width / 2 for x in x_positions], no_yawn_counts, width, label="no_yawn", color="#2f80ed")
    ax.bar([x + width / 2 for x in x_positions], yawn_counts, width, label="yawn", color="#f2994a")

    ax.set_title("Distribucion del dataset por split y clase")
    ax.set_xlabel("Split")
    ax.set_ylabel("Cantidad de imagenes")
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(SPLITS)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    for container in ax.containers:
        ax.bar_label(container, padding=3)

    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=160)
    plt.close(fig)
    return True


def print_summary(counts: dict[str, dict[str, int]], plot_created: bool) -> None:
    class_totals: defaultdict[str, int] = defaultdict(int)
    total_images = 0
    for split in SPLITS:
        split_total = sum(counts[split].values())
        total_images += split_total
        print(f"{split}: {split_total} imagenes")
        for class_name in CLASSES:
            count = counts[split][class_name]
            class_totals[class_name] += count
            print(f"  - {class_name}: {count}")

    print(f"Total: {total_images} imagenes")
    for class_name in CLASSES:
        print(f"Total {class_name}: {class_totals[class_name]}")

    if total_images:
        for class_name in CLASSES:
            percentage = class_totals[class_name] / total_images * 100
            print(f"Porcentaje {class_name}: {percentage:.2f}%")

    difference = abs(class_totals["yawn"] - class_totals["no_yawn"])
    print(f"Diferencia absoluta entre clases: {difference} imagenes")

    minimum_per_class = 300
    for class_name in CLASSES:
        status = "cumple" if class_totals[class_name] >= minimum_per_class else "no cumple"
        print(f"Minimo de {minimum_per_class} imagenes para {class_name}: {status}")

    print(f"CSV generado: {CSV_PATH}")
    if plot_created:
        print(f"Grafica generada: {PLOT_PATH}")
    else:
        print("Grafica no generada: matplotlib no esta disponible.")


def main() -> None:
    counts = collect_counts()
    write_csv(counts)
    plot_created = write_plot(counts)
    print_summary(counts, plot_created)


if __name__ == "__main__":
    main()
