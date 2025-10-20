#!/usr/bin/env python3

"""Generate quick diagnostics for the combined dataset."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
from PIL import Image


DATA_ROOT = Path("data")
COMBINED_ROOT = DATA_ROOT / "combined"
PLOTS_ROOT = DATA_ROOT / "plots"
SPLITS = ("train", "valid", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def iter_image_files(root: Path) -> Iterable[Tuple[str, Path, Path]]:
    for split in SPLITS:
        images_dir = root / split / "images"
        labels_dir = root / split / "labels"
        if not images_dir.exists():
            continue
        for image_path in sorted(images_dir.iterdir()):
            if not image_path.is_file():
                continue
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            label_path = labels_dir / f"{image_path.stem}.txt"
            yield split, image_path, label_path


def collect_statistics(root: Path) -> tuple[Counter, List[int]]:
    resolution_counts: Counter = Counter()
    label_counts: List[int] = []

    for _, image_path, label_path in iter_image_files(root):
        with Image.open(image_path) as image:
            width, height = image.size
        resolution_key = f"{width}x{height}"
        resolution_counts[resolution_key] += 1

        count = 0
        if label_path.exists():
            with label_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        count += 1
        label_counts.append(count)

    return resolution_counts, label_counts


def plot_resolution_counts(resolution_counts: Counter, output_path: Path) -> None:
    if not resolution_counts:
        print("No resolution data to plot.")
        return

    ordered = resolution_counts.most_common()
    top_limit = 20
    if len(ordered) > top_limit:
        top = ordered[:top_limit]
        others_total = sum(count for _, count in ordered[top_limit:])
        top.append(("Other", others_total))
    else:
        top = ordered

    labels, counts = zip(*top)

    plt.figure(figsize=(10, 6))
    plt.bar(labels, counts, color="#4C72B0")
    plt.title("Image Resolution Counts")
    plt.xlabel("Resolution (width x height)")
    plt.ylabel("Number of Images")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_label_histogram(label_counts: List[int], output_path: Path) -> None:
    if not label_counts:
        print("No label data to plot.")
        return

    plt.figure(figsize=(8, 6))
    plt.hist(label_counts, bins=range(0, max(label_counts) + 2), color="#55A868", edgecolor="black")
    plt.title("Labels per Image")
    plt.xlabel("Number of Labels")
    plt.ylabel("Image Count")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def write_resolution_text(resolution_counts: Counter, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        if not resolution_counts:
            handle.write("No resolution data available.\n")
            return

        handle.write("resolution,count\n")
        for resolution, count in resolution_counts.most_common():
            handle.write(f"{resolution},{count}\n")


def main() -> int:
    if not COMBINED_ROOT.exists():
        print(f"Combined dataset not found at {COMBINED_ROOT}")
        return 1

    resolution_counts, label_counts = collect_statistics(COMBINED_ROOT)

    plot_resolution_counts(resolution_counts, PLOTS_ROOT / "resolution_counts.png")
    plot_label_histogram(label_counts, PLOTS_ROOT / "labels_per_image_hist.png")
    write_resolution_text(resolution_counts, PLOTS_ROOT / "resolutions.txt")

    print(f"Saved plots to {PLOTS_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
