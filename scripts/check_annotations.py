#!/usr/bin/env python3

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import yaml


DEFAULT_SAMPLE_SIZE = 10


def load_class_names(data_yaml: Path) -> Sequence[str]:
    if not data_yaml.exists():
        raise FileNotFoundError(f"Missing data.yaml at {data_yaml}")

    with data_yaml.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    names = config.get("names")
    if names is None:
        raise ValueError(f"No 'names' entry found in {data_yaml}")

    if isinstance(names, dict):
        try:
            ordered_keys = sorted(names.keys(), key=lambda key: int(key))
        except ValueError:
            ordered_keys = sorted(names.keys())
        return [names[key] for key in ordered_keys]

    if isinstance(names, Iterable):
        return list(names)

    raise TypeError(f"Unsupported 'names' format in {data_yaml}: {type(names)!r}")


def load_annotations(label_path: Path) -> List[List[float]]:
    if not label_path.exists():
        return []

    annotations: List[List[float]] = []
    with label_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            parts = raw_line.strip().split()
            if len(parts) != 5:
                continue
            try:
                numbers = [float(value) for value in parts]
            except ValueError:
                continue
            annotations.append(numbers)
    return annotations


def to_pixel_box(
    cx: float, cy: float, width: float, height: float, image_width: int, image_height: int
) -> tuple[float, float, float, float]:
    box_width = width * image_width
    box_height = height * image_height
    center_x = cx * image_width
    center_y = cy * image_height

    top_left_x = center_x - box_width / 2
    top_left_y = center_y - box_height / 2
    return top_left_x, top_left_y, box_width, box_height


def pick_samples(files: Sequence[Path], count: int) -> Sequence[Path]:
    if not files:
        return []

    if count >= len(files):
        return list(files)

    return random.sample(files, count)


def build_color_map(class_names: Sequence[str]) -> Dict[int, str]:
    cmap = plt.get_cmap("tab20")
    total = len(class_names)
    colors: Dict[int, str] = {}
    for idx in range(total):
        colors[idx] = cmap(idx % cmap.N)
    return colors


def annotate_image(
    image_path: Path,
    label_path: Path,
    class_names: Sequence[str],
    colors: Dict[int, str],
    output_path: Path,
) -> None:
    annotations = load_annotations(label_path)
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    figure, axis = plt.subplots(figsize=(12, 12))
    axis.imshow(image)
    axis.axis("off")

    legend_handles = []
    used_classes = set()

    for annotation in annotations:
        class_idx = int(annotation[0])
        if class_idx < 0 or class_idx >= len(class_names):
            continue

        cx, cy, box_w, box_h = annotation[1:]
        x, y, w, h = to_pixel_box(cx, cy, box_w, box_h, width, height)

        color = colors[class_idx]
        rect = Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor="none")
        axis.add_patch(rect)
        axis.text(
            x,
            y,
            class_names[class_idx],
            color=color,
            fontsize=10,
            bbox=dict(facecolor="black", alpha=0.4, pad=2),
        )

        if class_idx not in used_classes:
            legend_handles.append((rect, class_idx))
            used_classes.add(class_idx)

    if legend_handles:
        handles, class_indices = zip(*legend_handles)
        axis.legend(
            handles,
            [class_names[idx] for idx in class_indices],
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(figure)

def run(
    images_dir: Path,
    labels_dir: Path,
    output_dir: Path,
    data_yaml: Path,
    sample_size: int,
) -> int:
    random.seed()
    if not images_dir.exists():
        print(f"Images directory not found: {images_dir}", file=sys.stderr)
        return 1
    if not labels_dir.exists():
        print(f"Labels directory not found: {labels_dir}", file=sys.stderr)
        return 1

    image_files = sorted(
        path
        for path in images_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not image_files:
        print(f"No supported image files found in {images_dir}", file=sys.stderr)
        return 1

    if sample_size is None or sample_size <= 0:
        selected_images = image_files
    else:
        selected_images = pick_samples(image_files, sample_size)

    class_names = load_class_names(data_yaml)
    color_map = build_color_map(class_names)

    for index, image_path in enumerate(selected_images, start=1):
        label_path = labels_dir / f"{image_path.stem}.txt"
        output_path = output_dir / f"{image_path.stem}.png"
        annotate_image(image_path, label_path, class_names, color_map, output_path)
        print(f"[{index}/{len(selected_images)}] Saved {output_path}")

    return 0


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize YOLO labels by drawing boxes on images.")
    parser.add_argument(
        "--images",
        type=Path,
        required=True,
        help="Directory containing images.",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="Directory containing YOLO label files (.txt).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory to save annotated images.",
    )
    parser.add_argument(
        "--data-yaml",
        type=Path,
        required=True,
        help="Path to data.yaml containing class names.",
    )
    parser.add_argument(
        "-n",
        "--num",
        type=int,
        default=0,
        help="Number of images to annotate (0 or negative means all).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    return run(
        images_dir=args.images,
        labels_dir=args.labels,
        output_dir=args.output,
        data_yaml=args.data_yaml,
        sample_size=args.num,
    )


if __name__ == "__main__":
    raise SystemExit(main())
