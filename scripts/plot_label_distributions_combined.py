#!/usr/bin/env python3
"""
Plot original and merged class label distributions for front/scan/side on stacked axes.

Top subplot: original 11-class schema.
Bottom subplot: merged schema matching build_combinedclasses.py.

Bars are grouped by split with pastel colors and counts annotated above each bar.

Example:
    python scripts/plot_label_distributions_combined.py --output plots/label_distributions_combined.png
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import yaml

from build_combinedclasses import load_merge_config, load_original_classes

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data"
SPLITS = ("front", "scan", "side")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot original and merged label distributions.")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "plots" / "label_distributions_combined.png",
        help="File path to save the figure.",
    )
    parser.add_argument(
        "--merge-config",
        type=Path,
        default=None,
        help="Optional YAML merge config (same as build_combinedclasses.py).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively.",
    )
    return parser.parse_args()


def load_class_names() -> List[str]:
    cfg_path = DATA_ROOT / "data.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing class config: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    names = config.get("names")
    if not isinstance(names, list):
        raise ValueError("`names` must be a list in data/data.yaml")
    return [str(name) for name in names]


def count_labels(dataset: str, num_classes: int) -> Counter:
    labels_dir = DATA_ROOT / dataset / "labels"
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")
    counts: Counter = Counter()
    for txt in labels_dir.glob("*.txt"):
        with txt.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                try:
                    cls_idx = int(parts[0])
                except ValueError as exc:
                    raise ValueError(f"Malformed label in {txt}: '{line}'") from exc
                if not 0 <= cls_idx < num_classes:
                    raise ValueError(
                        f"Class index {cls_idx} from {txt} outside [0, {num_classes - 1}]"
                    )
                counts[cls_idx] += 1
    return counts


def build_merge_mapping(merge_cfg) -> Dict[int, str]:
    original_names = load_original_classes()
    name_to_idx = {name: idx for idx, name in enumerate(original_names)}
    mapping: Dict[int, str] = {}
    covered = set()
    for new_name, originals in merge_cfg.groups:
        for orig in originals:
            if orig not in name_to_idx:
                raise KeyError(f"Class '{orig}' not found in original names.")
            idx = name_to_idx[orig]
            mapping[idx] = new_name
            covered.add(orig)
    missing = [name for name in original_names if name not in covered]
    if missing:
        raise KeyError(f"Merge mapping missing classes: {', '.join(missing)}")
    return mapping


def count_labels_merged(dataset: str, idx_to_merged: Dict[int, str]) -> Counter:
    labels_dir = DATA_ROOT / dataset / "labels"
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")
    counts: Counter = Counter()
    for txt in labels_dir.glob("*.txt"):
        with txt.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                try:
                    cls_idx = int(parts[0])
                except ValueError as exc:
                    raise ValueError(f"Malformed label in {txt}: '{line}'") from exc
                new_name = idx_to_merged.get(cls_idx)
                if new_name is None:
                    raise ValueError(f"Class index {cls_idx} not in merge mapping.")
                counts[new_name] += 1
    return counts


def get_pastel_colors(n: int) -> List[str]:
    pastels = [
        "#AEC6CF",  # pastel blue
        "#FFB347",  # pastel orange
        "#CFCFC4",  # pastel gray
        "#77DD77",  # pastel green
        "#FF6961",  # pastel red
        "#FDFD96",  # pastel yellow
    ]
    if n <= len(pastels):
        return pastels[:n]
    # If more colors needed, use a light colormap.
    cmap = plt.get_cmap("Pastel1")
    return [cmap(i % cmap.N) for i in range(n)]


def annotate_bars(ax, bars):
    for bar in bars:
        height = bar.get_height()
        if height == 0:
            continue
        ax.annotate(
            f"{int(height)}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#333333",
        )


def plot_grouped_bars(ax, categories: List[str], per_split_counts: Dict[str, List[int]], title: str) -> None:
    x = list(range(len(categories)))
    bar_width = 0.25
    offsets = {"front": -bar_width, "scan": 0.0, "side": bar_width}
    colors = {split: color for split, color in zip(SPLITS, get_pastel_colors(len(SPLITS)))}

    for split, offset in offsets.items():
        counts = per_split_counts[split]
        positions = [pos + offset for pos in x]
        bars = ax.bar(
            positions,
            counts,
            width=bar_width,
            label=split.capitalize(),
            color=colors[split],
            edgecolor="white",
        )
        annotate_bars(ax, bars)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha="right")
    ax.set_ylabel("Annotations")
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend()


def main() -> None:
    args = parse_args()

    # Original counts
    original_names = load_class_names()
    num_classes = len(original_names)
    original_counts: Dict[str, List[int]] = {}
    for split in SPLITS:
        counter = count_labels(split, num_classes)
        original_counts[split] = [counter.get(idx, 0) for idx in range(num_classes)]

    # Merged counts
    merge_cfg = load_merge_config(args.merge_config)
    idx_to_merged = build_merge_mapping(merge_cfg)
    merged_names = [group for group, _ in merge_cfg.groups]
    merged_counts: Dict[str, List[int]] = {}
    for split in SPLITS:
        counter = count_labels_merged(split, idx_to_merged)
        merged_counts[split] = [counter.get(name, 0) for name in merged_names]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(max(12, len(original_names) * 1.2), 10),
        constrained_layout=True,
    )

    plot_grouped_bars(
        axes[0],
        original_names,
        original_counts,
        "Original Class Label Distribution",
    )

    plot_grouped_bars(
        axes[1],
        merged_names,
        merged_counts,
        "Merged Class Label Distribution",
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=220)
        print(f"[info] Plot saved to {args.output}")
    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
