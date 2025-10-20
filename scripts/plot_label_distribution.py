#!/usr/bin/env python3
"""
Visualize class label distributions for the `front`, `scan`, and `side` splits.

Example:
    python scripts/plot_label_distribution.py --output plots/label_distribution.png
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data"
SPLITS = ("front", "scan", "side")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot class count distributions for the front/scan/side datasets."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "plots" / "label_distribution.png",
        help="Where to save the generated bar plot.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot window instead of (or in addition to) saving it.",
    )
    return parser.parse_args()


def load_class_names() -> List[str]:
    cfg_path = DATA_ROOT / "data.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing class config at {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    names = config.get("names")
    if not isinstance(names, list):
        raise ValueError("Class names must be defined as a list in data/data.yaml.")
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
                        f"Class index {cls_idx} in {txt} outside [0, {num_classes - 1}]"
                    )
                counts[cls_idx] += 1
    return counts


def main() -> None:
    args = parse_args()
    class_names = load_class_names()
    num_classes = len(class_names)

    # Collect per-split counts keyed by class index.
    per_split_counts: Dict[str, Counter] = {}
    for split in SPLITS:
        per_split_counts[split] = count_labels(split, num_classes)

    # Prepare data for plotting.
    x = list(range(num_classes))
    bar_width = 0.25
    offsets = {
        "front": -bar_width,
        "scan": 0.0,
        "side": bar_width,
    }

    fig, ax = plt.subplots(figsize=(max(10, num_classes * 1.2), 6))
    for split, offset in offsets.items():
        counts = [per_split_counts[split].get(idx, 0) for idx in x]
        positions = [pos + offset for pos in x]
        ax.bar(positions, counts, width=bar_width, label=split.capitalize())

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_ylabel("Number of annotations")
    ax.set_title("Label Distribution per Split")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=200)
        print(f"[info] Plot saved to {args.output}")
    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
