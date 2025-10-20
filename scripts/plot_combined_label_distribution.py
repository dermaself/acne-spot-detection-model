#!/usr/bin/env python3
"""
Plot the class distribution after applying the same class merges used in build_combinedclasses.py.

Example:
    python scripts/plot_combined_label_distribution.py --output plots/combined_label_distribution.png
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import yaml

from build_combinedclasses import MergeConfig, load_merge_config, DEFAULT_GROUPS, load_original_classes

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data"
SPLITS = ("front", "scan", "side")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot label distribution with merged classes.")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "plots" / "combined_label_distribution.png",
        help="Path to save the plot image.",
    )
    parser.add_argument(
        "--merge-config",
        type=Path,
        default=None,
        help="Optional YAML mapping for custom class merges (same format as build_combinedclasses).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plot interactively.",
    )
    return parser.parse_args()


def load_class_mapping(merge_cfg: MergeConfig) -> Dict[int, str]:
    original_classes = load_original_classes()
    name_to_idx = {name: idx for idx, name in enumerate(original_classes)}
    mapping: Dict[int, str] = {}
    for new_name, originals in merge_cfg.groups:
        for orig in originals:
            if orig not in name_to_idx:
                raise KeyError(f"Original class '{orig}' not found in data/data.yaml")
            mapping[name_to_idx[orig]] = new_name
    missing = [name for name in original_classes if name not in {orig for _, group in merge_cfg.groups for orig in group}]
    if missing:
        raise KeyError(f"The following classes are not covered by the merge mapping: {', '.join(missing)}")
    return mapping


def count_labels_with_merge(dataset: str, idx_to_new: Dict[int, str]) -> Counter:
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
                new_name = idx_to_new.get(cls_idx)
                if new_name is None:
                    raise ValueError(f"Class index {cls_idx} from {txt} not found in merge mapping.")
                counts[new_name] += 1
    return counts


def main() -> None:
    args = parse_args()
    merge_cfg = load_merge_config(args.merge_config)
    idx_to_new = load_class_mapping(merge_cfg)
    merged_names = [group for group, _ in merge_cfg.groups]

    per_split_counts: Dict[str, Counter] = {}
    for split in SPLITS:
        per_split_counts[split] = count_labels_with_merge(split, idx_to_new)

    x = list(range(len(merged_names)))
    bar_width = 0.25
    offsets = {"front": -bar_width, "scan": 0.0, "side": bar_width}

    fig, ax = plt.subplots(figsize=(max(10, len(merged_names) * 1.2), 6))
    for split, offset in offsets.items():
        counts = [per_split_counts[split].get(name, 0) for name in merged_names]
        positions = [pos + offset for pos in x]
        ax.bar(positions, counts, width=bar_width, label=split.capitalize())

    ax.set_xticks(x)
    ax.set_xticklabels(merged_names, rotation=45, ha="right")
    ax.set_ylabel("Number of annotations")
    ax.set_title("Merged Class Label Distribution per Split")
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
