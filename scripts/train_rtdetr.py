#!/usr/bin/env python3
"""
Helper utilities shared by RT-DETR / YOLO trainers.

Responsible for:
  • Loading class metadata (names, nc)
  • Building per-run train/val/test splits in YOLO format
  • Writing Ultralytics dataset YAMLs
  • Resolving weight checkpoints
  • Extracting metrics from results.csv
  • Persisting artifacts (best/last checkpoints, results.csv)
  • Writing a simple metrics.txt summary
"""

from __future__ import annotations

import csv
import os
import random
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

CLASS_META_SOURCE = REPO_ROOT / "data" / "combined" / "data.yaml"

SOURCE_DATASETS = {
    "all": REPO_ROOT / "data" / "combined",
    "front": REPO_ROOT / "data" / "old" / "front",
    "scan": REPO_ROOT / "data" / "old" / "scan",
    "side": REPO_ROOT / "data" / "old" / "side",
}

SPLIT_NAMES = ("train", "valid", "test")


def load_class_metadata() -> Dict[str, List[str]]:
    if not CLASS_META_SOURCE.exists():
        raise FileNotFoundError(f"Class metadata YAML not found at {CLASS_META_SOURCE}")
    with CLASS_META_SOURCE.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    names = data.get("names")
    if not isinstance(names, list):
        raise ValueError("Expected 'names' list in metadata YAML.")
    return {"names": names, "nc": len(names)}


def _list_images(image_dir: Path) -> List[Path]:
    return sorted([p for p in image_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}])


def _ensure_symlink(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    try:
        rel = os.path.relpath(src, dest.parent)
        dest.symlink_to(rel)
    except OSError:
        shutil.copy2(src, dest)


def _build_simple_split(source_dir: Path, dest_root: Path, seed: int) -> Dict[str, int]:
    image_dir = source_dir / "images"
    label_dir = source_dir / "labels"
    if not image_dir.exists():
        raise FileNotFoundError(f"Expected 'images/' under {source_dir}")
    label_dir.mkdir(parents=True, exist_ok=True)

    images = _list_images(image_dir)
    if not images:
        raise ValueError(f"No images found in {image_dir}")

    rng = random.Random(seed)
    rng.shuffle(images)

    n = len(images)
    n_train = max(1, int(round(n * 0.8)))
    n_val = max(1, int(round(n * 0.1)))
    n_test = n - n_train - n_val
    if n_test <= 0:
        n_test = max(1, n - n_train - n_val)
        if n_train + n_val + n_test > n:
            n_train = max(1, n_train - 1)

    splits = {
        "train": images[:n_train],
        "valid": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:],
    }

    for split_name, split_images in splits.items():
        img_out = dest_root / split_name / "images"
        lbl_out = dest_root / split_name / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        for img_path in split_images:
            _ensure_symlink(img_path, img_out / img_path.name)
            label_src = label_dir / f"{img_path.stem}.txt"
            label_dest = lbl_out / f"{img_path.stem}.txt"
            if label_src.exists():
                _ensure_symlink(label_src, label_dest)
            else:
                label_dest.touch()

    return {split: len(paths) for split, paths in splits.items()}


def _split_from_structured(source_dir: Path) -> Dict[str, int]:
    counts = {}
    for split in SPLIT_NAMES:
        img_dir = source_dir / split / "images"
        if img_dir.exists():
            counts[split] = len(_list_images(img_dir))
    return counts


def build_split_dataset(dataset_dir: Path, seed: int) -> Tuple[Path, Dict[str, int]]:
    dataset_name = dataset_dir.name
    source_dir = SOURCE_DATASETS.get(dataset_name, dataset_dir)

    if not source_dir.exists():
        raise FileNotFoundError(f"Source dataset directory not found: {source_dir}")

    structured = (source_dir / "train" / "images").exists() and (source_dir / "valid" / "images").exists()
    if structured:
        split_root = source_dir.resolve()
        counts = _split_from_structured(split_root)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        return split_root, counts

    split_root = dataset_dir / f"splits_seed{seed}"
    if (split_root / "train" / "images").exists():
        counts = _split_from_structured(split_root)
        return split_root, counts

    split_root.mkdir(parents=True, exist_ok=True)
    counts = _build_simple_split(source_dir, split_root, seed)
    return split_root, counts


def write_dataset_yaml(dataset_dir: Path, split_root: Path, meta: Dict[str, Iterable[str]]) -> Path:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = dataset_dir / "dataset.yaml"
    data = {
        "path": str(split_root.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": int(meta["nc"]),
        "names": list(meta["names"]),
    }
    with yaml_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)
    return yaml_path


def resolve_weights_path(weights: str, repo_root: Path | None = None) -> Path:
    repo_root = repo_root or REPO_ROOT
    candidate = Path(weights)
    if candidate.is_file():
        return candidate.resolve()
    alt = repo_root / "weights" / weights
    if alt.is_file():
        return alt.resolve()
    return Path(weights)


def extract_metrics(results_csv_path: Path) -> Dict[str, float]:
    if not results_csv_path.exists():
        raise FileNotFoundError(f"results.csv not found at {results_csv_path}")
    with results_csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [row for row in reader]
    if not rows:
        return {}
    last = rows[-1]
    mapping = {
        "metrics/mAP50(B)": "map50",
        "metrics/mAP50-95(B)": "map5095",
        "metrics/precision(B)": "precision",
        "metrics/recall(B)": "recall",
    }
    metrics: Dict[str, float] = {}
    for key, out_key in mapping.items():
        value = last.get(key)
        if value in (None, ""):
            continue
        try:
            metrics[out_key] = float(value)
        except ValueError:
            continue
    return metrics


def persist_model_artifacts(run_dir: Path, dataset_dir: Path) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("best.pt", "last.pt", "results.csv"):
        src = run_dir / filename
        if src.exists():
            shutil.copy2(src, dataset_dir / filename)

    dataset_yaml = dataset_dir / "dataset.yaml"
    if dataset_yaml.exists():
        shutil.copy2(dataset_yaml, run_dir / "dataset.yaml")


def write_metrics_file(dataset_dir: Path, metrics: Dict[str, float], run_dir: Path, split_counts: Dict[str, int]) -> Path:
    metrics_path = dataset_dir / "metrics.txt"
    lines = [
        f"Run Directory: {run_dir}",
        "Split counts:",
    ]
    for split in SPLIT_NAMES:
        if split in split_counts:
            lines.append(f"  {split}: {split_counts[split]}")
    lines.extend(
        [
            "Metrics:",
            f"  mAP@50: {metrics.get('map50', 0.0):.4f}",
            f"  mAP@50-95: {metrics.get('map5095', 0.0):.4f}",
            f"  Precision: {metrics.get('precision', 0.0):.4f}",
            f"  Recall: {metrics.get('recall', 0.0):.4f}",
        ]
    )
    with metrics_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return metrics_path


__all__ = [
    "load_class_metadata",
    "build_split_dataset",
    "write_dataset_yaml",
    "resolve_weights_path",
    "extract_metrics",
    "persist_model_artifacts",
    "write_metrics_file",
]
