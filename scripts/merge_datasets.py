#!/usr/bin/env python3

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import shutil
import yaml


DATASETS = ("new", "closeup")
SPLITS = ("train", "valid", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    root: Path
    class_names: Sequence[str]


def load_class_names(data_yaml: Path) -> List[str]:
    with data_yaml.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if not data or "names" not in data:
        raise ValueError(f"Missing 'names' entry in {data_yaml}")

    names = data["names"]
    if isinstance(names, dict):
        try:
            ordered_keys = sorted(names.keys(), key=lambda key: int(key))
        except ValueError:
            ordered_keys = sorted(names.keys())
        return [names[key] for key in ordered_keys]

    if isinstance(names, Iterable):
        return list(names)

    raise TypeError(f"Unsupported 'names' format in {data_yaml}: {type(names)!r}")


def build_dataset_specs(data_root: Path) -> List[DatasetSpec]:
    specs: List[DatasetSpec] = []
    for dataset_name in DATASETS:
        dataset_root = data_root / dataset_name
        if not dataset_root.exists():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_root}")

        names = load_class_names(dataset_root / "data.yaml")
        specs.append(DatasetSpec(dataset_name, dataset_root, names))

    return specs


def compute_combined_names(specs: Sequence[DatasetSpec]) -> List[str]:
    combined: List[str] = []
    for spec in specs:
        for label in spec.class_names:
            if label not in combined:
                combined.append(label)
    return combined


def prepare_destination(dest_root: Path) -> None:
    if dest_root.exists():
        shutil.rmtree(dest_root)

    for split in SPLITS:
        (dest_root / split / "images").mkdir(parents=True, exist_ok=True)
        (dest_root / split / "labels").mkdir(parents=True, exist_ok=True)


def collect_image_tasks(specs: Sequence[DatasetSpec]) -> List[tuple[str, str, Path, Path]]:
    tasks: List[tuple[str, str, Path, Path]] = []
    for spec in specs:
        for split in SPLITS:
            images_dir = spec.root / split / "images"
            labels_dir = spec.root / split / "labels"
            if not images_dir.exists() or not labels_dir.exists():
                continue

            for image_path in sorted(images_dir.iterdir()):
                if not image_path.is_file():
                    continue
                if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue

                label_path = labels_dir / f"{image_path.stem}.txt"
                tasks.append((spec.name, split, image_path, label_path))

    return tasks


def convert_label(label_path: Path, class_map: Dict[int, int]) -> List[str]:
    if not label_path.exists():
        return []

    remapped_lines: List[str] = []
    with label_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 5:
                raise ValueError(f"Unexpected label format in {label_path}: {line!r}")

            old_idx = int(parts[0])
            if old_idx not in class_map:
                raise KeyError(f"Class index {old_idx} missing in mapping for {label_path}")

            new_idx = class_map[old_idx]
            remapped_lines.append(" ".join([str(new_idx), *parts[1:]]))

    return remapped_lines


def write_data_yaml(dest_yaml: Path, class_names: Sequence[str]) -> None:
    payload = {
        "path": ".",
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(class_names),
        "names": list(class_names),
    }
    with dest_yaml.open("w", encoding="utf-8") as handle:
        yaml.dump(payload, handle, sort_keys=False)


def run() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    data_root = repo_root / "data"
    dest_root = data_root / "combined"

    specs = build_dataset_specs(data_root)
    combined_names = compute_combined_names(specs)
    class_maps: Dict[str, Dict[int, int]] = {}
    for spec in specs:
        mapping = {}
        for idx, label in enumerate(spec.class_names):
            try:
                mapping[idx] = combined_names.index(label)
            except ValueError as exc:
                raise ValueError(f"Unable to find label '{label}' in combined class list") from exc
        class_maps[spec.name] = mapping

    prepare_destination(dest_root)

    tasks = collect_image_tasks(specs)
    total = len(tasks)
    if total == 0:
        print("No images found to process.", file=sys.stderr)
        write_data_yaml(dest_root / "data.yaml", combined_names)
        return 1

    skipped = 0
    copied = 0
    for index, (dataset_name, split, image_path, label_path) in enumerate(tasks, start=1):
        percent = (index / total) * 100
        print(f"{index}/{total} ({percent:.1f}%)", end="\r", flush=True)

        remapped = convert_label(label_path, class_maps[dataset_name])
        if not remapped:
            skipped += 1
            continue

        base_name = f"{dataset_name}__{image_path.stem}"
        dest_image = dest_root / split / "images" / f"{base_name}{image_path.suffix.lower()}"
        dest_label = dest_root / split / "labels" / f"{base_name}.txt"

        shutil.copy2(image_path, dest_image)
        with dest_label.open("w", encoding="utf-8") as handle:
            handle.write("\n".join(remapped) + "\n")
        copied += 1

    print()  # ensure trailing newline after progress line
    write_data_yaml(dest_root / "data.yaml", combined_names)

    print(f"Copied {copied} files into {dest_root}")
    print(f"Skipped {skipped} files without labels")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(run())
    except Exception as exc:  # surface rich message but exit non-zero
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
