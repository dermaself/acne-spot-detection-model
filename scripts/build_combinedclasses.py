#!/usr/bin/env python3
"""
Create a merged `combinedclasses` dataset by:
1. Copying images from the `front`, `scan`, and `side` views.
2. Renaming files with the source view as a prefix to avoid collisions.
3. Re-writing YOLO labels to collapse rarely used classes into broader buckets.
4. Emitting `data/combinedclasses/classes.yaml` so the trainer knows the new schema.

Usage:
    python scripts/build_combinedclasses.py
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data"
SOURCE_DATASETS = ("front", "scan", "side")


@dataclass(frozen=True)
class MergeConfig:
    """Defines how original classes map into the merged taxonomy."""

    groups: List[tuple[str, List[str]]]

    @property
    def new_class_names(self) -> List[str]:
        return [group for group, _ in self.groups]

    @property
    def orig_to_new(self) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for new_name, originals in self.groups:
            for orig in originals:
                if orig in mapping:
                    raise ValueError(f"Class '{orig}' mapped twice (check merge config).")
                mapping[orig] = new_name
        return mapping


DEFAULT_GROUPS: List[tuple[str, List[str]]] = [
    ("Comedones", ["Comedones"]),
    ("Microcysts", ["Microcysts"]),
    ("Freckles", ["Freckles"]),
    ("Mole", ["Mole"]),
    ("Papules", ["Papules"]),
    ("Post-Acne Scar", ["Post-Acne Scar"]),
    ("Post-Acne Spot", ["Post-Acne Spot", "Spot"]),
    ("Pustules", ["Pustules"]),
    ("Cysts/Nodules", ["Cysts", "Nodules"]),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge split datasets and collapse rare classes into broader groups."
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=DATA_ROOT / "combinedclasses",
        help="Output dataset root (will create images/ and labels/).",
    )
    parser.add_argument(
        "--merge-config",
        type=Path,
        default=None,
        help="Optional YAML file containing {new_class: [list_of_original_classes]} mapping.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print actions without writing files.",
    )
    return parser.parse_args()


def load_original_classes() -> List[str]:
    cfg_path = DATA_ROOT / "data.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing class configuration: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    names = payload.get("names")
    if not isinstance(names, list):
        raise ValueError(f"`names` missing or invalid inside {cfg_path}")
    return names


def load_merge_config(path: Path | None) -> MergeConfig:
    if path is None:
        return MergeConfig(groups=DEFAULT_GROUPS)
    if not path.exists():
        raise FileNotFoundError(f"Merge config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Merge config must be a mapping of new_class -> [old_class, ...].")
    groups: List[tuple[str, List[str]]] = []
    for new_name, originals in raw.items():
        if not isinstance(new_name, str):
            raise ValueError(f"Invalid new class name: {new_name!r}")
        if not isinstance(originals, Iterable):
            raise ValueError(f"Invalid value for {new_name}; expected list of original classes.")
        originals_list = [str(item) for item in originals]
        if not originals_list:
            raise ValueError(f"No source classes listed for '{new_name}'.")
        groups.append((new_name, originals_list))
    return MergeConfig(groups=groups)


def validate_sources() -> None:
    missing = [name for name in SOURCE_DATASETS if not (DATA_ROOT / name).exists()]
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(f"Missing dataset directories: {joined}")


def rewrite_label(
    label_path: Path,
    destination: Path,
    new_name_to_idx: Dict[str, int],
    orig_idx_to_new_name: Dict[int, str],
    dry_run: bool,
) -> None:
    lines_out: List[str] = []
    with label_path.open("r", encoding="utf-8") as src:
        for raw_line in src:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            parts = raw_line.split()
            try:
                class_idx = int(parts[0])
            except ValueError as exc:
                raise ValueError(f"Malformed label line '{raw_line}' in {label_path}") from exc
            new_name = orig_idx_to_new_name.get(class_idx)
            if new_name is None:
                raise KeyError(
                    f"Original class index {class_idx} from {label_path.name} is not mapped; "
                    "extend your merge config."
                )
            new_idx = new_name_to_idx[new_name]
            line_out = " ".join([str(new_idx), *parts[1:]])
            lines_out.append(line_out)
    if dry_run:
        print(f"[dry-run] would write {destination} with {len(lines_out)} labels")
        return
    with destination.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines_out) + ("\n" if lines_out else ""))


def copy_image(source: Path, destination: Path, dry_run: bool) -> None:
    if dry_run:
        print(f"[dry-run] would copy image {source} -> {destination}")
        return
    shutil.copy2(source, destination)


def build_dataset(dest_root: Path, merge_cfg: MergeConfig, dry_run: bool = False) -> None:
    validate_sources()
    original_classes = load_original_classes()
    orig_name_to_idx = {name: i for i, name in enumerate(original_classes)}
    new_class_names = merge_cfg.new_class_names
    new_name_to_idx = {name: i for i, name in enumerate(new_class_names)}

    orig_to_new = merge_cfg.orig_to_new
    missing_names = [name for name in original_classes if name not in orig_to_new]
    if missing_names:
        joined = ", ".join(missing_names)
        raise KeyError(f"Original classes missing from merge mapping: {joined}")

    # Map original class indices directly for quick lookup.
    orig_idx_to_new_name = {
        orig_name_to_idx[name]: dest
        for name, dest in orig_to_new.items()
    }

    images_dir = dest_root / "images"
    labels_dir = dest_root / "labels"
    if dry_run:
        print(f"[dry-run] would {'recreate' if dest_root.exists() else 'create'} {dest_root}")
    else:
        if dest_root.exists():
            shutil.rmtree(dest_root)
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

    for dataset_name in SOURCE_DATASETS:
        src_img_dir = DATA_ROOT / dataset_name / "images"
        src_lbl_dir = DATA_ROOT / dataset_name / "labels"
        if not src_img_dir.exists() or not src_lbl_dir.exists():
            raise FileNotFoundError(
                f"Expected images/labels directories under data/{dataset_name}"
            )
        for image_path in sorted(src_img_dir.iterdir()):
            if not image_path.is_file():
                continue
            stem = image_path.stem
            label_path = src_lbl_dir / f"{stem}.txt"
            if not label_path.exists():
                raise FileNotFoundError(f"Missing label for {image_path}: {label_path}")
            new_stem = f"{dataset_name}__{stem}"
            dest_img = images_dir / f"{new_stem}{image_path.suffix.lower()}"
            dest_lbl = labels_dir / f"{new_stem}.txt"
            if dest_img.exists() or dest_lbl.exists():
                raise FileExistsError(f"Collision detected for {new_stem} – aborting.")
            copy_image(image_path, dest_img, dry_run=dry_run)
            rewrite_label(
                label_path,
                dest_lbl,
                new_name_to_idx=new_name_to_idx,
                orig_idx_to_new_name=orig_idx_to_new_name,
                dry_run=dry_run,
            )

    classes_yaml = dest_root / "classes.yaml"
    payload = {"nc": len(new_class_names), "names": new_class_names}
    if dry_run:
        print(f"[dry-run] would write class schema to {classes_yaml}")
    else:
        with classes_yaml.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False)
    print(f"[done] Created merged dataset at {dest_root} with {len(new_class_names)} classes.")


def main() -> None:
    args = parse_args()
    merge_cfg = load_merge_config(args.merge_config)
    build_dataset(dest_root=args.dest, merge_cfg=merge_cfg, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
