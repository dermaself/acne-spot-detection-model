#!/usr/bin/env python3
"""
Copy YOLO label files from the consolidated `data/alllabels` directory into the
`labels` subdirectory for each dataset split (`front`, `scan`, `side`). A label
is matched to an image when the filenames share the same stem.
"""

from __future__ import annotations

import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SPLITS = ("front", "scan", "side")


def copy_labels(data_root: Path) -> None:
    source_dir = data_root / "alllabels"
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Label source directory not found: {source_dir}")

    total_copied = 0
    total_missing = 0

    for split in SPLITS:
        images_dir = data_root / split / "images"
        labels_dir = data_root / split / "labels"

        if not images_dir.is_dir():
            print(f"[warn] Skipping {split!r}: images directory missing at {images_dir}")
            continue

        labels_dir.mkdir(parents=True, exist_ok=True)

        copied = 0
        missing = 0

        for image_path in images_dir.iterdir():
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            label_name = f"{image_path.stem}.txt"
            source_label = source_dir / label_name
            target_label = labels_dir / label_name

            if not source_label.exists():
                missing += 1
                continue

            shutil.copy2(source_label, target_label)
            copied += 1

        total_copied += copied
        total_missing += missing

        print(
            f"[split:{split}] copied={copied} "
            f"missing_labels={missing}"
        )

    print(
        f"[done] total_copied={total_copied} "
        f"total_missing_labels={total_missing}"
    )


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    data_root = repo_root / "data"
    copy_labels(data_root)


if __name__ == "__main__":
    main()
