#!/usr/bin/env python3
"""
Tile a YOLO-format dataset into 640×640 crops (25% overlap) across train/valid/test splits.

Usage:
    python scripts/tile_dataset.py --source data/combined --dest data/new

Features:
- Processes train/valid/test splits recursively, preserving structure.
- Images larger than 640×640 are tiled with stride 480.
- Smaller or equal images are letterboxed/upscaled to exactly 640×640.
- YOLO labels are clipped/re-mapped; empty tiles can be skipped (default) or kept.
- Copies data.yaml to the destination with updated paths.
"""

from __future__ import annotations

import argparse
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple

import yaml
from PIL import Image, ImageFile, UnidentifiedImageError

ImageFile.LOAD_TRUNCATED_IMAGES = True

TILE_SIZE = 640
STRIDE = 480  # 25% overlap
MIN_BOX_PIXELS = 1
ALLOWED_EXTS = {".jpg", ".jpeg", ".png"}
SPLITS = ("train", "valid", "test")


@dataclass(frozen=True)
class Box:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

    @classmethod
    def from_yolo(cls, line: str) -> "Box":
        parts = line.strip().split()
        if len(parts) != 5:
            raise ValueError(f"Malformed YOLO label line: '{line}'")
        return cls(
            class_id=int(parts[0]),
            x_center=float(parts[1]),
            y_center=float(parts[2]),
            width=float(parts[3]),
            height=float(parts[4]),
        )

    def to_yolo(self) -> str:
        return (
            f"{self.class_id} {self.x_center:.6f} {self.y_center:.6f} "
            f"{self.width:.6f} {self.height:.6f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tile a YOLO dataset for 640×640 training.")
    parser.add_argument("--source", type=Path, required=True, help="Root of the source dataset (with train/valid/test).")
    parser.add_argument("--dest", type=Path, required=True, help="Destination dataset root.")
    parser.add_argument(
        "--keep-empty",
        action="store_true",
        help="Keep tiles without annotations (default: drop empty tiles).",
    )
    return parser.parse_args()


def iter_positions(length: int, tile: int, stride: int) -> Iterator[int]:
    if tile >= length:
        yield 0
        return
    positions = list(range(0, length - tile + 1, stride))
    if positions[-1] != length - tile:
        positions.append(length - tile)
    seen = set()
    for pos in positions:
        if pos not in seen:
            seen.add(pos)
            yield pos


def load_boxes(label_path: Path) -> List[Box]:
    if not label_path.exists():
        return []
    with label_path.open("r", encoding="utf-8") as handle:
        lines = [ln.strip() for ln in handle.readlines() if ln.strip()]
    return [Box.from_yolo(line) for line in lines]


def box_to_pixels(box: Box, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    cx = box.x_center * img_w
    cy = box.y_center * img_h
    bw = box.width * img_w
    bh = box.height * img_h
    x1 = cx - bw / 2
    y1 = cy - bh / 2
    x2 = cx + bw / 2
    y2 = cy + bh / 2
    return x1, y1, x2, y2


def clip_box_to_tile(
    box: Box,
    img_w: int,
    img_h: int,
    tile_x: int,
    tile_y: int,
    tile_size: int,
) -> Box | None:
    x1, y1, x2, y2 = box_to_pixels(box, img_w, img_h)
    tile_x2 = tile_x + tile_size
    tile_y2 = tile_y + tile_size
    clipped_x1 = max(x1, tile_x)
    clipped_y1 = max(y1, tile_y)
    clipped_x2 = min(x2, tile_x2)
    clipped_y2 = min(y2, tile_y2)
    clipped_w = clipped_x2 - clipped_x1
    clipped_h = clipped_y2 - clipped_y1
    if clipped_w <= 0 or clipped_h <= 0:
        return None
    if clipped_w < MIN_BOX_PIXELS or clipped_h < MIN_BOX_PIXELS:
        return None
    new_cx = (clipped_x1 + clipped_x2) / 2
    new_cy = (clipped_y1 + clipped_y2) / 2
    return Box(
        class_id=box.class_id,
        x_center=max(0.0, min(1.0, (new_cx - tile_x) / tile_size)),
        y_center=max(0.0, min(1.0, (new_cy - tile_y) / tile_size)),
        width=max(0.0, min(1.0, clipped_w / tile_size)),
        height=max(0.0, min(1.0, clipped_h / tile_size)),
    )


def letterbox_image(
    img: Image.Image,
    boxes: List[Box],
) -> Tuple[Image.Image, List[Box]]:
    width, height = img.size
    scale = TILE_SIZE / max(width, height)
    resized_w = min(TILE_SIZE, int(round(width * scale)))
    resized_h = min(TILE_SIZE, int(round(height * scale)))
    resized = img.resize((resized_w, resized_h), Image.BILINEAR)
    canvas = Image.new("RGB", (TILE_SIZE, TILE_SIZE))
    pad_x = (TILE_SIZE - resized_w) // 2
    pad_y = (TILE_SIZE - resized_h) // 2
    canvas.paste(resized, (pad_x, pad_y))

    transformed: List[Box] = []
    for box in boxes:
        x1, y1, x2, y2 = box_to_pixels(box, width, height)
        x1 = x1 * scale + pad_x
        x2 = x2 * scale + pad_x
        y1 = y1 * scale + pad_y
        y2 = y2 * scale + pad_y
        clipped_w = x2 - x1
        clipped_h = y2 - y1
        if clipped_w < MIN_BOX_PIXELS or clipped_h < MIN_BOX_PIXELS:
            continue
        new_cx = max(0.0, min(1.0, (x1 + x2) / 2 / TILE_SIZE))
        new_cy = max(0.0, min(1.0, (y1 + y2) / 2 / TILE_SIZE))
        transformed.append(
            Box(
                class_id=box.class_id,
                x_center=new_cx,
                y_center=new_cy,
                width=max(0.0, min(1.0, clipped_w / TILE_SIZE)),
                height=max(0.0, min(1.0, clipped_h / TILE_SIZE)),
            )
        )
    return canvas, transformed


def process_image(
    img_path: Path,
    label_path: Path,
    dest_images: Path,
    dest_labels: Path,
    keep_empty: bool,
) -> int:
    boxes = load_boxes(label_path)
    try:
        with Image.open(img_path) as img_raw:
            img = img_raw.convert("RGB")
            width, height = img.size
            tile_idx = 0
            saved_tiles = 0
            if width <= TILE_SIZE and height <= TILE_SIZE:
                tile_img, tile_boxes = letterbox_image(img, boxes)
                if tile_boxes or keep_empty:
                    out_name = f"{img_path.stem}_0{img_path.suffix.lower()}"
                    lbl_name = f"{img_path.stem}_0.txt"
                    tile_img.save(dest_images / out_name)
                    write_label_file(dest_labels / lbl_name, tile_boxes)
                    saved_tiles += 1
            else:
                for ty in iter_positions(height, TILE_SIZE, STRIDE):
                    for tx in iter_positions(width, TILE_SIZE, STRIDE):
                        tile_boxes: List[Box] = []
                        for box in boxes:
                            clipped = clip_box_to_tile(box, width, height, tx, ty, TILE_SIZE)
                            if clipped is not None:
                                tile_boxes.append(clipped)
                        if not tile_boxes and not keep_empty:
                            continue
                        cropped = img.crop((tx, ty, tx + TILE_SIZE, ty + TILE_SIZE))
                        out_name = f"{img_path.stem}_{tile_idx}{img_path.suffix.lower()}"
                        lbl_name = f"{img_path.stem}_{tile_idx}.txt"
                        cropped.save(dest_images / out_name)
                        write_label_file(dest_labels / lbl_name, tile_boxes)
                        tile_idx += 1
                        saved_tiles += 1
            return saved_tiles
    except (UnidentifiedImageError, OSError) as exc:
        print(f"[warn] Skipping corrupted or unreadable file: {img_path} ({exc})")
        return 0


def write_label_file(path: Path, boxes: List[Box]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for box in boxes:
            handle.write(box.to_yolo() + "\n")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def process_split(
    split: str,
    source_root: Path,
    dest_root: Path,
    keep_empty: bool,
) -> Tuple[int, int]:
    src_images = source_root / split / "images"
    src_labels = source_root / split / "labels"
    if not src_images.exists():
        print(f"[warn] Source split '{split}' not found at {src_images}")
        return 0, 0

    dest_images = dest_root / split / "images"
    dest_labels = dest_root / split / "labels"
    ensure_dir(dest_images)
    ensure_dir(dest_labels)

    image_files = sorted(p for p in src_images.iterdir() if p.suffix.lower() in ALLOWED_EXTS)
    processed = 0
    total_tiles = 0
    for img_path in image_files:
        label_path = src_labels / f"{img_path.stem}.txt"
        tiles = process_image(img_path, label_path, dest_images, dest_labels, keep_empty)
        processed += 1
        total_tiles += tiles
    print(f"[{split}] processed {processed} images → {total_tiles} tiles")
    return processed, total_tiles


def copy_data_yaml(source: Path, dest: Path) -> None:
    src_yaml = source / "data.yaml"
    if not src_yaml.exists():
        print(f"[warn] data.yaml not found at {src_yaml}; skipping copy.")
        return
    with src_yaml.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    data = data or {}
    data["path"] = "."
    data["train"] = "train/images"
    if "val" in data:
        data["val"] = "valid/images"
    elif "valid" in data:
        data["valid"] = "valid/images"
    else:
        data["val"] = "valid/images"
    if "test" in data:
        data["test"] = "test/images"

    if "names" in data:
        data["nc"] = len(data["names"])

    with (dest / "data.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def main() -> None:
    args = parse_args()
    start_time = time.perf_counter()

    source = args.source.resolve()
    dest = args.dest.resolve()

    if not source.exists():
        raise FileNotFoundError(f"Source dataset not found: {source}")

    if dest.exists():
        shutil.rmtree(dest)
    for split in SPLITS:
        ensure_dir(dest / split / "images")
        ensure_dir(dest / split / "labels")

    total_images = 0
    total_tiles = 0
    for split in SPLITS:
        processed, tiles = process_split(split, source, dest, keep_empty=args.keep_empty)
        total_images += processed
        total_tiles += tiles

    copy_data_yaml(source, dest)

    elapsed = time.perf_counter() - start_time
    print(f"[done] total {total_images} images → {total_tiles} tiles")
    print(f"Output saved to {dest}")
    print(f"Elapsed time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
