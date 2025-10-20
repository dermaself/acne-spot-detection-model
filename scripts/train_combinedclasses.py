#!/usr/bin/env python3
"""
Convenience wrapper to launch RT-DETR training on the merged & tiled dataset with tuned defaults.

Pipeline:
1. Run `python scripts/build_combinedclasses.py` to merge views + collapse classes.
2. Run `python scripts/tile_dataset.py --source data/combinedclasses --dest data/combinedclasses_tiled`.
3. Launch this script to train on the tiled dataset.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RT-DETR on the combined, tiled dataset.")
    parser.add_argument(
        "--dataset",
        default="combinedclasses_tiled",
        help="Dataset directory under data/ to train on.",
    )
    parser.add_argument("--weights", default="rtdetr-x.pt", help="Weights checkpoint to start from.")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=1024, help="Training image size (match tile size).")
    parser.add_argument("--batch", type=int, default=6, help="Batch size (adjust to fit GPU memory).")
    parser.add_argument("--device", default="cuda", help="Device string passed to Ultralytics (e.g. '0', 'cuda').")
    parser.add_argument("--workers", type=int, default=8, help="Data loader workers.")
    parser.add_argument("--seed", type=int, default=42, help="Reproducibility seed.")
    parser.add_argument("--mosaic", type=float, default=0.5, help="Mosaic augmentation probability.")
    parser.add_argument("--mixup", type=float, default=0.15, help="MixUp augmentation probability.")
    parser.add_argument("--copy-paste", type=float, default=0.1, help="Copy-Paste augmentation probability.")
    parser.add_argument(
        "--rect",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use rectangular training batches (recommended for varied aspect ratios).",
    )
    parser.add_argument(
        "--multi-scale",
        dest="multi_scale",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable multi-scale training for robustness.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Optional run name override.",
    )
    return parser.parse_args()


def format_bool_flag(base: str, enabled: bool) -> str:
    prefix = "" if enabled else "no-"
    return f"--{prefix}{base}"


def main() -> None:
    args = parse_args()
    dataset_dir = REPO_ROOT / "data" / args.dataset
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory {dataset_dir} not found. "
            "Run the merge and tiling scripts first."
        )

    train_script = REPO_ROOT / "scripts" / "train_rtdetr.py"
    if not train_script.exists():
        raise FileNotFoundError(f"Expected training entrypoint at {train_script}")

    cmd = [
        sys.executable,
        str(train_script),
        args.dataset,
        "--weights",
        args.weights,
        "--epochs",
        str(args.epochs),
        "--imgsz",
        str(args.imgsz),
        "--batch",
        str(args.batch),
        "--device",
        args.device,
        "--workers",
        str(args.workers),
        "--seed",
        str(args.seed),
        "--mosaic",
        str(args.mosaic),
        "--mixup",
        str(args.mixup),
        "--copy-paste",
        str(args.copy_paste),
        format_bool_flag("rect", args.rect),
        format_bool_flag("multi-scale", args.multi_scale),
    ]
    if args.name:
        cmd.extend(["--name", args.name])

    print("[info] Launching:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
