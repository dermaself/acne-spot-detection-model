#!/usr/bin/env python3
"""
Train Ultralytics YOLO11 detectors on the combined `data/all` dataset (or any existing subset).

- Reuses the same data split workflow as scripts/train_rtdetr.py.
- Produces identical artifacts: dataset.yaml, results.csv, metrics.txt, checkpoint.pt, last.pt.
- Defaults to the large YOLO11 checkpoint for best accuracy (commercial-friendly license).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO

import train_rtdetr as rtd_utils


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train YOLO11 on a specified dataset split.")
    p.add_argument("dataset", nargs="?", default="all",
                   help="Dataset subset under data/<dataset>/. Defaults to 'all'.")
    p.add_argument("--weights", default="yolo11x.pt",
                   help="YOLO weights checkpoint (e.g., yolo11{n,s,m,l,x}.pt).")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--project", default="runs", help="Ultralytics output root.")
    p.add_argument("--name", default=None, help="Run name; default '<dataset>_<weights_stem>'.")
    return p.parse_args()


def train_model(
    dataset_yaml: Path,
    weights: Path,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str,
    workers: int,
    project: Path,
    name: str,
    seed: int,
) -> Path:
    project.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights))
    # Mirror augmentation choices from the RT-DETR trainer for consistency.
    model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
        fliplr=0.5,
        flipud=0.0,
        degrees=15.0,
        mosaic=0.2,
        mixup=0.1,
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.4,
        project=str(project),
        name=name,
        exist_ok=False,
        seed=seed,
        patience=25,
    )
    trainer = getattr(model, "trainer", None)
    if trainer is None or not hasattr(trainer, "save_dir"):
        raise RuntimeError("Ultralytics training produced no save_dir; training failed?")
    return Path(trainer.save_dir)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    dataset_dir = repo_root / "data" / args.dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)

    meta = rtd_utils.load_class_metadata()
    split_root, split_counts = rtd_utils.build_split_dataset(dataset_dir, seed=args.seed)
    dataset_yaml = rtd_utils.write_dataset_yaml(dataset_dir, split_root, meta)

    default_name = args.name or f"{args.dataset}_{Path(args.weights).stem}"
    project_dir = (repo_root / args.project / args.dataset).resolve()
    weights_path = rtd_utils.resolve_weights_path(args.weights, repo_root)

    print(f"[info] Training '{args.dataset}' with YOLO weights='{weights_path.name}' "
          f"imgsz={args.imgsz} batch={args.batch} device={args.device}")
    run_dir = train_model(
        dataset_yaml=dataset_yaml,
        weights=weights_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=project_dir,
        name=default_name,
        seed=args.seed,
    )

    metrics = rtd_utils.extract_metrics(run_dir / "results.csv")
    rtd_utils.persist_model_artifacts(run_dir, dataset_dir)
    rtd_utils.write_metrics_file(dataset_dir, metrics, run_dir, split_counts)
    print(f"[done] Metrics saved to {dataset_dir / 'metrics.txt'}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.", file=sys.stderr)
        sys.exit(130)
