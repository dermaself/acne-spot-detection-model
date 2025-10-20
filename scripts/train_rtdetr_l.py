#!/usr/bin/env python3
"""
Train Ultralytics RT-DETR-L detectors on the acne dataset splits.

Ultralytics uses square letterbox resizing: each image is scaled to preserve
aspect ratio, then symmetrically padded to `imgsz`. Example: 2048×1536 with
`imgsz=1536` → scale=0.75 → resized=1536×1152 → pad top/bottom=192,
left/right=0. The metrics we persist (mAP@50, precision, recall, mAP@50-95)
and the saved COCO predictions assume predictions are restored (de-letterboxed)
with that scale + padding, matching the inference stack and Roboflow exports.

Common recipes:
  • Accuracy / throughput balance: python scripts/train_rtdetr_l.py all \
      --weights rtdetr-l.pt --imgsz 1280 --batch auto --epochs 150 --aug none
  • High-res fine-tune: python scripts/train_rtdetr_l.py all \
      --weights runs/.../best.pt --imgsz 1536 --batch 3 --epochs 20 --aug none --resume
  • Eval only: python scripts/train_rtdetr_l.py all \
      --weights runs/.../best.pt --imgsz 1280 --eval_only
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from PIL import Image

try:  # preferred import
    from ultralytics import RTDETR as _UltralyticsModel
    _MODEL_SOURCE = "RTDETR"
except ImportError:
    try:
        from ultralytics import YOLO as _UltralyticsModel  # fallback loader; routes RT-DETR weights
        _MODEL_SOURCE = "YOLO"
        print("[warn] `from ultralytics import RTDETR` unavailable, falling back to `YOLO` loader.", file=sys.stderr)
    except ImportError as exc:  # pragma: no cover - informative failure
        raise SystemExit(
            "Unable to import Ultralytics RT-DETR. Install Ultralytics>=8.2 "
            "(exposes `RTDETR`) or a build where `YOLO` accepts RT-DETR weights."
        ) from exc

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))
try:
    import train_rtdetr as rtd_utils
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Unable to locate 'train_rtdetr.py'. Place it alongside this script or adjust PYTHONPATH."
    ) from exc


AUG_PROFILES = {
    "none": dict(fliplr=0.0, flipud=0.0, degrees=0.0, mosaic=0.0, mixup=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0),
    "light": dict(fliplr=0.5, flipud=0.0, degrees=10.0, mosaic=0.0, mixup=0.0, hsv_h=0.015, hsv_s=0.2, hsv_v=0.2),
    "match_yolo": dict(fliplr=0.5, flipud=0.0, degrees=15.0, mosaic=0.2, mixup=0.1, hsv_h=0.015, hsv_s=0.4, hsv_v=0.4),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Ultralytics RT-DETR-L on a dataset split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        default="all",
        help="Dataset subset under data/<dataset>/ (e.g. all, new, front).",
    )
    parser.add_argument("--weights", default="rtdetr-l.pt", help="RT-DETR checkpoint or alias.")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--imgsz", type=int, default=1280, help="Must be a multiple of 32.")
    parser.add_argument("--batch", default="auto", help="Global batch size; 'auto' picks a safe default for 22GB VRAM.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project", default="runs", help="Ultralytics output root.")
    parser.add_argument("--name", default=None, help="Run name; default '<dataset>_<weights_stem>'.")
    parser.add_argument("--aug", choices=sorted(AUG_PROFILES), default="none", help="In-code augmentation profile.")
    parser.add_argument("--optimizer", default="adamw", help="Optimizer name understood by Ultralytics (e.g. adamw, sgd, rmsprop).")
    parser.add_argument("--lr", type=float, default=3e-4, help="Initial learning rate (lr0).")
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Warmup steps (converted to epochs internally).")
    parser.add_argument("--cosine", action=argparse.BooleanOptionalAction, default=True, help="Use cosine LR schedule.")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping value (0 to disable).")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience on mAP@50.")
    parser.add_argument("--no_ema", action="store_true", help="Disable EMA tracking.")
    parser.add_argument("--resume", action="store_true", help="Resume the most recent run in the project/name directory.")
    parser.add_argument("--eval_only", action="store_true", help="Skip training; run validation/evaluation only.")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")  # type: ignore[attr-defined]


def ensure_img_size_multiple(imgsz: int) -> None:
    if imgsz % 32 != 0:
        raise SystemExit(f"--imgsz must be a multiple of 32; received {imgsz}.")


def resolve_batch_size(batch_arg: str, imgsz: int, device: str) -> int:
    try:
        return int(batch_arg)
    except ValueError:
        pass

    if batch_arg.lower() != "auto":
        raise SystemExit(f"Invalid --batch value '{batch_arg}'. Provide an integer or 'auto'.")

    if "cuda" in device and imgsz <= 1280:
        batch = 6
    elif "cuda" in device and imgsz <= 1536:
        batch = 4
    else:
        batch = 3

    if device == "cpu":
        batch = min(batch, 2)

    print(f"[info] Auto-selected batch size {batch} for imgsz={imgsz} on device={device}")
    return batch


def memory_preflight(imgsz: int, batch: int) -> None:
    if imgsz >= 1536 and batch > 3:
        print(
            f"[warn] imgsz={imgsz} with batch={batch} often exceeds 22GB VRAM. "
            "Consider --batch 3 or --imgsz 1280.",
            file=sys.stderr,
        )


def maybe_compile_model(model) -> None:
    if hasattr(torch, "compile"):
        try:
            model.model = torch.compile(model.model, mode="reduce-overhead", fullgraph=False)  # type: ignore[attr-defined]
            print("[info] torch.compile enabled.")
        except Exception as exc:  # pragma: no cover - best effort only
            print(f"[warn] torch.compile failed ({exc}); continuing without compilation.", file=sys.stderr)


def describe_letterbox(samples: Iterable[Path], imgsz: int) -> None:
    samples = list(samples)
    if not samples:
        print("[warn] No images found to illustrate letterbox scaling.")
        return
    chosen = random.sample(samples, k=min(2, len(samples)))
    print("[info] Letterbox examples (original_w×original_h -> scale, resized_w×resized_h, pad_l/r, pad_t/b):")
    for path in chosen:
        try:
            with Image.open(path) as img:
                orig_w, orig_h = img.size
        except Exception as exc:
            print(f"    {path.name}: unable to read dimensions ({exc})")
            continue
        scale = min(imgsz / orig_w, imgsz / orig_h)
        new_w = int(round(orig_w * scale))
        new_h = int(round(orig_h * scale))
        pad_w = imgsz - new_w
        pad_h = imgsz - new_h
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        print(
            f"    {path.name}: {orig_w}×{orig_h} -> scale={scale:.4f}, "
            f"{new_w}×{new_h}, pad_l/r={pad_left}/{pad_right}, pad_t/b={pad_top}/{pad_bottom}"
        )


def augmentation_overrides(aug_choice: str) -> Dict[str, float]:
    params = AUG_PROFILES[aug_choice].copy()
    return params


def compute_warmup_epochs(warmup_steps: int, train_images: int, batch: int) -> float:
    if warmup_steps <= 0 or train_images <= 0:
        return 0.0
    steps_per_epoch = max(1, math.ceil(train_images / batch))
    return warmup_steps / steps_per_epoch


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
    aug_choice: str,
    optimizer: str,
    lr: float,
    weight_decay: float,
    warmup_epochs: float,
    cosine: bool,
    grad_clip: float,
    patience: int,
    use_ema: bool,
    resume: bool,
) -> Tuple[Path, object]:
    project.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)
    model = _UltralyticsModel(str(weights))
    maybe_compile_model(model)
    aug_params = augmentation_overrides(aug_choice)
    train_kwargs = dict(
        data=str(dataset_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
        project=str(project),
        name=name,
        exist_ok=False,
        seed=seed,
        patience=patience,
        optimizer=optimizer.lower(),
        lr0=lr,
        weight_decay=weight_decay,
        cos_lr=cosine,
        warmup_epochs=warmup_epochs,
        grad_clip=grad_clip,
        ema=use_ema,
        amp=True,
        val_metric="map50",
        pin_memory=True,
        persistent_workers=workers > 0,
        resume=resume,
    )
    train_kwargs.update(aug_params)
    sanitized = train_kwargs.copy()
    unsupported = {"ema", "persistent_workers", "val_metric", "grad_clip", "pin_memory"}
    removed = sorted(k for k in unsupported if sanitized.pop(k, None) is not None)
    if removed:
        print(f"[warn] The following trainer options were skipped: {', '.join(removed)}", file=sys.stderr)
    model.train(**sanitized)
    trainer = getattr(model, "trainer", None)
    if trainer is None or not hasattr(trainer, "save_dir"):
        raise RuntimeError("Ultralytics training produced no save_dir; training failed?")
    return Path(trainer.save_dir), model


def copy_best_last_weights(run_dir: Path) -> Tuple[Path, Path]:
    best_src = run_dir / "weights" / "best.pt"
    last_src = run_dir / "weights" / "last.pt"
    if not best_src.exists() or not last_src.exists():
        raise FileNotFoundError("Ultralytics did not produce best.pt/last.pt under run_dir/weights/")
    best_dst = run_dir / "best.pt"
    last_dst = run_dir / "last.pt"
    shutil.copy2(best_src, best_dst)
    shutil.copy2(last_src, last_dst)
    return best_dst, last_dst


def run_validation(
    weights_path: Path,
    dataset_yaml: Path,
    imgsz: int,
    batch: int,
    device: str,
    workers: int,
    project: Path,
    name: str,
) -> Tuple[Path, object]:
    seed_everything(0)  # deterministic val augmentations
    val_model = _UltralyticsModel(str(weights_path))
    results = val_model.val(
        data=str(dataset_yaml),
        split="val",
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
        save_json=True,
        project=str(project),
        name=f"{name}_val",
        exist_ok=True,
    )
    predictions_path = Path(results.save_dir) / "predictions.json"
    if not predictions_path.exists():
        raise FileNotFoundError(f"Expected predictions.json under {results.save_dir}")
    return predictions_path, results


def metrics_from_val_results(results) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if hasattr(results, "metrics") and hasattr(results.metrics, "box"):
        box = results.metrics.box
        for key in ("map50", "map", "mp", "mr"):
            if hasattr(box, key):
                metrics_key = {"map": "map5095", "mp": "precision", "mr": "recall"}.get(key, key)
                metrics[metrics_key] = float(getattr(box, key))
    if not metrics and hasattr(results, "results_dict"):
        mapping = {
            "metrics/mAP50(B)": "map50",
            "metrics/mAP50-95(B)": "map5095",
            "metrics/precision(B)": "precision",
            "metrics/recall(B)": "recall",
        }
        for k, out_key in mapping.items():
            if k in results.results_dict:
                metrics[out_key] = float(results.results_dict[k])
    return metrics


def write_metrics_json(metrics: Dict[str, float], path: Path) -> None:
    payload = {
        "map50": float(metrics.get("map50", 0.0)),
        "precision": float(metrics.get("precision", 0.0)),
        "recall": float(metrics.get("recall", 0.0)),
        "map5095": float(metrics.get("map5095", metrics.get("map", 0.0))),
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def gather_sample_paths(split_root: Path) -> List[Path]:
    image_dir = split_root / "train" / "images"
    return sorted(p for p in image_dir.glob("*") if p.is_file())


def main() -> None:
    args = parse_args()
    ensure_img_size_multiple(args.imgsz)

    repo_root = Path(__file__).resolve().parents[1]
    dataset_dir = repo_root / "data" / args.dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)

    meta = rtd_utils.load_class_metadata()
    split_root, split_counts = rtd_utils.build_split_dataset(dataset_dir, seed=args.seed)
    dataset_yaml = rtd_utils.write_dataset_yaml(dataset_dir, split_root, meta)
    sample_paths = gather_sample_paths(split_root)
    describe_letterbox(sample_paths, args.imgsz)

    weights_path = rtd_utils.resolve_weights_path(args.weights, repo_root)
    batch_size = resolve_batch_size(str(args.batch), args.imgsz, args.device)
    memory_preflight(args.imgsz, batch_size)

    default_name = args.name or f"{args.dataset}_{Path(args.weights).stem}"
    project_dir = (repo_root / args.project / args.dataset).resolve()

    summary_line = (
        f"RT-DETR-L[{_MODEL_SOURCE}] | imgsz={args.imgsz} batch={batch_size} aug={args.aug} "
        f"epochs={args.epochs} opt={args.optimizer} lr={args.lr} wd={args.weight_decay} device={args.device}"
    )
    print(summary_line)

    train_images = int(split_counts.get("train", 0))
    warmup_epochs = compute_warmup_epochs(args.warmup_steps, train_images, batch_size)

    if args.eval_only:
        seed_everything(args.seed)
        predictions_path, val_results = run_validation(
            weights_path=weights_path,
            dataset_yaml=dataset_yaml,
            imgsz=args.imgsz,
            batch=batch_size,
            device=args.device,
            workers=args.workers,
            project=project_dir,
            name=default_name,
        )
        metrics = metrics_from_val_results(val_results)
        metrics_json_path = dataset_dir / "metrics.json"
        write_metrics_json(metrics, metrics_json_path)
        rtd_utils.write_metrics_file(dataset_dir, metrics, Path(val_results.save_dir), split_counts)
        shutil.copy2(predictions_path, Path(val_results.save_dir) / "val_predictions.coco.json")
        shutil.copy2(predictions_path, dataset_dir / "val_predictions.coco.json")
        print(
            f"[done] mAP50={metrics.get('map50', 0):.4f} precision={metrics.get('precision', 0):.4f} "
            f"recall={metrics.get('recall', 0):.4f} | run_dir={val_results.save_dir}"
        )
        return

    run_dir, _ = train_model(
        dataset_yaml=dataset_yaml,
        weights=weights_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=batch_size,
        device=args.device,
        workers=args.workers,
        project=project_dir,
        name=default_name,
        seed=args.seed,
        aug_choice=args.aug,
        optimizer=args.optimizer,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=warmup_epochs,
        cosine=args.cosine,
        grad_clip=args.grad_clip,
        patience=args.patience,
        use_ema=not args.no_ema,
        resume=args.resume,
    )

    best_path, last_path = copy_best_last_weights(run_dir)

    metrics = rtd_utils.extract_metrics(run_dir / "results.csv")
    metrics_json_path = dataset_dir / "metrics.json"
    write_metrics_json(metrics, metrics_json_path)
    rtd_utils.persist_model_artifacts(run_dir, dataset_dir)
    rtd_utils.write_metrics_file(dataset_dir, metrics, run_dir, split_counts)

    predictions_path, val_results = run_validation(
        weights_path=best_path,
        dataset_yaml=dataset_yaml,
        imgsz=args.imgsz,
        batch=batch_size,
        device=args.device,
        workers=args.workers,
        project=project_dir,
        name=default_name,
    )
    shutil.copy2(predictions_path, run_dir / "val_predictions.coco.json")
    shutil.copy2(predictions_path, dataset_dir / "val_predictions.coco.json")

    map50 = metrics.get("map50", 0.0)
    precision = metrics.get("precision", 0.0)
    recall = metrics.get("recall", 0.0)
    print(
        f"[done] mAP50={map50:.4f} precision={precision:.4f} recall={recall:.4f} | run_dir={run_dir}"
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.", file=sys.stderr)
        sys.exit(130)
