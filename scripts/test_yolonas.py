#!/usr/bin/env python3

"""Evaluate a trained YOLO-NAS model on the test split and export annotated images."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from super_gradients.training.processing.processing import ComposeProcessing  # noqa: E402

from train_yolonas import (  # type: ignore  # noqa: E402
    LetterboxYoloDataset,
    build_coco_ground_truth,
    eval_collate_fn,
    evaluate_coco_metrics,
    load_class_names,
    normalize_predictions,
    save_metrics,
)

try:
    from super_gradients.common.object_names import Models
    from super_gradients.training import models
except ImportError as exc:  # pragma: no cover - informative failure
    raise SystemExit(
        "super-gradients package is required. Install it with `pip install super-gradients==3.6.0`."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO-NAS inference on the test split.")
    parser.add_argument("--data", type=Path, default=Path("data/combined"), help="Dataset root in YOLO layout.")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to evaluate.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("runs/RUN_20251012_195511_427487/ckpt_best.pth"),
        help="Path to a trained checkpoint (ckpt_best.pth).",
    )
    parser.add_argument("--model", type=str, default="yolo_nas_m", help="Model variant (e.g. yolo_nas_m).")
    parser.add_argument("--imgsz", type=int, default=1024, help="Target square dimension used during training.")
    parser.add_argument("--batch", type=int, default=4, help="Batch size for inference.")
    parser.add_argument("--num-workers", type=int, default=None, help="Dataloader workers (defaults to cpu_count/2).")
    parser.add_argument("--score-threshold", type=float, default=0.25, help="Confidence threshold for drawing boxes.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Directory to save annotated images and metrics (defaults to data/<split>/output).",
    )
    parser.add_argument("--device", type=str, default=None, help="Device override (cpu or cuda:N).")
    return parser.parse_args()


def determine_output_dir(dataset_root: Path, split: str, override: Path | None) -> Path:
    if override is not None:
        return override.resolve()
    return (dataset_root / split / "output").resolve()


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_palette(num_classes: int) -> List[Tuple[int, int, int]]:
    if num_classes <= 0:
        return []
    hues = [(i / num_classes) for i in range(num_classes)]
    palette: List[Tuple[int, int, int]] = []
    for hue in hues:
        r, g, b = _hsv_to_rgb(hue, 0.6, 1.0)
        palette.append((r, g, b))
    return palette


def _hsv_to_rgb(h: float, s: float, v: float) -> Tuple[int, int, int]:
    import colorsys

    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return int(r * 255), int(g * 255), int(b * 255)


def load_trained_model(
    checkpoint_path: Path,
    model_name: str,
    num_classes: int,
    device: torch.device,
    class_names: Sequence[str],
    score_threshold: float,
) -> torch.nn.Module:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = None
    if isinstance(checkpoint, dict):
        for key in ("ema_state_dict", "state_dict", "net"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                state_dict = checkpoint[key]
                break
        if state_dict is None:
            state_dict = checkpoint
    if state_dict is None:
        raise ValueError(f"Unrecognized checkpoint format from {checkpoint_path}")

    model_key = Models.YOLO_NAS_M if model_name == "yolo_nas_m" else model_name
    model = models.get(model_key, num_classes=num_classes, pretrained_weights=None)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    if hasattr(model, "set_dataset_processing_params"):
        model.set_dataset_processing_params(
            class_names=list(class_names),
            image_processor=ComposeProcessing([]),
            conf=score_threshold,
            iou=0.5,
            nms_top_k=300,
            max_predictions=300,
            multi_label_per_box=False,
            class_agnostic_nms=False,
        )
    return model


def annotate_and_collect_predictions(
    model: torch.nn.Module,
    dataloader: DataLoader,
    class_names: Sequence[str],
    output_dir: Path,
    device: torch.device,
    score_threshold: float,
) -> List[Dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    palette = generate_palette(len(class_names))
    font = ImageFont.load_default()
    predictions: List[Dict[str, object]] = []

    model.eval()

    total_images = len(dataloader.dataset) if hasattr(dataloader, "dataset") else None
    processed = 0

    with torch.inference_mode():
        for batch in dataloader:
            if not isinstance(batch, (tuple, list)) or len(batch) != 3:
                raise ValueError("Expected batches of (images, targets, meta) from eval_collate_fn.")
            images, _targets, meta = batch
            images = images.to(device, non_blocking=True)

            if hasattr(model, "predict"):
                # Convert the normalized tensor batch back to uint8 numpy so the SG pipeline
                # applies its own preprocessing without double-scaling.
                images_np = (
                    images.detach()
                    .cpu()
                    .clamp(0, 1)
                    .permute(0, 2, 3, 1)
                    .mul(255.0)
                    .byte()
                    .numpy()
                )
                raw_preds = model.predict(images_np, fuse_model=False, skip_image_resizing=True)
            else:
                raw_preds = model(images)

            normalized = normalize_predictions(raw_preds, len(meta))

            for sample_idx, sample_pred in enumerate(normalized):
                boxes = torch.as_tensor(sample_pred["boxes"], dtype=torch.float32)
                scores = torch.as_tensor(sample_pred["scores"], dtype=torch.float32)
                labels = torch.as_tensor(sample_pred["labels"], dtype=torch.int64)

                if boxes.ndim == 1:
                    boxes = boxes.unsqueeze(0)
                if scores.ndim == 0:
                    scores = scores.unsqueeze(0)
                if labels.ndim == 0:
                    labels = labels.unsqueeze(0)

                keep = scores >= score_threshold
                boxes = boxes[keep]
                scores = scores[keep]
                labels = labels[keep]

                meta_item = meta[sample_idx]
                orig_h, orig_w = [float(v) for v in meta_item["orig_size"].tolist()]
                scale_x, scale_y = [float(v) for v in meta_item["scale"].tolist()]
                pad_left, pad_top, _, _ = [float(v) for v in meta_item["padding"].tolist()]
                image_id_tensor = meta_item["image_id"]
                image_id = int(image_id_tensor.item() if hasattr(image_id_tensor, "item") else image_id_tensor)
                image_path = Path(meta_item["path"])

                if boxes.numel():
                    boxes = boxes.clone()
                    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_left).clamp(min=0)
                    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_top).clamp(min=0)
                    boxes[:, [0, 2]] /= max(scale_x, 1e-6)
                    boxes[:, [1, 3]] /= max(scale_y, 1e-6)
                    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp_(0, orig_w)
                    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp_(0, orig_h)

                annotated = Image.open(image_path).convert("RGB")
                draw = ImageDraw.Draw(annotated)

                for box, score, label in zip(boxes.tolist(), scores.tolist(), labels.tolist()):
                    x_min, y_min, x_max, y_max = box
                    width = max(x_max - x_min, 0.0)
                    height = max(y_max - y_min, 0.0)
                    predictions.append(
                        {
                            "image_id": image_id,
                            "category_id": int(label),
                            "bbox": [float(x_min), float(y_min), float(width), float(height)],
                            "score": float(score),
                        }
                    )

                    class_idx = int(label)
                    color = palette[class_idx % len(palette)] if palette else (255, 0, 0)
                    draw.rectangle([(x_min, y_min), (x_max, y_max)], outline=color, width=3)
                    caption = f"{class_names[class_idx]} {score:.2f}"
                    text_bbox = draw.textbbox((0, 0), caption, font=font)
                    text_width = text_bbox[2] - text_bbox[0]
                    text_height = text_bbox[3] - text_bbox[1]
                    text_top = max(y_min - text_height - 4, 0)
                    text_background = [
                        x_min,
                        text_top,
                        x_min + text_width + 4,
                        text_top + text_height + 4,
                    ]
                    draw.rectangle(text_background, fill=color)
                    draw.text((x_min + 2, text_top + 2), caption, fill=(0, 0, 0), font=font)

                output_path = output_dir / image_path.name
                annotated.save(output_path)

                processed += 1
                if total_images is not None and processed % max(1, total_images // 20) == 0:
                    print(f"Processed {processed}/{total_images} images", flush=True)

    if total_images is not None and processed % max(1, total_images // 20) != 0:
        print(f"Processed {processed}/{total_images} images", flush=True)

    return predictions


def main() -> int:
    args = parse_args()
    dataset_root = args.data.resolve()
    checkpoint_path = args.checkpoint.resolve()
    output_dir = determine_output_dir(dataset_root, args.split, args.output)

    class_names = load_class_names(dataset_root)
    num_workers = args.num_workers or max(1, (os.cpu_count() or 2) // 2)
    device = resolve_device(args.device)

    dataset = LetterboxYoloDataset(dataset_root, args.split, args.imgsz, class_names)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        collate_fn=eval_collate_fn,
    )

    model = load_trained_model(
        checkpoint_path,
        args.model,
        len(class_names),
        device,
        class_names,
        args.score_threshold,
    )

    predictions = annotate_and_collect_predictions(
        model=model,
        dataloader=dataloader,
        class_names=class_names,
        output_dir=output_dir,
        device=device,
        score_threshold=args.score_threshold,
    )

    ground_truth = build_coco_ground_truth(dataset, class_names)
    metrics = evaluate_coco_metrics(ground_truth, predictions, output_dir)
    save_metrics(output_dir, metrics)

    print(json.dumps(metrics, indent=2))
    print(f"Annotated images written to: {output_dir}")
    print(f"Checkpoint evaluated: {checkpoint_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
