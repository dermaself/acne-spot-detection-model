#!/usr/bin/env python3

"""Utility for training YOLO-NAS models on YOLO-formatted datasets.

This script targets the specification supplied by the user:
* Letterbox resize to 1536×1536 without any other augmentation.
* Uses SuperGradients' YOLO-NAS-M model with AdamW + cosine LR.
* Reports mAP@50, precision, recall for validation and saves artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

try:
    from super_gradients.common.object_names import Models
    from super_gradients.training import Trainer, models
    from super_gradients.training.losses import PPYoloELoss
    try:
        from super_gradients.training.utils.callbacks.lr_warmup import (  # type: ignore
            LRWarmupModes,
            LR_WARMUP_CLS_DICT,
        )
    except ImportError:
        try:
            from super_gradients.training.utils.callbacks.lr_warmup_callback import (  # type: ignore
                LRWarmupModes,
                LR_WARMUP_CLS_DICT,
            )
        except ImportError:
            LRWarmupModes = None  # type: ignore
            LR_WARMUP_CLS_DICT = {}
    try:
        from super_gradients.training.utils.callbacks import EarlyStoppingCallback
    except ImportError:
        try:
            from super_gradients.training.utils.callbacks.early_stopping import EarlyStoppingCallback
        except ImportError:
            EarlyStoppingCallback = None  # type: ignore[assignment]
    try:
        from super_gradients.training.utils.predict.prediction_results import (
            ImageDetectionPrediction,
            ImagesDetectionPrediction,
        )
    except ImportError:
        ImageDetectionPrediction = None  # type: ignore[assignment]
        ImagesDetectionPrediction = None  # type: ignore[assignment]
except ImportError as exc:  # pragma: no cover - informative failure
    raise SystemExit(
        "super-gradients package is required for this script. Install it with "
        "`pip install super-gradients==3.6.0` (or a compatible version)."
    ) from exc

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except ImportError as exc:  # pragma: no cover
    raise SystemExit("pycocotools is required. Install it via `pip install pycocotools`.") from exc


try:
    import torchvision.transforms.functional as F
except ImportError as exc:  # pragma: no cover
    raise SystemExit("torchvision is required. Install it via `pip install torchvision`.") from exc


YOLO_LABEL_SUFFIX = ".txt"
DEFAULT_OUTPUT_DIR = "runs/yolonas_m_1536"
DEFAULT_MODEL_NAME = "yolo_nas_m"


def resolve_linear_warmup_mode():
    if "LR_WARMUP_CLS_DICT" in globals() and LR_WARMUP_CLS_DICT:
        for key in LR_WARMUP_CLS_DICT:
            key_str = getattr(key, "value", str(key))
            if "linear" in key_str.lower():
                return key
    if "LRWarmupModes" in globals() and LRWarmupModes is not None:
        for candidate_name in ("LINEAR", "LINEAR_LR", "LINEAR_STEP"):
            if hasattr(LRWarmupModes, candidate_name):
                return getattr(LRWarmupModes, candidate_name)
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO-NAS on a YOLO-format dataset.")
    parser.add_argument("--data", type=Path, default=Path("data/combined"), help="Dataset root in YOLO layout.")
    parser.add_argument("--imgsz", type=int, default=1536, help="Target square dimension for letterbox resize.")
    parser.add_argument("--epochs", type=int, default=250, help="Number of training epochs.")
    parser.add_argument("--batch", type=int, default=4, help="Batch size (per device).")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate for AdamW.")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay for AdamW.")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Number of warmup steps for cosine schedule.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME, help="Model variant to train (e.g. yolo_nas_m).")
    parser.add_argument("--out", type=Path, default=Path(DEFAULT_OUTPUT_DIR), help="Output directory for checkpoints and logs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--num_workers", type=int, default=None, help="Override dataloader workers (defaults to cpu_count/2).")
    parser.add_argument("--device", type=str, default=None, help="Optional device override (cpu/cuda:N).")
    return parser.parse_args()


def load_class_names(dataset_root: Path) -> List[str]:
    data_yaml = dataset_root / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset metadata not found: {data_yaml}")

    with data_yaml.open("r", encoding="utf-8") as handle:
        raw = handle.read()

    import yaml

    config = yaml.safe_load(raw)
    names = config.get("names")
    if names is None:
        raise ValueError(f"No 'names' entry found in {data_yaml}")

    if isinstance(names, dict):
        try:
            ordered = sorted(names.items(), key=lambda item: int(item[0]))
        except ValueError:
            ordered = sorted(names.items(), key=lambda item: item[0])
        return [name for _, name in ordered]

    if isinstance(names, Iterable):
        return list(names)

    raise TypeError(f"Unsupported 'names' format in {data_yaml}: {type(names)!r}")


def discover_dataset_split(dataset_root: Path, split: str) -> List[Tuple[Path, Path]]:
    images_dir = dataset_root / split / "images"
    labels_dir = dataset_root / split / "labels"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory missing for split '{split}': {images_dir}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Labels directory missing for split '{split}': {labels_dir}")

    pairs: List[Tuple[Path, Path]] = []
    for image_path in sorted(images_dir.iterdir()):
        if not image_path.is_file():
            continue
        label_path = labels_dir / f"{image_path.stem}{YOLO_LABEL_SUFFIX}"
        pairs.append((image_path, label_path))
    if not pairs:
        raise ValueError(f"No image files found under {images_dir}")
    return pairs


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def letterbox(image: Image.Image, target_size: int) -> Tuple[torch.Tensor, Tuple[float, float], Tuple[int, int, int, int]]:
    width, height = image.size
    scale = target_size / max(width, height)
    resized_w = int(round(width * scale))
    resized_h = int(round(height * scale))

    resized = image.resize((resized_w, resized_h), Image.Resampling.BILINEAR)

    pad_w = target_size - resized_w
    pad_h = target_size - resized_h
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top

    padded = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    padded.paste(resized, (pad_left, pad_top))

    tensor = F.to_tensor(padded)
    return tensor, (scale, scale), (pad_left, pad_top, pad_right, pad_bottom)


def load_yolo_annotations(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    if not label_path.exists():
        return []

    annotations: List[Tuple[int, float, float, float, float]] = []
    with label_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                continue
            cls = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:])
            annotations.append((cls, cx, cy, bw, bh))
    return annotations


def yolo_to_pascal_voc(
    records: List[Tuple[int, float, float, float, float]],
    width: int,
    height: int,
) -> List[Tuple[int, float, float, float, float]]:
    abs_records: List[Tuple[int, float, float, float, float]] = []
    for cls, cx, cy, bw, bh in records:
        abs_cx = cx * width
        abs_cy = cy * height
        abs_bw = bw * width
        abs_bh = bh * height
        x_min = abs_cx - abs_bw / 2
        y_min = abs_cy - abs_bh / 2
        x_max = abs_cx + abs_bw / 2
        y_max = abs_cy + abs_bh / 2
        abs_records.append((cls, x_min, y_min, x_max, y_max))
    return abs_records


def scale_boxes(
    boxes: List[Tuple[int, float, float, float, float]],
    scale: Tuple[float, float],
    padding: Tuple[int, int, int, int],
) -> List[Tuple[int, float, float, float, float]]:
    scaled: List[Tuple[int, float, float, float, float]] = []
    scale_x, scale_y = scale
    pad_left, pad_top, _, _ = padding
    for cls, x_min, y_min, x_max, y_max in boxes:
        x_min = x_min * scale_x + pad_left
        x_max = x_max * scale_x + pad_left
        y_min = y_min * scale_y + pad_top
        y_max = y_max * scale_y + pad_top
        scaled.append((cls, x_min, y_min, x_max, y_max))
    return scaled


class LetterboxYoloDataset(Dataset):
    """Minimal DetectionDataset wrapper applying letterbox preprocessing."""

    def __init__(
        self,
        dataset_root: Path,
        split: str,
        img_size: int,
        class_names: Sequence[str],
    ) -> None:
        self.dataset_root = dataset_root
        self.split = split
        self.img_size = img_size
        self.class_names = class_names
        self.samples = discover_dataset_split(dataset_root, split)
        self.num_classes = len(class_names)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        image_path, label_path = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        tensor, scale, padding = letterbox(image, self.img_size)

        annotations = load_yolo_annotations(label_path)
        abs_boxes = yolo_to_pascal_voc(annotations, width, height)
        letterboxed_boxes = scale_boxes(abs_boxes, scale, padding)

        if letterboxed_boxes:
            boxes_tensor = torch.tensor(
                [[x_min, y_min, x_max, y_max] for (_, x_min, y_min, x_max, y_max) in letterboxed_boxes],
                dtype=torch.float32,
            )
            labels_tensor = torch.tensor([cls for (cls, *_rest) in letterboxed_boxes], dtype=torch.int64)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)

        sample = {
            "image": tensor,
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "orig_size": torch.tensor([height, width], dtype=torch.int32),
            "scale": torch.tensor(scale, dtype=torch.float32),
            "padding": torch.tensor(padding, dtype=torch.float32),
            "path": str(image_path),
            "image_id": torch.tensor(index, dtype=torch.int64),
        }
        return sample


def _collect_batch_components(
    batch: Sequence[Dict[str, torch.Tensor]]
) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, torch.Tensor]]]:
    images = torch.stack([sample["image"] for sample in batch], dim=0)
    target_rows: List[List[float]] = []
    meta: List[Dict[str, torch.Tensor]] = []

    for i, sample in enumerate(batch):
        boxes = sample["boxes"]
        labels = sample["labels"]
        if boxes.numel():
            for box, label in zip(boxes.tolist(), labels.tolist()):
                x_min, y_min, x_max, y_max = box
                target_rows.append([float(i), float(label), float(x_min), float(y_min), float(x_max), float(y_max)])

        # store metadata for downstream evaluation
        meta.append(
            {
                "orig_size": sample["orig_size"].detach().cpu(),
                "scale": sample["scale"].detach().cpu(),
                "padding": sample["padding"].detach().cpu(),
                "path": sample["path"],
                "image_id": sample["image_id"].detach().cpu(),
            }
        )

    target_tensor = (
        torch.tensor(target_rows, dtype=torch.float32) if target_rows else torch.zeros((0, 6), dtype=torch.float32)
    )

    return images, target_tensor, meta


def train_collate_fn(batch: Sequence[Dict[str, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    images, target_tensor, _ = _collect_batch_components(batch)
    return images, target_tensor


def eval_collate_fn(batch: Sequence[Dict[str, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, torch.Tensor]]]:
    images, target_tensor, meta = _collect_batch_components(batch)
    return images, target_tensor, meta


def build_coco_ground_truth(dataset: LetterboxYoloDataset, class_names: Sequence[str]) -> Dict[str, object]:
    annotations: List[Dict[str, object]] = []
    images: List[Dict[str, object]] = []
    ann_id = 1

    for image_id, (image_path, label_path) in enumerate(dataset.samples):
        with Image.open(image_path) as img:
            width, height = img.size

        images.append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "width": width,
                "height": height,
            }
        )

        labels = load_yolo_annotations(label_path)
        abs_boxes = yolo_to_pascal_voc(labels, width, height)
        for cls, x_min, y_min, x_max, y_max in abs_boxes:
            bw = max(x_max - x_min, 0.0)
            bh = max(y_max - y_min, 0.0)
            area = bw * bh
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": cls,
                    "bbox": [x_min, y_min, bw, bh],
                    "area": area,
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    categories = [
        {"id": idx, "name": name, "supercategory": "lesion"}
        for idx, name in enumerate(class_names)
    ]
    return {"images": images, "annotations": annotations, "categories": categories}


def normalize_predictions(raw_predictions, batch_size: int) -> List[Dict[str, np.ndarray]]:
    if isinstance(raw_predictions, dict):
        for key in ("predictions", "detections", "outputs"):
            if key in raw_predictions:
                raw_predictions = raw_predictions[key]
                break

    if ImagesDetectionPrediction is not None and isinstance(raw_predictions, ImagesDetectionPrediction):
        converted: List[Dict[str, np.ndarray]] = []
        for image_pred in raw_predictions:
            detection = getattr(image_pred, "prediction", None)
            if detection is None:
                raise ValueError("ImagesDetectionPrediction item missing 'prediction' attribute.")
            converted.append(
                {
                    "boxes": np.asarray(detection.bboxes_xyxy),
                    "scores": np.asarray(detection.confidence),
                    "labels": np.asarray(detection.labels, dtype=np.int64),
                }
            )
        raw_predictions = converted
    elif ImageDetectionPrediction is not None and isinstance(raw_predictions, ImageDetectionPrediction):
        detection = getattr(raw_predictions, "prediction", None)
        if detection is None:
            raise ValueError("ImageDetectionPrediction missing 'prediction' attribute.")
        raw_predictions = [
            {
                "boxes": np.asarray(detection.bboxes_xyxy),
                "scores": np.asarray(detection.confidence),
                "labels": np.asarray(detection.labels, dtype=np.int64),
            }
        ]

    if not isinstance(raw_predictions, (list, tuple)):
        raise TypeError(f"Unsupported prediction type: {type(raw_predictions)!r}")

    if len(raw_predictions) != batch_size:
        raise ValueError(f"Expected {batch_size} predictions; received {len(raw_predictions)}")

    normalized: List[Dict[str, np.ndarray]] = []
    for item in raw_predictions:
        if isinstance(item, dict):
            boxes = item.get("boxes")
            if boxes is None:
                boxes = item.get("bboxes")
            if boxes is None:
                boxes = item.get("pred_boxes")

            scores = item.get("scores")
            if scores is None:
                scores = item.get("conf")
            if scores is None:
                scores = item.get("confidence")

            labels = item.get("labels")
            if labels is None:
                labels = item.get("classes")
            if labels is None:
                labels = item.get("pred_classes")
        else:
            boxes = getattr(item, "boxes", None)
            if boxes is None:
                boxes = getattr(item, "bboxes", None)
            if boxes is None:
                boxes = getattr(item, "pred_boxes", None)

            scores = getattr(item, "scores", None)
            if scores is None:
                scores = getattr(item, "conf", None)
            if scores is None:
                scores = getattr(item, "pred_scores", None)

            labels = getattr(item, "labels", None)
            if labels is None:
                labels = getattr(item, "classes", None)
            if labels is None:
                labels = getattr(item, "pred_classes", None)

        if boxes is None or scores is None or labels is None:
            raise ValueError("Prediction item missing required keys (boxes, scores, labels).")

        normalized.append(
            {
                "boxes": np.asarray(boxes),
                "scores": np.asarray(scores),
                "labels": np.asarray(labels, dtype=np.int64),
            }
        )
    return normalized


def generate_coco_predictions(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    score_threshold: float = 0.001,
) -> List[Dict[str, object]]:
    model.eval()
    predictions: List[Dict[str, object]] = []

    with torch.inference_mode():
        for batch in dataloader:
            meta = []
            if isinstance(batch, (list, tuple)):
                if len(batch) == 3:
                    images, _targets, meta = batch
                elif len(batch) == 2:
                    images, _targets = batch
                elif len(batch) == 1:
                    images = batch[0]
                else:
                    raise ValueError("Unexpected batch tuple length.")
            elif isinstance(batch, dict):
                images = batch.get("image")
                meta = batch.get("meta", [])
            else:
                raise TypeError(f"Unsupported batch type: {type(batch)!r}")

            if images is None:
                raise ValueError("Batch does not contain image tensor.")

            images = images.to(device, non_blocking=True)

            if hasattr(model, "predict"):
                raw_preds = model.predict(images)
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
                pad_left, pad_top, pad_right, pad_bottom = [float(v) for v in meta_item["padding"].tolist()]
                image_id_tensor = meta_item["image_id"]
                image_id = int(image_id_tensor.item() if hasattr(image_id_tensor, "item") else image_id_tensor)

                if boxes.numel() == 0:
                    continue

                # Remove padding and rescale back to original image space
                boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_left).clamp(min=0)
                boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_top).clamp(min=0)
                boxes[:, [0, 2]] /= max(scale_x, 1e-6)
                boxes[:, [1, 3]] /= max(scale_y, 1e-6)
                boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp_(0, orig_w)
                boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp_(0, orig_h)

                widths = (boxes[:, 2] - boxes[:, 0]).clamp(min=0)
                heights = (boxes[:, 3] - boxes[:, 1]).clamp(min=0)

                for box, score, label, width, height in zip(boxes, scores, labels, widths, heights):
                    predictions.append(
                        {
                            "image_id": image_id,
                            "category_id": int(label.item()),
                            "bbox": [
                                float(box[0].item()),
                                float(box[1].item()),
                                float(width.item()),
                                float(height.item()),
                            ],
                            "score": float(score.item()),
                        }
                    )

    return predictions


def evaluate_coco_metrics(
    ground_truth: Dict[str, object],
    predictions: List[Dict[str, object]],
    output_dir: Path,
) -> Dict[str, float]:
    output_dir.mkdir(parents=True, exist_ok=True)
    gt_path = output_dir / "val_ground_truth.coco.json"
    pred_path = output_dir / "val_predictions.coco.json"

    with gt_path.open("w", encoding="utf-8") as handle:
        json.dump(ground_truth, handle)
    with pred_path.open("w", encoding="utf-8") as handle:
        json.dump(predictions, handle)

    coco_gt = COCO(str(gt_path))
    if predictions:
        coco_dt = coco_gt.loadRes(str(pred_path))
    else:
        coco_dt = coco_gt.loadRes([])

    evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    iou_thresholds = evaluator.params.iouThrs.tolist()
    try:
        iou_index = iou_thresholds.index(0.5)
    except ValueError:
        iou_index = 0

    precision = evaluator.eval["precision"]  # shape: [T, R, K, A, M]
    recall = evaluator.eval["recall"]  # shape: [T, K, A, M]
    area_index = list(evaluator.params.areaRngLbl).index("all")
    max_det_index = list(evaluator.params.maxDets).index(100)

    precision_slice = precision[iou_index, :, :, area_index, max_det_index]
    valid_precision = precision_slice[precision_slice > -1]
    overall_precision = float(valid_precision.mean()) if valid_precision.size else 0.0

    recall_slice = recall[iou_index, :, area_index, max_det_index]
    valid_recall = recall_slice[recall_slice > -1]
    overall_recall = float(valid_recall.mean()) if valid_recall.size else 0.0

    map50 = float(evaluator.stats[1])  # AP at IoU=0.50
    map50_95 = float(evaluator.stats[0])  # AP@[.50:.95]

    return {
        "map50": map50,
        "precision": overall_precision,
        "recall": overall_recall,
        "map5095": map50_95,
    }


def build_dataloaders(
    dataset_root: Path,
    img_size: int,
    class_names: Sequence[str],
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader]:
    train_dataset = LetterboxYoloDataset(dataset_root, "train", img_size, class_names)
    valid_dataset = LetterboxYoloDataset(dataset_root, "valid", img_size, class_names)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
        collate_fn=train_collate_fn,
        drop_last=False,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        collate_fn=train_collate_fn,
        drop_last=False,
    )
    eval_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        collate_fn=eval_collate_fn,
        drop_last=False,
    )
    return train_loader, valid_loader, eval_loader


def compute_warmup_epochs(
    warmup_steps: int,
    train_loader: DataLoader,
    epochs: int,
) -> float:
    steps_per_epoch = len(train_loader)
    if steps_per_epoch == 0:
        return 0.0
    warmup_epochs = warmup_steps / steps_per_epoch
    return min(warmup_epochs, float(epochs))


def train_model(
    args: argparse.Namespace,
    class_names: Sequence[str],
    train_loader: DataLoader,
    valid_loader: DataLoader,
) -> Tuple[Trainer, Path]:
    experiment_name = f"{args.model}_img{args.imgsz}"
    trainer = Trainer(experiment_name=experiment_name, ckpt_root_dir=str(args.out))

    num_classes = len(class_names)
    model = models.get(
        Models.YOLO_NAS_M if args.model == "yolo_nas_m" else args.model,
        num_classes=num_classes,
        pretrained_weights=None,
    )

    warmup_mode = resolve_linear_warmup_mode()

    training_params = {
        "max_epochs": args.epochs,
        "lr_mode": "cosine",
        "initial_lr": args.lr,
        "cosine_final_lr_ratio": 0.01,
        "lr_warmup_steps": args.warmup_steps if warmup_mode is not None else 0,
        "warmup_initial_lr": args.lr * 0.1,
    }
    if warmup_mode is not None:
        training_params["warmup_mode"] = warmup_mode

    callbacks: List[object] = []
    if EarlyStoppingCallback is not None:
        callbacks.append(
            EarlyStoppingCallback(
                monitor="PPYoloELoss/loss",
                patience=20,
                min_delta=0.0,
                mode="min",
            )
        )

    training_params.update(
        {
        "optimizer": "AdamW",
        "optimizer_params": {
            "betas": (0.9, 0.999),
            "weight_decay": args.weight_decay,
        },
        "loss": PPYoloELoss(num_classes=num_classes, use_static_assigner=True),
        "ema": True,
        "ema_params": {
            "decay": 0.9999,
            "ema_start": 0,
            "decay_type": "exp",
            "beta": 15,
        },
        "mixed_precision": True,
        "valid_metrics_list": [],
        "metric_to_watch": "PPYoloELoss/loss",
        "greater_is_better": False,
        "save_checkpoints_dir": str(args.out / "checkpoints"),
        "save_model": True,
        "average_best_models": False,
        "run_validation_freq": 1,
        "train_logging_frequency": 50,
        "train_metrics_list": [],
    }
    )
    if callbacks:
        training_params["callbacks"] = callbacks

    trainer.train(  # type: ignore[arg-type]
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        training_params=training_params,
    )

    best_ckpt_candidates = sorted(args.out.glob(f"{experiment_name}/**/ckpt_best.pth"))
    if not best_ckpt_candidates:
        raise FileNotFoundError("Unable to locate ckpt_best.pth after training.")
    best_ckpt_path = best_ckpt_candidates[0]
    return trainer, best_ckpt_path


def load_model_from_checkpoint(
    checkpoint_path: Path,
    args: argparse.Namespace,
    class_names: Sequence[str],
    device: torch.device,
) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = None
    for key in ("ema_state_dict", "state_dict", "net"):
        if isinstance(checkpoint, dict) and key in checkpoint:
            state_dict = checkpoint[key]
            break
    if state_dict is None and isinstance(checkpoint, dict):
        state_dict = checkpoint
    if state_dict is None:
        raise ValueError(f"Unrecognized checkpoint structure: keys={list(checkpoint.keys())}")

    num_classes = len(class_names)
    model_name = Models.YOLO_NAS_M if args.model == "yolo_nas_m" else args.model
    model = models.get(model_name, num_classes=num_classes, pretrained_weights=None)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def evaluate_model(
    model: torch.nn.Module,
    valid_loader: DataLoader,
    class_names: Sequence[str],
    out_dir: Path,
) -> Dict[str, float]:
    if not isinstance(valid_loader.dataset, LetterboxYoloDataset):
        raise TypeError("Validation loader dataset must be LetterboxYoloDataset for evaluation.")

    ground_truth = build_coco_ground_truth(valid_loader.dataset, class_names)
    device = next(model.parameters()).device
    predictions = generate_coco_predictions(model, valid_loader, device)
    metrics = evaluate_coco_metrics(ground_truth, predictions, out_dir)
    return metrics


def save_metrics(out_dir: Path, metrics: Dict[str, float]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)


def export_onnx(model, out_dir: Path, input_size: int = 1536) -> Path:
    onnx_path = out_dir / "best.onnx"
    original_device = next(model.parameters()).device
    model = model.to("cpu")
    dummy = torch.zeros((1, 3, input_size, input_size), device="cpu")
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["images"],
        output_names=["predictions"],
        opset_version=12,
        dynamic_axes={"images": {0: "batch"}, "predictions": {0: "batch"}},
    )
    model.to(original_device)
    return onnx_path


def export_quantized_onnx(onnx_path: Path, out_dir: Path) -> Path:
    int8_path = out_dir / "best_int8.onnx"
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError:
        print("onnxruntime not installed; skipping INT8 export.")
        return int8_path

    quantize_dynamic(
        model_input=str(onnx_path),
        model_output=str(int8_path),
        weight_type=QuantType.QInt8,
    )
    return int8_path


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    dataset_root = args.data.resolve()
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    class_names = load_class_names(dataset_root)
    num_workers = args.num_workers or max(1, (os.cpu_count() or 2) // 2)

    train_loader, valid_loader, eval_loader = build_dataloaders(
        dataset_root=dataset_root,
        img_size=args.imgsz,
        class_names=class_names,
        batch_size=args.batch,
        num_workers=num_workers,
    )

    trainer, best_ckpt_path = train_model(args, class_names, train_loader, valid_loader)
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_model = load_model_from_checkpoint(best_ckpt_path, args, class_names, device)

    metrics = evaluate_model(eval_model, eval_loader, class_names, out_dir)
    save_metrics(out_dir, metrics)
    print(json.dumps(metrics, indent=2))

    best_ckpt_copy = out_dir / "best.ckpt"
    shutil.copy2(best_ckpt_path, best_ckpt_copy)

    onnx_path = export_onnx(eval_model, out_dir, args.imgsz)
    export_quantized_onnx(onnx_path, out_dir)

    print(f"Artifacts saved under: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
