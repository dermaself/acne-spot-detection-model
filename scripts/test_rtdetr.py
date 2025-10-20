#!/usr/bin/env python3
"""
Run inference with a trained Ultralytics RT-DETR model on a folder of images,
save annotated predictions, and report test-set metrics.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

try:
    from ultralytics import RTDETR as _UltralyticsModel
    _MODEL_SOURCE = "RTDETR"
except ImportError:  # pragma: no cover - informative fallback
    try:
        from ultralytics import YOLO as _UltralyticsModel
        _MODEL_SOURCE = "YOLO"
        print("[warn] Falling back to `ultralytics.YOLO`; ensure it supports RT-DETR weights.", file=sys.stderr)
    except ImportError as exc:  # pragma: no cover - informative failure
        raise SystemExit(
            "Unable to import Ultralytics RT-DETR. Install ultralytics>=8.3.20."
        ) from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = REPO_ROOT / "runs/new/rtdetr_l_safe/best.pt"
DEFAULT_IMAGE_DIR = REPO_ROOT / "data/new/test/images"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/new/test/output"
DEFAULT_METRICS_FILE = REPO_ROOT / "data/new/test/output.txt"
DEFAULT_DATA_YAML = REPO_ROOT / "data/new/data.yaml"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RT-DETR inference on a folder and evaluate the test split."
    )
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="Path to trained RT-DETR weights (*.pt).")
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGE_DIR, help="Directory containing images to predict.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory to save annotated outputs.")
    parser.add_argument(
        "--metrics-file",
        type=Path,
        default=DEFAULT_METRICS_FILE,
        help="Path to write evaluation metrics summary.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_YAML,
        help="Dataset YAML describing train/val/test splits (used for metrics).",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size (square).")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for drawing predictions.")
    parser.add_argument("--iou", type=float, default=0.7, help="IoU threshold used for NMS.")
    parser.add_argument("--max-det", type=int, default=300, help="Maximum detections per image.")
    parser.add_argument("--batch", type=int, default=4, help="Batch size for the evaluation pass.")
    parser.add_argument("--device", type=str, default=None, help="Device override (e.g. 'cpu', 'cuda:0').")
    parser.add_argument("--line-width", type=int, default=3, help="Bounding box line width in pixels.")
    parser.add_argument("--save-conf", action="store_true", help="Append confidence values to class labels.")
    return parser.parse_args()


def resolve_device(device_arg: str | None) -> str:
    if device_arg:
        if device_arg.startswith("cuda") and not torch.cuda.is_available():
            print("[warn] CUDA requested but unavailable. Falling back to CPU.", file=sys.stderr)
            return "cpu"
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def ensure_path(path: Path, kind: str) -> None:
    if kind == "file":
        if not path.is_file():
            raise FileNotFoundError(f"{path} does not exist or is not a file.")
    elif kind == "dir":
        if not path.is_dir():
            raise FileNotFoundError(f"{path} does not exist or is not a directory.")
    else:
        raise ValueError(f"Unsupported kind '{kind}'.")


def list_images(image_dir: Path) -> List[Path]:
    return sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file())


def load_model(weights: Path):
    ensure_path(weights, "file")
    model = _UltralyticsModel(str(weights))
    return model


def extract_class_names(model) -> List[str]:
    names = getattr(model, "names", None)
    if isinstance(names, dict):
        return [names[k] for k in sorted(names)]
    if isinstance(names, (list, tuple)):
        return list(names)
    raise ValueError("Unable to determine class names from the loaded model.")


def generate_palette(num_classes: int) -> List[Tuple[int, int, int]]:
    if num_classes <= 0:
        return [(255, 0, 0)]
    import colorsys

    palette: List[Tuple[int, int, int]] = []
    for idx in range(num_classes):
        hue = (idx / max(1, num_classes)) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 1.0)
        palette.append((int(r * 255), int(g * 255), int(b * 255)))
    return palette


def _convert_result_image(result) -> Image.Image:
    orig = result.orig_img
    if orig is None:
        raise ValueError("Result does not carry the original image array.")
    if isinstance(orig, np.ndarray):
        if orig.ndim == 2:
            orig = np.stack([orig] * 3, axis=-1)
        if orig.shape[2] == 3:
            rgb = orig[:, :, ::-1]
        elif orig.shape[2] == 4:
            rgb = orig[:, :, [2, 1, 0, 3]]
        else:
            raise ValueError(f"Unexpected channel count: {orig.shape}")
        return Image.fromarray(rgb).convert("RGB")
    raise TypeError("Unsupported image array type.")


def annotate_result(
    result,
    class_names: Sequence[str],
    palette: Sequence[Tuple[int, int, int]],
    line_width: int,
    show_conf: bool,
) -> Image.Image:
    image = _convert_result_image(result)
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return image

    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    classes = boxes.cls.cpu().numpy().astype(int)

    width, height = image.size
    palette_size = max(1, len(palette))

    for coords, conf, cls in zip(xyxy, confs, classes):
        x1, y1, x2, y2 = map(float, coords)
        cls_name = class_names[cls] if 0 <= cls < len(class_names) else f"class_{cls}"
        label_text = f"{cls_name} {conf:.2f}" if show_conf else cls_name
        color = palette[cls % palette_size]

        draw.rectangle([x1, y1, x2, y2], outline=color, width=max(1, line_width))

        text_bbox = draw.textbbox((0, 0), label_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        text_x = max(0.0, min(x1, width - text_width - 4))
        text_y = y1 - text_height - 4
        if text_y < 0:
            text_y = min(y2 + 4, height - text_height - 4)

        background_coords = [
            text_x,
            text_y,
            text_x + text_width + 4,
            text_y + text_height + 4,
        ]
        draw.rectangle(background_coords, fill=color)
        draw.text((text_x + 2, text_y + 2), label_text, fill=(255, 255, 255), font=font)

    return image


def run_predictions(
    model,
    image_paths: Sequence[Path],
    output_dir: Path,
    imgsz: int,
    conf: float,
    iou: float,
    max_det: int,
    device: str,
    class_names: Sequence[str],
    palette: Sequence[Tuple[int, int, int]],
    line_width: int,
    show_conf: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for image_path in image_paths:
        result = model.predict(
            source=str(image_path),
            stream=False,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            max_det=max_det,
            device=device,
            verbose=False,
        )
        result_item = result[0] if isinstance(result, list) else result
        annotated = annotate_result(
            result_item,
            class_names=class_names,
            palette=palette,
            line_width=line_width,
            show_conf=show_conf,
        )
        annotated.save(output_dir / image_path.name)


def extract_metrics(results) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if hasattr(results, "metrics") and hasattr(results.metrics, "box"):
        box = results.metrics.box
        for attr, key in (("map50", "map50"), ("map", "map5095"), ("mp", "precision"), ("mr", "recall")):
            value = getattr(box, attr, None)
            if value is not None:
                metrics[key] = float(value)
    if hasattr(results, "results_dict"):
        mapping = {
            "metrics/mAP50(B)": "map50",
            "metrics/mAP50-95(B)": "map5095",
            "metrics/precision(B)": "precision",
            "metrics/recall(B)": "recall",
        }
        for src, dst in mapping.items():
            if dst not in metrics and src in results.results_dict:
                metrics[dst] = float(results.results_dict[src])
    return metrics


def write_metrics_report(
    metrics_file: Path,
    weights: Path,
    data_yaml: Path,
    results,
    extracted: Dict[str, float],
) -> None:
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = [
        "RT-DETR test metrics",
        f"weights: {weights}",
        f"data: {data_yaml}",
    ]
    if extracted:
        for key in ("map50", "map5095", "precision", "recall"):
            if key in extracted:
                lines.append(f"{key}: {extracted[key]:.4f}")
    else:
        lines.append("No metrics extracted from evaluation output.")

    speed = getattr(results, "speed", None)
    if isinstance(speed, dict):
        for key in ("preprocess", "inference", "postprocess"):
            if key in speed:
                lines.append(f"speed_{key}_ms: {float(speed[key]):.2f}")

    with metrics_file.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def evaluate_model(
    model,
    data_yaml: Path,
    imgsz: int,
    batch: int,
    device: str,
    iou: float,
    max_det: int,
):
    ensure_path(data_yaml, "file")
    results = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=imgsz,
        batch=batch,
        device=device,
        iou=iou,
        max_det=max_det,
        verbose=False,
    )
    return results


def main() -> None:
    args = parse_args()
    weights = args.weights.resolve()
    images_dir = args.images.resolve()
    output_dir = args.output_dir.resolve()
    metrics_file = args.metrics_file.resolve()
    data_yaml = args.data.resolve()
    device = resolve_device(args.device)

    ensure_path(images_dir, "dir")
    image_paths = list_images(images_dir)
    if not image_paths:
        raise SystemExit(f"No images with extensions {sorted(IMAGE_EXTENSIONS)} found under {images_dir}")

    model = load_model(weights)
    class_names = extract_class_names(model)
    palette = generate_palette(len(class_names))

    run_predictions(
        model=model,
        image_paths=image_paths,
        output_dir=output_dir,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        device=device,
        class_names=class_names,
        palette=palette,
        line_width=args.line_width,
        show_conf=args.save_conf,
    )

    results = evaluate_model(
        model=model,
        data_yaml=data_yaml,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        iou=args.iou,
        max_det=args.max_det,
    )
    extracted_metrics = extract_metrics(results)
    write_metrics_report(
        metrics_file=metrics_file,
        weights=weights,
        data_yaml=data_yaml,
        results=results,
        extracted=extracted_metrics,
    )

    print(f"Annotated predictions saved to: {output_dir}")
    print(f"Metrics summary written to: {metrics_file}")


if __name__ == "__main__":
    main()
