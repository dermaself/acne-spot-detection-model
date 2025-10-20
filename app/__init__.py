from __future__ import annotations

import base64
import io
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from flask import Flask, Response, jsonify, render_template, request, stream_with_context
from PIL import Image, UnidentifiedImageError

try:
    from ultralytics import RTDETR as _UltralyticsModel
except ImportError:  # pragma: no cover - informative failure
    try:
        from ultralytics import YOLO as _UltralyticsModel
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "Ultralytics is required to serve the RT-DETR detector. Install ultralytics>=8.3.20."
        ) from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO_ROOT / "runs" / "new" / "rtdetr_l_safe" / "best.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMGSZ = 640
TILE_SIZE = 640
TILE_STRIDE = 480
INFERENCE_CONF = 0.05
INFERENCE_IOU = 0.85
DEFAULT_DISPLAY_CONF = 0.30
DEFAULT_DISPLAY_IOU = 0.70


@dataclass(frozen=True)
class TileSpec:
    canvas: np.ndarray
    origin_x: int
    origin_y: int
    pad_left: int
    pad_top: int
    patch_w: int
    patch_h: int


_model = _UltralyticsModel(str(MODEL_PATH))
_CLASS_NAMES: Sequence[str]
names = getattr(_model, "names", None)
if isinstance(names, dict):
    _CLASS_NAMES = [names[k] for k in sorted(names)]
elif isinstance(names, (list, tuple)):
    _CLASS_NAMES = list(names)
else:  # pragma: no cover - defensive
    raise ValueError("Unable to determine class names from RT-DETR weights.")


def _generate_palette(num_classes: int) -> List[Tuple[int, int, int]]:
    if num_classes <= 0:
        return [(255, 0, 0)]
    import colorsys

    palette: List[Tuple[int, int, int]] = []
    for idx in range(num_classes):
        hue = (idx / max(1, num_classes)) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 1.0)
        palette.append((int(r * 255), int(g * 255), int(b * 255)))
    return palette


PALETTE = _generate_palette(len(_CLASS_NAMES))


def _tile_positions(length: int, tile: int, stride: int) -> List[int]:
    if tile >= length:
        return [0]
    positions = list(range(0, length - tile + 1, stride))
    last_pos = length - tile
    if not positions or positions[-1] != last_pos:
        positions.append(last_pos)
    deduped: List[int] = []
    seen = set()
    for pos in positions:
        if pos not in seen:
            deduped.append(pos)
            seen.add(pos)
    return deduped


def _build_tile(patch: np.ndarray, origin_x: int, origin_y: int) -> TileSpec:
    patch_h, patch_w = patch.shape[:2]
    canvas = np.zeros((TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8)
    pad_left = (TILE_SIZE - patch_w) // 2
    pad_top = (TILE_SIZE - patch_h) // 2
    canvas[pad_top : pad_top + patch_h, pad_left : pad_left + patch_w] = patch
    return TileSpec(
        canvas=canvas,
        origin_x=origin_x,
        origin_y=origin_y,
        pad_left=pad_left,
        pad_top=pad_top,
        patch_w=patch_w,
        patch_h=patch_h,
    )


def _tile_iterator(image: np.ndarray, x_positions: List[int], y_positions: List[int]) -> Iterable[TileSpec]:
    height, width = image.shape[:2]
    for top in y_positions:
        bottom = min(top + TILE_SIZE, height)
        for left in x_positions:
            right = min(left + TILE_SIZE, width)
            patch = image[top:bottom, left:right]
            yield _build_tile(patch, origin_x=left, origin_y=top)


def _extract_thumbnail(image: np.ndarray, x1: float, y1: float, x2: float, y2: float, pad_ratio: float = 0.5, thumb_size: int = 160) -> str:
    h, w = image.shape[:2]
    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)
    pad_w = box_w * pad_ratio
    pad_h = box_h * pad_ratio

    crop_x1 = max(0, int(math.floor(x1 - pad_w)))
    crop_y1 = max(0, int(math.floor(y1 - pad_h)))
    crop_x2 = min(w, int(math.ceil(x2 + pad_w)))
    crop_y2 = min(h, int(math.ceil(y2 + pad_h)))

    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        crop_x1 = max(0, int(math.floor(x1)))
        crop_y1 = max(0, int(math.floor(y1)))
        crop_x2 = min(w, int(math.ceil(x2)))
        crop_y2 = min(h, int(math.ceil(y2)))

    crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
    if crop.size == 0:
        crop = np.zeros((thumb_size, thumb_size, 3), dtype=np.uint8)

    pil_crop = Image.fromarray(crop.astype(np.uint8))
    if thumb_size > 0:
        pil_crop = pil_crop.resize((thumb_size, thumb_size), Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)

    buffer = io.BytesIO()
    pil_crop.save(buffer, format='JPEG', quality=85)
    encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f'data:image/jpeg;base64,{encoded}'


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024  # 15 MB uploads

    @app.get("/")
    def index():
        return render_template(
            "index.html",
            default_conf=DEFAULT_DISPLAY_CONF,
            default_iou=DEFAULT_DISPLAY_IOU,
        )

    @app.post("/api/detect")
    def api_detect():
        file = request.files.get("photo")
        if file is None or file.filename == "":
            return jsonify({"error": "No image file provided."}), 400

        try:
            pil_image = Image.open(file.stream).convert("RGB")
        except UnidentifiedImageError:
            return jsonify({"error": "Unsupported image format."}), 400
        except Exception as exc:  # pragma: no cover - informative failure
            return jsonify({"error": f"Unable to read the image: {exc}"}), 400

        image_np = np.array(pil_image)
        height, width = image_np.shape[:2]

        x_positions = _tile_positions(width, TILE_SIZE, TILE_STRIDE)
        y_positions = _tile_positions(height, TILE_SIZE, TILE_STRIDE)
        total_tiles = max(1, len(x_positions) * len(y_positions))

        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG", quality=95)
        image_data = base64.b64encode(buffer.getvalue()).decode("utf-8")

        palette_size = max(1, len(PALETTE))

        def stream() -> Iterable[str]:
            boxes_payload: List[Dict[str, float]] = []
            detection_counter = 0
            try:
                start_payload = {
                    "type": "start",
                    "total_tiles": int(total_tiles),
                    "image_width": int(width),
                    "image_height": int(height),
                    "image_data": image_data,
                }
                yield json.dumps(start_payload) + "\n"

                for index, tile in enumerate(_tile_iterator(image_np, x_positions, y_positions), start=1):
                    results = _model.predict(
                        source=tile.canvas,
                        stream=False,
                        imgsz=IMGSZ,
                        conf=INFERENCE_CONF,
                        iou=INFERENCE_IOU,
                        max_det=300,
                        device=DEVICE,
                        verbose=False,
                    )
                    prediction = results[0] if isinstance(results, list) else results
                    boxes = getattr(prediction, "boxes", None)
                    if boxes is not None and len(boxes) > 0:
                        xyxy = boxes.xyxy.cpu().numpy()
                        confs = boxes.conf.cpu().numpy()
                        classes = boxes.cls.cpu().numpy().astype(int)
                        for coords, conf, cls_idx in zip(xyxy, confs, classes):
                            x1, y1, x2, y2 = coords.tolist()
                            adj_x1 = max(0.0, float(x1) - tile.pad_left)
                            adj_y1 = max(0.0, float(y1) - tile.pad_top)
                            adj_x2 = min(float(tile.patch_w), float(x2) - tile.pad_left)
                            adj_y2 = min(float(tile.patch_h), float(y2) - tile.pad_top)
                            if adj_x2 <= adj_x1 or adj_y2 <= adj_y1:
                                continue
                            global_x1 = max(0.0, min(float(width), tile.origin_x + adj_x1))
                            global_y1 = max(0.0, min(float(height), tile.origin_y + adj_y1))
                            global_x2 = max(0.0, min(float(width), tile.origin_x + adj_x2))
                            global_y2 = max(0.0, min(float(height), tile.origin_y + adj_y2))
                            if global_x2 - global_x1 <= 0.0 or global_y2 - global_y1 <= 0.0:
                                continue
                            class_name = (
                                _CLASS_NAMES[cls_idx] if 0 <= cls_idx < len(_CLASS_NAMES) else f"class_{cls_idx}"
                            )
                            color = PALETTE[cls_idx % palette_size]
                            thumbnail_data = _extract_thumbnail(image_np, global_x1, global_y1, global_x2, global_y2)
                            boxes_payload.append(
                                {
                                    "id": int(detection_counter),
                                    "label": class_name,
                                    "confidence": round(float(conf), 4),
                                    "x1": float(global_x1),
                                    "y1": float(global_y1),
                                    "x2": float(global_x2),
                                    "y2": float(global_y2),
                                    "color": f"rgb({color[0]}, {color[1]}, {color[2]})",
                                    "class_id": int(cls_idx),
                                    "thumbnail": thumbnail_data,
                                }
                            )
                            detection_counter += 1

                    yield json.dumps({"type": "progress", "current": index, "total": int(total_tiles)}) + "\n"

                yield json.dumps({"type": "complete", "boxes": boxes_payload}) + "\n"
            except Exception as exc:  # pragma: no cover - best effort streaming error
                yield json.dumps({"type": "error", "message": str(exc)}) + "\n"

        return Response(stream_with_context(stream()), mimetype="application/x-ndjson")

    return app


app = create_app()
