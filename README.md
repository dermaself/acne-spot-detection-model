# Acne Spot Detection Toolkit

Full pipeline for acne lesion detection: prepare split-specific datasets, train RT-DETR models, and demo detections through an interactive tiling-based Flask app.

![Tiled detection demo](screenshot.png)

## Features
- Automates dataset curation and label syncing for `front`, `scan`, and `side` splits.
- Trains RT-DETRv3 detectors with Ultralytics and records metrics per run.
- Serves a Flask UI that tiles large uploads, performs per-tile inference, and shows zoomed-in crops per class.

## Project Structure
- `scripts/` – dataset preparation, tiling, and training utilities.
- `app/` – Flask application for inference and visualisation.
- `plots/` – helper notebooks or plots generated during experiments.
- `runs/` – Ultralytics outputs (created during training).
- `screenshot.png` – demo capture embedded above.

## Script Reference
- `scripts/analyse_data.py` – Summarise combined dataset image resolutions and label counts, writing plots and CSVs.
- `scripts/build_combinedclasses.py` – Merge the view-specific datasets, collapse rare classes, and emit a merged taxonomy yaml.
- `scripts/check_annotations.py` – Spot-check randomly sampled annotations by drawing bounding boxes for manual review.
- `scripts/combine_data.py` – Copy images and labels from `front`, `scan`, and `side` into a single `data/all` dataset.
- `scripts/merge_datasets.py` – Merge disparate YOLO datasets while reconciling class indices and preserving split structure.
- `scripts/plot_label_distribution.py` – Plot per-class label counts for the original splits.
- `scripts/plot_combined_label_distribution.py` – Visualise label counts after applying the merged-class mapping.
- `scripts/plot_label_distributions_combined.py` – Render original versus merged class distributions on stacked subplots.
- `scripts/sync_labels.py` – Copy consolidated YOLO labels into each split’s `labels/` directory.
- `scripts/tile_dataset.py` – Tile large images into 640×640 crops (25% overlap) and remap annotations.
- `scripts/train_rtdetr.py` – Train RT-DETRv3 on any dataset split with predefined augmentations and logging.
- `scripts/train_rtdetr_l.py` – Variant trainer targeting RT-DETR-L with configurable augmentation profiles.
- `scripts/train_combinedclasses.py` – Convenience wrapper to train RT-DETR on the merged, tiled dataset with tuned defaults.
- `scripts/train_yolo11.py` – Train Ultralytics YOLO11 detectors using the same split preparation utilities as RT-DETR.
- `scripts/train_yolonas.py` – Launch a SuperGradients YOLO-NAS training run with letterbox preprocessing and COCO metrics.
- `scripts/test_rtdetr.py` – Run RT-DETR inference on a folder of images, export annotated outputs, and compute metrics.
- `scripts/test_yolonas.py` – Evaluate a trained YOLO-NAS checkpoint and save annotated examples plus COCO-style metrics.

## Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Prerequisites:
- Python 3.10+
- NVIDIA GPU with recent CUDA drivers (tested against CUDA 11.8+)
- [PyTorch](https://pytorch.org/get-started/locally/) compiled for your CUDA version

## Dataset Preparation
1. Place raw images under `data/<split>/images` for `front`, `scan`, and `side`.
2. Put the consolidated YOLO labels (`.txt`) inside `data/alllabels`.
3. Populate each split’s `labels` directory:
   ```bash
   python scripts/sync_labels.py
   ```

`train_rtdetr.py` automatically builds an 80/10/10 train/val/test split (using symlinks when available) before each run.

## Training RT-DETR
Train RT-DETRv3-XL for any split with augmentation (horizontal flips, ±15° rotation, light colour jitter, mosaic=0.2, mixup=0.1) and imgsz=1280, epochs=150:
```bash
# front, scan, or side
python scripts/train_rtdetr.py front
```

- The script downloads `rtdetrv3_xl.pt` if missing (Ultralytics cache → Hugging Face mirror). For offline runs, download once and place it at the project root or `weights/`:
  ```bash
  mkdir -p weights
  # yolo download model=rtdetrv3_xl.pt
  mv /path/to/rtdetrv3_xl.pt weights/
  ```
- Key outputs copied into `data/<split>/`:
  - `checkpoint.pt` (best weights) and `last.pt`
  - `metrics.txt` with mAP@50, mAP@50-95, precision, recall, validation losses, and split image counts
  - `results.csv`
- Ultralytics artefacts remain in `runs/<split>/<run_name>/`.
- Auto-generated dataset config at `data/<split>/dataset.yaml`.

### Advanced Options
```bash
python scripts/train_rtdetr.py scan \
  --epochs 200 \
  --batch 12 \
  --weights rtdetrv3_xl.pt \
  --device 0 \
  --name scan_experiment
```

All CLI arguments mirror Ultralytics options for convenient overrides.

## Flask Inference App
The web app tiles large uploads, runs the detector on each tile, stitches detections back together, and surfaces zoomed-in thumbnails per class for rapid review.

1. Ensure the trained weights referenced in `app/__init__.py` exist (default path: `runs/new/rtdetr_l_safe/best.pt`).
2. Activate your virtual environment and launch the server:
   ```bash
   python -m app
   ```
3. Open the printed URL (defaults to `http://127.0.0.1:5000`) and upload an image. Use the sliders to adjust confidence/IoU thresholds, explore detections, and inspect per-class crops.
