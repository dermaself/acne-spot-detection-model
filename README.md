# Acne Spot Detection Toolkit

Unified pipeline for acne lesion detection: curate the consolidated dataset, train RT-DETR models, and demo predictions through an interactive tiling-based Flask app.

![Tiled detection demo](screenshot.png)

## Features
- Automates dataset curation and label syncing for the unified `data/all` dataset.
- Trains RT-DETRv3 detectors with Ultralytics and records metrics per run.
- Serves a Flask UI that tiles large uploads, performs per-tile inference, and shows zoomed-in crops per class.

## Project Structure
- `scripts/` – dataset preparation, tiling, and training utilities.
- `app/` – Flask application for inference and visualisation.
- `plots/` – helper notebooks or plots generated during experiments.
- `runs/` – Ultralytics outputs (created during training).
- `screenshot.png` – demo capture embedded above.

## Script Reference
- `scripts/analyse_data.py` – Summarise image resolutions and annotation counts across the combined dataset.
- `scripts/build_combinedclasses.py` – Collapse rare classes into broader groups and emit a merged taxonomy YAML.
- `scripts/check_annotations.py` – Spot-check randomly sampled annotations by drawing bounding boxes for manual review.
- `scripts/combine_data.py` – Legacy helper to rebuild `data/all` from angle-specific folders if you still maintain them.
- `scripts/merge_datasets.py` – Merge multiple YOLO datasets while reconciling class indices and preserving structure.
- `scripts/plot_label_distribution.py` – Plot class count distributions for datasets that still track separate views.
- `scripts/plot_combined_label_distribution.py` – Visualise class counts after applying a merged-class mapping.
- `scripts/plot_label_distributions_combined.py` – Render original versus merged class distributions on stacked subplots.
- `scripts/sync_labels.py` – Copy consolidated YOLO labels into `data/all/labels` when sourced from a central label store.
- `scripts/tile_dataset.py` – Tile large images into 640×640 crops (25% overlap) and remap annotations.
- `scripts/train_rtdetr.py` – Train RT-DETRv3 on the unified dataset with predefined augmentations and logging.
- `scripts/train_rtdetr_l.py` – Variant trainer targeting RT-DETR-L with configurable augmentation profiles.
- `scripts/train_combinedclasses.py` – Convenience wrapper to train RT-DETR on the merged-and-tiled dataset with tuned defaults.
- `scripts/train_yolo11.py` – Train Ultralytics YOLO11 detectors using the same dataset preparation utilities.
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
1. Place all training images under `data/all/images`.
2. Ensure each image has a matching YOLO label file inside `data/all/labels` (same stem, `.txt` extension). If labels are stored in a consolidated directory, use `python scripts/sync_labels.py` to copy them across.
3. Keep raw consolidated labels under `data/alllabels/` if you rely on the sync helper or other downstream tooling.

`scripts/train_rtdetr.py` automatically materialises an 80/10/10 train/val/test split (using symlinks whenever possible) from `data/all` before each training run.

## Training RT-DETR
Train RT-DETRv3-XL on the unified dataset with augmentation (horizontal flips, ±15° rotation, light colour jitter, mosaic=0.2, mixup=0.1) and imgsz=1280, epochs=150:
```bash
python scripts/train_rtdetr.py all
```

- The script downloads `rtdetrv3_xl.pt` if it is missing (Ultralytics cache → Hugging Face mirror). For offline runs, download once and place it at the project root or inside `weights/`:
  ```bash
  mkdir -p weights
  # yolo download model=rtdetrv3_xl.pt
  mv /path/to/rtdetrv3_xl.pt weights/
  ```
- Key outputs copied into `data/all/`:
  - `checkpoint.pt` (best weights) and `last.pt`
  - `metrics.txt` with mAP@50, mAP@50-95, precision, recall, validation losses, and image counts
  - `results.csv`
- Ultralytics artefacts remain in `runs/all/<run_name>/`.
- Auto-generated dataset config at `data/all/dataset.yaml`.

### Advanced Options
```bash
python scripts/train_rtdetr.py all \
  --epochs 200 \
  --batch 12 \
  --weights rtdetrv3_xl.pt \
  --device 0 \
  --name all_experiment
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

## Tiled Training
To boost performance on tiny lesions, you can generate a tiled dataset and fine-tune with stronger augmentation:
1. Collapse classes as desired:
   ```bash
   python scripts/build_combinedclasses.py
   ```
2. Tile the merged dataset so each patch is 1024×1024 with 25% overlap:
   ```bash
   python scripts/tile_dataset.py \
     --source data/combinedclasses \
     --dest data/combinedclasses_tiled \
     --tile-size 1024 \
     --stride 768
   ```
3. Launch training with tuned augmentation on the tiled dataset:
   ```bash
   python scripts/train_combinedclasses.py
   ```