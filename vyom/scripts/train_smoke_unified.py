"""Initial Custom YOLOv8-OBB Smoke Training Pipeline on Final Unified Dataset.

Configured according to exact hardware and constraint specifications:
- Model: YOLOv8n-OBB pretrained checkpoint (d:/space_eye/vyom/models/yolov8n-obb.pt)
- Dataset: d:/space_eye/vyom/data/yolo_obb_final/dataset.yaml
- Image size: 640x640
- Epochs: 5
- Batch size: 2 (conservative memory limit for 4GB VRAM)
- Device: CUDA 0 (NVIDIA GeForce RTX 3050 Laptop GPU)
- AMP: True
- Workers: 0 (safe single-process dataloading on Windows)
- Cache: False
- Output checkpoint: models/vyom_yolov8n_obb_smoke.pt
- Separate evaluation breakdown: Real-world classes vs Provisional synthetic-only classes
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO

ROOT_DIR = Path("d:/space_eye/vyom")
DATASET_YAML = ROOT_DIR / "data" / "yolo_obb_final" / "dataset.yaml"
BASE_MODEL_PT = ROOT_DIR / "models" / "yolov8n-obb.pt"
OUTPUT_DIR = ROOT_DIR / "models"
RUNS_DIR = Path("d:/space_eye/runs/obb")

FINAL_CLASSES = [
    "centrifuge_tube",     # 0: Real (LabOS) + Synthetic v2
    "tube_rack",           # 1: Real (LabOS + ChemEq25)
    "rack_hole",           # 2: Real (LabOS)
    "tube_cap",            # 3: Real (LabOS)
    "pipette",             # 4: Real (ChemEq25) + Synthetic v2
    "beaker",              # 5: Real (ChemEq25)
    "conical_flask",       # 6: Real (ChemEq25)
    "sample_vial",         # 7: Provisional (Synthetic v2)
    "multiwell_plate",     # 8: Provisional (Synthetic v2)
    "rig_slot",            # 9: Provisional (Synthetic v2)
]

PROVISIONAL_SYNTHETIC_CLASSES = {"sample_vial", "multiwell_plate", "rig_slot"}
REAL_WORLD_CLASSES = {"centrifuge_tube", "tube_rack", "rack_hole", "tube_cap", "pipette", "beaker", "conical_flask"}


def compute_file_sha256(file_path: Path) -> str:
    """Compute SHA-256 hash digest of a model artifact."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def run_smoke_training(
    epochs: int = 5,
    batch_size: int = 2,
    img_size: int = 640,
    device_id: str = "0",
) -> Dict[str, Any]:
    """Execute smoke training on final unified dataset and save vyom_yolov8n_obb_smoke.pt."""
    print("=" * 65)
    print("VYOM INITIAL CUSTOM YOLOV8-OBB SMOKE TRAINING")
    print("=" * 65)

    assert torch.cuda.is_available(), "CUDA is not available in PyTorch!"
    gpu_name = torch.cuda.get_device_name(0)
    total_vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
    print(f"Target Device:     CUDA:{device_id} ({gpu_name}, {total_vram_mb:.0f} MiB VRAM)")
    print(f"Dataset YAML:      {DATASET_YAML}")
    print(f"Base Pretrained:   {BASE_MODEL_PT}")
    print(f"Image Size:        {img_size}x{img_size}")
    print(f"Batch Size:        {batch_size} (conservative for 4GB VRAM)")
    print(f"Epochs:            {epochs}")
    print(f"Workers:           0 (Windows-safe)")
    print(f"AMP:               True")
    print(f"Cache:             False")
    print("=" * 65)

    # 1. Initialize YOLO model with pretrained weights
    print(f"\nLoading pretrained weights: {BASE_MODEL_PT}")
    model = YOLO(str(BASE_MODEL_PT))

    # 2. Train model
    exp_name = "vyom_smoke_unified"
    t0 = time.time()
    train_results = model.train(
        data=str(DATASET_YAML),
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device_id,
        workers=0,
        amp=True,
        project=str(RUNS_DIR),
        name=exp_name,
        exist_ok=True,
        save=True,
        cache=False,
        patience=5,
        verbose=True,
    )
    training_duration_s = time.time() - t0

    # Max GPU memory allocated
    peak_vram_mb = torch.cuda.max_memory_allocated(0) / (1024 * 1024)

    # 3. Locate best checkpoint and copy to models/vyom_yolov8n_obb_smoke.pt
    run_best_pt = RUNS_DIR / exp_name / "weights" / "best.pt"
    if not run_best_pt.exists():
        run_best_pt = RUNS_DIR / exp_name / "weights" / "last.pt"

    dest_smoke_pt = OUTPUT_DIR / "vyom_yolov8n_obb_smoke.pt"
    shutil.copy2(run_best_pt, dest_smoke_pt)
    sha256_hash = compute_file_sha256(dest_smoke_pt)
    print(f"\nSaved smoke checkpoint: {dest_smoke_pt} (SHA-256: {sha256_hash})")

    # 4. Formal Holdout Validation
    print("\nRunning formal evaluation on holdout validation split (val)...")
    val_model = YOLO(str(dest_smoke_pt))
    metrics = val_model.val(data=str(DATASET_YAML), split="val", imgsz=img_size, device=device_id)

    # Extract overall metrics
    box_metrics = getattr(metrics, "box", None) or getattr(metrics, "obb", None)
    precision = float(box_metrics.mp) if box_metrics and hasattr(box_metrics, "mp") else 0.0
    recall = float(box_metrics.mr) if box_metrics and hasattr(box_metrics, "mr") else 0.0
    map50 = float(box_metrics.map50) if box_metrics and hasattr(box_metrics, "map50") else 0.0
    map50_95 = float(box_metrics.map) if box_metrics and hasattr(box_metrics, "map") else 0.0

    class_names = getattr(val_model, "names", {})
    per_class_map50: Dict[str, float] = {}
    if box_metrics and hasattr(box_metrics, "maps") and box_metrics.maps is not None:
        for idx, score in enumerate(box_metrics.maps):
            cname = class_names.get(idx, f"class_{idx}")
            per_class_map50[cname] = round(float(score), 4)

    # Separate real-world vs synthetic metrics
    real_class_scores = {k: v for k, v in per_class_map50.items() if k in REAL_WORLD_CLASSES}
    syn_class_scores = {k: v for k, v in per_class_map50.items() if k in PROVISIONAL_SYNTHETIC_CLASSES}

    mean_real_map50 = float(np.mean(list(real_class_scores.values()))) if real_class_scores else 0.0
    mean_syn_map50 = float(np.mean(list(syn_class_scores.values()))) if syn_class_scores else 0.0

    # 5. Generate visual prediction previews
    print("\nGenerating prediction previews on sample validation images...")
    preview_dir = OUTPUT_DIR / "smoke_eval_previews"
    preview_dir.mkdir(parents=True, exist_ok=True)

    val_imgs_dir = ROOT_DIR / "data" / "yolo_obb_final" / "images" / "val"
    sample_val_imgs = []
    for prefix in ["labos", "chemeq", "syn"]:
        matches = sorted(list(val_imgs_dir.glob(f"{prefix}*")))
        sample_val_imgs.extend(matches[:2])

    for s_img in sample_val_imgs:
        if s_img.exists():
            res = val_model.predict(str(s_img), imgsz=img_size, conf=0.25, device=device_id)
            if res:
                annotated = res[0].plot()
                out_p = preview_dir / f"pred_{s_img.stem}.jpg"
                cv2.imwrite(str(out_p), annotated)
                print(f"  • Preview saved: {out_p.name}")

    # 6. Save comprehensive metadata
    meta = {
        "model_name": "VYOM Custom YOLOv8n-OBB Smoke Model",
        "checkpoint_filename": dest_smoke_pt.name,
        "checkpoint_path": str(dest_smoke_pt),
        "sha256": sha256_hash,
        "base_pretrained": str(BASE_MODEL_PT),
        "dataset_yaml": str(DATASET_YAML),
        "num_classes": len(class_names),
        "class_names": class_names,
        "provisional_synthetic_classes": list(PROVISIONAL_SYNTHETIC_CLASSES),
        "real_world_classes": list(REAL_WORLD_CLASSES),
        "hardware": {
            "device_name": gpu_name,
            "total_vram_mb": round(total_vram_mb, 1),
            "peak_vram_used_mb": round(peak_vram_mb, 1),
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        },
        "training_hyperparameters": {
            "epochs": epochs,
            "batch_size": batch_size,
            "imgsz": img_size,
            "amp": True,
            "workers": 0,
            "cache": False,
            "experiment_dir": str(RUNS_DIR / exp_name),
        },
        "training_duration_seconds": round(training_duration_s, 1),
        "metrics": {
            "overall_precision": round(precision, 4),
            "overall_recall": round(recall, 4),
            "overall_mAP50": round(map50, 4),
            "overall_mAP50_95": round(map50_95, 4),
            "real_world_classes_mean_mAP50": round(mean_real_map50, 4),
            "provisional_synthetic_classes_mean_mAP50": round(mean_syn_map50, 4),
            "per_class_mAP50": per_class_map50,
        },
    }

    meta_path = OUTPUT_DIR / "smoke_model_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved smoke model metadata to: {meta_path}")

    print("\n" + "=" * 65)
    print("YOLOV8-OBB SMOKE TRAINING COMPLETE")
    print("=" * 65)
    print(f"Checkpoint:           {dest_smoke_pt}")
    print(f"Training Duration:    {training_duration_s:.1f}s")
    print(f"Peak VRAM Allocated:  {peak_vram_mb:.1f} MiB")
    print(f"Overall Precision:    {precision:.2%}")
    print(f"Overall Recall:       {recall:.2%}")
    print(f"Overall mAP@0.50:     {map50:.2%}")
    print(f"Overall mAP@0.50:0.95:{map50_95:.2%}")
    print("Real-World Categories Performance:")
    for cname in sorted(REAL_WORLD_CLASSES):
        score = per_class_map50.get(cname, 0.0)
        print(f"  • {cname:20s}: mAP50 = {score:.2%}")
    print("Provisional Synthetic-Only Categories Performance:")
    for cname in sorted(PROVISIONAL_SYNTHETIC_CLASSES):
        score = per_class_map50.get(cname, 0.0)
        print(f"  • {cname:20s}: mAP50 = {score:.2%} [PROVISIONAL]")
    print("=" * 65)

    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLOv8-OBB initial smoke training.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs (default: 5).")
    parser.add_argument("--batch", type=int, default=2, help="Batch size (default: 2 for 4GB VRAM).")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution.")
    args = parser.parse_args()

    run_smoke_training(
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.imgsz,
    )


if __name__ == "__main__":
    main()
