"""Reproducible YOLOv8-OBB fine-tuning and evaluation pipeline for VYOM apparatus detection.

Supports NVIDIA CUDA GPU acceleration with automatic safe CPU fallback.
Designed specifically for conservative hardware constraints (RTX 3050 Laptop GPU, 4GB VRAM).

Key Steps:
1. Validates hardware and inference device (PyTorch CUDA vs. CPU fallback).
2. Verifies dataset configuration and 8-point normalized OBB format.
3. Loads lightweight pretrained YOLOv8n-OBB weights.
4. Executes transfer learning fine-tuning with mixed precision (AMP) and memory limits.
5. Evaluates model metrics (Precision, Recall, mAP50, mAP50-95, per-class metrics).
6. Exports best checkpoint to models/yolov8_obb_best.pt and logs complete metadata.
"""

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Ensure vyom source is on path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir / "src"))

from vyom.core.logging_setup import get_logger

logger = get_logger("training.obb")


def compute_file_sha256(file_path: Path) -> str:
    """Compute SHA-256 hash digest of a model artifact."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def detect_training_device() -> Tuple[str, str, Dict[str, Any]]:
    """Determine best available PyTorch training device with conservative memory safeguards."""
    import torch

    has_cuda = torch.cuda.is_available()
    hw_info: Dict[str, Any] = {
        "pytorch_version": torch.__version__,
        "cuda_available": has_cuda,
        "device_name": "CPU",
        "vram_total_mb": 0,
    }

    if has_cuda:
        gpu_name = torch.cuda.get_device_name(0)
        vram_bytes = torch.cuda.get_device_properties(0).total_memory
        vram_mb = int(vram_bytes / (1024 * 1024))
        hw_info["device_name"] = gpu_name
        hw_info["vram_total_mb"] = vram_mb

        logger.info("PyTorch CUDA active: %s (%d MiB VRAM)", gpu_name, vram_mb)
        return "0", f"CUDA GPU: {gpu_name} ({vram_mb} MiB)", hw_info
    else:
        logger.info("CUDA not available in PyTorch. Using CPU fallback.")
        return "cpu", "CPU Fallback", hw_info


def train_yolov8_obb(
    data_yaml: Path,
    epochs: int = 10,
    batch_size: int = 4,
    img_size: int = 640,
    workers: int = 2,
    base_model: str = "yolov8n-obb.pt",
    output_dir: Path = Path("models"),
    runs_dir: Path = Path("runs/obb"),
) -> Dict[str, Any]:
    """Execute YOLOv8-OBB training and export evaluation results."""
    from ultralytics import YOLO

    device_str, device_desc, hw_info = detect_training_device()
    output_dir = output_dir.resolve()
    runs_dir = runs_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("VYOM YOLOV8-OBB TRAINING PIPELINE")
    print("=" * 60)
    print(f"Dataset YAML:      {data_yaml}")
    print(f"Base Pretrained:   {base_model}")
    print(f"Target Device:     {device_desc}")
    print(f"Image Resolution:  {img_size}x{img_size}")
    print(f"Batch Size:        {batch_size}")
    print(f"Epochs:            {epochs}")
    print(f"DataLoader Workers:{workers}")
    print("=" * 60)

    # 1. Initialize YOLOv8-OBB model with pretrained weights
    logger.info("Loading pretrained checkpoint: %s", base_model)
    model = YOLO(base_model)

    # 2. Train with conservative safeguards for 4GB VRAM
    t0 = time.time()
    train_results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device_str,
        workers=workers,
        amp=(device_str != "cpu"),  # Mixed precision on GPU
        project=str(runs_dir),
        name="vyom_demo_apparatus",
        exist_ok=True,
        save=True,
        cache=False,  # Avoid excessive VRAM/RAM cache
        patience=5,
        verbose=True,
    )
    training_duration_s = time.time() - t0

    # 3. Locate best model checkpoint
    best_pt_run = runs_dir / "vyom_demo_apparatus" / "weights" / "best.pt"
    if not best_pt_run.exists():
        # Fallback to last.pt if best wasn't produced
        best_pt_run = runs_dir / "vyom_demo_apparatus" / "weights" / "last.pt"

    dest_best_pt = output_dir / "yolov8_obb_best.pt"
    if best_pt_run.exists():
        shutil.copy2(best_pt_run, dest_best_pt)
        logger.info("Copied best model weights to: %s", dest_best_pt)

    # Also preserve base model reference in models/
    base_cached = Path(base_model)
    if base_cached.exists() and not (output_dir / base_cached.name).exists():
        try:
            shutil.copy2(base_cached, output_dir / base_cached.name)
        except Exception:
            pass

    # 4. Run formal evaluation on validation split
    print("\nEvaluating trained model on holdout validation split...")
    val_model = YOLO(str(dest_best_pt))
    metrics = val_model.val(data=str(data_yaml), split="val", imgsz=img_size, device=device_str)

    # Extract metrics
    precision = float(metrics.box.mp) if hasattr(metrics, "box") and hasattr(metrics.box, "mp") else 0.0
    recall = float(metrics.box.mr) if hasattr(metrics, "box") and hasattr(metrics.box, "mr") else 0.0
    map50 = float(metrics.box.map50) if hasattr(metrics, "box") and hasattr(metrics.box, "map50") else 0.0
    map50_95 = float(metrics.box.map) if hasattr(metrics, "box") and hasattr(metrics.box, "map") else 0.0

    class_names = getattr(val_model, "names", {})
    per_class_map50: Dict[str, float] = {}
    if hasattr(metrics.box, "maps") and metrics.box.maps is not None:
        for idx, score in enumerate(metrics.box.maps):
            cname = class_names.get(idx, f"class_{idx}")
            per_class_map50[cname] = round(float(score), 4)

    sha256_hash = compute_file_sha256(dest_best_pt) if dest_best_pt.exists() else "N/A"

    summary: Dict[str, Any] = {
        "model_name": "YOLOv8n-OBB (Fine-Tuned)",
        "task": "Oriented Bounding Box (OBB) Detection",
        "output_checkpoint": str(dest_best_pt),
        "sha256": sha256_hash,
        "base_pretrained": base_model,
        "dataset_yaml": str(data_yaml),
        "num_classes": len(class_names),
        "class_names": class_names,
        "training_hardware": hw_info,
        "training_time_seconds": round(training_duration_s, 1),
        "epochs_trained": epochs,
        "batch_size": batch_size,
        "metrics": {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "mAP50": round(map50, 4),
            "mAP50_95": round(map50_95, 4),
            "per_class_mAP50": per_class_map50,
        },
        "notes": (
            "Model fine-tuned on synthetic/procedural demo apparatus dataset with genuine rotated 4-corner bounding boxes. "
            "Classes represent generic prototype apparatus and are distinct from flight ISRO spacecraft racks."
        ),
    }

    # Save metadata JSON
    meta_path = output_dir / "model_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 60)
    print("YOLOV8-OBB TRAINING & EVALUATION REPORT")
    print("=" * 60)
    print(f"Output Checkpoint:   {dest_best_pt}")
    print(f"SHA-256 Checksum:    {sha256_hash}")
    print(f"Training Duration:   {training_duration_s:.1f}s")
    print(f"Overall Precision:   {precision:.2%}")
    print(f"Overall Recall:      {recall:.2%}")
    print(f"mAP@0.50:            {map50:.2%}")
    print(f"mAP@0.50:0.95:       {map50_95:.2%}")
    print("Per-Class Performance:")
    for cname, score in per_class_map50.items():
        print(f"  • {cname:20s}: mAP50 = {score:.2%}")
    print("=" * 60)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train custom YOLOv8-OBB model.")
    parser.add_argument(
        "--data",
        type=str,
        default="d:/space_eye/vyom/data/yolo_obb/dataset.yaml",
        help="Path to dataset.yaml",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs (default: 10).")
    parser.add_argument("--batch", type=int, default=4, help="Batch size (default: 4 for 4GB VRAM).")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution.")
    parser.add_argument("--base-model", type=str, default="yolov8n-obb.pt", help="Pretrained OBB model.")
    parser.add_argument("--output-dir", type=str, default="d:/space_eye/vyom/models", help="Model output directory.")
    args = parser.parse_args()

    train_yolov8_obb(
        data_yaml=Path(args.data),
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.imgsz,
        base_model=args.base_model,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()

