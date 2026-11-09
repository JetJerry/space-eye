"""Comprehensive dataset inspection and audit script across all downloaded and synthetic sources.
Audits:
1. LabOS (Real Laboratory Instance Segmentation)
2. ChemEq25 (Real Chemistry Laboratory Object Detection)
3. ENIGMA-360 (Real Industrial / Laboratory Procedure Videos)
4. Synthetic v1 (Initial Procedural Baseline)
5. Synthetic v2 (Improved Disjoint Backgrounds Procedural Baseline)
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple
import cv2
import yaml

ROOT_DIR = Path("d:/space_eye/vyom")
DATASETS_DIR = Path("D:/space_eye/datasets")


def audit_labos() -> Dict[str, Any]:
    """Inspect LabOS instance segmentation dataset."""
    labos_dir = DATASETS_DIR / "LabOS"
    images_dir = labos_dir / "images"
    labels_dir = labos_dir / "labels"

    imgs = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg"))
    lbls = list(labels_dir.glob("*.txt"))

    # Load COCO annotations
    ann_train = json.loads((labos_dir / "annotations_train.json").read_text(encoding="utf-8"))
    ann_val = json.loads((labos_dir / "annotations_val.json").read_text(encoding="utf-8"))

    # Check image resolutions and corrupted files
    shapes: Set[Tuple[int, int, int]] = set()
    corrupted = 0
    for img_p in imgs:
        im = cv2.imread(str(img_p))
        if im is None:
            corrupted += 1
        else:
            shapes.add(im.shape)

    # Class statistics from COCO
    categories = {c["id"]: c["name"] for c in ann_train["categories"]}
    train_cat_counts = Counter(a["category_id"] for a in ann_train["annotations"])
    val_cat_counts = Counter(a["category_id"] for a in ann_val["annotations"])

    class_stats = []
    for cid, cname in categories.items():
        tr_c = train_cat_counts.get(cid, 0)
        val_c = val_cat_counts.get(cid, 0)
        class_stats.append({
            "id": cid,
            "name": cname,
            "train": tr_c,
            "val": val_c,
            "total": tr_c + val_c,
        })

    return {
        "name": "LabOS",
        "location": str(labos_dir),
        "license": "CC BY-NC 4.0",
        "format": "COCO Instance Segmentation & YOLO Polygon",
        "total_images": len(imgs),
        "train_images": len(ann_train["images"]),
        "val_images": len(ann_val["images"]),
        "total_annotations": len(ann_train["annotations"]) + len(ann_val["annotations"]),
        "corrupted_images": corrupted,
        "image_shapes": [list(s) for s in shapes],
        "classes": class_stats,
    }


def audit_chemeq25() -> Dict[str, Any]:
    """Inspect ChemEq25 chemistry laboratory object detection dataset."""
    chemeq_dir = DATASETS_DIR / "ChemEq25"

    with open(chemeq_dir / "data.yaml", "r", encoding="utf-8") as f:
        data_yaml = yaml.safe_load(f)

    class_names = data_yaml.get("names", [])
    roboflow_meta = data_yaml.get("roboflow", {})

    splits = ["train", "valid", "test"]
    split_counts = {}
    class_counts_by_split: Dict[str, Counter] = {}

    corrupted = 0
    shapes: Set[Tuple[int, int, int]] = set()

    for s in splits:
        img_dir = chemeq_dir / s / "images"
        lbl_dir = chemeq_dir / s / "labels"
        imgs = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
        split_counts[s] = len(imgs)

        cnt = Counter()
        for lbl_p in lbl_dir.glob("*.txt"):
            lines = lbl_p.read_text().splitlines()
            for l in lines:
                parts = l.strip().split()
                if len(parts) == 5:
                    cnt[int(parts[0])] += 1
        class_counts_by_split[s] = cnt

        # Check sample shapes
        for img_p in imgs[:25]:
            im = cv2.imread(str(img_p))
            if im is None:
                corrupted += 1
            else:
                shapes.add(im.shape)

    class_stats = []
    for cid, cname in enumerate(class_names):
        tr_c = class_counts_by_split["train"].get(cid, 0)
        val_c = class_counts_by_split["valid"].get(cid, 0)
        te_c = class_counts_by_split["test"].get(cid, 0)
        class_stats.append({
            "id": cid,
            "name": cname,
            "train": tr_c,
            "val": val_c,
            "test": te_c,
            "total": tr_c + val_c + te_c,
        })

    total_ann = sum(s["total"] for s in class_stats)

    return {
        "name": "ChemEq25",
        "location": str(chemeq_dir),
        "license": roboflow_meta.get("license", "CC BY 4.0"),
        "url": roboflow_meta.get("url", ""),
        "format": "YOLO Axis-Aligned Bounding Box (5 tokens)",
        "total_images": sum(split_counts.values()),
        "train_images": split_counts["train"],
        "val_images": split_counts["valid"],
        "test_images": split_counts["test"],
        "total_annotations": total_ann,
        "corrupted_images": corrupted,
        "image_shapes": [list(s) for s in shapes],
        "classes": class_stats,
    }


def audit_enigma360() -> Dict[str, Any]:
    """Inspect ENIGMA-360 video and annotation properties."""
    enigma_dir = DATASETS_DIR / "ENIGMA-360"
    train_txt = enigma_dir / "annotations" / "train.txt"
    val_txt = enigma_dir / "annotations" / "val.txt"

    train_vids = train_txt.read_text().splitlines() if train_txt.exists() else []
    val_vids = val_txt.read_text().splitlines() if val_txt.exists() else []

    all_vids = list((enigma_dir / "data" / "videos").glob("*/*_third_person.mp4"))

    return {
        "name": "ENIGMA-360",
        "location": str(enigma_dir),
        "license": "Research / Academic Use Only",
        "format": "Exocentric Third-Person MP4 Videos (672x376 @ 30fps)",
        "total_videos": len(all_vids),
        "train_videos_count": len(train_vids),
        "val_videos_count": len(val_vids),
        "total_duration_hours": 36.35,
        "annotation_compatibility": (
            "Third-person exocentric videos provide authentic laboratory workbench backgrounds. "
            "HOI instance masks are egocentric Aria glasses (2272x1278) and coarse (hand/object only); "
            "they cannot be used directly for third-person exocentric object detection."
        ),
    }


def audit_synthetic_v1() -> Dict[str, Any]:
    """Inspect Synthetic v1 dataset."""
    v1_dir = ROOT_DIR / "data" / "yolo_obb"
    train_imgs = list((v1_dir / "images" / "train").glob("*.jpg"))
    val_imgs = list((v1_dir / "images" / "val").glob("*.jpg"))

    class_names = ["sample_vial", "pipette", "multiwell_plate", "centrifuge_tube", "rig_slot"]
    cnt_train: Counter = Counter()
    cnt_val: Counter = Counter()

    for p in (v1_dir / "labels" / "train").glob("*.txt"):
        for l in p.read_text().splitlines():
            parts = l.strip().split()
            if len(parts) == 9:
                cnt_train[int(parts[0])] += 1

    for p in (v1_dir / "labels" / "val").glob("*.txt"):
        for l in p.read_text().splitlines():
            parts = l.strip().split()
            if len(parts) == 9:
                cnt_val[int(parts[0])] += 1

    return {
        "name": "Synthetic v1 (yolo_obb)",
        "location": str(v1_dir),
        "format": "YOLOv8-OBB (9 tokens)",
        "total_images": len(train_imgs) + len(val_imgs),
        "train_images": len(train_imgs),
        "val_images": len(val_imgs),
        "total_instances": sum(cnt_train.values()) + sum(cnt_val.values()),
        "status": "Historical Baseline Only (Preserved, Do Not Merge)",
        "limitations": "Simple 2D OpenCV shapes, train/val backgrounds both sampled from video 103 (leakage).",
        "classes": {
            class_names[i]: {"train": cnt_train[i], "val": cnt_val[i], "total": cnt_train[i] + cnt_val[i]}
            for i in range(5)
        },
    }


def audit_synthetic_v2() -> Dict[str, Any]:
    """Inspect Synthetic v2 dataset."""
    v2_dir = ROOT_DIR / "data" / "yolo_obb_v2"
    train_imgs = list((v2_dir / "images" / "train").glob("*.jpg"))
    val_imgs = list((v2_dir / "images" / "val").glob("*.jpg"))

    manifest = json.loads((v2_dir / "manifest.json").read_text(encoding="utf-8"))

    return {
        "name": "Synthetic v2 (yolo_obb_v2)",
        "location": str(v2_dir),
        "format": "YOLOv8-OBB (9 tokens)",
        "total_images": len(train_imgs) + len(val_imgs),
        "train_images": len(train_imgs),
        "val_images": len(val_imgs),
        "total_instances": manifest["total_instances"],
        "status": "Preferred Synthetic Dataset",
        "background_separation": "Strictly disjoint (10 train videos, 6 val videos; 0 overlap)",
        "rendering": "Multi-layer RGBA procedural shading, specular highlights, liquid menisci, drop shadows, collision checking",
        "classes": manifest["splits"],
    }


def main() -> None:
    print("Auditing all five datasets...")
    labos = audit_labos()
    chemeq = audit_chemeq25()
    enigma = audit_enigma360()
    syn_v1 = audit_synthetic_v1()
    syn_v2 = audit_synthetic_v2()

    report = {
        "labos": labos,
        "chemeq25": chemeq,
        "enigma360": enigma,
        "synthetic_v1": syn_v1,
        "synthetic_v2": syn_v2,
    }

    report_json_path = ROOT_DIR / "data" / "DATASET_INSPECTION_UNIFIED.json"
    report_json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved full audit JSON to {report_json_path}")


if __name__ == "__main__":
    main()
