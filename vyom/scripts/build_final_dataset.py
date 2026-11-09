"""Build Final Unified YOLOv8-OBB Dataset for VYOM Apparatus Detection.

Applies strict corrections from Prompt 6 audit:
1. Never maps Weighing_Bottle to sample_vial (excludes Weighing_Bottle due to insufficient verified samples: 8 train, 0 val).
2. Keeps rack_hole and rig_slot strictly separate. LabOS rack holes are labelled as rack_hole.
3. Explicitly flags synthetic-only provisional categories: sample_vial, multiwell_plate, rig_slot.
4. Standardizes pipette to handheld micropipettes/pipetting devices, excluding Volumetric_Pipet (rare chemical transfer tube).
5. Keeps tube_rack and rack_hole strictly separate.
6. Preserves semantic differences across tubes, vials, and caps.
7. ChemEq25: Uses exclusively the 304 verified matching image-label pairs (199 train, 36 val, 69 test).
8. Synthetic v1 is excluded. Synthetic v2 is integrated.
9. LabOS instance segmentations converted to genuine rotated OBB (cv2.minAreaRect).
10. ChemEq25 bounding boxes converted to zero-rotation OBB.

Output destination: data/yolo_obb_final/
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
import yaml

ROOT_DIR = Path("d:/space_eye/vyom")
DATASETS_DIR = Path("D:/space_eye/datasets")

# 10-Class Final Approved Taxonomy
FINAL_CLASSES = [
    "centrifuge_tube",     # 0: LabOS 50ml eppendorf tube + Synthetic v2 centrifuge_tube
    "tube_rack",           # 1: LabOS 50Ml 4 way rack, 14ml rack + ChemEq25 TestTube_Holder
    "rack_hole",           # 2: LabOS 4 way rack 50ml hole + 14ml rack hole (Real LabOS holes)
    "tube_cap",            # 3: LabOS 50Ml eppendorf cap/orange cap/blue cap/cap top
    "pipette",             # 4: ChemEq25 Pipette + Synthetic v2 pipette
    "beaker",              # 5: ChemEq25 Beaker
    "conical_flask",       # 6: ChemEq25 Conical_Flask
    "sample_vial",         # 7: Synthetic v2 sample_vial (Provisional - Synthetic Only)
    "multiwell_plate",     # 8: Synthetic v2 multiwell_plate (Provisional - Synthetic Only)
    "rig_slot",            # 9: Synthetic v2 rig_slot (Provisional - Synthetic Only)
]

# Source-to-Final Category Mappings
LABOS_MAPPING = {
    # 50ml eppendorf tube -> centrifuge_tube
    3: "centrifuge_tube",
    # Eppendorf caps -> tube_cap
    4: "tube_cap",
    13: "tube_cap",
    14: "tube_cap",
    15: "tube_cap",
    16: "tube_cap",
    17: "tube_cap",
    # Racks -> tube_rack
    5: "tube_rack",
    12: "tube_rack",
    # Rack holes -> rack_hole (STRICTLY SEPARATED FROM rig_slot)
    6: "rack_hole",
    11: "rack_hole",
}

CHEMEQ_MAPPING = {
    0: "beaker",            # Beaker
    4: "conical_flask",     # Conical_Flask
    10: "pipette",          # Pipette (handheld/dropper)
    19: "tube_rack",        # TestTube_Holder
    # Note: Weighing_Bottle (24) is EXCLUDED (never mapped to sample_vial)
    # Note: Test_Tube (20) and Volumetric_Pipet (22) excluded from core initial model
}

SYN_V2_MAPPING = {
    0: "sample_vial",       # Provisional - Synthetic Only
    1: "pipette",
    2: "multiwell_plate",   # Provisional - Synthetic Only
    3: "centrifuge_tube",
    4: "rig_slot",          # Provisional - Synthetic Only (Experimental rig mounting receptacle)
}


def polygon_to_obb(pts_norm: np.ndarray, width: float, height: float) -> Optional[np.ndarray]:
    """Convert normalized polygon coordinates to 4-corner normalized OBB via minAreaRect."""
    if len(pts_norm) < 3:
        return None

    pts_px = (pts_norm * np.array([width, height], dtype=np.float32)).astype(np.float32)
    rect = cv2.minAreaRect(pts_px)
    (cx, cy), (w, h), angle = rect
    if min(w, h) < 3.0 or (w * h) < 20.0:
        return None

    box_px = cv2.boxPoints(rect)
    if not cv2.isContourConvex(box_px.astype(np.int32)):
        return None
    if cv2.contourArea(box_px) < 16.0:
        return None

    box_norm = box_px / np.array([width, height], dtype=np.float32)
    box_norm = np.clip(box_norm, 0.0, 1.0)

    # Re-verify post-clipping convex geometry
    box_norm_px = (box_norm * np.array([width, height], dtype=np.float32)).astype(np.int32)
    if not cv2.isContourConvex(box_norm_px) or cv2.contourArea(box_norm_px) < 12.0:
        return None

    return box_norm


def aabb_to_obb(xc: float, yc: float, w: float, h: float) -> np.ndarray:
    """Convert YOLO axis-aligned box (xc, yc, w, h) to 4-corner zero-rotation OBB representation."""
    p0 = [xc - w / 2.0, yc - h / 2.0]
    p1 = [xc + w / 2.0, yc - h / 2.0]
    p2 = [xc + w / 2.0, yc + h / 2.0]
    p3 = [xc - w / 2.0, yc + h / 2.0]
    box = np.array([p0, p1, p2, p3], dtype=np.float32)
    return np.clip(box, 0.0, 1.0)


def process_labos(
    dest_dir: Path,
    class_to_id: Dict[str, int],
) -> Tuple[int, int, Dict[str, Dict[str, int]]]:
    """Convert LabOS segmentation annotations to OBB preserving native train/val splits."""
    labos_dir = DATASETS_DIR / "LabOS"
    images_dir = labos_dir / "images"

    coco_files = {
        "train": labos_dir / "annotations_train.json",
        "val": labos_dir / "annotations_val.json",
    }

    images_count = 0
    boxes_count = 0
    split_class_counts: Dict[str, Dict[str, int]] = {"train": Counter(), "val": Counter(), "test": Counter()}

    for split in ["train", "val"]:
        coco_p = coco_files.get(split)
        if not coco_p or not coco_p.exists():
            continue

        coco_data = json.loads(coco_p.read_text(encoding="utf-8"))
        images_meta = {img["id"]: img for img in coco_data["images"]}
        anns_by_img: Dict[int, List[Dict[str, Any]]] = {}
        for ann in coco_data["annotations"]:
            anns_by_img.setdefault(ann["image_id"], []).append(ann)

        dest_img_dir = dest_dir / "images" / split
        dest_lbl_dir = dest_dir / "labels" / split
        dest_img_dir.mkdir(parents=True, exist_ok=True)
        dest_lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_id, meta in images_meta.items():
            fn = meta["file_name"]
            src_img_p = images_dir / fn
            if not src_img_p.exists():
                continue

            w_img = float(meta["width"])
            h_img = float(meta["height"])

            out_stem = f"labos_{split}_{Path(fn).stem}"
            dest_img_p = dest_img_dir / f"{out_stem}{src_img_p.suffix}"
            dest_lbl_p = dest_lbl_dir / f"{out_stem}.txt"

            lines: List[str] = []
            for ann in anns_by_img.get(img_id, []):
                raw_cat_id = ann["category_id"]
                target_cname = LABOS_MAPPING.get(raw_cat_id - 1) or LABOS_MAPPING.get(raw_cat_id)
                if not target_cname or target_cname not in class_to_id:
                    continue

                unified_cid = class_to_id[target_cname]

                seg = ann.get("segmentation")
                if not seg or not isinstance(seg, list) or len(seg) == 0:
                    continue

                poly_raw = seg[0]
                if len(poly_raw) < 6:
                    continue

                pts = np.array(poly_raw, dtype=np.float32).reshape(-1, 2)
                pts[:, 0] /= w_img
                pts[:, 1] /= h_img

                obb = polygon_to_obb(pts, w_img, h_img)
                if obb is not None:
                    line = f"{unified_cid} " + " ".join(f"{coord:.6f}" for pt in obb for coord in pt)
                    lines.append(line)
                    split_class_counts[split][target_cname] += 1
                    boxes_count += 1

            if lines:
                shutil.copy2(src_img_p, dest_img_p)
                dest_lbl_p.write_text("\n".join(lines) + "\n", encoding="utf-8")
                images_count += 1

    return images_count, boxes_count, {s: dict(c) for s, c in split_class_counts.items()}


def process_chemeq25(
    dest_dir: Path,
    class_to_id: Dict[str, int],
) -> Tuple[int, int, Dict[str, Dict[str, int]]]:
    """Convert ChemEq25 bounding boxes to OBB for verified stem-matching pairs only."""
    chemeq_dir = DATASETS_DIR / "ChemEq25"

    split_map = {
        "train": "train",
        "val": "valid",
        "test": "test",
    }

    images_count = 0
    boxes_count = 0
    split_class_counts: Dict[str, Dict[str, int]] = {"train": Counter(), "val": Counter(), "test": Counter()}

    for out_split, src_split in split_map.items():
        src_img_dir = chemeq_dir / src_split / "images"
        src_lbl_dir = chemeq_dir / src_split / "labels"

        dest_img_dir = dest_dir / "images" / out_split
        dest_lbl_dir = dest_dir / "labels" / out_split
        dest_img_dir.mkdir(parents=True, exist_ok=True)
        dest_lbl_dir.mkdir(parents=True, exist_ok=True)

        # Build index of verified matching stems
        img_stems = {p.stem.upper(): p for p in src_img_dir.glob("*.*")}
        lbl_stems = {p.stem.upper(): p for p in src_lbl_dir.glob("*.txt")}
        matched_stems = set(img_stems.keys()).intersection(set(lbl_stems.keys()))

        for stem in sorted(matched_stems):
            src_img_p = img_stems[stem]
            src_lbl_p = lbl_stems[stem]

            lines_in = src_lbl_p.read_text().splitlines()
            lines_out: List[str] = []

            for l in lines_in:
                parts = l.strip().split()
                if len(parts) != 5:
                    continue

                raw_cid = int(parts[0])
                target_cname = CHEMEQ_MAPPING.get(raw_cid)
                if not target_cname or target_cname not in class_to_id:
                    continue

                unified_cid = class_to_id[target_cname]
                xc, yc, w, h = [float(x) for x in parts[1:]]

                obb = aabb_to_obb(xc, yc, w, h)
                line = f"{unified_cid} " + " ".join(f"{coord:.6f}" for pt in obb for coord in pt)
                lines_out.append(line)
                split_class_counts[out_split][target_cname] += 1
                boxes_count += 1

            if lines_out:
                out_stem = f"chemeq_{out_split}_{src_img_p.stem}"
                dest_img_p = dest_img_dir / f"{out_stem}{src_img_p.suffix}"
                dest_lbl_p = dest_lbl_dir / f"{out_stem}.txt"

                shutil.copy2(src_img_p, dest_img_p)
                dest_lbl_p.write_text("\n".join(lines_out) + "\n", encoding="utf-8")
                images_count += 1

    return images_count, boxes_count, {s: dict(c) for s, c in split_class_counts.items()}


def process_synthetic_v2(
    dest_dir: Path,
    class_to_id: Dict[str, int],
) -> Tuple[int, int, Dict[str, Dict[str, int]]]:
    """Integrate Synthetic v2 OBB dataset preserving disjoint background splits."""
    v2_dir = ROOT_DIR / "data" / "yolo_obb_v2"

    images_count = 0
    boxes_count = 0
    split_class_counts: Dict[str, Dict[str, int]] = {"train": Counter(), "val": Counter(), "test": Counter()}

    for split in ["train", "val"]:
        src_img_dir = v2_dir / "images" / split
        src_lbl_dir = v2_dir / "labels" / split
        if not src_img_dir.exists():
            continue

        dest_img_dir = dest_dir / "images" / split
        dest_lbl_dir = dest_dir / "labels" / split
        dest_img_dir.mkdir(parents=True, exist_ok=True)
        dest_lbl_dir.mkdir(parents=True, exist_ok=True)

        for src_img_p in src_img_dir.glob("*.jpg"):
            src_lbl_p = src_lbl_dir / f"{src_img_p.stem}.txt"
            if not src_lbl_p.exists():
                continue

            lines_in = src_lbl_p.read_text().splitlines()
            lines_out: List[str] = []

            for l in lines_in:
                parts = l.strip().split()
                if len(parts) != 9:
                    continue

                raw_cid = int(parts[0])
                target_cname = SYN_V2_MAPPING.get(raw_cid)
                if not target_cname or target_cname not in class_to_id:
                    continue

                unified_cid = class_to_id[target_cname]
                coords_str = " ".join(parts[1:])
                lines_out.append(f"{unified_cid} {coords_str}")
                split_class_counts[split][target_cname] += 1
                boxes_count += 1

            if lines_out:
                out_stem = f"syn_v2_{split}_{src_img_p.stem}"
                dest_img_p = dest_img_dir / f"{out_stem}{src_img_p.suffix}"
                dest_lbl_p = dest_lbl_dir / f"{out_stem}.txt"

                shutil.copy2(src_img_p, dest_img_p)
                dest_lbl_p.write_text("\n".join(lines_out) + "\n", encoding="utf-8")
                images_count += 1

    return images_count, boxes_count, {s: dict(c) for s, c in split_class_counts.items()}


def generate_previews(dataset_dir: Path, num_previews_per_source: int = 4) -> None:
    """Generate visual preview images with oriented bounding box overlays for inspection."""
    previews_dir = dataset_dir / "previews"
    previews_dir.mkdir(parents=True, exist_ok=True)

    palette = [
        (0, 230, 115),   # 0: centrifuge_tube
        (255, 140, 0),   # 1: tube_rack
        (0, 215, 255),   # 2: rack_hole
        (200, 200, 200), # 3: tube_cap
        (220, 60, 220),  # 4: pipette
        (50, 150, 255),  # 5: beaker
        (150, 50, 255),  # 6: conical_flask
        (0, 165, 255),   # 7: sample_vial (provisional)
        (255, 100, 100), # 8: multiwell_plate (provisional)
        (100, 255, 100), # 9: rig_slot (provisional)
    ]

    img_dir = dataset_dir / "images" / "train"
    lbl_dir = dataset_dir / "labels" / "train"

    for prefix in ["labos_train_", "chemeq_train_", "syn_v2_train_"]:
        sample_imgs = sorted(list(img_dir.glob(f"{prefix}*.*")))[:num_previews_per_source]
        for img_p in sample_imgs:
            lbl_p = lbl_dir / f"{img_p.stem}.txt"
            if not lbl_p.exists():
                continue

            im = cv2.imread(str(img_p))
            if im is None:
                continue

            h, w = im.shape[:2]
            for l in lbl_p.read_text().splitlines():
                parts = l.strip().split()
                if len(parts) != 9:
                    continue
                cid = int(parts[0])
                coords = [float(x) for x in parts[1:]]
                pts = (np.array(coords).reshape(4, 2) * [w, h]).astype(np.int32)
                color = palette[cid % len(palette)]
                cv2.polylines(im, [pts], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)
                cname = FINAL_CLASSES[cid]
                cv2.putText(
                    im,
                    f"{cid}:{cname}",
                    (pts[0, 0], max(16, pts[0, 1] - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    color,
                    1,
                    cv2.LINE_AA,
                )

            out_prev = previews_dir / f"{img_p.stem}_preview.jpg"
            cv2.imwrite(str(out_prev), im, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

    print(f"Generated inspection previews under {previews_dir}")


def main() -> None:
    dest_dir = ROOT_DIR / "data" / "yolo_obb_final"
    dest_dir.mkdir(parents=True, exist_ok=True)
    class_to_id = {cname: idx for idx, cname in enumerate(FINAL_CLASSES)}

    print("=" * 65)
    print("VYOM FINAL UNIFIED OBB DATASET PREPARATION")
    print("=" * 65)
    print(f"Destination: {dest_dir}")
    print(f"Taxonomy ({len(FINAL_CLASSES)} classes):")
    for idx, c in enumerate(FINAL_CLASSES):
        prov = " [SYNTHETIC ONLY - PROVISIONAL]" if c in ["sample_vial", "multiwell_plate", "rig_slot"] else " [REAL/VALIDATED]"
        print(f"  [{idx}] {c:20s}{prov}")
    print("=" * 65)

    # 1. Process LabOS
    print("\n[1/3] Converting LabOS (real instance segmentations -> genuine OBB)...")
    labos_imgs, labos_boxes, labos_counts = process_labos(dest_dir, class_to_id)
    print(f"  -> Processed {labos_imgs} images, {labos_boxes} genuine rotated OBBs.")

    # 2. Process ChemEq25
    print("\n[2/3] Converting ChemEq25 (verified matching pairs -> zero-rotation OBB)...")
    chemeq_imgs, chemeq_boxes, chemeq_counts = process_chemeq25(dest_dir, class_to_id)
    print(f"  -> Processed {chemeq_imgs} images, {chemeq_boxes} zero-rotation OBBs.")

    # 3. Process Synthetic v2
    print("\n[3/3] Integrating Synthetic v2 (disjoint background procedural OBB)...")
    syn_imgs, syn_boxes, syn_counts = process_synthetic_v2(dest_dir, class_to_id)
    print(f"  -> Processed {syn_imgs} images, {syn_boxes} procedural OBBs.")

    # Aggregate statistics
    total_imgs = labos_imgs + chemeq_imgs + syn_imgs
    total_boxes = labos_boxes + chemeq_boxes + syn_boxes

    split_img_counts = {
        "train": len(list((dest_dir / "images" / "train").glob("*.*"))),
        "val": len(list((dest_dir / "images" / "val").glob("*.*"))),
        "test": len(list((dest_dir / "images" / "test").glob("*.*"))),
    }

    # Generate dataset.yaml
    yaml_content = f"""# Ultralytics YOLOv8-OBB dataset configuration for VYOM Final Unified Laboratory Apparatus
path: {dest_dir.as_posix()}
train: images/train
val: images/val
test: images/test

names:
"""
    for idx, cname in enumerate(FINAL_CLASSES):
        yaml_content += f"  {idx}: {cname}\n"
    (dest_dir / "dataset.yaml").write_text(yaml_content, encoding="utf-8")

    # Generate class_mapping.yaml
    mapping_data = {
        "final_classes": {idx: cname for idx, cname in enumerate(FINAL_CLASSES)},
        "provisional_synthetic_only_classes": ["sample_vial", "multiwell_plate", "rig_slot"],
        "real_world_classes": ["centrifuge_tube", "tube_rack", "rack_hole", "tube_cap", "pipette", "beaker", "conical_flask"],
        "source_mappings": {
            "LabOS": {
                "license": "CC BY-NC 4.0",
                "annotation_type": "Derived minimum-area rotated bounding box (from polygon)",
                "mapping": LABOS_MAPPING,
                "notes": "Rack holes strictly mapped to rack_hole, never rig_slot.",
            },
            "ChemEq25": {
                "license": "CC BY 4.0",
                "annotation_type": "Derived zero-rotation bounding box (from axis-aligned bounding box)",
                "mapping": CHEMEQ_MAPPING,
                "notes": "Uses only verified stem-matching pairs. Weighing_Bottle excluded.",
            },
            "Synthetic_v2": {
                "license": "VYOM Procedural Synthetic",
                "annotation_type": "Procedural genuine rotated bounding box (from rendered geometry)",
                "mapping": SYN_V2_MAPPING,
            },
        },
    }
    with open(dest_dir / "class_mapping.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(mapping_data, f, sort_keys=False)

    # Generate dataset_manifest.yaml
    manifest_data = {
        "dataset_name": "VYOM Final Unified Laboratory Apparatus Dataset (OBB)",
        "version": "1.0.0",
        "taxonomy": FINAL_CLASSES,
        "image_counts": split_img_counts,
        "total_images": total_imgs,
        "total_annotations": total_boxes,
        "sources": {
            "LabOS": {"images": labos_imgs, "annotations": labos_boxes, "split_counts": labos_counts},
            "ChemEq25": {"images": chemeq_imgs, "annotations": chemeq_boxes, "split_counts": chemeq_counts},
            "Synthetic_v2": {"images": syn_imgs, "annotations": syn_boxes, "split_counts": syn_counts},
        },
    }
    with open(dest_dir / "dataset_manifest.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(manifest_data, f, sort_keys=False)

    # Generate SOURCES.md
    sources_md = f"""# Final Unified Dataset Sources and License Attributions

## 1. Source Provenance & Licensing
- **LabOS:** `D:\\space_eye\\datasets\\LabOS` | **CC BY-NC 4.0** | Real instance segmentations $\\rightarrow$ genuine rotated OBB.
- **ChemEq25:** `D:\\space_eye\\datasets\\ChemEq25` | **CC BY 4.0** | Verified matching pairs only $\\rightarrow$ zero-rotation OBB.
- **Synthetic v2:** `d:\\space_eye\\vyom\\data\\yolo_obb_v2` | Procedural synthetic on disjoint ENIGMA-360 backgrounds.

## 2. Image and Split Distribution
- **Train Images:** {split_img_counts['train']}
- **Val Images:** {split_img_counts['val']}
- **Test Images:** {split_img_counts['test']}
- **Total Images:** {total_imgs}
- **Total Annotations:** {total_boxes}

## 3. Important Generalization Notice
- `sample_vial`, `multiwell_plate`, and `rig_slot` contain synthetic-only training examples and are marked **provisional**.
- ChemEq25 bounding boxes are axis-aligned and do not provide genuine rotation variety.
- LabOS is licensed under **CC BY-NC 4.0** (Non-Commercial). Any future commercial deployment requires licensing review.
"""
    (dest_dir / "SOURCES.md").write_text(sources_md, encoding="utf-8")

    # Generate previews
    generate_previews(dest_dir, num_previews_per_source=4)

    print("\n" + "=" * 65)
    print("FINAL DATASET SUMMARY")
    print("=" * 65)
    print(f"Total Converted Images:      {total_imgs} (Train: {split_img_counts['train']}, Val: {split_img_counts['val']}, Test: {split_img_counts['test']})")
    print(f"Total Converted Annotations: {total_boxes}")
    print("=" * 65)


if __name__ == "__main__":
    main()
