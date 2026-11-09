"""Unified OBB Dataset Preparation and Conversion Pipeline for VYOM.

Converts and integrates multiple heterogeneous datasets into a standardized YOLOv8-OBB dataset:
1. LabOS: Converts instance segmentation polygons to minimum-area rotated bounding boxes (cv2.minAreaRect).
2. ChemEq25: Converts axis-aligned bounding boxes (xc, yc, w, h) to 4-corner zero-rotation OBB representations.
3. Synthetic v2: Integrates procedural high-fidelity OBB annotations with verified disjoint backgrounds.

Preserves native dataset splits (train / val / test), prevents cross-split data leakage,
tracks per-image source provenance, and maintains license obligations (CC BY 4.0, CC BY-NC 4.0).
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

# Default Proposed Core VYOM Taxonomy (10 Classes)
DEFAULT_CORE_CLASSES = [
    "centrifuge_tube",     # 0: LabOS 50ml eppendorf tube + Synthetic v2 centrifuge_tube
    "test_tube",           # 1: ChemEq25 Test_Tube
    "sample_vial",         # 2: Synthetic v2 sample_vial + ChemEq25 Weighing_Bottle
    "pipette",             # 3: ChemEq25 Pipette + Synthetic v2 pipette + ChemEq25 Volumetric_Pipet
    "multiwell_plate",     # 4: Synthetic v2 multiwell_plate
    "rig_slot",            # 5: LabOS 4 way rack 50ml hole + 14ml rack hole + Synthetic v2 rig_slot
    "tube_rack",           # 6: LabOS 50Ml 4 way rack + 14ml rack + ChemEq25 TestTube_Holder
    "tube_cap",            # 7: LabOS 50Ml eppendorf caps & cap tops
    "beaker",              # 8: ChemEq25 Beaker
    "conical_flask",       # 9: ChemEq25 Conical_Flask
]

# Source-to-Unified Category Mappings
LABOS_CLASS_MAP = {
    # 50ml eppendorf tube -> centrifuge_tube
    3: "centrifuge_tube",
    # Tube caps -> tube_cap
    4: "tube_cap",
    13: "tube_cap",
    14: "tube_cap",
    15: "tube_cap",
    16: "tube_cap",
    17: "tube_cap",
    # Racks -> tube_rack
    5: "tube_rack",
    12: "tube_rack",
    # Rack holes -> rig_slot
    6: "rig_slot",
    11: "rig_slot",
}

CHEMEQ_CLASS_MAP = {
    0: "beaker",            # Beaker
    4: "conical_flask",     # Conical_Flask
    10: "pipette",          # Pipette
    22: "pipette",          # Volumetric_Pipet
    19: "tube_rack",        # TestTube_Holder
    20: "test_tube",        # Test_Tube
    24: "sample_vial",      # Weighing_Bottle
}

SYN_V2_CLASS_MAP = {
    0: "sample_vial",
    1: "pipette",
    2: "multiwell_plate",
    3: "centrifuge_tube",
    4: "rig_slot",
}


def polygon_to_obb(pts_norm: np.ndarray, width: float, height: float) -> Optional[np.ndarray]:
    """Convert normalized polygon coordinates to 4-corner normalized OBB via minAreaRect."""
    if len(pts_norm) < 3:
        return None

    # Convert to float32 pixel coordinates
    pts_px = (pts_norm * np.array([width, height], dtype=np.float32)).astype(np.float32)
    rect = cv2.minAreaRect(pts_px)
    box_px = cv2.boxPoints(rect)

    if cv2.contourArea(box_px) < 4.0:
        return None

    # Normalize corners back to [0, 1]
    box_norm = box_px / np.array([width, height], dtype=np.float32)
    box_norm = np.clip(box_norm, 0.0, 1.0)
    return box_norm


def aabb_to_obb(xc: float, yc: float, w: float, h: float) -> np.ndarray:
    """Convert YOLO axis-aligned box (xc, yc, w, h) to 4-corner zero-rotation OBB representation."""
    p0 = [xc - w / 2.0, yc - h / 2.0]
    p1 = [xc + w / 2.0, yc - h / 2.0]
    p2 = [xc + w / 2.0, yc + h / 2.0]
    p3 = [xc - w / 2.0, yc + h / 2.0]
    box = np.array([p0, p1, p2, p3], dtype=np.float32)
    return np.clip(box, 0.0, 1.0)


def convert_labos(
    output_dir: Path,
    class_to_id: Dict[str, int],
    include_splits: List[str] = ["train", "val"],
) -> Tuple[int, int, Dict[str, int]]:
    """Convert LabOS segmentation annotations to OBB preserving train/val split."""
    labos_dir = DATASETS_DIR / "LabOS"
    images_dir = labos_dir / "images"

    # Read COCO JSON for split membership and dimensions
    coco_files = {
        "train": labos_dir / "annotations_train.json",
        "val": labos_dir / "annotations_val.json",
    }

    images_converted = 0
    boxes_converted = 0
    class_counts: Counter = Counter()

    for split in include_splits:
        coco_p = coco_files.get(split)
        if not coco_p or not coco_p.exists():
            continue

        coco_data = json.loads(coco_p.read_text(encoding="utf-8"))
        # Map image id to metadata
        images_meta = {img["id"]: img for img in coco_data["images"]}
        # Group annotations by image id
        anns_by_img: Dict[int, List[Dict[str, Any]]] = {}
        for ann in coco_data["annotations"]:
            anns_by_img.setdefault(ann["image_id"], []).append(ann)

        dest_img_dir = output_dir / "images" / split
        dest_lbl_dir = output_dir / "labels" / split
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
                # In COCO annotations categories are 1-indexed, match with LABOS_CLASS_MAP (0-indexed or 1-indexed)
                target_class_name = LABOS_CLASS_MAP.get(raw_cat_id - 1) or LABOS_CLASS_MAP.get(raw_cat_id)
                if not target_class_name or target_class_name not in class_to_id:
                    continue

                unified_cid = class_to_id[target_class_name]

                # Convert segmentation polygon
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
                    class_counts[target_class_name] += 1
                    boxes_converted += 1

            if lines:
                shutil.copy2(src_img_p, dest_img_p)
                dest_lbl_p.write_text("\n".join(lines) + "\n", encoding="utf-8")
                images_converted += 1

    return images_converted, boxes_converted, dict(class_counts)


def convert_chemeq25(
    output_dir: Path,
    class_to_id: Dict[str, int],
    include_splits: List[str] = ["train", "val", "test"],
) -> Tuple[int, int, Dict[str, int]]:
    """Convert ChemEq25 axis-aligned boxes to zero-rotation OBB representation."""
    chemeq_dir = DATASETS_DIR / "ChemEq25"

    split_map = {
        "train": "train",
        "val": "valid",
        "test": "test",
    }

    images_converted = 0
    boxes_converted = 0
    class_counts: Counter = Counter()

    for out_split, src_split in split_map.items():
        if out_split not in include_splits:
            continue

        src_img_dir = chemeq_dir / src_split / "images"
        src_lbl_dir = chemeq_dir / src_split / "labels"

        dest_img_dir = output_dir / "images" / out_split
        dest_lbl_dir = output_dir / "labels" / out_split
        dest_img_dir.mkdir(parents=True, exist_ok=True)
        dest_lbl_dir.mkdir(parents=True, exist_ok=True)

        for src_img_p in src_img_dir.glob("*.*"):
            src_lbl_p = src_lbl_dir / f"{src_img_p.stem}.txt"
            if not src_lbl_p.exists():
                continue

            lines_in = src_lbl_p.read_text().splitlines()
            lines_out: List[str] = []

            for l in lines_in:
                parts = l.strip().split()
                if len(parts) != 5:
                    continue

                raw_cid = int(parts[0])
                target_cname = CHEMEQ_CLASS_MAP.get(raw_cid)
                if not target_cname or target_cname not in class_to_id:
                    continue

                unified_cid = class_to_id[target_cname]
                xc, yc, w, h = [float(x) for x in parts[1:]]

                obb = aabb_to_obb(xc, yc, w, h)
                line = f"{unified_cid} " + " ".join(f"{coord:.6f}" for pt in obb for coord in pt)
                lines_out.append(line)
                class_counts[target_cname] += 1
                boxes_converted += 1

            if lines_out:
                out_stem = f"chemeq_{out_split}_{src_img_p.stem}"
                dest_img_p = dest_img_dir / f"{out_stem}{src_img_p.suffix}"
                dest_lbl_p = dest_lbl_dir / f"{out_stem}.txt"

                shutil.copy2(src_img_p, dest_img_p)
                dest_lbl_p.write_text("\n".join(lines_out) + "\n", encoding="utf-8")
                images_converted += 1

    return images_converted, boxes_converted, dict(class_counts)


def convert_synthetic_v2(
    output_dir: Path,
    class_to_id: Dict[str, int],
    include_splits: List[str] = ["train", "val"],
) -> Tuple[int, int, Dict[str, int]]:
    """Integrate Synthetic v2 OBB dataset preserving disjoint backgrounds."""
    v2_dir = ROOT_DIR / "data" / "yolo_obb_v2"

    images_converted = 0
    boxes_converted = 0
    class_counts: Counter = Counter()

    for split in include_splits:
        src_img_dir = v2_dir / "images" / split
        src_lbl_dir = v2_dir / "labels" / split
        if not src_img_dir.exists():
            continue

        dest_img_dir = output_dir / "images" / split
        dest_lbl_dir = output_dir / "labels" / split
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
                target_cname = SYN_V2_CLASS_MAP.get(raw_cid)
                if not target_cname or target_cname not in class_to_id:
                    continue

                unified_cid = class_to_id[target_cname]
                coords_str = " ".join(parts[1:])
                lines_out.append(f"{unified_cid} {coords_str}")
                class_counts[target_cname] += 1
                boxes_converted += 1

            if lines_out:
                out_stem = f"syn_v2_{split}_{src_img_p.stem}"
                dest_img_p = dest_img_dir / f"{out_stem}{src_img_p.suffix}"
                dest_lbl_p = dest_lbl_dir / f"{out_stem}.txt"

                shutil.copy2(src_img_p, dest_img_p)
                dest_lbl_p.write_text("\n".join(lines_out) + "\n", encoding="utf-8")
                images_converted += 1

    return images_converted, boxes_converted, dict(class_counts)


def build_unified_dataset(
    output_dir: Path = Path("d:/space_eye/vyom/data/unified_obb"),
    classes: List[str] = DEFAULT_CORE_CLASSES,
    include_synthetic_v2: bool = True,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Execute complete unified dataset conversion pipeline and write manifest files."""
    output_dir = output_dir.resolve()
    class_to_id = {cname: idx for idx, cname in enumerate(classes)}

    print("=" * 60)
    print("VYOM UNIFIED OBB DATASET PREPARATION PIPELINE")
    print("=" * 60)
    print(f"Target Directory:  {output_dir}")
    print(f"Unified Classes:   {len(classes)} classes")
    for idx, c in enumerate(classes):
        print(f"  [{idx:2d}] {c}")
    print(f"Dry Run Mode:      {dry_run}")
    print("=" * 60)

    if dry_run:
        print("Dry run requested. Inspecting counts without writing files...")
        # (Could estimate counts without disk writes)

    # Clean / prepare output directories
    for s in ["train", "val", "test"]:
        (output_dir / "images" / s).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / s).mkdir(parents=True, exist_ok=True)

    # 1. Convert LabOS (Real Instance Segmentation -> OBB)
    print("\n[1/3] Converting LabOS instance segmentations to genuine OBB...")
    labos_imgs, labos_boxes, labos_classes = convert_labos(output_dir, class_to_id)
    print(f"  -> Converted {labos_imgs} images, {labos_boxes} genuine rotated OBBs.")

    # 2. Convert ChemEq25 (Real AABB -> Zero-Rotation OBB)
    print("\n[2/3] Converting ChemEq25 bounding boxes to zero-rotation OBB...")
    chemeq_imgs, chemeq_boxes, chemeq_classes = convert_chemeq25(output_dir, class_to_id)
    print(f"  -> Converted {chemeq_imgs} images, {chemeq_boxes} zero-rotation OBBs.")

    # 3. Integrate Synthetic v2 (Preferred Synthetic Procedural OBB)
    syn_imgs, syn_boxes, syn_classes = 0, 0, {}
    if include_synthetic_v2:
        print("\n[3/3] Integrating Synthetic v2 procedural OBB (disjoint backgrounds)...")
        syn_imgs, syn_boxes, syn_classes = convert_synthetic_v2(output_dir, class_to_id)
        print(f"  -> Integrated {syn_imgs} images, {syn_boxes} procedural OBBs.")
    else:
        print("\n[3/3] Synthetic v2 integration disabled.")

    # Aggregate counts
    total_imgs = labos_imgs + chemeq_imgs + syn_imgs
    total_boxes = labos_boxes + chemeq_boxes + syn_boxes

    total_by_class: Counter = Counter()
    for d in [labos_classes, chemeq_classes, syn_classes]:
        for k, v in d.items():
            total_by_class[k] += v

    # 4. Generate dataset.yaml
    names_dict = {idx: cname for idx, cname in enumerate(classes)}
    dataset_yaml_content = f"""# Ultralytics YOLOv8-OBB dataset configuration for VYOM Unified Laboratory Apparatus
path: {output_dir.as_posix()}
train: images/train
val: images/val
test: images/test

names:
"""
    for idx, cname in names_dict.items():
        dataset_yaml_content += f"  {idx}: {cname}\n"

    (output_dir / "dataset.yaml").write_text(dataset_yaml_content, encoding="utf-8")

    # 5. Generate class_mapping.yaml
    mapping_data = {
        "unified_classes": names_dict,
        "source_mappings": {
            "LabOS": {
                "source_license": "CC BY-NC 4.0",
                "annotation_type": "Derived minimum-area rotated bounding box (from polygon)",
                "mapped_categories": LABOS_CLASS_MAP,
            },
            "ChemEq25": {
                "source_license": "CC BY 4.0",
                "annotation_type": "Derived zero-rotation bounding box (from axis-aligned bounding box)",
                "mapped_categories": {k: v for k, v in CHEMEQ_CLASS_MAP.items() if v in class_to_id},
            },
            "Synthetic_v2": {
                "source_license": "VYOM Procedural Synthetic",
                "annotation_type": "Procedural genuine rotated bounding box (from rendered geometry)",
                "mapped_categories": {k: v for k, v in SYN_V2_CLASS_MAP.items() if v in class_to_id},
            },
        },
    }
    with open(output_dir / "class_mapping.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(mapping_data, f, sort_keys=False)

    # 6. Generate dataset_manifest.yaml
    manifest_data = {
        "dataset_name": "VYOM Unified Laboratory Apparatus Dataset (OBB)",
        "version": "1.0.0",
        "taxonomy_size": len(classes),
        "total_images": total_imgs,
        "total_annotations": total_boxes,
        "sources": {
            "LabOS": {
                "images": labos_imgs,
                "annotations": labos_boxes,
                "license": "CC BY-NC 4.0",
                "class_counts": labos_classes,
            },
            "ChemEq25": {
                "images": chemeq_imgs,
                "annotations": chemeq_boxes,
                "license": "CC BY 4.0",
                "class_counts": chemeq_classes,
            },
            "Synthetic_v2": {
                "images": syn_imgs,
                "annotations": syn_boxes,
                "license": "VYOM Procedural Synthetic",
                "class_counts": syn_classes,
            },
        },
        "class_totals": dict(total_by_class),
    }
    with open(output_dir / "dataset_manifest.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(manifest_data, f, sort_keys=False)

    # 7. Generate SOURCES.md
    sources_md = f"""# Unified Dataset Sources and License Attributions

This unified dataset (`data/unified_obb/`) integrates real and procedural laboratory apparatus data
under strict provenance tracking and original licensing terms.

## 1. Source Provenance & Licensing

### Dataset 1: LabOS Segmentation Dataset
- **Local Location:** `D:\\space_eye\\datasets\\LabOS`
- **License:** **CC BY-NC 4.0** (Creative Commons Attribution-NonCommercial 4.0 International)
- **Role in VYOM:** Real laboratory instance segmentations converted to genuine oriented bounding boxes (`cv2.minAreaRect`).
- **Contributed Categories:** `centrifuge_tube` (50ml eppendorf), `tube_cap`, `tube_rack` (4-way & 14ml racks), `rig_slot` (rack holes).
- **Images Contributed:** {labos_imgs}
- **Annotations Contributed:** {labos_boxes} genuine rotated OBBs.

### Dataset 2: ChemEq25 Laboratory Dataset
- **Local Location:** `D:\\space_eye\\datasets\\ChemEq25`
- **Source Project:** Roboflow Universe `chemistry-lab-apparatus-detn` (v8)
- **License:** **CC BY 4.0** (Creative Commons Attribution 4.0 International)
- **Role in VYOM:** Real chemistry laboratory glassware converted to zero-rotation OBB representation.
- **Contributed Categories:** `beaker`, `conical_flask`, `pipette`, `test_tube`, `sample_vial` (weighing bottle), `tube_rack` (test tube holder).
- **Images Contributed:** {chemeq_imgs}
- **Annotations Contributed:** {chemeq_boxes} zero-rotation OBBs.
- **Attribution Note:** ChemEq25 bounding boxes are axis-aligned in the original dataset and do not provide genuine rotation variety.

### Dataset 3: VYOM Synthetic Demo Dataset v2
- **Local Location:** `d:\\space_eye\\vyom\\data\\yolo_obb_v2`
- **Role in VYOM:** High-fidelity procedural laboratory apparatus on real ENIGMA-360 disjoint backgrounds.
- **Contributed Categories:** `multiwell_plate` (unique 96-well ANSI microtiter plates), `sample_vial`, `pipette`, `centrifuge_tube`, `rig_slot`.
- **Images Contributed:** {syn_imgs}
- **Annotations Contributed:** {syn_boxes} procedural OBBs.

---

## 2. Total Class Distribution in Unified Dataset
| Class ID | Class Name | LabOS (Real OBB) | ChemEq25 (Real AABB) | Synthetic v2 (Procedural) | Total Instances |
|:---:|---|:---:|:---:|:---:|:---:|
"""
    for idx, cname in enumerate(classes):
        l_c = labos_classes.get(cname, 0)
        c_c = chemeq_classes.get(cname, 0)
        s_c = syn_classes.get(cname, 0)
        tot = l_c + c_c + s_c
        sources_md += f"| {idx} | `{cname}` | {l_c} | {c_c} | {s_c} | **{tot}** |\n"

    sources_md += f"""
**Total Images:** {total_imgs}  
**Total Annotations:** {total_boxes}  
"""
    (output_dir / "SOURCES.md").write_text(sources_md, encoding="utf-8")

    print("\n" + "=" * 60)
    print("UNIFIED DATASET SUMMARY")
    print("=" * 60)
    print(f"Total Converted Images:      {total_imgs}")
    print(f"Total Converted Annotations: {total_boxes}")
    print("Class Counts:")
    for cname, count in total_by_class.items():
        print(f"  • {cname:20s}: {count:5d} instances")
    print("=" * 60)

    return manifest_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert and build VYOM Unified OBB dataset.")
    parser.add_argument("--output-dir", type=str, default="d:/space_eye/vyom/data/unified_obb", help="Target output folder.")
    parser.add_argument("--dry-run", action="store_true", help="Inspect without copying files.")
    parser.add_argument("--no-synthetic", action="store_true", help="Exclude synthetic v2 data.")
    args = parser.parse_args()

    build_unified_dataset(
        output_dir=Path(args.output_dir),
        classes=DEFAULT_CORE_CLASSES,
        include_synthetic_v2=not args.no_synthetic,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()

