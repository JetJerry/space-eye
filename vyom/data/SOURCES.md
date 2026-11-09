# Public Dataset Sourcing Manifest

This document records the provenance, licensing, citation, and characteristics of public datasets evaluated and utilized for VYOM research.

---

## 1. ENIGMA-360 (Primary Procedural & Interaction Source)

- **Source Name:** ENIGMA-360 (Egocentric and Exocentric Procedural Video Dataset)
- **Official URL:** https://fpv-iplab.github.io/ENIGMA-360/
- **Repository:** https://github.com/fpv-iplab/ENIGMA-360
- **Local Configured Path:** `D:\space_eye\datasets\ENIGMA-360`
- **License:** CC BY-NC 4.0 (Non-commercial research / demo only)
- **Modality:** Synchronized Exocentric (`exo`) and Egocentric (`ego`) video pairs, spatial/temporal annotations, hand-object interaction (HOI), action segmentation.
- **Target Selection:** Exocentric (`exo`) training (103/103 complete) and validation (25/29 complete) splits. Fixed-payload camera perspective matches the VYOM hardware setup.
- **Video Properties:** 672×376 @ 30.0 FPS exocentric video stream (`third_person.mp4`), total verified duration: 36.35 hours (3,926,209 frames).
- **Inspection Findings:**
  - HOI instance segmentation masks (`annotations/hoi/train.json` & `val.json`) have resolution 2272×1278 and correspond to the egocentric Aria first-person perspective.
  - Categories are coarse binary indicators (`hand` and `object`), not fine-grained apparatus items.
- **Safety Directive:** Read-only external reference. Do not modify or delete files during downloads or ingestion.

---

## 2. ENIGMA-51 (Tool & Hand Interaction Reference)

- **Source Name:** ENIGMA-51
- **Official URL:** https://fpv-iplab.github.io/ENIGMA-51/
- **License:** Inspect dataset terms before redistribution (research use only).
- **Modality:** Egocentric electrical-board repair, 51 videos, 45,505 frames, 25 object classes, tool manipulation.
- **Target Selection:** Textual procedures, temporal interaction labels, and tool bounding boxes.

---

## 3. Annotated Chemical Apparatus Image Dataset (Object Detection Baseline)

- **Source Name:** Annotated Chemical Apparatus Image Dataset
- **Official URL:** https://data.mendeley.com/datasets/8p2hvgdvpn/1 (DOI: 10.17632/8p2hvgdvpn.1)
- **License:** CC BY 4.0
- **Size:** 4.50 GB archive
- **Modality:** 5,078 lab images across 7 classes (hands + 6 chemistry apparatus items).
- **Annotation Format:** Axis-aligned YOLO bounding boxes (`class x_center y_center width height`).
- **Policy Directive:** Requires 4.50 GB disk space; multi-gigabyte download requires user confirmation before acquisition. Labels are standard axis-aligned rectangles, **not genuine OBB**.

---

## 4. Physics Lab Equipment Image Dataset

- **Source Name:** Physics Lab Equipment Image Dataset
- **Official URL:** https://doi.org/10.6084/m9.figshare.30984658.v1
- **License:** CC BY 4.0
- **Modality:** 5,121 images, 27 instrument classes, YOLO-compatible TXT annotations.

---

## 5. VYOM Synthetic Demo Apparatus OBB Dataset (Active OBB Ground Truth)

- **Source Name:** VYOM Synthetic Demo Apparatus OBB Dataset
- **Local Location:** `data/yolo_obb/`
- **Generator Script:** `scripts/generate_synthetic_obb_dataset.py`
- **Validator Script:** `scripts/validate_obb_labels.py`
- **Format:** Strict YOLOv8-OBB 4-corner normalized coordinates: `class_id x1 y1 x2 y2 x3 y3 x4 y4`
- **Total Images:** 150 images (120 train / 30 val)
- **Backgrounds:** Extracted directly from real laboratory workbench frames in ENIGMA-360 `third_person.mp4`.
- **Classes:**
  - `0: sample_vial` (amber/clear cylindrical specimen container)
  - `1: pipette` (laboratory liquid transfer tool)
  - `2: multiwell_plate` (specimen reaction tray with grid wells)
  - `3: centrifuge_tube` (conical-bottom sample tube)
  - `4: rig_slot` (receptacle bay on the experiment mounting plate)
- **Orientation:** Genuine $0^\circ - 360^\circ$ continuous rotations (95.7% non-horizontal OBB).
- **Validation:** 100% convex non-degenerate quadrilaterals verified by `scripts/validate_obb_labels.py`.
- **Transparency Statement:** Classes represent generic scientific apparatus for algorithm and pipeline verification; they are not ISRO flight equipment.

---

## Ethical and Licensing Commitments

1. No unauthorized scraping or unverified mirrors are used.
2. Original dataset annotations and directory structures remain intact.
3. Axis-aligned bounding boxes are never falsely represented as genuine oriented bounding boxes.
4. Model evaluation scores are calculated only from verified holdout sets.
