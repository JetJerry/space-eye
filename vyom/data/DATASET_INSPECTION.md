# ENIGMA-360 Dataset Inspection and Perception Audit Report

Generated automatically by `scripts/inspect_dataset.py` on ENIGMA-360.

## 1. Executive Summary

| Parameter | Value | Notes |
|---|---|---|
| **Dataset Root** | `D:\space_eye\datasets\ENIGMA-360` | Read-only external storage (D: drive) |
| **Integrity Status** | `Accessible` | Source files preserved untouched |
| **Total MP4 Files** | `129` | Exocentric fixed-camera views (`third_person.mp4`) |
| **Fully Verified Videos** | `128` | Fully readable headers and intact moov atoms |
| **Active Downloads** | `1` | Actively written by background process PID 16792 |
| **Train Split Complete** | `103 / 103` | **100% of official training split is available** |
| **Validation Split Complete** | `25 / 29` | **25/29 validation videos available** |
| **Total Duration** | `36.35 hours` | **3,926,209 total frames** |
| **Exocentric Resolution** | `672x376 @ 30.0 FPS` | Fixed third-person procedural view |

## 2. Annotation Audit & Object Detection Compatibility

| Annotation Source | Format | Target Perspective | Classes Present | Detection Suitability |
|---|---|---|---|---|
| `annotations/raw_json/` | JSON clip timestamps | Exocentric & Egocentric | 68 procedural action verbs | **Temporal action labels only**; no spatial object bounding boxes. |
| `annotations/action_taxonomy.json` | JSON list | Procedural action catalog | 68 procedure descriptions | High-level procedure text (e.g. "Takes high-voltage card"). |
| `annotations/hoi/train.json` | COCO JSON (3,479 imgs, 9,418 annos) | **Egocentric (first_person)** (2272x1278) | `hand` (4,709), `object` (4,709) | **Coarse binary classes only** (`hand` vs `object`). Not ISRO apparatus. |
| `annotations/hoi/val.json` | COCO JSON (1,218 imgs, 3,419 annos) | **Egocentric (first_person)** (2272x1278) | `hand` (1,709), `object` (1,710) | Coarse binary classes. |

## 3. Critical Architectural Findings

1. **Perspective Mismatch:** The detailed instance segmentation masks in `annotations/hoi/` have dimensions `2272x1278`, matching the egocentric Aria first-person perspective. The downloaded `third_person.mp4` fixed-camera videos are `672x376`. Third-person spatial annotations are not part of the standard ENIGMA-360 HOI release.
2. **Coarse Class Limitations:** ENIGMA-360 HOI provides only `hand` and generic `object`. It does not provide fine-grained lab or spacecraft apparatus classes.
3. **Axis-Aligned vs. OBB:** COCO bounding boxes in `annotations/hoi/` are axis-aligned (`[x, y, w, h]`). Converting an axis-aligned box to four corners yields a horizontal rectangle with zero rotation ($	heta=0^\circ$), which **does not provide genuine oriented-box training variety** for YOLOv8-OBB.

## 4. Training Strategy for Custom YOLOv8-OBB

Per Section 1A and Prompt 6 of `VYOM_Antigravity_Build_Guide_v4.md`:
- **Step 1:** Establish a reproducible baseline object detection pipeline on validated public data.
- **Step 2:** Build an OBB-specific dataset generation pipeline that produces genuine rotated bounding boxes (`class_id x1 y1 x2 y2 x3 y3 x4 y4`) with varying orientations, lighting, backgrounds, and aspect ratios.
- **Step 3:** Fine-tune Ultralytics `yolov8n-obb.pt` using transfer learning on the target system GPU (NVIDIA RTX 3050 Laptop GPU, 4GB VRAM) with automatic CPU fallback.
