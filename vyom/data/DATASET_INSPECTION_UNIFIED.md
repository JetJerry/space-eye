# VYOM Unified Multi-Source Dataset Inspection & Proposed Class Mapping Report

**Audit Date:** 2026-11-09  
**Specification:** SIH 26174 / `VYOM_Antigravity_Build_Guide_v4.md` (Prompt 6 Continuation)  
**Target Unified Directory:** `d:/space_eye/vyom/data/unified_obb/`  
**Pipeline Script:** `vyom/scripts/convert_to_unified_obb.py`  
**Status:** Inspection Complete — **Awaiting User Class Mapping Approval Before Training**  

---

## 1. Executive Summary Across All Evaluated Datasets

| Dataset | Type / Origin | Total Media | Total Annotations | Annotation Format | Native Splits | License | Role in VYOM |
|---|---|---|---|---|---|---|---|
| **LabOS** | Real Lab Imagery | 314 images | 3,671 instances | COCO Polygon & YOLO Seg | Train (251) / Val (63) | **CC BY-NC 4.0** | Real segmented apparatus (tubes, racks, slots, caps) converted to **genuine rotated OBB**. |
| **ChemEq25** | Real Chemistry Glassware | 4,598 images | 6,790 instances | YOLO AABB (5 tokens) + `Metadata.csv` | Train (3,220) / Val (920) / Test (458) | **CC BY 4.0** | Real laboratory glassware converted to **zero-rotation OBB**. |
| **ENIGMA-360** | Real Lab/Industrial Video | 132 videos (36.35 hrs) | Task taxonomy (68 actions) | Video frames & action timestamps | Train (103 vids) / Val (29 vids) | Academic / Research | Authentic workbench textures, lighting, shadows, and human activity monitoring. |
| **Synthetic v1** (`data/yolo_obb/`) | Procedural Demo (v1) | 150 images | 373 instances | YOLOv8-OBB (9 tokens) | Train (120) / Val (30) | VYOM Internal | **Historical baseline only** (simple 2D shapes, single video 103 background leakage). |
| **Synthetic v2** (`data/yolo_obb_v2/`) | Procedural Demo (v2) | 300 images | 919 instances | YOLOv8-OBB (9 tokens) | Train (240) / Val (60) | VYOM Internal | **Preferred synthetic baseline** (multi-layer shading, menisci, drop shadows, 16 disjoint videos). |

---

## 2. Deep-Dive Inspection of Real Datasets

### A. LabOS Dataset (`D:\space_eye\datasets\LabOS`)
- **Structure:**
  - `images/`: 314 PNG images. Resolutions: $1280 \times 720$ (HD workbench views: 308 images), $540 \times 360$ (4 images), $300 \times 185$ (2 images). Zero corrupt files.
  - `labels/`: 314 TXT files containing YOLO segmentation polygons (`class_id x1 y1 x2 y2 ... xN yN`).
  - `annotations_train.json` (251 images, 2,989 annotations) and `annotations_val.json` (63 images, 682 annotations) in standard COCO instance segmentation JSON format.
  - `label_counts.txt`: Verified post-merge class census across all 18 categories.
- **Classes (18 categories):**
  - High-value apparatus: `50ml eppendorf tube` (320 instances), `50Ml 4 way rack` (120 instances), `14ml rack` (35 instances).
  - High-value interaction targets: `4 way rack 50ml hole` (867 instances), `14ml rack hole` (1,322 instances).
  - Tube caps & closures: `50Ml eppendorf cap` / `orange cap` / `blue cap` / `cap top` (559 instances).
  - Instrumentation: `Vortex Genie 2` (108 instances), `Vortex Genie Top Plate` (84 instances), `Vortex Genie Hole` (77 instances).
- **Licensing:** CC BY-NC 4.0. Requires non-commercial attribution.

### B. ChemEq25 Dataset (`D:\space_eye\datasets\ChemEq25`)
- **Structure:**
  - Roboflow Universe export (`chemistry-lab-apparatus-detn`, version 8).
  - Partition directories: `train/` (3,220 images), `valid/` (920 images), `test/` (458 images). All images are standard $640 \times 640 \times 3$. Zero corrupt images.
  - `Metadata.csv`: Exhaustive database containing 6,790 annotated bounding boxes across 4,548 images with image filenames, split sets, class IDs, class names, and normalized coordinates ($X, Y, W, H$).
- **Critical Quality Audit Finding (8.3 Filename Decoupling):**
  - In the official author's archive, long file paths exceeded Windows path limits during archiving, causing ~93% of the image files in `images/` and `.txt` files in `labels/` to be renamed into arbitrary 8.3 short names (e.g., `2000C8~1.JPG` vs `2000D3~1.TXT`).
  - **Verified Stem Matches:** Exactly **304 images** across the dataset (`199` train, `36` val, `69` test) have verified identical image and label filenames.
  - To prevent false supervision and label corruption, our conversion pipeline strictly converts only verified stem-matched files.
- **Classes (25 categories):**
  - High-volume glassware: `Conical_Flask` (452), `Beaker` (388), `Test_Tube` (260), `Measuring_Cylinder` (252), `Volumetric_Pipet` (245), `Reagent_Bottle` (243), `Pipette` (237), `Weighing_Bottle` (230), `TestTube_Holder` (268).
  - Specialized apparatus: `Burette_Stands` (305), `Funnel` (282), `Precision_Weight_Scale` (256), `Mechanical_Balance_Scale` (232), `Glass_Rod` (363).
- **Licensing:** CC BY 4.0. Permissive open access with attribution.

### C. ENIGMA-360 Dataset (`D:\space_eye\datasets\ENIGMA-360`)
- **Structure:**
  - 132 third-person exocentric fixed camera MP4 videos ($672 \times 376$ at 30 fps, 36.35 hours total duration).
  - Partition: 103 training videos (`annotations/train.txt`) and 29 validation videos (`annotations/val.txt`).
- **Perspective Mismatch & Annotation Compatibility:**
  - Third-person exocentric camera captures real workbench surfaces, natural laboratory illumination, and physical activity.
  - HOI instance segmentation masks in `annotations/hoi/` are egocentric Aria glasses captures at $2272 \times 1278$. They cannot be projected directly onto exocentric video frames without projective distortion, and they only provide coarse binary labels (`hand` / `object`).
  - Role: Source for real workbench backgrounds, lighting variation, and temporal procedure evaluation.

### D. Synthetic Datasets Comparison
1. **Synthetic v1 (`data/yolo_obb/`):**
   - 150 images (120 train / 30 val), 373 instances.
   - Generated using flat 2D OpenCV geometric shapes without shading or drop shadows.
   - Train and validation backgrounds were both sampled from a single video (`103_third_person.mp4`), causing data leakage.
   - **Recommendation:** Retained for historical baseline comparison only. Excluded from unified training.
2. **Synthetic v2 (`data/yolo_obb_v2/`):**
   - 300 images (240 train / 60 val), 919 instances.
   - Multi-layer RGBA procedural rendering with cylindrical cosine lighting, specular reflection streaks, liquid columns with menisci, graduations, knurled caps, and contact drop shadows.
   - Collision detection (`cv2.intersectConvexConvex`) prevents mutual occlusion.
   - 16 strictly disjoint source videos from ENIGMA-360 (10 train, 6 val; zero overlap).
   - **Recommendation:** Preferred synthetic baseline; provides high-quality coverage for `multiwell_plate` (which does not exist in LabOS or ChemEq25).

---

## 3. Proposed Unified Class Mapping (Part 2)

We propose two well-defined taxonomy options for user review and approval:

### Option 1: Core VYOM Experiment Taxonomy (10 Focused Classes — RECOMMENDED)
This taxonomy directly models the physical objects manipulated during laboratory experiment protocols (liquid transfer, tube insertion, capping, plate manipulation):

| Unified ID | Unified Class Name | LabOS Source Categories | ChemEq25 Source Categories | Synthetic v2 Categories | Physical Semantics in Procedure |
|:---:|---|---|---|---|---|
| **0** | `centrifuge_tube` | `50ml eppendorf tube` | — | `centrifuge_tube` | Conical polypropylene tube with screw cap. |
| **1** | `test_tube` | — | `Test_Tube` | — | Cylindrical glass laboratory tube. |
| **2** | `sample_vial` | — | `Weighing_Bottle` | `sample_vial` | Small screw-capped/stoppered reagent vial. |
| **3** | `pipette` | — | `Pipette`, `Volumetric_Pipet` | `pipette` | Handheld micropipette / volumetric transfer pipette. |
| **4** | `multiwell_plate` | — | — | `multiwell_plate` | Standard ANSI/SBS 96-well / 24-well microtiter plate. |
| **5** | `rig_slot` | `4 way rack 50ml hole`, `14ml rack hole` | — | `rig_slot` | Individual mounting socket / receptacle slot. |
| **6** | `tube_rack` | `50Ml 4 way rack`, `14ml rack` | `TestTube_Holder` | — | Multi-tube apparatus rack / holder. |
| **7** | `tube_cap` | `50Ml eppendorf cap`, `orange cap`, `blue cap`, `cap top` | — | — | Tube screw cap (verifies open/closed tube status). |
| **8** | `beaker` | — | `Beaker` | — | Flat-bottomed cylindrical reagent beaker. |
| **9** | `conical_flask` | — | `Conical_Flask` | — | Erlenmeyer narrow-neck flask. |

*Instance Distribution in Proposed Option 1:*
- `rig_slot`: 1,769 instances (real rack holes + procedural slots)
- `centrifuge_tube`: 464 instances (real eppendorfs + procedural tubes)
- `tube_cap`: 418 instances (real tube closures)
- `pipette`: 206 instances (real pipettes + procedural micropipettes)
- `sample_vial`: 192 instances (procedural vials + real weighing bottles)
- `multiwell_plate`: 184 instances (procedural microtiter plates)
- `tube_rack`: 123 instances (real 4-way racks & tube holders)
- `beaker`: 40 instances (real beakers)
- `conical_flask`: 26 instances (real flasks)
- `test_tube`: 18 instances (real test tubes)
- **Total:** **3,440 annotated instances across 623 images.**

---

### Option 2: Extended Multi-Apparatus Chemistry Taxonomy (18 Classes)
Retains all primary experiment objects plus secondary chemistry apparatus from ChemEq25:
- Classes 0–9 as defined in Option 1.
- Class 10: `vortex_mixer` (LabOS `Vortex Genie 2`, 108 instances)
- Class 11: `measuring_cylinder` (ChemEq25 `Measuring_Cylinder`, 252 instances)
- Class 12: `reagent_bottle` (ChemEq25 `Reagent_Bottle` + `Nessler_Reagent_Bottle`, 458 instances)
- Class 13: `burette_stand` (ChemEq25 `Burette_Stands`, 305 instances)
- Class 14: `funnel` (ChemEq25 `Funnel` + `Buchner_Funnel` + `Separating_Funnel`, 825 instances)
- Class 15: `glass_rod` (ChemEq25 `Glass_Rod`, 363 instances)
- Class 16: `precision_scale` (ChemEq25 `Precision_Weight_Scale` + `Mechanical_Balance_Scale`, 488 instances)
- Class 17: `round_bottom_flask` (ChemEq25 1-neck, 2-neck, 3-neck flasks, 796 instances)

*Trade-off Assessment:* Option 1 provides a highly focused, balanced model for the primary VYOM physical rig protocol without cluttering the detector with secondary distillation or titration apparatus.

---

## 4. Annotation Conversion Pipeline (`convert_to_unified_obb.py`)

1. **Polygon-to-OBB Conversion (LabOS):**
   $$\text{rect} = \text{cv2.minAreaRect}(\text{pts}_{\text{px}})$$
   Extracts minimum-area bounding box from high-resolution segmentation polygons, computing the exact 4 rotated corners $[p_0, p_1, p_2, p_3]$ normalized to $[0.0, 1.0]$.
2. **Axis-Aligned to OBB Conversion (ChemEq25):**
   Transforms $(x_c, y_c, w, h)$ into 4 explicit corner coordinates $[(x_c - w/2, y_c - h/2), (x_c + w/2, y_c - h/2), (x_c + w/2, y_c + h/2), (x_c - w/2, y_c + h/2)]$ with zero rotation.
   > [!NOTE]
   > ChemEq25 annotations are explicitly tagged in `SOURCES.md` and `class_mapping.yaml` as derived zero-rotation boxes; the pipeline does not claim they provide genuine rotation training variety.
3. **Split Preservation & Provenance:**
   - All source splits are preserved without leakage: ChemEq25 `train`/`valid`/`test` $\rightarrow$ `train`/`val`/`test`; LabOS `train`/`val` $\rightarrow$ `train`/`val`; Synthetic v2 `train`/`val` $\rightarrow$ `train`/`val`.
   - File prefixes (`labos_`, `chemeq_`, `syn_v2_`) guarantee 100% trace-back to original files and licenses.

---

## 5. Unified Dataset Structure & Delivered Artifacts

The unified dataset architecture is established under `d:/space_eye/vyom/data/unified_obb/`:

```text
data/unified_obb/
├── images/
│   ├── train/            (488 images: 153 LabOS, 95 ChemEq, 240 Synthetic v2)
│   ├── val/              (111 images: 40 LabOS, 11 ChemEq, 60 Synthetic v2)
│   └── test/             (24 images: 24 ChemEq)
├── labels/
│   ├── train/            (Matching YOLOv8-OBB annotations)
│   ├── val/
│   └── test/
├── previews/             (Visual inspection previews across all 3 source domains)
├── dataset.yaml          (Ultralytics YOLOv8-OBB dataset configuration)
├── class_mapping.yaml    (Formal cross-domain mapping dictionary)
├── dataset_manifest.yaml (Machine-readable image & instance counts per source)
└── SOURCES.md            (Licensing, citations, and attribution documentation)
```

---

## 6. Training Status & Next Action

> [!IMPORTANT]
> **TRAINING IS CURRENTLY PAUSED.**
> As instructed, no training run will be initiated until the user reviews and confirms the proposed class mapping.

Please review the proposed class mapping above:
- **Confirm Option 1 (Core 10-Class Taxonomy)** or request modifications to classes/mappings.
- Upon your approval, we will proceed to train the detector on this unified real + synthetic dataset or proceed to Prompt 7 (ONNX export, parity verification, and desktop runtime integration).
