# VYOM YOLOv8-OBB Dataset Manifest (v2)

## 1. Overview
- **Dataset Name:** VYOM Procedural Demo Apparatus Dataset v2
- **Format:** YOLOv8-OBB (`class_id x1 y1 x2 y2 x3 y3 x4 y4`, normalized $[0, 1]$)
- **Target Size:** 640 $\times$ 640
- **Total Images:** 300 (240 train, 60 val)
- **Total Annotated Instances:** 919

## 2. Background Source Video Separation
| Split | Image Count | Background Source Videos (ENIGMA-360) | Video Partition Origin | Overlap Check |
|---|---|---|---|---|
| **Train** | 240 | 103, 104, 105, 109, 110, 111, 112, 113, 114, 124 | `annotations/train.txt` | **0 overlap (Disjoint)** |
| **Val** | 60 | 100, 101, 102, 115, 116, 127 | `annotations/val.txt` | **0 overlap (Disjoint)** |

> [!IMPORTANT]
> Train and validation splits use completely disjoint source videos. No background frame or video from the training partition appears in the validation partition.

## 3. Class Taxonomy & Distribution
| Class ID | Class Name | Train Instances | Val Instances | Total Instances | Percentage |
|---|---|---|---|---|---|
| 0 | `sample_vial` | 148 | 36 | 184 | 20.0% |
| 1 | `pipette` | 148 | 36 | 184 | 20.0% |
| 2 | `multiwell_plate` | 148 | 36 | 184 | 20.0% |
| 3 | `centrifuge_tube` | 148 | 36 | 184 | 20.0% |
| 4 | `rig_slot` | 148 | 35 | 183 | 19.9% |
| **Total** | **All 5 Classes** | **740** | **179** | **919** | **100.0%** |

## 4. Visual Improvements in Version 2
1. **Multi-layer RGBA Procedural Rendering:**
   - Shading gradients (cylindrical lighting).
   - Glass transparency and refraction alpha-blending onto real workbench wood/laminate textures.
   - Reagent liquid columns with curved menisci.
   - Volumetric graduations and molded knurled cap ribs.
   - ANSI/SBS microplate A1 corner notch and well bottom shadows.
   - Machined aluminum rig receptacle with counterbored socket screws.
2. **Contact Drop Shadows:**
   - Directional soft contact shadows beneath objects, realistically grounding them onto the workbench.
3. **Collision Detection & Occlusion Prevention:**
   - Non-overlapping object placement using `cv2.intersectConvexConvex`.
   - Prevents unannotated mutual occlusions and ensures 100% boundary tightness.
