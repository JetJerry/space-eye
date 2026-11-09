# Unified Dataset Sources and License Attributions

This unified dataset (`data/unified_obb/`) integrates real and procedural laboratory apparatus data
under strict provenance tracking and original licensing terms.

## 1. Source Provenance & Licensing

### Dataset 1: LabOS Segmentation Dataset
- **Local Location:** `D:\space_eye\datasets\LabOS`
- **License:** **CC BY-NC 4.0** (Creative Commons Attribution-NonCommercial 4.0 International)
- **Role in VYOM:** Real laboratory instance segmentations converted to genuine oriented bounding boxes (`cv2.minAreaRect`).
- **Contributed Categories:** `centrifuge_tube` (50ml eppendorf), `tube_cap`, `tube_rack` (4-way & 14ml racks), `rig_slot` (rack holes).
- **Images Contributed:** 193
- **Annotations Contributed:** 2378 genuine rotated OBBs.

### Dataset 2: ChemEq25 Laboratory Dataset
- **Local Location:** `D:\space_eye\datasets\ChemEq25`
- **Source Project:** Roboflow Universe `chemistry-lab-apparatus-detn` (v8)
- **License:** **CC BY 4.0** (Creative Commons Attribution 4.0 International)
- **Role in VYOM:** Real chemistry laboratory glassware converted to zero-rotation OBB representation.
- **Contributed Categories:** `beaker`, `conical_flask`, `pipette`, `test_tube`, `sample_vial` (weighing bottle), `tube_rack` (test tube holder).
- **Images Contributed:** 130
- **Annotations Contributed:** 143 zero-rotation OBBs.
- **Attribution Note:** ChemEq25 bounding boxes are axis-aligned in the original dataset and do not provide genuine rotation variety.

### Dataset 3: VYOM Synthetic Demo Dataset v2
- **Local Location:** `d:\space_eye\vyom\data\yolo_obb_v2`
- **Role in VYOM:** High-fidelity procedural laboratory apparatus on real ENIGMA-360 disjoint backgrounds.
- **Contributed Categories:** `multiwell_plate` (unique 96-well ANSI microtiter plates), `sample_vial`, `pipette`, `centrifuge_tube`, `rig_slot`.
- **Images Contributed:** 300
- **Annotations Contributed:** 919 procedural OBBs.

---

## 2. Total Class Distribution in Unified Dataset
| Class ID | Class Name | LabOS (Real OBB) | ChemEq25 (Real AABB) | Synthetic v2 (Procedural) | Total Instances |
|:---:|---|:---:|:---:|:---:|:---:|
| 0 | `centrifuge_tube` | 280 | 0 | 184 | **464** |
| 1 | `test_tube` | 0 | 18 | 0 | **18** |
| 2 | `sample_vial` | 0 | 8 | 184 | **192** |
| 3 | `pipette` | 0 | 22 | 184 | **206** |
| 4 | `multiwell_plate` | 0 | 0 | 184 | **184** |
| 5 | `rig_slot` | 1586 | 0 | 183 | **1769** |
| 6 | `tube_rack` | 94 | 29 | 0 | **123** |
| 7 | `tube_cap` | 418 | 0 | 0 | **418** |
| 8 | `beaker` | 0 | 40 | 0 | **40** |
| 9 | `conical_flask` | 0 | 26 | 0 | **26** |

**Total Images:** 623  
**Total Annotations:** 3440  
