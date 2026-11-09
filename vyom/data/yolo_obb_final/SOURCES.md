# Final Unified Dataset Sources and License Attributions

## 1. Source Provenance & Licensing
- **LabOS:** `D:\space_eye\datasets\LabOS` | **CC BY-NC 4.0** | Real instance segmentations $\rightarrow$ genuine rotated OBB.
- **ChemEq25:** `D:\space_eye\datasets\ChemEq25` | **CC BY 4.0** | Verified matching pairs only $\rightarrow$ zero-rotation OBB.
- **Synthetic v2:** `d:\space_eye\vyom\data\yolo_obb_v2` | Procedural synthetic on disjoint ENIGMA-360 backgrounds.

## 2. Image and Split Distribution
- **Train Images:** 448
- **Val Images:** 125
- **Test Images:** 18
- **Total Images:** 591
- **Total Annotations:** 3396

## 3. Important Generalization Notice
- `sample_vial`, `multiwell_plate`, and `rig_slot` contain synthetic-only training examples and are marked **provisional**.
- ChemEq25 bounding boxes are axis-aligned and do not provide genuine rotation variety.
- LabOS is licensed under **CC BY-NC 4.0** (Non-Commercial). Any future commercial deployment requires licensing review.
