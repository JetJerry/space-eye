# VYOM Utility & Pipeline Scripts

This directory houses standalone utility, training, export, and dataset preparation scripts.

## Planned Utilities (Prompt Sequence)

| Script | Purpose | Guide Phase |
|---|---|---|
| `inspect_dataset.py` | Verify dataset integrity, class distribution, image dimensions, corrupt files | Phase 6 |
| `validate_obb_labels.py` | Check 4-corner polygon label validity and zero-rotation warnings | Phase 6 |
| `split_dataset.py` | Create session-aware train/val/test splits without frame leakage | Phase 6 |
| `train_obb.py` | YOLOv8-OBB training script with conservative VRAM/batch sizing | Phase 6 |
| `export_onnx.py` | Export validated `.pt` checkpoints to ONNX with dynamic/static shapes | Phase 7 |
| `compare_pt_onnx.py` | Parity verification comparing PyTorch and ONNX predictions | Phase 7 |
| `generate_aruco_markers.py` | Procedural ArUco marker generator for physical rig printing | Phase 4 |

