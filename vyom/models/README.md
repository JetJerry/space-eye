# VYOM Model Weights Directory

This directory stores offline model artifacts and weights. **Model weights and checkpoint files are intentionally excluded from version control (`.gitignore`).**

## Expected Model Assets

| Asset File | Target Architecture / Source | Purpose | Required For |
|---|---|---|---|
| `object_detector.onnx` | ONNX Runtime (CPU / CUDA) | Oriented bounding-box (OBB) detector for experiment apparatus and tools | Phase 7+ |
| `yolov8_obb_best.pt` | Ultralytics PyTorch Checkpoint | Custom fine-tuned YOLOv8n-OBB model weights (trained Phase 6) | Phase 6, 7+ |
| `yolov8n-obb.pt` | Ultralytics Base Pretrained | Base pretrained YOLOv8n-OBB weights | Phase 6 |
| `hand_landmarker.task` | Google MediaPipe Tasks Hand Landmarker | 21 3D hand landmarks per detected hand (offline bundle) | Phase 5+ |

### Custom Model Provenance: `yolov8_obb_best.pt`
- **Architecture:** YOLOv8n-OBB (3.1M parameters)
- **Base Checkpoint:** `yolov8n-obb.pt`
- **Output File:** `models/yolov8_obb_best.pt`
- **File Size:** 6,405,649 bytes (~6.4 MB)
- **SHA-256 Checksum:** `014f3dcf79ac591b4243944c1bab961025468c8c33dfb98890428d059a7ccc63`
- **Training Device:** NVIDIA GeForce RTX 3050 Laptop GPU (4GB VRAM, AMP mixed precision)
- **Training Time:** 54.1 seconds (5 epochs)
- **Classes (5):** `sample_vial`, `pipette`, `multiwell_plate`, `centrifuge_tube`, `rig_slot`
- **Validation Metrics:**
  - Precision: `68.44%`
  - Recall: `74.60%`
  - mAP@0.50: `78.79%`
  - mAP@0.50:0.95: `71.29%`
- **Metadata Log:** [`models/model_metadata.json`](file:///d:/space_eye/vyom/models/model_metadata.json)

### Official Model Provenance: `hand_landmarker.task`
- **Official URL:** `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task`
- **File Size:** 7,819,105 bytes (~7.8 MB)
- **SHA-256 Checksum:** `fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1`
- **License / Terms:** Google MediaPipe Tasks Open Source Apache 2.0 / Pretrained Model Bundle
- **Running Mode:** Offline CPU inference with TensorFlow Lite XNNPACK delegate
- **Outputs:** 21 normalized image landmarks (x, y, z), 21 hand-relative metric world landmarks (x, y, z in meters), and left/right handedness classification with confidence score.

## Security and Integrity Guidelines

- Every loaded model file is verified at startup using SHA-256 checksums recorded in session audit logs.
- Never commit binary model weights to the Git repository.
- Download the official `hand_landmarker.task` only from the verified Google MediaPipe repository.
- If a model file is missing or not configured, VYOM gracefully displays `MODEL NOT CONFIGURED` rather than fabricating predictions.

