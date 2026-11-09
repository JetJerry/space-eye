# VYOM Model Weights Directory

This directory stores offline model artifacts and weights. **Model weights and checkpoint files are intentionally excluded from version control (`.gitignore`).**

## Expected Model Assets

| Asset File | Target Architecture / Source | Purpose | Required For |
|---|---|---|---|
| `object_detector.onnx` | ONNX Runtime (CPU / CUDA) | Oriented bounding-box (OBB) detector for experiment apparatus and tools | Phase 7+ |
| `yolov8_obb.pt` | Ultralytics PyTorch Checkpoint | Training / initial validation checkpoint before ONNX export | Training Phase 6 |
| `hand_landmarker.task` | Google MediaPipe Tasks Hand Landmarker | 21 3D hand landmarks per detected hand (offline bundle) | Phase 5+ |

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

