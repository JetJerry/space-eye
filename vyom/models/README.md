# VYOM Model Weights Directory

This directory stores offline model artifacts and weights. **Model weights and checkpoint files are intentionally excluded from version control (`.gitignore`).**

## Expected Model Assets

| Asset File | Target Architecture / Source | Purpose | Required For |
|---|---|---|---|
| `object_detector.onnx` | ONNX Runtime (CPU / CUDA) | Oriented bounding-box (OBB) detector for experiment apparatus and tools | Phase 7+ |
| `yolov8_obb.pt` | Ultralytics PyTorch Checkpoint | Training / initial validation checkpoint before ONNX export | Training Phase 6 |
| `hand_landmarker.task` | Google MediaPipe Tasks Hand Landmarker | 21 3D hand landmarks per detected hand (offline bundle) | Phase 5+ |

## Security and Integrity Guidelines

- Every loaded model file is verified at startup using SHA-256 checksums recorded in session audit logs.
- Never commit binary model weights to the Git repository.
- Download the official `hand_landmarker.task` only from the verified Google MediaPipe repository.
- If a model file is missing or not configured, VYOM gracefully displays `MODEL NOT CONFIGURED` rather than fabricating predictions.

