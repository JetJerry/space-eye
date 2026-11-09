"""Perception pipeline combining YOLOv8-OBB, MediaPipe Hands, and ArUco markers.

Planned components (Prompts 4, 5, 7):
- pipeline.py: Central perception loop and multi-model synchronization.
- object_detector.py: YOLOv8-OBB detector (PyTorch & ONNX Runtime with CPU/CUDA).
- hand_tracker.py: MediaPipe Tasks Hand Landmarker (21 landmarks).
- aruco_tracker.py: OpenCV ArUco detector and pose estimator.
- calibration.py: Camera calibration matrix and rig-relative transformation.
- drawing.py: Modular OpenCV overlay renderer for detections and landmarks.
"""

