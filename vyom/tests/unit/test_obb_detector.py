"""Focused unit and integration verification for YOLOv8-OBB detector subsystem."""

import json
from pathlib import Path
import cv2
import numpy as np
import pytest
import torch

from vyom.vision.drawing import OBB_CLASS_COLORS, draw_obb_overlays
from vyom.vision.obb_detector import (
    FINAL_CLASSES,
    OBBDetectorConfig,
    OBBDetectorStatus,
    OBBDetection,
    OBBFrameDetections,
    PROVISIONAL_SYNTHETIC_CLASSES,
    REAL_WORLD_CLASSES,
    YOLOOBBDetector,
)
from vyom.vision.observations import (
    ObjectObservation,
    PerceptionFrameObservation,
)

SMOKE_MODEL_PT = Path("d:/space_eye/vyom/models/vyom_yolov8n_obb_smoke.pt")
SMOKE_META_JSON = Path("d:/space_eye/vyom/models/smoke_model_metadata.json")


def test_missing_model_handling():
    """Verify detector reports MODEL_MISSING and stops without substituting generic weights."""
    cfg = OBBDetectorConfig(model_path=Path("models/non_existent_model_checkpoint.pt"))
    detector = YOLOOBBDetector(cfg)
    success, status, msg = detector.initialize()

    assert success is False
    assert status == OBBDetectorStatus.MODEL_MISSING
    assert "does not exist" in msg
    assert not detector.is_initialized


def test_model_loading_and_class_taxonomy():
    """Verify custom smoke checkpoint loads, verifies task='obb', and matches 10 classes."""
    if not SMOKE_MODEL_PT.exists():
        pytest.skip(f"Smoke checkpoint not found at {SMOKE_MODEL_PT}")

    cfg = OBBDetectorConfig(
        model_path=SMOKE_MODEL_PT,
        metadata_path=SMOKE_META_JSON,
        device_preference="AUTO",
    )
    detector = YOLOOBBDetector(cfg)
    success, status, msg = detector.initialize()

    assert success is True
    assert status == OBBDetectorStatus.READY
    assert detector.is_initialized

    # Verify 10 classes match expected taxonomy exactly
    cnames = detector.class_names
    assert len(cnames) == 10
    for idx, expected_name in enumerate(FINAL_CLASSES):
        assert cnames[idx] == expected_name

    # Verify provenance metadata
    if SMOKE_META_JSON.exists():
        assert detector.metadata is not None
        assert detector.metadata["num_classes"] == 10
        assert detector.metadata["metrics"]["overall_mAP50"] > 0.50

    detector.close()


def test_cpu_and_gpu_device_selection():
    """Verify CPU preference and GPU preference reporting."""
    if not SMOKE_MODEL_PT.exists():
        pytest.skip(f"Smoke checkpoint not found at {SMOKE_MODEL_PT}")

    # 1. Force CPU
    cfg_cpu = OBBDetectorConfig(
        model_path=SMOKE_MODEL_PT,
        device_preference="CPU",
    )
    det_cpu = YOLOOBBDetector(cfg_cpu)
    success_cpu, _, _ = det_cpu.initialize()
    assert success_cpu is True
    assert det_cpu.active_device == "cpu"
    det_cpu.close()

    # 2. Force GPU if CUDA available
    if torch.cuda.is_available():
        cfg_gpu = OBBDetectorConfig(
            model_path=SMOKE_MODEL_PT,
            device_preference="GPU",
        )
        det_gpu = YOLOOBBDetector(cfg_gpu)
        success_gpu, _, _ = det_gpu.initialize()
        assert success_gpu is True
        assert "cuda" in det_gpu.active_device
        det_gpu.close()


def test_inference_on_real_sample_image():
    """Verify detection, four corners, center, dimensions, and angle on actual validation image."""
    if not SMOKE_MODEL_PT.exists():
        pytest.skip(f"Smoke checkpoint not found at {SMOKE_MODEL_PT}")

    # Use a sample validation image from yolo_obb_final
    sample_img_path = Path("d:/space_eye/vyom/data/yolo_obb_final/images/val/labos_val_102.png")
    if not sample_img_path.exists():
        # Fallback to any image in val
        val_imgs = list(Path("d:/space_eye/vyom/data/yolo_obb_final/images/val").glob("*.*"))
        if not val_imgs:
            pytest.skip("No validation images found")
        sample_img_path = val_imgs[0]

    img = cv2.imread(str(sample_img_path))
    assert img is not None

    cfg = OBBDetectorConfig(
        model_path=SMOKE_MODEL_PT,
        conf_threshold=0.20,
        device_preference="AUTO",
    )
    detector = YOLOOBBDetector(cfg)
    detector.initialize()

    result = detector.detect(img, timestamp_s=1.234, frame_id=42)

    assert isinstance(result, OBBFrameDetections)
    assert result.frame_id == 42
    assert result.timestamp_s == 1.234
    assert result.inference_time_ms > 0.0
    assert result.frame_shape == img.shape[:2]

    # Verify detection structure
    for det in result.detections:
        assert isinstance(det, OBBDetection)
        assert 0 <= det.class_id < 10
        assert det.class_name in FINAL_CLASSES
        assert 0.0 <= det.confidence <= 1.0
        assert det.corners.shape == (4, 2)
        assert len(det.center) == 2
        assert det.width > 0
        assert det.height > 0
        assert det.frame_id == 42

    detector.close()


def test_obb_drawing_overlay():
    """Verify draw_obb_overlays generates valid annotated images without exceptions."""
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    corners = np.array([
        [100, 100],
        [200, 120],
        [180, 220],
        [80, 200],
    ], dtype=np.float32)

    det = OBBDetection(
        class_id=4,
        class_name="pipette",
        confidence=0.92,
        corners=corners,
        center=(140.0, 160.0),
        width=100.0,
        height=100.0,
        angle_deg=15.0,
        frame_id=1,
        timestamp_s=0.033,
        is_provisional=False,
    )

    result = OBBFrameDetections(
        frame_id=1,
        timestamp_s=0.033,
        detections=[det],
        inference_time_ms=5.5,
        device_used="cuda:0",
        frame_shape=(480, 640),
    )

    annotated = draw_obb_overlays(dummy_frame, result, draw_labels=True, draw_centroid=True)
    assert annotated.shape == dummy_frame.shape
    # Check that drawing actually modified pixels
    assert np.any(annotated > 0)


def test_structured_observation_interface():
    """Verify PerceptionFrameObservation serializes all fields required for Prompt 8."""
    corners = np.array([[10, 10], [50, 10], [50, 50], [10, 50]], dtype=np.float32)
    det = OBBDetection(
        class_id=0,
        class_name="centrifuge_tube",
        confidence=0.88,
        corners=corners,
        center=(30.0, 30.0),
        width=40.0,
        height=40.0,
        angle_deg=0.0,
        frame_id=10,
        timestamp_s=0.333,
        is_provisional=False,
    )

    obb_res = OBBFrameDetections(
        frame_id=10,
        timestamp_s=0.333,
        detections=[det],
        inference_time_ms=6.1,
        device_used="cuda:0",
        frame_shape=(720, 1280),
    )

    obs = PerceptionFrameObservation(
        frame_id=10,
        timestamp_s=0.333,
        aruco_result=None,
        hand_result=None,
        obb_result=obb_res,
    )

    assert obs.has_objects
    assert not obs.has_hands
    assert not obs.has_rig

    objects = obs.get_objects()
    assert len(objects) == 1
    assert isinstance(objects[0], ObjectObservation)
    assert objects[0].object_id == "centrifuge_tube_0"
    assert objects[0].class_name == "centrifuge_tube"

    d = obs.to_dict()
    assert d["frame_id"] == 10
    assert d["objects"]["count"] == 1
    assert d["objects"]["items"][0]["object_id"] == "centrifuge_tube_0"

