"""Unit tests for dataset inspection, OBB label validation, and model metadata verification."""

import json
import sys
from pathlib import Path
import numpy as np
import pytest
import yaml

# Add vyom project root so scripts package is accessible
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.validate_obb_labels import (
    calculate_polygon_properties,
    parse_obb_label_line,
    validate_obb_file,
)


def test_parse_obb_label_line():
    """Verify parsing of valid and invalid YOLOv8-OBB lines."""
    # Valid 9-token line (class_id + 8 normalized coordinates)
    valid_line = "0 0.10 0.20 0.30 0.20 0.30 0.40 0.10 0.40"
    parsed = parse_obb_label_line(valid_line)
    assert parsed is not None
    cid, pts = parsed
    assert cid == 0
    assert pts.shape == (4, 2)
    np.testing.assert_allclose(pts[0], [0.10, 0.20])

    # Invalid line (wrong number of tokens)
    assert parse_obb_label_line("0 0.1 0.2 0.3 0.4") is None
    # Invalid token (non-numeric)
    assert parse_obb_label_line("0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 abc") is None


def test_calculate_polygon_properties_rotation_detection():
    """Verify that rotated and horizontal polygons are correctly differentiated."""
    # Horizontal box (0 degree rotation)
    h_box = np.array([
        [0.1, 0.1],
        [0.3, 0.1],
        [0.3, 0.4],
        [0.1, 0.4],
    ], dtype=np.float32)
    h_props = calculate_polygon_properties(h_box, 640, 640)
    assert h_props["is_convex"]
    assert not h_props["is_genuine_obb"]  # Horizontal

    # Rotated box (~45 degrees)
    r_box = np.array([
        [0.3, 0.1],
        [0.5, 0.3],
        [0.3, 0.5],
        [0.1, 0.3],
    ], dtype=np.float32)
    r_props = calculate_polygon_properties(r_box, 640, 640)
    assert r_props["is_convex"]
    assert r_props["is_genuine_obb"]  # Rotated


def test_dataset_yaml_structure():
    """Verify dataset.yaml schema for YOLOv8-OBB."""
    yaml_path = Path("d:/space_eye/vyom/data/yolo_obb/dataset.yaml")
    if not yaml_path.exists():
        pytest.skip("dataset.yaml not found")

    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    assert "train" in cfg
    assert "val" in cfg
    assert "names" in cfg
    assert len(cfg["names"]) == 5
    assert cfg["names"][0] == "sample_vial"
    assert cfg["names"][4] == "rig_slot"


def test_model_metadata_verification():
    """Verify that model_metadata.json exists and accurately reflects the trained checkpoint."""
    meta_path = Path("d:/space_eye/vyom/models/model_metadata.json")
    if not meta_path.exists():
        pytest.skip("model_metadata.json not found")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["model_name"] == "YOLOv8n-OBB (Fine-Tuned)"
    assert meta["task"] == "Oriented Bounding Box (OBB) Detection"
    assert meta["num_classes"] == 5
    assert "metrics" in meta
    assert meta["metrics"]["mAP50"] > 0.50
    assert meta["metrics"]["mAP50_95"] > 0.40
    assert meta["training_hardware"]["cuda_available"] is True


def test_dataset_v2_manifest_and_separation():
    """Verify dataset v2 manifest, class balance, and strict background separation."""
    manifest_path = Path("d:/space_eye/vyom/data/yolo_obb_v2/manifest.json")
    if not manifest_path.exists():
        pytest.skip("dataset v2 manifest not found")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["version"] == "2.0.0"
    assert manifest["splits"]["train"]["image_count"] == 240
    assert manifest["splits"]["val"]["image_count"] == 60

    # Verify zero background video overlap
    train_vids = set(manifest["background_separation"]["train_source_videos"])
    val_vids = set(manifest["background_separation"]["val_source_videos"])
    overlap = train_vids.intersection(val_vids)
    assert len(overlap) == 0, f"Found overlapping background videos: {overlap}"
    assert manifest["background_separation"]["overlap_video_count"] == 0

    # Verify class balance in train split (all classes present and balanced)
    tr_counts = manifest["splits"]["train"]["class_counts"]
    for cname in ["sample_vial", "pipette", "multiwell_plate", "centrifuge_tube", "rig_slot"]:
        assert cname in tr_counts
        assert tr_counts[cname] > 100


def test_object_patch_geometry_tightness():
    """Verify that all 5 procedural renderers generate mathematically tight OBB bounding boxes."""
    from scripts.generate_synthetic_obb_dataset_v2 import (
        create_sample_vial_patch,
        create_pipette_patch,
        create_multiwell_plate_patch,
        create_centrifuge_tube_patch,
        create_rig_slot_patch,
        check_collision,
    )

    creators = [
        ("vial", create_sample_vial_patch, 34, 85),
        ("pipette", create_pipette_patch, 26, 140),
        ("plate", create_multiwell_plate_patch, 120, 85),
        ("tube", create_centrifuge_tube_patch, 36, 100),
        ("slot", create_rig_slot_patch, 75, 75),
    ]

    for name, fn, w, h in creators:
        patch, corners = fn(w, h)
        assert patch.shape[2] == 4  # RGBA
        assert corners.shape == (4, 2)
        min_x, min_y = corners.min(axis=0)
        max_x, max_y = corners.max(axis=0)
        # Bounding box width and height must match requested size within 1 pixel
        assert abs((max_x - min_x) - w) <= 1.0, f"{name} width mismatch"
        assert abs((max_y - min_y) - h) <= 1.0, f"{name} height mismatch"

    # Test collision detection
    box1 = np.array([[10, 10], [50, 10], [50, 50], [10, 50]], dtype=np.float32)
    box2_overlap = np.array([[30, 30], [70, 30], [70, 70], [30, 70]], dtype=np.float32)
    box3_disjoint = np.array([[100, 100], [140, 100], [140, 140], [100, 140]], dtype=np.float32)

    assert check_collision(box1, [box2_overlap]) is True
    assert check_collision(box1, [box3_disjoint]) is False

