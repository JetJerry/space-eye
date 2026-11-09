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
