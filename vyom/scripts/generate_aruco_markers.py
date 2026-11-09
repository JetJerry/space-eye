#!/usr/bin/env python3
"""Procedural ArUco marker and printable rig sheet generator for VYOM."""

import argparse
import sys
from pathlib import Path
import cv2
import numpy as np

# Ensure src/ is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vyom.core.config import get_config_manager
from vyom.vision.aruco_tracker import ARUCO_DICTIONARY_MAP


def generate_marker_image(
    aruco_dict,
    marker_id: int,
    marker_size_px: int = 500,
    margin_px: int = 60,
    label_text: str = "",
) -> np.ndarray:
    """Generate high-resolution ArUco marker PNG with quiet white border and text metadata."""
    # Generate pure black/white marker bit matrix
    marker_bits = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size_px, borderBits=1)

    total_w = marker_size_px + 2 * margin_px
    label_height = 50 if label_text else 0
    total_h = marker_size_px + 2 * margin_px + label_height

    # White canvas
    canvas = np.full((total_h, total_w), 255, dtype=np.uint8)

    # Place marker inside canvas
    canvas[margin_px : margin_px + marker_size_px, margin_px : margin_px + marker_size_px] = marker_bits

    # Draw corner alignment tick marks
    tick_len = 15
    for (x, y) in [
        (margin_px, margin_px),
        (margin_px + marker_size_px, margin_px),
        (margin_px + marker_size_px, margin_px + marker_size_px),
        (margin_px, margin_px + marker_size_px),
    ]:
        cv2.drawMarker(canvas, (x, y), 180, markerType=cv2.MARKER_TILTED_CROSS, markerSize=tick_len, thickness=1)

    # Render label text
    if label_text:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (tw, th), _ = cv2.getTextSize(label_text, font, font_scale, thickness)
        tx = max(10, (total_w - tw) // 2)
        ty = total_h - 20
        cv2.putText(canvas, label_text, (tx, ty), font, font_scale, 30, thickness, cv2.LINE_AA)

    return canvas


def generate_composite_rig_sheet(
    aruco_dict,
    marker_ids: list[int],
    output_path: Path,
    marker_size_px: int = 350,
) -> None:
    """Generate a printable 2x2 grid calibration rig sheet with all markers and origin layout."""
    margin = 50
    gap = 80
    sample_img = generate_marker_image(aruco_dict, 0, marker_size_px=marker_size_px, margin_px=20, label_text="Test")
    mh, mw = sample_img.shape

    sheet_w = 2 * mw + 2 * margin + gap
    sheet_h = 2 * mh + 2 * margin + gap + 100

    sheet = np.full((sheet_h, sheet_w), 255, dtype=np.uint8)

    # Sheet Title
    header = "VYOM EXPERIMENT RIG FIDUCIAL BOARD — DICT_4X4_50 [50mm x 50mm]"
    cv2.putText(sheet, header, (margin, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, 0, 1, cv2.LINE_AA)
    sub = "Print at 100% scale (Do not fit to page). Marker 0 = Rig Origin (0,0,0)"
    cv2.putText(sheet, sub, (margin, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.42, 100, 1, cv2.LINE_AA)

    # 2x2 Positions
    positions = [
        (margin, margin + 60),
        (margin + mw + gap, margin + 60),
        (margin, margin + mh + gap + 60),
        (margin + mw + gap, margin + mh + gap + 60),
    ]

    for idx, mid in enumerate(marker_ids[:4]):
        if idx >= len(positions):
            break
        px, py = positions[idx]
        role = " [RIG ORIGIN (0,0,0)]" if mid == 0 else ""
        label = f"ID: {mid}{role} | 50mm"
        m_img = generate_marker_image(
            aruco_dict,
            mid,
            marker_size_px=marker_size_px,
            margin_px=20,
            label_text=label,
        )
        h, w = m_img.shape
        sheet[py : py + h, px : px + w] = m_img

    cv2.imwrite(str(output_path), sheet)
    print(f" • Generated composite printable sheet: {output_path.name} ({sheet.shape[1]}x{sheet.shape[0]} px)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate printable ArUco markers for VYOM experiment rig")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "markers", help="Target output directory")
    parser.add_argument("--dictionary", type=str, default=None, help="ArUco dictionary name (default from rig.json)")
    parser.add_argument("--size-px", type=int, default=500, help="Marker resolution in pixels")
    args = parser.parse_args()

    config = get_config_manager().config.rig
    dict_name = args.dictionary or config.dictionary_name
    marker_ids = config.marker_ids
    marker_size_mm = int(config.marker_size_m * 1000)

    if dict_name not in ARUCO_DICTIONARY_MAP:
        print(f"ERROR: Unknown dictionary '{dict_name}'. Available: {list(ARUCO_DICTIONARY_MAP.keys())}", file=sys.stderr)
        return 1

    dict_id = ARUCO_DICTIONARY_MAP[dict_name]
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)

    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {len(marker_ids)} ArUco markers for dictionary {dict_name}...")

    for mid in marker_ids:
        role = " [RIG ORIGIN]" if mid == config.origin_marker_id else ""
        label = f"VYOM FIDUCIAL | ID: {mid}{role} | {marker_size_mm}mm x {marker_size_mm}mm | {dict_name}"
        img = generate_marker_image(
            aruco_dict,
            mid,
            marker_size_px=args.size_px,
            margin_px=60,
            label_text=label,
        )
        out_file = out_dir / f"aruco_marker_{mid}.png"
        cv2.imwrite(str(out_file), img)
        print(f" • Generated: {out_file.name} ({img.shape[1]}x{img.shape[0]} px)")

    # Composite printable sheet
    composite_path = out_dir / "aruco_rig_sheet.png"
    generate_composite_rig_sheet(aruco_dict, marker_ids, composite_path)

    print(f"\nAll markers successfully generated in: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
