#!/usr/bin/env python3
"""Interactive or batch camera calibration utility using planar chessboard patterns."""

import argparse
import sys
from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vyom.core.config import get_config_manager
from vyom.vision.calibration import CameraCalibration, CheckerboardCalibrator


def calibrate_from_images(image_paths: list[Path], rows: int, cols: int, square_size_m: float) -> CameraCalibration | None:
    calibrator = CheckerboardCalibrator(board_rows=rows, board_cols=cols, square_size_m=square_size_m)
    print(f"Processing {len(image_paths)} calibration images (Pattern: {cols}x{rows}, Square: {square_size_m*1000:.1f}mm)...")

    valid_views = 0
    for img_p in image_paths:
        frame = cv2.imread(str(img_p))
        if frame is None:
            continue
        ok, corners = calibrator.add_frame(frame)
        if ok:
            valid_views += 1
            print(f" ✓ Detected corners in: {img_p.name}")
        else:
            print(f" ✗ Pattern not found in: {img_p.name}")

    if valid_views < 8:
        print(f"ERROR: Only {valid_views} valid views detected. Need at least 8 to compute reliable calibration.", file=sys.stderr)
        return None

    return calibrator.compute_calibration()


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate camera intrinsics using planar chessboard pattern")
    parser.add_argument("--images-dir", type=Path, default=None, help="Directory containing calibration chessboard images")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "config" / "camera_calibration.json", help="Output calibration JSON path")
    parser.add_argument("--rows", type=int, default=6, help="Interior chessboard rows (corners)")
    parser.add_argument("--cols", type=int, default=9, help="Interior chessboard columns (corners)")
    parser.add_argument("--square-size-mm", type=float, default=25.0, help="Square side length in mm")
    args = parser.parse_args()

    sq_m = args.square_size_mm / 1000.0

    if args.images_dir and args.images_dir.exists():
        img_paths = sorted(list(args.images_dir.glob("*.jpg")) + list(args.images_dir.glob("*.png")))
        calib = calibrate_from_images(img_paths, args.rows, args.cols, sq_m)
        if calib:
            calib.save_to_file(args.output)
            print(f"\nSUCCESS: Camera calibration saved to: {args.output}")
            print(f"Reprojection Error: {calib.reprojection_error:.4f} px")
            print(f"Camera Matrix:\n{calib.camera_matrix}")
            return 0
        return 1

    print("INFO: No calibration images directory supplied.")
    print("Usage example: python scripts/calibrate_camera.py --images-dir data/calibration_images/")
    return 0


if __name__ == "__main__":
    sys.exit(main())

