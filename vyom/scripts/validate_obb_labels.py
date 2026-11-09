"""YOLOv8-OBB label validator and visual verification utility.

Validates that annotations conform to the strict YOLOv8-OBB 4-corner normalized format:
    class_id x1 y1 x2 y2 x3 y3 x4 y4

Where:
- class_id is an integer >= 0
- (x1, y1) through (x4, y4) are normalized between 0.0 and 1.0
- The 4 points form a non-degenerate convex quadrilateral
- Distinguishes genuine oriented bounding boxes from horizontal (zero-rotation) boxes
"""

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np


def parse_obb_label_line(line: str) -> Optional[Tuple[int, np.ndarray]]:
    """Parse single YOLOv8-OBB label line."""
    parts = line.strip().split()
    if len(parts) != 9:
        return None
    try:
        class_id = int(parts[0])
        coords = [float(x) for x in parts[1:]]
        pts = np.array(coords, dtype=np.float32).reshape(4, 2)
        return class_id, pts
    except ValueError:
        return None


def calculate_polygon_properties(pts_norm: np.ndarray, width: int = 640, height: int = 640) -> Dict[str, Any]:
    """Compute geometric properties of the 4-corner polygon in pixel space."""
    pts_px = pts_norm.copy()
    pts_px[:, 0] *= width
    pts_px[:, 1] *= height

    area = cv2.contourArea(pts_px)
    perimeter = cv2.arcLength(pts_px, closed=True)

    # Compute minimum-area oriented bounding box to assess genuine rotation
    rect = cv2.minAreaRect(pts_px)
    (cx, cy), (w, h), angle = rect

    # Is it convex?
    is_convex = cv2.isContourConvex(pts_px.astype(np.int32))

    # Calculate angle of primary edge (p0 -> p1)
    dx = pts_px[1, 0] - pts_px[0, 0]
    dy = pts_px[1, 1] - pts_px[0, 1]
    edge_angle = float(np.degrees(np.arctan2(dy, dx))) % 180.0

    # A box is horizontal if its edges are near 0/90/180 degrees
    is_horizontal = (abs(edge_angle - 0.0) < 1.5 or abs(edge_angle - 90.0) < 1.5 or abs(edge_angle - 180.0) < 1.5)

    return {
        "area_px": round(float(area), 1),
        "perimeter_px": round(float(perimeter), 1),
        "width_px": round(float(w), 1),
        "height_px": round(float(h), 1),
        "aspect_ratio": round(float(max(w, h) / max(min(w, h), 1e-4)), 2),
        "min_area_rect_angle": round(float(angle), 1),
        "edge_angle": round(edge_angle, 1),
        "is_convex": bool(is_convex),
        "is_genuine_obb": not is_horizontal,
    }


def validate_obb_file(label_path: Path, width: int = 640, height: int = 640) -> Dict[str, Any]:
    """Validate all labels inside a single YOLOv8-OBB TXT file."""
    res = {
        "file": str(label_path),
        "valid_count": 0,
        "invalid_count": 0,
        "genuine_obb_count": 0,
        "horizontal_count": 0,
        "classes_present": [],
        "errors": [],
        "labels": [],
    }

    if not label_path.exists():
        res["errors"].append("File not found")
        return res

    lines = label_path.read_text(encoding="utf-8").splitlines()
    for line_idx, line in enumerate(lines, 1):
        parsed = parse_obb_label_line(line)
        if parsed is None:
            res["invalid_count"] += 1
            res["errors"].append(f"Line {line_idx}: Invalid format (expected 9 tokens, got {len(line.split())})")
            continue

        class_id, pts = parsed

        # Validate range [0.0, 1.0]
        if np.any(pts < -0.01) or np.any(pts > 1.01):
            res["invalid_count"] += 1
            res["errors"].append(f"Line {line_idx}: Coordinates out of normalized [0, 1] range: {pts}")
            continue

        props = calculate_polygon_properties(pts, width, height)
        if not props["is_convex"] or props["area_px"] < 4.0:
            res["invalid_count"] += 1
            res["errors"].append(f"Line {line_idx}: Degenerate or non-convex polygon (area={props['area_px']})")
            continue

        res["valid_count"] += 1
        res["classes_present"].append(class_id)
        if props["is_genuine_obb"]:
            res["genuine_obb_count"] += 1
        else:
            res["horizontal_count"] += 1

        res["labels"].append({"class_id": class_id, "points": pts.tolist(), "properties": props})

    return res


def render_obb_preview(
    image_bgr: np.ndarray,
    labels: List[Dict[str, Any]],
    class_names: Optional[List[str]] = None,
) -> np.ndarray:
    """Render oriented bounding boxes and orientation vectors onto image."""
    canvas = image_bgr.copy()
    h, w = canvas.shape[:2]

    # High-contrast palette for classes
    palette = [
        (0, 230, 115),   # Green
        (255, 140, 0),   # Cyan
        (0, 165, 255),   # Orange
        (220, 60, 220),  # Violet
        (0, 215, 255),   # Yellow
        (255, 100, 100), # Soft Blue
    ]

    for item in labels:
        cid = item["class_id"]
        pts = np.array(item["points"], dtype=np.float32)
        pts[:, 0] *= w
        pts[:, 1] *= h
        pts_int = pts.astype(np.int32)

        color = palette[cid % len(palette)]

        # 1. Draw 4-corner polygon
        cv2.polylines(canvas, [pts_int], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)

        # 2. Highlight corner 0 (leading corner) and corner 1 to show orientation direction
        cv2.circle(canvas, (int(pts[0, 0]), int(pts[0, 1])), 4, (0, 0, 255), -1, lineType=cv2.LINE_AA)
        cv2.line(canvas, (int(pts[0, 0]), int(pts[0, 1])), (int(pts[1, 0]), int(pts[1, 1])), (0, 0, 255), 2, lineType=cv2.LINE_AA)

        # 3. Class label pill
        cname = class_names[cid] if class_names and cid < len(class_names) else f"cls_{cid}"
        angle_deg = item["properties"]["edge_angle"]
        label_txt = f"{cname} ({angle_deg:.0f}°)"

        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), baseline = cv2.getTextSize(label_txt, font, 0.42, 1)
        lx = max(4, int(pts[0, 0]))
        ly = max(th + 6, int(pts[0, 1]) - 4)

        cv2.rectangle(canvas, (lx - 2, ly - th - 3), (lx + tw + 2, ly + baseline + 1), (20, 26, 36), -1)
        cv2.rectangle(canvas, (lx - 2, ly - th - 3), (lx + tw + 2, ly + baseline + 1), color, 1)
        cv2.putText(canvas, label_txt, (lx, ly), font, 0.42, (240, 245, 250), 1, cv2.LINE_AA)

    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate YOLOv8-OBB dataset labels.")
    parser.add_argument("--labels-dir", type=str, required=True, help="Directory containing YOLO OBB label txt files.")
    parser.add_argument("--images-dir", type=str, default="", help="Optional images directory for preview generation.")
    parser.add_argument("--output-preview-dir", type=str, default="", help="Optional output directory for visual previews.")
    args = parser.parse_args()

    labels_dir = Path(args.labels_dir)
    files = list(labels_dir.glob("*.txt"))
    print(f"Validating {len(files)} label files in {labels_dir}...")

    total_valid = 0
    total_invalid = 0
    total_genuine_obb = 0
    total_horizontal = 0

    for f in files:
        res = validate_obb_file(f)
        total_valid += res["valid_count"]
        total_invalid += res["invalid_count"]
        total_genuine_obb += res["genuine_obb_count"]
        total_horizontal += res["horizontal_count"]
        if res["errors"]:
            print(f"  [ERROR] {f.name}: {res['errors']}")

    print("=" * 60)
    print("OBB VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Total Files Scanned:       {len(files)}")
    print(f"Total Valid Boxes:         {total_valid}")
    print(f"Total Invalid Boxes:       {total_invalid}")
    print(f"Genuine Rotated OBBs:      {total_genuine_obb} ({total_genuine_obb / max(total_valid, 1):.1%})")
    print(f"Horizontal (Zero-Rot) OBBs:{total_horizontal} ({total_horizontal / max(total_valid, 1):.1%})")
    print("=" * 60)


if __name__ == "__main__":
    main()

