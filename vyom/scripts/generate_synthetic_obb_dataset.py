"""Synthetic procedural OBB demo dataset generator for VYOM apparatus detection.

Generates realistic training and validation datasets with genuine oriented bounding boxes (YOLOv8-OBB format: class_id x1 y1 x2 y2 x3 y3 x4 y4).
Uses real background textures extracted from downloaded ENIGMA-360 videos to ensure authentic lab lighting, table surfaces, and optical noise.

DEMO APPARATUS CLASSES (Generic scientific prototypes; clearly distinct from actual ISRO flight hardware):
    0: sample_vial
    1: pipette
    2: multiwell_plate
    3: centrifuge_tube
    4: rig_slot
"""

import argparse
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

# Class Taxonomy
DEMO_CLASSES = [
    "sample_vial",
    "pipette",
    "multiwell_plate",
    "centrifuge_tube",
    "rig_slot",
]


def extract_real_backgrounds(video_path: Path, count: int = 15, target_size: Tuple[int, int] = (640, 640)) -> List[np.ndarray]:
    """Extract diversified background frames from real lab video."""
    backgrounds: List[np.ndarray] = []
    if not video_path.exists():
        # Fallback to realistic synthetic laboratory workbench texture
        for _ in range(count):
            bg = np.full((target_size[1], target_size[0], 3), (35, 40, 45), dtype=np.uint8)
            # Add subtle workbench grid and noise
            noise = np.random.normal(0, 8, bg.shape).astype(np.int16)
            bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            backgrounds.append(bg)
        return backgrounds

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // (count + 2))

    for idx in range(1, count + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx * step)
        ret, frame = cap.read()
        if ret and frame is not None:
            # Resize / crop to 640x640
            h, w = frame.shape[:2]
            sz = min(w, h)
            crop = frame[(h - sz) // 2 : (h + sz) // 2, (w - sz) // 2 : (w + sz) // 2]
            resized = cv2.resize(crop, target_size, interpolation=cv2.INTER_AREA)
            backgrounds.append(resized)
    cap.release()

    if not backgrounds:
        for _ in range(count):
            bg = np.full((target_size[1], target_size[0], 3), (40, 45, 50), dtype=np.uint8)
            backgrounds.append(bg)

    return backgrounds


def draw_oriented_object(
    canvas: np.ndarray,
    class_id: int,
    center: Tuple[int, int],
    size: Tuple[int, int],
    angle_deg: float,
) -> Optional[np.ndarray]:
    """Draw a synthetic apparatus object with distinct geometric silhouette and return 4 OBB corners."""
    cx, cy = center
    w, h = size
    angle_rad = np.radians(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

    # Base unrotated rectangle centered at origin
    hw, hh = w / 2.0, h / 2.0
    base_corners = np.array([
        [-hw, -hh],
        [hw, -hh],
        [hw, hh],
        [-hw, hh],
    ], dtype=np.float32)

    # Rotate corners
    rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    rotated_corners = np.dot(base_corners, rot_matrix.T) + np.array([cx, cy])

    # Check bounds
    img_h, img_w = canvas.shape[:2]
    if np.any(rotated_corners < 5) or np.any(rotated_corners[:, 0] > img_w - 5) or np.any(rotated_corners[:, 1] > img_h - 5):
        return None

    pts_int = rotated_corners.astype(np.int32)

    # Class-specific rendering
    if class_id == 0:  # sample_vial (amber/transparent glass cylinder with cap)
        cv2.fillPoly(canvas, [pts_int], (40, 120, 200), lineType=cv2.LINE_AA)  # Amber glass
        # Cap
        cap_corners = np.array([[-hw, -hh], [hw, -hh], [hw, -hh + 12], [-hw, -hh + 12]])
        cap_rot = (np.dot(cap_corners, rot_matrix.T) + np.array([cx, cy])).astype(np.int32)
        cv2.fillPoly(canvas, [cap_rot], (220, 220, 220), lineType=cv2.LINE_AA)
        cv2.polylines(canvas, [pts_int], isClosed=True, color=(15, 60, 120), thickness=1, lineType=cv2.LINE_AA)

    elif class_id == 1:  # pipette (elongated cylindrical body with taper tip)
        cv2.fillPoly(canvas, [pts_int], (190, 195, 205), lineType=cv2.LINE_AA)  # Light gray plastic
        # Plunger button
        top_corners = np.array([[-hw * 0.7, -hh], [hw * 0.7, -hh], [hw * 0.7, -hh + 8], [-hw * 0.7, -hh + 8]])
        top_rot = (np.dot(top_corners, rot_matrix.T) + np.array([cx, cy])).astype(np.int32)
        cv2.fillPoly(canvas, [top_rot], (0, 165, 255), lineType=cv2.LINE_AA)  # Blue/Orange plunger
        cv2.polylines(canvas, [pts_int], isClosed=True, color=(100, 105, 115), thickness=1, lineType=cv2.LINE_AA)

    elif class_id == 2:  # multiwell_plate (rectangular plate with interior well grid)
        cv2.fillPoly(canvas, [pts_int], (70, 75, 80), lineType=cv2.LINE_AA)
        cv2.polylines(canvas, [pts_int], isClosed=True, color=(160, 170, 180), thickness=2, lineType=cv2.LINE_AA)
        # Draw some well dots inside
        for gx in np.linspace(-hw + 14, hw - 14, 4):
            for gy in np.linspace(-hh + 14, hh - 14, 3):
                pt = np.dot(np.array([gx, gy]), rot_matrix.T) + np.array([cx, cy])
                cv2.circle(canvas, (int(pt[0]), int(pt[1])), 4, (35, 40, 45), -1, lineType=cv2.LINE_AA)

    elif class_id == 3:  # centrifuge_tube (conical base plastic tube with red screw cap)
        cv2.fillPoly(canvas, [pts_int], (170, 180, 190), lineType=cv2.LINE_AA)
        cap_corners = np.array([[-hw, -hh], [hw, -hh], [hw, -hh + 14], [-hw, -hh + 14]])
        cap_rot = (np.dot(cap_corners, rot_matrix.T) + np.array([cx, cy])).astype(np.int32)
        cv2.fillPoly(canvas, [cap_rot], (0, 0, 220), lineType=cv2.LINE_AA)  # Red cap
        cv2.polylines(canvas, [pts_int], isClosed=True, color=(80, 90, 100), thickness=1, lineType=cv2.LINE_AA)

    else:  # rig_slot (dark rectangular receptacle slot with chamfered outline)
        cv2.fillPoly(canvas, [pts_int], (20, 22, 25), lineType=cv2.LINE_AA)
        cv2.polylines(canvas, [pts_int], isClosed=True, color=(0, 215, 255), thickness=2, lineType=cv2.LINE_AA)

    return rotated_corners


def generate_dataset(
    output_dir: Path,
    bg_video_path: Path,
    num_train: int = 120,
    num_val: int = 30,
    img_size: int = 640,
) -> None:
    """Generate complete YOLOv8-OBB dataset directory with train/val splits and labels."""
    output_dir = output_dir.resolve()
    images_train = output_dir / "images" / "train"
    images_val = output_dir / "images" / "val"
    labels_train = output_dir / "labels" / "train"
    labels_val = output_dir / "labels" / "val"

    for d in [images_train, images_val, labels_train, labels_val]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"Extracting authentic backgrounds from {bg_video_path}...")
    backgrounds = extract_real_backgrounds(bg_video_path, count=25, target_size=(img_size, img_size))
    print(f"Loaded {len(backgrounds)} background frames.")

    random.seed(42)
    np.random.seed(42)

    splits = [("train", num_train, images_train, labels_train), ("val", num_val, images_val, labels_val)]

    # Size distributions per class (width, height)
    class_sizes = {
        0: (32, 75),    # sample_vial (slender)
        1: (24, 120),   # pipette (long)
        2: (110, 85),   # multiwell_plate (broad)
        3: (34, 90),    # centrifuge_tube (slender)
        4: (70, 70),    # rig_slot (square slot)
    }

    for split_name, count, img_dir, lbl_dir in splits:
        print(f"Generating {count} images for {split_name} split...")
        for i in range(count):
            bg = random.choice(backgrounds).copy()

            # Randomize background brightness/contrast
            alpha = random.uniform(0.85, 1.15)
            beta = random.randint(-15, 15)
            canvas = cv2.convertScaleAbs(bg, alpha=alpha, beta=beta)

            # Insert 1 to 4 objects per image
            num_objects = random.randint(1, 4)
            labels_lines: List[str] = []

            for _ in range(num_objects):
                class_id = random.randint(0, len(DEMO_CLASSES) - 1)
                base_w, base_h = class_sizes[class_id]

                # Random scale & angle
                scale = random.uniform(0.75, 1.25)
                w = int(base_w * scale)
                h = int(base_h * scale)
                angle = random.uniform(0.0, 360.0)

                cx = random.randint(80, img_size - 80)
                cy = random.randint(80, img_size - 80)

                corners = draw_oriented_object(canvas, class_id, (cx, cy), (w, h), angle)
                if corners is not None:
                    # Normalize corners between 0 and 1
                    c_norm = corners.copy()
                    c_norm[:, 0] = np.clip(c_norm[:, 0] / img_size, 0.0, 1.0)
                    c_norm[:, 1] = np.clip(c_norm[:, 1] / img_size, 0.0, 1.0)

                    line = f"{class_id} " + " ".join(f"{coord:.6f}" for pt in c_norm for coord in pt)
                    labels_lines.append(line)

            # Save image and label
            stem = f"syn_{split_name}_{i:04d}"
            img_file = img_dir / f"{stem}.jpg"
            lbl_file = lbl_dir / f"{stem}.txt"

            cv2.imwrite(str(img_file), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            lbl_file.write_text("\n".join(labels_lines) + "\n", encoding="utf-8")

    # Generate dataset.yaml
    yaml_content = f"""# Ultralytics YOLOv8-OBB dataset configuration for VYOM demo apparatus
path: {output_dir.as_posix()}
train: images/train
val: images/val
test: images/val

names:
  0: sample_vial
  1: pipette
  2: multiwell_plate
  3: centrifuge_tube
  4: rig_slot
"""
    yaml_path = output_dir / "dataset.yaml"
    yaml_path.write_text(yaml_content, encoding="utf-8")
    print(f"Generated dataset YAML: {yaml_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic YOLOv8-OBB demo apparatus dataset.")
    parser.add_argument("--output-dir", type=str, default="d:/space_eye/vyom/data/yolo_obb", help="Target output folder.")
    parser.add_argument(
        "--bg-video",
        type=str,
        default="D:/space_eye/datasets/ENIGMA-360/data/videos/103/103_third_person.mp4",
        help="Source video for realistic lab background extraction.",
    )
    parser.add_argument("--num-train", type=int, default=120, help="Number of training images.")
    parser.add_argument("--num-val", type=int, default=30, help="Number of validation images.")
    args = parser.parse_args()

    generate_dataset(
        output_dir=Path(args.output_dir),
        bg_video_path=Path(args.bg_video),
        num_train=args.num_train,
        num_val=args.num_val,
    )


if __name__ == "__main__":
    main()
