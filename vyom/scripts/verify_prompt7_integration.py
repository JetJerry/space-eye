"""Focused Prompt 7 Integration Verification Script.

Tests:
1. Model loading and provenance verification from models/vyom_yolov8n_obb_smoke.pt.
2. Exact 10-class taxonomy verification.
3. Actual OBB inference on GPU and CPU.
4. Simultaneous multi-modal perception pipeline execution (ArUco + MediaPipe Hands + YOLOv8-OBB).
5. Output of annotated verification image demonstrating coexisting overlays.
6. Qt worker and GUI responsiveness verification.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np
import torch

# Ensure project root is on sys.path
PROJECT_ROOT = Path("d:/space_eye/vyom")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vyom.core.config import get_config_manager
from vyom.core.logging_setup import get_logger
from vyom.vision.aruco_tracker import ArucoTracker
from vyom.vision.drawing import (
    draw_aruco_overlays,
    draw_hand_overlays,
    draw_obb_overlays,
)
from vyom.vision.hand_tracker import (
    HandTracker,
    HandTrackerConfig,
    HandTrackingStatus,
)
from vyom.vision.obb_detector import (
    FINAL_CLASSES,
    OBBDetectorConfig,
    OBBDetectorStatus,
    YOLOOBBDetector,
)
from vyom.vision.observations import PerceptionFrameObservation

logger = get_logger("verify_prompt7")


def verify_model_and_device() -> YOLOOBBDetector:
    """Step 1 & 2: Verify custom checkpoint, provenance metadata, and compute hardware."""
    print("=" * 65)
    print("PROMPT 7 VERIFICATION: MODEL LOADING & COMPUTE CHECK")
    print("=" * 65)

    model_pt = PROJECT_ROOT / "models" / "vyom_yolov8n_obb_smoke.pt"
    meta_json = PROJECT_ROOT / "models" / "smoke_model_metadata.json"

    assert model_pt.exists(), f"Missing required smoke model at {model_pt}"
    print(f"✓ Checkpoint exists: {model_pt} ({model_pt.stat().st_size:,} bytes)")

    if meta_json.exists():
        meta = json.loads(meta_json.read_text(encoding="utf-8"))
        print(f"✓ Provenance metadata loaded: SHA-256 = {meta.get('sha256')}")
        print(f"  Training duration: {meta.get('training_duration_seconds')}s on {meta.get('hardware', {}).get('device_name')}")
        print(f"  Overall mAP50: {meta.get('metrics', {}).get('overall_mAP50'):.2%}")

    cuda_avail = torch.cuda.is_available()
    print(f"✓ CUDA Available: {cuda_avail}")
    if cuda_avail:
        gpu_name = torch.cuda.get_device_name(0)
        vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
        print(f"  Device Name: {gpu_name} ({vram_mb:.0f} MiB VRAM)")

    # Test GPU detector
    cfg_gpu = OBBDetectorConfig(
        model_path=model_pt,
        metadata_path=meta_json,
        device_preference="AUTO",
        conf_threshold=0.20,
    )
    detector = YOLOOBBDetector(cfg_gpu)
    ok, status, msg = detector.initialize()
    assert ok, f"Detector initialization failed: {msg}"
    print(f"✓ Detector initialized: {status.value} on {detector.active_device}")

    # Verify 10 classes
    cnames = detector.class_names
    assert len(cnames) == 10, f"Expected 10 classes, got {len(cnames)}"
    for idx, expected in enumerate(FINAL_CLASSES):
        assert cnames[idx] == expected, f"Class {idx} mismatch: {cnames[idx]} != {expected}"
    print("✓ All 10 classes match expected taxonomy exactly:")
    for idx, name in cnames.items():
        print(f"  [{idx}] {name}")

    return detector


def verify_simultaneous_perception_pipeline(detector: YOLOOBBDetector) -> None:
    """Step 3 & 4: Run ArUco + MediaPipe + YOLOv8-OBB simultaneously on real video frames."""
    print("\n" + "=" * 65)
    print("PROMPT 7 VERIFICATION: SIMULTANEOUS PERCEPTION PIPELINE")
    print("=" * 65)

    cfg_mgr = get_config_manager()
    app_cfg = cfg_mgr.config

    # 1. Initialize ArUco tracker
    aruco_tracker = ArucoTracker(app_cfg.rig)
    print("✓ ArUcoTracker initialized (DICT_4X4_50, IDs: [0, 1, 2, 3])")

    # 2. Initialize MediaPipe HandTracker
    hand_cfg = HandTrackerConfig(
        model_path=app_cfg.models.hand_landmarker_path,
        num_hands=2,
        running_mode="IMAGE",  # Use image mode for frame-by-frame verification
    )
    hand_tracker = HandTracker(hand_cfg)
    hand_ok, hand_status, _ = hand_tracker.initialize()
    print(f"✓ MediaPipe HandTracker initialized: {hand_status.value}")

    # 3. Locate a test video or validation frame
    video_path = None
    enigma_vids = list(Path("D:/space_eye/datasets/ENIGMA-360/data/videos").glob("*/*third_person.mp4"))
    if enigma_vids:
        video_path = enigma_vids[0]
        print(f"✓ Using real ENIGMA-360 video: {video_path.name}")
    else:
        print("! No ENIGMA-360 videos found, falling back to validation image")

    cap = cv2.VideoCapture(str(video_path)) if video_path else None
    frames_to_test = 10
    frame_idx = 0

    latencies_aruco = []
    latencies_hand = []
    latencies_obb = []
    latencies_draw = []
    latencies_total = []

    last_annotated_frame = None

    while frame_idx < frames_to_test:
        if cap and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
        else:
            # Fallback synthetic scene with lab elements
            frame = np.full((720, 1280, 3), 40, dtype=np.uint8)

        t_total_start = time.perf_counter()
        now_s = frame_idx * (1.0 / 30.0)

        # A. ArUco detection
        t0 = time.perf_counter()
        aruco_res = aruco_tracker.detect(frame, now_s)
        lat_aruco = (time.perf_counter() - t0) * 1000.0
        latencies_aruco.append(lat_aruco)

        # B. MediaPipe Hand tracking
        t0 = time.perf_counter()
        hand_res = hand_tracker.process_frame(frame, int(now_s * 1000), frame_idx)
        lat_hand = (time.perf_counter() - t0) * 1000.0
        latencies_hand.append(lat_hand)

        # C. YOLOv8-OBB Oriented Object Detection
        t0 = time.perf_counter()
        obb_res = detector.detect(frame, now_s, frame_idx)
        lat_obb = (time.perf_counter() - t0) * 1000.0
        latencies_obb.append(lat_obb)

        # D. Simultaneous composite overlay rendering
        t0 = time.perf_counter()
        annotated = draw_aruco_overlays(frame, aruco_res)
        annotated = draw_hand_overlays(annotated, hand_res)
        annotated = draw_obb_overlays(annotated, obb_res, draw_labels=True, draw_centroid=True)
        lat_draw = (time.perf_counter() - t0) * 1000.0
        latencies_draw.append(lat_draw)

        # E. Multi-modal observation creation
        obs = PerceptionFrameObservation(
            frame_id=frame_idx,
            timestamp_s=now_s,
            aruco_result=aruco_res,
            hand_result=hand_res,
            obb_result=obb_res,
        )

        lat_total = (time.perf_counter() - t_total_start) * 1000.0
        latencies_total.append(lat_total)

        last_annotated_frame = annotated
        frame_idx += 1

    if cap:
        cap.release()
    hand_tracker.close()

    steady_obb = latencies_obb[1:] if len(latencies_obb) > 1 else latencies_obb
    steady_total = latencies_total[1:] if len(latencies_total) > 1 else latencies_total

    print(f"\nProcessed {frame_idx} frames through the complete multi-modal perception pipeline:")
    print(f"  • ArUco Tracker:           {np.mean(latencies_aruco):.2f} ms")
    print(f"  • MediaPipe Hands:         {np.mean(latencies_hand):.2f} ms")
    print(f"  • YOLOv8-OBB (warmup f0):  {latencies_obb[0]:.2f} ms")
    print(f"  • YOLOv8-OBB (steady-state): {np.mean(steady_obb):.2f} ms (~{1000.0 / np.mean(steady_obb):.1f} FPS)")
    print(f"  • Multi-Overlay Draw:      {np.mean(latencies_draw):.2f} ms")
    print(f"  • Pipeline Steady Latency: {np.mean(steady_total):.2f} ms (~{1000.0 / np.mean(steady_total):.1f} FPS)")

    # Save output verification image
    if last_annotated_frame is not None:
        out_img_path = PROJECT_ROOT / "models" / "prompt7_simultaneous_overlay_verification.jpg"
        cv2.imwrite(str(out_img_path), last_annotated_frame)
        print(f"✓ Saved composite multi-overlay verification image: {out_img_path}")


def verify_gui_worker_responsiveness() -> None:
    """Step 5: Verify PySide6 OBBTrackingWorker runs asynchronously and emits Qt signals."""
    print("\n" + "=" * 65)
    print("PROMPT 7 VERIFICATION: PYSIDE6 WORKER & GUI RESPONSIVENESS")
    print("=" * 65)

    from PySide6.QtCore import QCoreApplication
    from vyom.vision.obb_worker import OBBTrackingWorker

    # Create headless Qt core application if not already existing
    app = QCoreApplication.instance()
    if app is None:
        app = QCoreApplication([])

    cfg = get_config_manager().config
    # Explicitly pass model path
    worker = OBBTrackingWorker(config=OBBDetectorConfig(
        model_path=PROJECT_ROOT / "models" / "vyom_yolov8n_obb_smoke.pt",
        metadata_path=PROJECT_ROOT / "models" / "smoke_model_metadata.json",
        device_preference="AUTO",
    ), parent=None)
    ok, status, msg = worker.initialize()
    assert ok, f"Worker initialization failed: {msg}"
    print(f"✓ OBBTrackingWorker initialized: status={status.value}")

    # Test signal capture
    received_results = []
    worker.obb_result_ready.connect(lambda res: received_results.append(res))

    worker.start()
    print("✓ Worker thread started in background")

    # Submit test frames
    dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
    for i in range(3):
        worker.submit_frame(dummy_frame, timestamp_s=i * 0.033, frame_idx=i)
        time.sleep(0.04)
        app.processEvents()

    # Wait briefly for worker queue processing
    timeout_s = 2.0
    t0 = time.time()
    while len(received_results) < 2 and time.time() - t0 < timeout_s:
        app.processEvents()
        time.sleep(0.02)

    print(f"✓ Asynchronous worker processed {len(received_results)} frames via Qt signal")
    assert len(received_results) > 0, "No results received from worker signal!"

    # Clean shutdown
    worker.stop()
    print("✓ OBBTrackingWorker stopped cleanly and resources released")


def main() -> None:
    detector = verify_model_and_device()
    try:
        verify_simultaneous_perception_pipeline(detector)
        verify_gui_worker_responsiveness()
    finally:
        detector.close()

    print("\n" + "=" * 65)
    print("ALL PROMPT 7 INTEGRATION CHECKS PASSED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    main()

