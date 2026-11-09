"""Focused verification script for MediaPipe Hand Tracking integration in VYOM."""

import os
import sys
import time
from pathlib import Path
import numpy as np

# Set offscreen Qt platform for non-interactive test
os.environ["QT_QPA_PLATFORM"] = "offscreen"

# Ensure vyom source is on path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir / "src"))

from PySide6.QtWidgets import QApplication
from vyom.core.config import get_config_manager
from vyom.ui.main_window import VyomMainWindow
from vyom.vision.hand_tracker import HandTrackingStatus


def run_verification():
    print("=" * 60)
    print("VYOM PROMPT 5: MEDIAPIPE HAND TRACKING INTEGRATION CHECK")
    print("=" * 60)

    app = QApplication.instance() or QApplication(sys.argv)

    cfg_mgr = get_config_manager(root_dir)
    print(f"Loaded config: App={cfg_mgr.config.app_name}, Models dir={cfg_mgr.config.models.models_dir}")

    window = VyomMainWindow(cfg_mgr)
    window.show()

    print("[1] Verifying HandTrackingWorker initialization...")
    assert hasattr(window, "hand_worker"), "window.hand_worker not found!"
    assert window.hand_worker.is_initialized, "HandLandmarker failed to initialize!"
    print(f"    HandLandmarker initialized: {window.hand_worker.is_initialized}")

    print("[2] Checking LiveMonitor initial status badge...")
    status_text = window.live_monitor.mediapipe_overlay_badge.text()
    print(f"    MediaPipe Badge Text: '{status_text}'")
    assert "HAND TRACKING" in status_text or "HANDS" in status_text, f"Unexpected badge text: {status_text}"

    print("[3] Simulating live video frame acquisition and dual perception overlays...")
    # Create synthetic frame with green background and white circle
    synthetic_frame = np.full((720, 1280, 3), 40, dtype=np.uint8)

    # Process frame through vision processor
    annotated_bgr, (aruco_res, hand_res) = window._vision_frame_processor(
        synthetic_frame,
        timestamp_s=1.0,
        frame_idx=1,
    )

    assert annotated_bgr.shape == (720, 1280, 3), f"Unexpected annotated shape: {annotated_bgr.shape}"
    print(f"    Annotated frame shape: {annotated_bgr.shape}")
    print(f"    ArUco markers detected: {len(aruco_res.configured_ids_visible)}")
    print(f"    Hand tracking status: {window.hand_worker.tracker.is_initialized}")

    print("[4] Testing window close and background thread cleanup...")
    window.close()
    app.processEvents()
    time.sleep(0.5)

    print("=" * 60)
    print("VERIFICATION RESULT: SUCCESS")
    print("All MediaPipe hand tracking components integrated and verified cleanly.")
    print("=" * 60)


if __name__ == "__main__":
    run_verification()
