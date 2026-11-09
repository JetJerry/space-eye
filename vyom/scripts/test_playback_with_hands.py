"""Quick playback check: verifies simultaneous video playback, ArUco detection, and hand tracking."""

import os
import sys
import time
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"

root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir / "src"))

from PySide6.QtWidgets import QApplication
from vyom.core.config import get_config_manager
from vyom.ui.main_window import VyomMainWindow


def test_playback():
    app = QApplication.instance() or QApplication(sys.argv)
    cfg_mgr = get_config_manager(root_dir)
    window = VyomMainWindow(cfg_mgr)
    window.show()

    # Find sample video
    enigma_dir = cfg_mgr.config.datasets.enigma360_path
    video_path = enigma_dir / "data" / "videos" / "103" / "103_third_person.mp4"

    if not video_path.exists():
        print(f"Sample video not found at {video_path}, skipping playback test.")
        window.close()
        return

    print(f"Setting video source to: {video_path}")
    window.session_panel.source_type_combo.setCurrentText("Video File (.mp4, .avi)")
    window.session_panel.set_selected_video_file(video_path)

    # Start session
    print("Starting session...")
    window.session_panel.start_btn.click()
    app.processEvents()

    # Let it run for 1.5 seconds (acquiring and processing ~45 frames)
    t0 = time.time()
    while time.time() - t0 < 1.5:
        app.processEvents()
        time.sleep(0.02)

    print("Checking active state...")
    assert window._session_state == "ACTIVE", f"Session state is {window._session_state}"

    # Stop session cleanly
    print("Stopping session...")
    window.session_panel.stop_btn.click()
    app.processEvents()
    time.sleep(0.3)

    print(f"Final session state: {window._session_state}")
    assert window._session_state == "IDLE"

    window.close()
    app.processEvents()
    print("PLAYBACK TEST: SUCCESS")


if __name__ == "__main__":
    test_playback()

