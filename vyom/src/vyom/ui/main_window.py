"""Primary desktop application window for VYOM with OpenCV capture and ArUco rig tracking."""

import collections
from datetime import datetime
from pathlib import Path
import time
from typing import Any, Optional, Tuple
import numpy as np
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QCloseEvent, QImage
from PySide6.QtWidgets import (
    QMainWindow,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from vyom.capture.camera_source import CameraSource
from vyom.capture.capture_worker import CaptureWorker
from vyom.capture.frame_source import FrameSource
from vyom.capture.network_source import NetworkSource
from vyom.capture.recorder import SessionRecorder
from vyom.capture.video_file_source import VideoFileSource
from vyom.core.config import ConfigManager, get_config_manager
from vyom.core.logging_setup import get_logger
from vyom.ui.event_panel import EventPanel
from vyom.ui.live_monitor import LiveMonitor
from vyom.ui.protocol_panel import ProtocolPanel
from vyom.ui.session_panel import SessionPanel
from vyom.ui.settings_dialog import SettingsDialog
from vyom.ui.styles import APPLICATION_STYLESHEET
from vyom.ui.top_bar import TopBar
from vyom.vision.aruco_tracker import ArucoDetectionResult, ArucoTracker, RigTrackingState
from vyom.vision.drawing import draw_aruco_overlays, draw_hand_overlays
from vyom.vision.hand_tracker import (
    HandTrackerConfig,
    HandTrackingResult,
    HandTrackingStatus,
)
from vyom.vision.hand_worker import HandTrackingWorker

logger = get_logger("ui.main_window")


class VyomMainWindow(QMainWindow):
    """Main desktop monitoring workstation integrating OpenCV capture and ArUco tracking."""

    def __init__(self, config_manager: ConfigManager | None = None) -> None:
        super().__init__()
        self.config_manager = config_manager or get_config_manager()
        self.config = self.config_manager.config

        self.setWindowTitle("VYOM — Experiment Activity & Procedure Compliance Workstation")
        self.setMinimumSize(1200, 750)
        self.resize(1360, 840)
        self.setStyleSheet(APPLICATION_STYLESHEET)

        self._session_state: str = "IDLE"
        self._current_source_type: str = "device_index"
        self._current_source_param: Any = self.config.camera.device_index

        self._capture_worker: Optional[CaptureWorker] = None
        self._active_recorder: Optional[SessionRecorder] = None
        self._active_session_id: Optional[str] = None

        # ArUco Fiducial and Rig Reference Tracking Engine
        self.aruco_tracker = ArucoTracker(self.config.rig)
        self._last_logged_aruco_state: Optional[RigTrackingState] = None

        # MediaPipe Hand Tracking Subsystem (Off GUI Thread)
        hand_cfg = HandTrackerConfig(
            model_path=self.config.models.hand_landmarker_path,
            num_hands=2,
            min_detection_confidence=0.5,
            min_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode="VIDEO",
        )
        self.hand_worker = HandTrackingWorker(hand_cfg, self)
        self._last_logged_hand_count: int = 0
        self._hand_fps_history = collections.deque(maxlen=20)

        self._init_ui()
        self._init_data()
        self._connect_signals()

        # Check for pre-existing downloaded ENIGMA-360 videos for easy 1-click loading
        self._check_initial_dataset_video()

        logger.info("VyomMainWindow initialized with ArUco and MediaPipe hand tracking subsystems")

    def _init_ui(self) -> None:
        """Construct the mission control desktop layout."""
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        root_layout = QVBoxLayout(central_widget)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # 1. Top Bar
        self.top_bar = TopBar(self)
        root_layout.addWidget(self.top_bar)

        # 2. Main Work Area (Vertical Splitter: Top Splitter + Event Panel)
        v_splitter = QSplitter(Qt.Vertical)

        # Horizontal Splitter: Left Sidebar + Center Monitor + Right Protocol Panel
        h_splitter = QSplitter(Qt.Horizontal)

        # Left Sidebar (Session Controls, Camera/File Selection, Protocols)
        self.session_panel = SessionPanel(self)
        h_splitter.addWidget(self.session_panel)

        # Central Live Monitor (Video Display & Overlay Zones)
        self.live_monitor = LiveMonitor(self)
        h_splitter.addWidget(self.live_monitor)

        # Right Procedure Panel (Checklist, Current Step, Next Step, Progress)
        self.protocol_panel = ProtocolPanel(self)
        h_splitter.addWidget(self.protocol_panel)

        # Proportions: 280px left, flex center, 340px right
        h_splitter.setStretchFactor(0, 0)
        h_splitter.setStretchFactor(1, 1)
        h_splitter.setStretchFactor(2, 0)
        h_splitter.setSizes([280, 740, 340])

        v_splitter.addWidget(h_splitter)

        # Bottom Event & Subsystem Status Panel
        self.event_panel = EventPanel(self)
        v_splitter.addWidget(self.event_panel)

        # Proportions: monitor top flex, bottom fixed/compact
        v_splitter.setStretchFactor(0, 1)
        v_splitter.setStretchFactor(1, 0)
        v_splitter.setSizes([580, 220])

        root_layout.addWidget(v_splitter, stretch=1)

    def _init_data(self) -> None:
        """Populate initial configurations, 4-tier hardware report, and protocols."""
        hw_report = self.config_manager.get_hardware_report()
        self.top_bar.set_compute_status(hw_report.active_device_selected, hw_report.device_selection_reason)
        self.event_panel.set_hardware_report(hw_report)

        protocols_dir = self.config.protocols_dir
        protocol_files = []
        if protocols_dir.exists():
            protocol_files = [p.name for p in protocols_dir.glob("*.json")]

        self.session_panel.set_available_protocols(
            protocol_files,
            default_selected=self.config.default_protocol,
        )

        if self.config.default_protocol in protocol_files:
            try:
                proto = self.config_manager.load_protocol(self.config.default_protocol)
                self.protocol_panel.load_protocol(proto)
            except Exception as e:
                logger.error("Failed to load initial protocol: %s", e)

        self.session_panel.set_session_state("IDLE")
        self.top_bar.set_session_status("IDLE")
        self.event_panel.set_subsystem_statuses(camera_connected=False)
        self.live_monitor.set_aruco_tracking_status(RigTrackingState.NOT_INITIALIZED, 0)

        # Initialize MediaPipe Hand Tracking worker
        success, status, err = self.hand_worker.initialize()
        if success:
            self.live_monitor.set_hand_tracking_status(HandTrackingStatus.SEARCHING)
            self.hand_worker.start()
            logger.info("MediaPipe HandTrackingWorker initialized and background thread started")
        else:
            self.live_monitor.set_hand_tracking_status(status)
            if err:
                self.event_panel.add_event(
                    severity="WARNING",
                    source="MEDIAPIPE",
                    description=err,
                    hash_digest="mediapipe:init_warning",
                )

    def _check_initial_dataset_video(self) -> None:
        """Pre-populate file selector with first available ENIGMA-360 exocentric video if present."""
        enigma_dir = self.config.datasets.enigma360_path
        videos_dir = enigma_dir / "data" / "videos"
        if videos_dir.exists():
            for mp4_file in videos_dir.glob("*/*third_person.mp4"):
                if mp4_file.stat().st_size > 1000:
                    logger.info("Discovered available ENIGMA-360 video: %s", mp4_file.name)
                    self.session_panel.set_selected_video_file(mp4_file)
                    break

    def _connect_signals(self) -> None:
        """Wire UI events across panels using Qt signals and slots."""
        self.top_bar.settings_clicked.connect(self._open_settings_dialog)

        self.session_panel.start_requested.connect(self._handle_start_session)
        self.session_panel.pause_requested.connect(self._handle_pause_session)
        self.session_panel.stop_requested.connect(self._handle_stop_session)
        self.session_panel.reset_requested.connect(self._handle_reset_session)
        self.session_panel.protocol_changed.connect(self._handle_protocol_selected)
        self.session_panel.source_changed.connect(self._handle_source_changed)

        # Wire MediaPipe Hand Tracking Worker output
        self.hand_worker.hand_result_ready.connect(self._on_hand_result_ready)

    # ==========================================================================
    # Background Vision Processing Pipeline (Off GUI Thread)
    # ==========================================================================

    def _vision_frame_processor(
        self,
        frame_bgr: np.ndarray,
        timestamp_s: float,
        frame_idx: int,
    ) -> Tuple[np.ndarray, Any]:
        """Runs inside CaptureWorker QThread to process frame and render ArUco and Hand overlays."""
        # 1. Asynchronously submit frame to HandTrackingWorker (non-blocking latest-frame queue)
        self.hand_worker.submit_frame(frame_bgr, timestamp_s, frame_idx)

        # 2. Detect ArUco fiducials and compute rig reference
        detection_result = self.aruco_tracker.detect(frame_bgr, timestamp_s)

        # 3. Retrieve freshest non-stale hand tracking result (<= 200ms latency)
        hand_result = self.hand_worker.get_latest_result(max_age_s=0.200)

        # 4. Draw ArUco fiducials and 3D axes (if calibrated) onto preview frame
        calib = self.aruco_tracker.rig_coords.calibration if self.aruco_tracker.rig_coords.is_calibrated else None
        cam_mtx = calib.camera_matrix if calib else None
        dist_coef = calib.dist_coeffs if calib else None

        annotated_bgr = draw_aruco_overlays(
            frame_bgr,
            detection_result,
            camera_matrix=cam_mtx,
            dist_coeffs=dist_coef,
            axis_length_m=self.aruco_tracker.marker_size_m * 0.8,
        )

        # 5. Draw Hand Landmark overlays (coexists with ArUco overlays)
        if hand_result is not None and hand_result.has_hands:
            annotated_bgr = draw_hand_overlays(
                annotated_bgr,
                hand_result,
                draw_connections=True,
                draw_landmarks=True,
                draw_handedness_pill=True,
            )

        return annotated_bgr, (detection_result, hand_result)

    # ==========================================================================
    # Video Capture Worker Management
    # ==========================================================================

    def _create_frame_source(self) -> Optional[FrameSource]:
        """Instantiate selected concrete FrameSource."""
        if self._current_source_type == "device_index":
            dev_idx = int(self._current_source_param)
            return CameraSource(
                device_index=dev_idx,
                target_width=self.config.camera.width,
                target_height=self.config.camera.height,
                target_fps=self.config.camera.target_fps,
            )
        elif self._current_source_type == "video_file":
            file_path = self.session_panel.selected_video_file
            if not file_path or not file_path.exists():
                self.event_panel.add_event(
                    severity="WARNING",
                    source="CAPTURE",
                    description="No valid video file selected. Click 'Browse...' to select an MP4/AVI file.",
                )
                return None
            return VideoFileSource(
                file_path=file_path,
                loop=self.session_panel.should_loop_video,
            )
        elif self._current_source_type == "rtsp_stream":
            url = str(self._current_source_param)
            return NetworkSource(stream_url=url, target_fps=self.config.camera.target_fps)
        return None

    @Slot()
    def _handle_start_session(self) -> None:
        """Start acquisition worker thread and optional recording."""
        if self._capture_worker is not None and self._capture_worker.isRunning():
            logger.warning("Capture worker already running")
            return

        source = self._create_frame_source()
        if source is None:
            return

        self._active_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        recorder: Optional[SessionRecorder] = None
        if self.session_panel.is_recording_enabled:
            rec_dir = self.config.storage.sessions_dir / "recordings"
            recorder = SessionRecorder(
                output_dir=rec_dir,
                session_id=self._active_session_id,
                width=self.config.camera.width,
                height=self.config.camera.height,
                fps=float(self.config.camera.target_fps),
            )
            self._active_recorder = recorder
            self.live_monitor.set_recording_status(True)

        # Reset ArUco tracking state
        self._last_logged_aruco_state = None

        # Initialize CaptureWorker thread with vision processor
        self._capture_worker = CaptureWorker(self)
        self._capture_worker.configure(
            source=source,
            recorder=recorder,
            frame_processor=self._vision_frame_processor,
        )

        # Wire thread signals
        self._capture_worker.frame_ready.connect(self._on_frame_ready)
        self._capture_worker.vision_result_ready.connect(self._on_vision_result_ready)
        self._capture_worker.source_opened.connect(self._on_source_opened)
        self._capture_worker.source_closed.connect(self._on_source_closed)
        self._capture_worker.source_error.connect(self._on_source_error)
        self._capture_worker.fps_measured.connect(self._on_fps_measured)
        self._capture_worker.recording_started.connect(self._on_recording_started)
        self._capture_worker.recording_stopped.connect(self._on_recording_stopped)

        self._session_state = "ACTIVE"
        self.session_panel.set_session_state("ACTIVE")
        self.top_bar.set_session_status("ACTIVE")
        self.live_monitor.set_camera_status(True, source.get_metadata().source_label)

        self._capture_worker.start()

    @Slot(QImage, int, float)
    def _on_frame_ready(self, qimg: QImage, frame_idx: int, timestamp_s: float) -> None:
        """Handle incoming frame from worker thread and render to viewport."""
        self.live_monitor.update_frame_qimage(qimg, frame_idx, timestamp_s)

    @Slot(object)
    def _on_vision_result_ready(self, result: object) -> None:
        """Handle perception detection result emitted from background worker."""
        aruco_res = None
        if isinstance(result, tuple) and len(result) >= 1:
            aruco_res = result[0]
        elif isinstance(result, ArucoDetectionResult):
            aruco_res = result

        if not isinstance(aruco_res, ArucoDetectionResult):
            return

        # 1. Update Monitor ArUco badge
        self.live_monitor.set_aruco_tracking_status(
            state=aruco_res.state,
            configured_count=len(aruco_res.configured_ids_visible),
            total_configured=len(self.aruco_tracker.configured_marker_ids),
        )

        # 2. Update Event Panel ArUco indicator
        self.event_panel.set_aruco_status(aruco_res)

        # 3. Log state changes to audit timeline (without spamming every frame)
        if aruco_res.state != self._last_logged_aruco_state:
            self._last_logged_aruco_state = aruco_res.state
            if aruco_res.state == RigTrackingState.RIG_TRACKING:
                self.event_panel.add_event(
                    severity="INFO",
                    source="ARUCO",
                    description=f"Rig workspace reference established. Origin marker ID {self.aruco_tracker.origin_marker_id} visible (Active IDs: {aruco_res.configured_ids_visible}).",
                    hash_digest="aruco:rig_tracking",
                )
            elif aruco_res.state == RigTrackingState.CALIBRATION_REQUIRED:
                self.event_panel.add_event(
                    severity="WARNING",
                    source="ARUCO",
                    description=f"ArUco marker IDs {aruco_res.configured_ids_visible} detected. 2D pixels tracked; 3D metric coordinates unavailable (camera calibration required).",
                    hash_digest="aruco:uncalibrated",
                )
            elif aruco_res.state == RigTrackingState.SEARCHING:
                self.event_panel.add_event(
                    severity="INFO",
                    source="ARUCO",
                    description="ArUco fiducials not currently in view. Searching for DICT_4X4_50...",
                )

    @Slot(object)
    def _on_hand_result_ready(self, result: object) -> None:
        """Handle MediaPipe hand tracking result emitted from background HandTrackingWorker."""
        if not isinstance(result, HandTrackingResult):
            return

        # 1. Compute Hand Tracking FPS
        now = time.monotonic()
        self._hand_fps_history.append(now)
        fps = 0.0
        if len(self._hand_fps_history) >= 2:
            dur = self._hand_fps_history[-1] - self._hand_fps_history[0]
            if dur > 0.001:
                fps = (len(self._hand_fps_history) - 1) / dur

        # 2. Update Live Monitor badge
        handedness_labels = [f"{h.handedness} ({h.confidence:.0%})" for h in result.hands]
        total_lms = sum(len(h.landmarks) for h in result.hands)
        self.live_monitor.set_hand_tracking_status(
            status=result.status,
            hand_count=result.hand_count,
            handedness_labels=handedness_labels,
            landmark_count=total_lms,
            processing_fps=fps,
            latency_ms=result.latency_ms,
        )

        # 3. Update Event Panel status label
        self.event_panel.set_hand_tracking_status(result)

        # 4. Log hand presence state transitions to audit timeline
        if result.hand_count != self._last_logged_hand_count:
            self._last_logged_hand_count = result.hand_count
            if result.hand_count > 0:
                self.event_panel.add_event(
                    severity="INFO",
                    source="HAND_TRACKER",
                    description=f"Hand detected: {', '.join(handedness_labels)} with 21 landmarks each (latency: {result.latency_ms:.1f}ms).",
                    hash_digest="hand:detected",
                )
            else:
                self.event_panel.add_event(
                    severity="INFO",
                    source="HAND_TRACKER",
                    description="Hand(s) exited camera view.",
                    hash_digest="hand:cleared",
                )

    @Slot(str, int, int, float)
    def _on_source_opened(self, source_name: str, width: int, height: int, fps: float) -> None:
        self.live_monitor.set_source_info(source_name, width, height, fps)
        self.event_panel.set_subsystem_statuses(
            camera_connected=True,
            camera_desc=f"{source_name} ({width}x{height})",
        )
        self.event_panel.add_event(
            severity="INFO",
            source="CAPTURE",
            description=f"Capture stream opened: {source_name} ({width}x{height} @ {fps:.1f} FPS)",
            hash_digest="stream:opened",
        )

    @Slot()
    def _on_source_closed(self) -> None:
        self._session_state = "IDLE"
        self.session_panel.set_session_state("IDLE")
        self.top_bar.set_session_status("IDLE")
        self.live_monitor.set_recording_status(False)
        self.live_monitor.set_camera_status(False, "Stopped")
        self.event_panel.set_subsystem_statuses(camera_connected=False)
        self.event_panel.add_event(
            severity="INFO",
            source="CAPTURE",
            description="Capture stream closed and resources released.",
            hash_digest="stream:closed",
        )

    @Slot(str)
    def _on_source_error(self, error_msg: str) -> None:
        self.event_panel.add_event(
            severity="WARNING",
            source="CAPTURE",
            description=error_msg,
        )

    @Slot(float, int)
    def _on_fps_measured(self, fps: float, dropped: int) -> None:
        self.live_monitor.set_measured_fps(fps)
        self.event_panel.set_subsystem_statuses(
            camera_connected=(self._session_state == "ACTIVE"),
            camera_desc=self._get_current_source_info(),
            fps=fps,
            dropped=dropped,
        )

    @Slot(str)
    def _on_recording_started(self, file_path: str) -> None:
        self.live_monitor.set_recording_status(True)
        self.event_panel.add_event(
            severity="INFO",
            source="RECORDER",
            description=f"Session recording started -> {Path(file_path).name}",
        )

    @Slot(str, int, float)
    def _on_recording_stopped(self, file_path: str, total_frames: int, duration_s: float) -> None:
        self.live_monitor.set_recording_status(False)
        self.event_panel.add_event(
            severity="INFO",
            source="RECORDER",
            description=f"Recording finalized: {Path(file_path).name} ({total_frames} frames, {duration_s:.1f}s)",
        )

    @Slot()
    def _handle_pause_session(self) -> None:
        if self._capture_worker is None:
            return

        if self._session_state == "ACTIVE":
            self._session_state = "PAUSED"
            self._capture_worker.pause_capture()
            self.session_panel.set_session_state("PAUSED")
            self.top_bar.set_session_status("PAUSED")
            self.event_panel.add_event(
                severity="WARNING",
                source="SESSION",
                description="Capture stream paused by operator.",
            )
        else:
            self._session_state = "ACTIVE"
            self._capture_worker.resume_capture()
            self.session_panel.set_session_state("ACTIVE")
            self.top_bar.set_session_status("ACTIVE")
            self.event_panel.add_event(
                severity="INFO",
                source="SESSION",
                description="Capture stream resumed by operator.",
            )

    @Slot()
    def _handle_stop_session(self) -> None:
        if self._capture_worker is not None:
            self._capture_worker.stop_capture()
            self._capture_worker = None

        if hasattr(self, "hand_worker") and self.hand_worker is not None:
            self.hand_worker.reset()

        self._last_logged_hand_count = 0
        self._session_state = "IDLE"
        self.session_panel.set_session_state("IDLE")
        self.top_bar.set_session_status("IDLE")
        self.live_monitor.set_recording_status(False)
        self.live_monitor.set_camera_status(False, self._get_current_source_info())
        self.live_monitor.set_hand_tracking_status(HandTrackingStatus.SEARCHING)
        self.event_panel.set_subsystem_statuses(camera_connected=False)

    @Slot()
    def _handle_reset_session(self) -> None:
        self._handle_stop_session()
        self.protocol_panel.set_current_step(0)
        self.event_panel.clear_timeline()
        self.event_panel.add_event(
            severity="INFO",
            source="SESSION",
            description="Session reset. Compliance progress returned to STEP_01.",
        )

    @Slot(str)
    def _handle_protocol_selected(self, protocol_name: str) -> None:
        if not protocol_name:
            return
        try:
            proto = self.config_manager.load_protocol(protocol_name)
            self.protocol_panel.load_protocol(proto)
            self.event_panel.add_event(
                severity="INFO",
                source="PROTOCOL",
                description=f"Loaded protocol '{proto.experiment_name}' (ID: {proto.protocol_id}) with {len(proto.steps)} steps.",
            )
        except Exception as e:
            self.event_panel.add_event(
                severity="CRITICAL",
                source="PROTOCOL",
                description=f"Failed to load protocol {protocol_name}: {e}",
            )

    @Slot(str, object)
    def _handle_source_changed(self, source_type: str, source_val: object) -> None:
        self._current_source_type = source_type
        self._current_source_param = source_val
        info = self._get_current_source_info()
        self.live_monitor.set_camera_status(self._session_state == "ACTIVE", info)
        self.event_panel.add_event(
            severity="INFO",
            source="CAPTURE",
            description=f"Source configured: {info}",
        )

    def _get_current_source_info(self) -> str:
        st = self.session_panel.source_type_combo.currentText()
        if "Webcam" in st:
            return f"Device {self.session_panel.device_spin.value()} (1280x720 @ 30 FPS)"
        if "File" in st:
            f = self.session_panel.selected_video_file
            return f"File: {f.name}" if f else "File: None selected"
        return "Network Stream (RTSP)"

    @Slot()
    def _open_settings_dialog(self) -> None:
        dialog = SettingsDialog(self.config_manager, self)
        if dialog.exec():
            hw_report = self.config_manager.get_hardware_report()
            self.top_bar.set_compute_status(hw_report.active_device_selected, hw_report.device_selection_reason)
            self.event_panel.set_hardware_report(hw_report)

    def closeEvent(self, event: QCloseEvent) -> None:
        """Ensure capture worker thread, hand tracking worker, and recording are stopped cleanly."""
        if self._capture_worker is not None:
            logger.info("Stopping capture worker during application shutdown")
            self._capture_worker.stop_capture()
            self._capture_worker = None

        if hasattr(self, "hand_worker") and self.hand_worker is not None:
            logger.info("Stopping hand tracking worker during application shutdown")
            self.hand_worker.stop()

        event.accept()
