"""Primary desktop application window for VYOM with OpenCV capture and ArUco rig tracking."""

from vyom.interaction import InteractionEventType
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
from vyom.vision.drawing import (
    draw_aruco_overlays,
    draw_hand_overlays,
    draw_obb_overlays,
)
from vyom.vision.hand_tracker import (
    HandTrackerConfig,
    HandTrackingResult,
    HandTrackingStatus,
)
from vyom.vision.hand_worker import HandTrackingWorker
from vyom.vision.obb_detector import (
    OBBDetectorConfig,
    OBBDetectorStatus,
    OBBFrameDetections,
)
from vyom.vision.obb_worker import OBBTrackingWorker
from vyom.alerts import AlertManager, VoiceAlertConfig, VoiceAlertItem
from vyom.audit.hash_chain import AuditChainLogger
from vyom.audit.reporter import ExperimentReporter
from vyom.audit.verifier import verify_audit_chain
from vyom.interaction.drawing import draw_interaction_overlays
from vyom.interaction.engine import (
    HandObjectRelationship,
    InteractionConfig,
    InteractionFrameResult,
)
from vyom.interaction.events import InteractionEvent, InteractionState
from vyom.interaction.worker import InteractionWorker
from vyom.protocol import ProcedureEngine, ProtocolDefinition
from vyom.core.profiler import SystemPerformanceProfiler, get_performance_profiler
from vyom.storage.database import DatabaseManager
from vyom.storage.session_manager import SessionManager
from vyom.ui.diagnostics_dialog import DiagnosticsDialog
from vyom.ui.readiness_dialog import SessionReadinessChecker, SessionReadinessDialog
from vyom.ui.session_history_dialog import SessionHistoryDialog
from vyom.vision.observations import PerceptionFrameObservation

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

        # YOLOv8-OBB Oriented Object Detection Subsystem (Off GUI Thread)
        smoke_model_path = self.config.models.yolo_pt_path or Path("models/vyom_yolov8n_obb_smoke.pt")
        smoke_meta_path = getattr(self.config.models, "smoke_metadata_path", None) or Path("models/smoke_model_metadata.json")
        obb_cfg = OBBDetectorConfig(
            model_path=smoke_model_path,
            metadata_path=smoke_meta_path,
            conf_threshold=0.25,
            iou_threshold=0.45,
            imgsz=640,
            device_preference=self.config.inference.device_preference.value,
            allow_cpu_fallback=self.config.inference.allow_cpu_fallback,
        )
        self.obb_worker = OBBTrackingWorker(obb_cfg, self)
        self._last_logged_obb_count: int = 0

        # Temporal Hand-Object Interaction Recognition Subsystem (Off GUI Thread)
        self.interaction_worker = InteractionWorker(InteractionConfig(), self)
        self._last_logged_interaction_state: Optional[str] = None

        # Deterministic Procedure Validation & Compliance Engine (Prompt 9)
        self.procedure_engine = ProcedureEngine()

        # Offline Voice Alerts & Experiment Guidance System (Prompt 10)
        voice_cfg = getattr(self.config, "voice_alerts", VoiceAlertConfig())
        self.alert_manager = AlertManager(voice_cfg, self)
        self.alert_manager.start()

        # SQLite Persistent Storage and Hash-Chain Audit Logging (Prompt 11)
        self.sessions_dir = Path("sessions")
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.db_manager = DatabaseManager(self.sessions_dir / "vyom_sessions.db")
        self.db_manager.initialize()
        self.session_manager = SessionManager(self.db_manager)
        self.reporter = ExperimentReporter(output_root=self.sessions_dir)
        self.audit_logger: Optional[AuditChainLogger] = None

        # System Performance Profiler & Runtime Telemetry (Prompt 12)
        self.profiler = get_performance_profiler()

        self._init_ui()
        self._init_data()
        self._connect_signals()

        # Check for pre-existing downloaded ENIGMA-360 videos for easy 1-click loading
        self._check_initial_dataset_video()

        logger.info("VyomMainWindow initialized with ArUco, MediaPipe, YOLOv8-OBB, and Interaction subsystems")

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

        # Recover any orphaned sessions from unexpected shutdowns or power loss (Prompt 12)
        recovered = self.session_manager.recover_interrupted_sessions()
        if recovered:
            self.event_panel.add_event(
                severity="WARNING",
                source="STORAGE",
                description=f"Recovered {len(recovered)} interrupted session(s) from prior runs ({', '.join(recovered[:2])}).",
                hash_digest="session:recovery",
            )

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
                if isinstance(proto, ProtocolDefinition):
                    self.procedure_engine.set_protocol(proto)
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

        # Initialize YOLOv8-OBB Object Detection worker
        obb_ok, obb_status, obb_msg = self.obb_worker.initialize()
        if obb_ok:
            cnames = list(self.obb_worker.detector.class_names.values())
            self.live_monitor.set_yolo_status(
                status=obb_status,
                device=self.obb_worker.active_device,
                class_names=cnames,
            )
            self.obb_worker.start()
            logger.info("YOLOv8-OBB OBBTrackingWorker initialized and background thread started")
            self.event_panel.add_event(
                severity="INFO",
                source="YOLO_OBB",
                description=f"Model loaded on {self.obb_worker.active_device.upper()} ({len(cnames)} classes: {', '.join(cnames[:4])}...)",
                hash_digest="yolo_obb:init_ok",
            )
        else:
            self.live_monitor.set_yolo_status(
                status=obb_status,
                device=self.obb_worker.active_device,
            )
            self.event_panel.add_event(
                severity="WARNING" if "MISSING" in obb_status.value else "ERROR",
                source="YOLO_OBB",
                description=obb_msg,
                hash_digest="yolo_obb:init_err",
            )

        # Start Interaction Recognition worker
        self.interaction_worker.start()
        logger.info("InteractionWorker background thread started")
        self.live_monitor.set_interaction_badge("INTERACTION: IDLE")

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
        self.top_bar.diagnostics_clicked.connect(self._open_diagnostics_dialog)

        self.session_panel.start_requested.connect(self._handle_start_session)
        self.session_panel.pause_requested.connect(self._handle_pause_session)
        self.session_panel.stop_requested.connect(self._handle_stop_session)
        self.session_panel.reset_requested.connect(self._handle_reset_session)
        self.session_panel.history_requested.connect(self._open_session_history_dialog)
        self.session_panel.protocol_changed.connect(self._handle_protocol_selected)
        self.session_panel.source_changed.connect(self._handle_source_changed)

        # Wire MediaPipe Hand Tracking Worker output
        self.hand_worker.hand_result_ready.connect(self._on_hand_result_ready)

        # Wire YOLOv8-OBB Object Detection Worker output
        self.obb_worker.obb_result_ready.connect(self._on_obb_result_ready)

        # Wire Temporal Interaction Recognition Worker outputs
        self.interaction_worker.interaction_result_ready.connect(self._on_interaction_result_ready)
        self.interaction_worker.interaction_event_emitted.connect(self._on_interaction_event_emitted)

        # Wire Offline Voice Alert Manager signals (Prompt 10)
        self.alert_manager.voice_status_changed.connect(self._on_voice_status_changed)
        self.alert_manager.voice_alert_queued.connect(self._on_voice_alert_queued)
        self.alert_manager.voice_speech_started.connect(self._on_voice_speech_started)
        self.alert_manager.voice_speech_finished.connect(self._on_voice_speech_finished)
        self.alert_manager.voice_speech_error.connect(self._on_voice_speech_error)
        self.alert_manager.voice_alert_dropped.connect(self._on_voice_alert_dropped)

    # ==========================================================================
    # Background Vision Processing Pipeline (Off GUI Thread)
    # ==========================================================================

    def _vision_frame_processor(
        self,
        frame_bgr: np.ndarray,
        timestamp_s: float,
        frame_idx: int,
    ) -> Tuple[np.ndarray, Any]:
        """Runs inside CaptureWorker QThread to process frame and render ArUco, Hand, YOLO, and Interaction overlays."""
        # 1. Asynchronously submit frame to HandTrackingWorker (non-blocking latest-frame queue)
        self.hand_worker.submit_frame(frame_bgr, timestamp_s, frame_idx)

        # 2. Asynchronously submit frame to OBBTrackingWorker (non-blocking latest-frame queue)
        self.obb_worker.submit_frame(frame_bgr, timestamp_s, frame_idx)

        # 3. Detect ArUco fiducials and compute rig reference
        t_ar_0 = time.perf_counter()
        detection_result = self.aruco_tracker.detect(frame_bgr, timestamp_s)
        self.profiler.record_subsystem_latency("aruco", (time.perf_counter() - t_ar_0) * 1000.0)

        # 4. Retrieve freshest non-stale hand tracking result (<= 200ms latency)
        hand_result = self.hand_worker.get_latest_result(max_age_s=0.200)

        # 5. Retrieve freshest non-stale OBB detection result (<= 250ms latency)
        obb_result = self.obb_worker.get_latest_result(max_age_s=0.250)

        # 6. Draw ArUco fiducials and 3D axes (if calibrated) onto preview frame
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

        # 7. Draw Hand Landmark overlays (coexists with ArUco overlays)
        if hand_result is not None and hand_result.has_hands:
            annotated_bgr = draw_hand_overlays(
                annotated_bgr,
                hand_result,
                draw_connections=True,
                draw_landmarks=True,
                draw_handedness_pill=True,
            )

        # 8. Draw YOLOv8-OBB Oriented Bounding Box Overlays (coexists with ArUco and Hand overlays)
        if obb_result is not None and obb_result.has_detections:
            annotated_bgr = draw_obb_overlays(
                annotated_bgr,
                obb_result,
                draw_labels=True,
                draw_centroid=True,
                alpha_fill=0.16,
            )

        # Unified Perception Frame Observation
        frame_obs = PerceptionFrameObservation(
            frame_id=frame_idx,
            timestamp_s=timestamp_s,
            aruco_result=detection_result,
            hand_result=hand_result,
            obb_result=obb_result,
        )

        # 9. Asynchronously submit observation to InteractionWorker
        self.interaction_worker.submit_observation(frame_obs)

        # 10. Retrieve freshest non-stale interaction result (<= 250ms latency)
        interaction_res = self.interaction_worker.get_latest_result(max_age_s=0.250)

        # 11. Draw Interaction Overlays (coexists with all perception overlays)
        if interaction_res is not None:
            annotated_bgr = draw_interaction_overlays(
                annotated_bgr,
                interaction_res,
                draw_connecting_lines=True,
                draw_badges=True,
            )

        return annotated_bgr, (frame_obs, interaction_res)

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

        # Prompt 12: Pre-session validation and calibration interlocks
        checker = SessionReadinessChecker(self.config_manager, self.aruco_tracker, self.alert_manager)
        proto = self.procedure_engine.protocol
        can_start, items = checker.perform_check(
            protocol=proto,
            source_type=self._current_source_type,
            source_param=self._current_source_param,
            selected_video_file=self.session_panel.selected_video_file,
            yolo_ready=getattr(self.obb_worker, "is_ready", True),
            yolo_status_msg=self.obb_worker.status.value,
            hand_ready=getattr(self.hand_worker, "is_ready", True),
            hand_status_msg=self.hand_worker.status.value,
        )
        if not can_start:
            dialog = SessionReadinessDialog(can_start=False, items=items, parent=self)
            dialog.exec()
            return

        self._active_session_id = self.session_manager.generate_session_id()
        session_folder = self.sessions_dir / self._active_session_id
        session_folder.mkdir(parents=True, exist_ok=True)

        proto = self.procedure_engine.protocol
        if proto:
            hw_report = self.config_manager.get_hardware_report()
            self.session_manager.create_session(
                protocol=proto,
                camera_source=self._get_current_source_info(),
                hardware_device=hw_report.active_device_selected,
                hardware_backend="OpenCV",
                model_version="smoke_v1",
                session_id=self._active_session_id,
            )
            self.session_manager.start_session(self._active_session_id)

        # Initialize AuditChainLogger for append-only SHA-256 chain
        self.audit_logger = AuditChainLogger(
            session_id=self._active_session_id,
            audit_dir=session_folder,
            db_manager=self.db_manager,
        )
        proto_id = proto.protocol_id if proto else "GENERIC"
        proto_ver = proto.protocol_version if proto else "1.0"
        proto_name = proto.experiment_name if proto else "Experiment"
        self.audit_logger.initialize_genesis(
            protocol_id=proto_id,
            protocol_version=proto_ver,
            experiment_name=proto_name,
            metadata={
                "camera_source": self._get_current_source_info(),
                "device": self.config.inference.device_preference.value,
            },
        )
        self.audit_logger.append_event("SESSION_STARTED", {"session_id": self._active_session_id})

        recorder: Optional[SessionRecorder] = None
        if self.session_panel.is_recording_enabled:
            rec_dir = session_folder / "recordings"
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

        # Start procedure compliance monitoring and voice guidance (Prompt 10)
        eval_res = self.procedure_engine.start()
        self.protocol_panel.update_evaluation_result(eval_res)
        proto_name = self.procedure_engine.protocol.experiment_name if self.procedure_engine.protocol else "Experiment"
        self.alert_manager.on_session_started(proto_name)
        self.alert_manager.process_procedure_result(eval_res, self.procedure_engine.protocol)

        self._capture_worker.start()

    @Slot(QImage, int, float)
    def _on_frame_ready(self, qimg: QImage, frame_idx: int, timestamp_s: float) -> None:
        """Handle incoming frame from worker thread and render to viewport."""
        self.live_monitor.update_frame_qimage(qimg, frame_idx, timestamp_s)
        self.profiler.record_capture_frame(timestamp_s)

    @Slot(object)
    def _on_vision_result_ready(self, result: object) -> None:
        """Handle perception detection result emitted from background worker."""
        aruco_res = None
        if isinstance(result, PerceptionFrameObservation):
            aruco_res = result.aruco_result
        elif isinstance(result, tuple) and len(result) >= 1:
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
        self.profiler.record_subsystem_latency("hand", result.latency_ms)

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

    @Slot(object)
    def _on_obb_result_ready(self, result: object) -> None:
        """Handle YOLOv8-OBB detection result emitted from background OBBTrackingWorker."""
        if not isinstance(result, OBBFrameDetections):
            return

        fps = self.obb_worker.measured_fps
        cnames = list(self.obb_worker.detector.class_names.values())
        self.profiler.record_subsystem_latency("yolo", result.inference_time_ms, provider=result.device_used)

        # 1. Update Live Monitor YOLO overlay badge
        self.live_monitor.set_yolo_status(
            status=self.obb_worker.detector.status,
            detection_count=result.count,
            device=result.device_used,
            latency_ms=result.inference_time_ms,
            fps=fps,
            class_names=cnames,
        )

        # 2. Log significant detection state transitions to Event Panel
        if result.count > 0 and self._last_logged_obb_count == 0:
            det_summary = [f"{d.class_name} ({d.confidence:.0%})" for d in result.detections[:3]]
            self.event_panel.add_event(
                severity="INFO",
                source="YOLO_OBB",
                description=f"Objects detected ({result.count}): {', '.join(det_summary)}",
                hash_digest="yolo_obb:detected",
            )
        elif result.count == 0 and self._last_logged_obb_count > 0:
            self.event_panel.add_event(
                severity="INFO",
                source="YOLO_OBB",
                description="Objects no longer in camera view.",
                hash_digest="yolo_obb:cleared",
            )
        self._last_logged_obb_count = result.count

    @Slot(object)
    def _on_interaction_result_ready(self, result: object) -> None:
        """Handle interaction frame recognition result emitted from background InteractionWorker."""
        if not isinstance(result, InteractionFrameResult):
            return

        # 1. Update LiveMonitor interaction badge
        active_cnt = len(result.active_contacts)
        if active_cnt > 0:
            rel = result.active_contacts[0]
            txt = f"{rel.handedness} -> {rel.object_track_id} [{rel.current_state.value}]"
            color = "#2ea043" if "CONTACT" in rel.current_state.value else "#d29922"
            self.live_monitor.set_interaction_badge(txt, color=color)
            self.event_panel.set_interaction_status(f"{txt} ({result.processing_time_ms:.1f}ms)", active=True)
        elif len(result.active_relationships) > 0:
            rel = result.active_relationships[0]
            txt = f"{rel.handedness} -> {rel.object_track_id} [{rel.current_state.value}]"
            self.live_monitor.set_interaction_badge(txt, color="#2f81f7")
            self.event_panel.set_interaction_status(f"{txt} ({result.processing_time_ms:.1f}ms)", active=False)
        else:
            self.live_monitor.set_interaction_badge("INTERACTION: IDLE", color="#65768d")
            self.event_panel.set_interaction_status(f"IDLE ({len(result.tracked_objects)} tracked objs)", active=False)

        self.profiler.record_subsystem_latency("interaction", result.processing_time_ms)

    @Slot(object)
    def _on_interaction_event_emitted(self, event: object) -> None:
        """Handle discrete temporal interaction event, update compliance state, and append to audit timeline."""
        if not isinstance(event, InteractionEvent):
            return

        severity = "INFO"
        if event.event_type in (InteractionEventType.PICKUP_CANDIDATE, InteractionEventType.PLACEMENT_CANDIDATE):
            severity = "NOTICE"
        elif event.event_type == InteractionEventType.CONTACT_CANDIDATE:
            severity = "INFO"

        self.event_panel.add_event(
            severity=severity,
            source="INTERACTION",
            description=event.summary_line(),
            hash_digest=event.event_id,
        )

        # Prompt 11: Persist interaction event and append to audit chain
        if self._active_session_id:
            self.session_manager.record_interaction_event(self._active_session_id, event)
        if self.audit_logger:
            self.audit_logger.append_event("INTERACTION_EVENT", {
                "event_id": event.event_id,
                "event_type": event.event_type.value,
                "object_class": event.object_class,
                "confidence": round(event.interaction_confidence, 3),
            })

        # Prompt 12: Profiler end-to-end freshness from video frame capture
        self.profiler.record_end_to_end_freshness(event.timestamp_s)

        # Prompt 9: Evaluate event against deterministic ProcedureEngine
        t_proc_0 = time.perf_counter()
        prev_step_id = self.procedure_engine.current_step_id
        eval_res = self.procedure_engine.evaluate_event(event)
        self.profiler.record_subsystem_latency("procedure", (time.perf_counter() - t_proc_0) * 1000.0)
        self.protocol_panel.update_evaluation_result(eval_res)

        # Prompt 11: Sync evaluation results and step progress to SQLite
        if self._active_session_id:
            self.session_manager.sync_evaluation_result(self._active_session_id, eval_res)

        # Step progression notification
        if prev_step_id is not None and eval_res.current_step_id != prev_step_id:
            self.event_panel.add_event(
                severity="NOTICE",
                source="COMPLIANCE",
                description=f"Step Completed: [{prev_step_id}] -> Advancing to [{eval_res.current_step_id or 'FINISH'}]",
                hash_digest=f"step_complete:{prev_step_id}",
            )
            if self.audit_logger:
                self.audit_logger.append_event("STEP_COMPLETED", {
                    "step_id": prev_step_id,
                    "next_step_id": eval_res.current_step_id,
                })

        # Full procedure completion notification
        if eval_res.is_success:
            self.event_panel.add_event(
                severity="NOTICE",
                source="COMPLIANCE",
                description=f"Experiment Procedure Fully Completed: {eval_res.protocol_id}",
                hash_digest=f"proto_complete:{eval_res.protocol_id}",
            )
            if self.audit_logger:
                self.audit_logger.append_event("PROCEDURE_COMPLETED", {
                    "protocol_id": eval_res.protocol_id,
                    "completed_steps": eval_res.completed_steps,
                })

        # Log new compliance violations
        for violation in eval_res.new_violations:
            viol_sev = "CRITICAL" if violation.severity == "critical" else "WARNING"
            self.event_panel.add_event(
                severity=viol_sev,
                source="COMPLIANCE",
                description=f"[{violation.violation_type.value}] {violation.explanation}",
                hash_digest=violation.violation_id,
            )
            # Prompt 11: Persist violation to SQLite and audit chain
            if self._active_session_id:
                self.session_manager.record_violation(self._active_session_id, violation)
            if self.audit_logger:
                self.audit_logger.append_event("COMPLIANCE_VIOLATION", {
                    "violation_id": violation.violation_id,
                    "violation_type": violation.violation_type.value,
                    "step_id": violation.step_id,
                    "severity": violation.severity,
                    "explanation": violation.explanation,
                })

        # Prompt 10: Dispatch spoken guidance and prioritized offline audio alerts
        self.alert_manager.process_procedure_result(eval_res, self.procedure_engine.protocol)

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

        # Prompt 11: Finalize recording in SQLite with SHA-256 and log to audit chain
        if self._active_session_id:
            w = self.config.camera.width
            h = self.config.camera.height
            fps = float(self.config.camera.target_fps)
            rec_row = self.session_manager.finalize_recording(
                session_id=self._active_session_id,
                file_path=file_path,
                total_frames=total_frames,
                duration_s=duration_s,
                fps=fps,
                width=w,
                height=h,
            )
            if self.audit_logger:
                self.audit_logger.append_event("RECORDING_FINALIZED", {
                    "recording_id": rec_row.recording_id,
                    "file_path": rec_row.file_path,
                    "sha256": rec_row.sha256_checksum,
                    "total_frames": total_frames,
                    "duration_s": round(duration_s, 2),
                })

    @Slot()
    def _handle_pause_session(self) -> None:
        if self._capture_worker is None:
            return

        if self._session_state == "ACTIVE":
            self._session_state = "PAUSED"
            self._capture_worker.pause_capture()
            self.session_panel.set_session_state("PAUSED")
            self.top_bar.set_session_status("PAUSED")
            self.protocol_panel.update_evaluation_result(self.procedure_engine.pause())
            if self._active_session_id:
                self.session_manager.pause_session(self._active_session_id)
            if self.audit_logger:
                self.audit_logger.append_event("SESSION_PAUSED", {"session_id": self._active_session_id})
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
            self.protocol_panel.update_evaluation_result(self.procedure_engine.resume())
            if self._active_session_id:
                self.session_manager.resume_session(self._active_session_id)
            if self.audit_logger:
                self.audit_logger.append_event("SESSION_RESUMED", {"session_id": self._active_session_id})
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

        if hasattr(self, "obb_worker") and self.obb_worker is not None:
            self.obb_worker.reset()

        if hasattr(self, "interaction_worker") and self.interaction_worker is not None:
            self.interaction_worker.reset()

        self.procedure_engine.stop()
        final_state = self.procedure_engine.get_state()
        self.protocol_panel.update_evaluation_result(final_state)
        if hasattr(self, "alert_manager") and self.alert_manager is not None:
            self.alert_manager.on_session_aborted()

        # Prompt 11: Finalize session in SQLite, verify hash chain, export certification reports
        if self._active_session_id and self.audit_logger:
            self.audit_logger.append_event("SESSION_STOPPED", {
                "session_id": self._active_session_id,
                "completed": (final_state.status.value == "COMPLETED"),
            })
            is_completed = (final_state.status.value == "COMPLETED")
            self.session_manager.stop_session(
                session_id=self._active_session_id,
                is_completed=is_completed,
                final_digest=self.audit_logger.latest_hash,
            )

            # Replay and verify cryptographic hash chain
            records = self.audit_logger.records
            ver_res = verify_audit_chain(records)
            self.session_manager.update_audit_status(
                session_id=self._active_session_id,
                status=ver_res.status.value,
                final_digest=ver_res.final_hash,
            )

            # Generate certification reports package
            session_rec = self.session_manager.get_session(self._active_session_id)
            if session_rec:
                self.reporter.export_all_reports(
                    session=session_rec,
                    step_results=self.session_manager.get_step_results(self._active_session_id),
                    violations=self.session_manager.get_violations(self._active_session_id),
                    voice_alerts=self.session_manager.get_voice_alerts(self._active_session_id),
                    recordings=self.session_manager.get_recordings(self._active_session_id),
                    verification_result=ver_res,
                )

            self.event_panel.add_event(
                severity="SUCCESS" if ver_res.is_valid else "CRITICAL",
                source="AUDIT",
                description=(
                    f"Session {self._active_session_id} finalized: {ver_res.status.value} "
                    f"({ver_res.total_records} records, digest: {ver_res.final_hash[:16]}...)"
                ),
                hash_digest=ver_res.final_hash,
            )

        self._last_logged_hand_count = 0
        self._last_logged_obb_count = 0
        self._session_state = "IDLE"
        self.session_panel.set_session_state("IDLE")
        self.top_bar.set_session_status("IDLE")
        self.live_monitor.set_recording_status(False)
        self.live_monitor.set_camera_status(False, self._get_current_source_info())
        self.live_monitor.set_hand_tracking_status(HandTrackingStatus.SEARCHING)
        self.live_monitor.set_interaction_badge("INTERACTION: IDLE")
        self.event_panel.set_subsystem_statuses(camera_connected=False)
        self.event_panel.set_interaction_status("IDLE", active=False)

    @Slot()
    def _handle_reset_session(self) -> None:
        self._handle_stop_session()
        self.procedure_engine.reset()
        if hasattr(self, "alert_manager") and self.alert_manager is not None:
            self.alert_manager.reset_announcement_state()
        self.protocol_panel.update_evaluation_result(self.procedure_engine.get_state())
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
            if isinstance(proto, ProtocolDefinition):
                self.procedure_engine.set_protocol(proto)
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
        dialog = SettingsDialog(self.config_manager, self, alert_manager=self.alert_manager)
        if dialog.exec():
            hw_report = self.config_manager.get_hardware_report()
            self.top_bar.set_compute_status(hw_report.active_device_selected, hw_report.device_selection_reason)
            self.event_panel.set_hardware_report(hw_report)

    @Slot()
    def _open_session_history_dialog(self) -> None:
        dialog = SessionHistoryDialog(
            session_manager=self.session_manager,
            reporter=self.reporter,
            parent=self,
            audit_dir=self.sessions_dir,
        )
        dialog.exec()

    @Slot()
    def _open_diagnostics_dialog(self) -> None:
        dialog = DiagnosticsDialog(self.profiler, parent=self)
        dialog.exec()

    # -------------------------------------------------------------------------
    # Offline Voice Alert Listeners (Prompt 10)
    # -------------------------------------------------------------------------

    @Slot(str, str)
    def _on_voice_status_changed(self, status_val: str, status_text: str) -> None:
        self.top_bar.set_voice_status(status_text)

    @Slot(object)
    def _on_voice_alert_queued(self, item: VoiceAlertItem) -> None:
        self.event_panel.add_event(
            severity="INFO",
            source="VOICE",
            description=f"Alert queued: '{item.text}' (priority={item.priority_rank})",
            hash_digest=item.alert_id,
        )
        if self._active_session_id:
            self.session_manager.record_voice_alert(
                session_id=self._active_session_id,
                alert_id=item.alert_id,
                category=item.category,
                priority=item.priority,
                text=item.text,
                reason=item.reason,
                event_id=item.event_id,
                step_id=item.step_id,
                status="QUEUED",
            )
        if self.audit_logger:
            self.audit_logger.append_event("VOICE_ALERT_QUEUED", {
                "alert_id": item.alert_id,
                "category": item.category,
                "text": item.text,
                "priority": item.priority,
            })

    @Slot(object)
    def _on_voice_speech_started(self, item: VoiceAlertItem) -> None:
        self.event_panel.add_event(
            severity="NOTICE",
            source="VOICE",
            description=f"Speaking alert: '{item.text}'",
            hash_digest=item.alert_id,
        )

    @Slot(object)
    def _on_voice_speech_finished(self, item: VoiceAlertItem) -> None:
        self.event_panel.add_event(
            severity="SUCCESS",
            source="VOICE",
            description=f"Alert spoken: '{item.text}'",
            hash_digest=item.alert_id,
        )
        if self._active_session_id:
            self.session_manager.record_voice_alert(
                session_id=self._active_session_id,
                alert_id=item.alert_id,
                category=item.category,
                priority=item.priority,
                text=item.text,
                reason=item.reason,
                status="SPOKEN",
            )
        if self.audit_logger:
            self.audit_logger.append_event("VOICE_ALERT_SPOKEN", {
                "alert_id": item.alert_id,
                "text": item.text,
            })

    @Slot(object, str)
    def _on_voice_speech_error(self, item: VoiceAlertItem, err: str) -> None:
        self.event_panel.add_event(
            severity="CRITICAL",
            source="VOICE",
            description=f"Alert failed: '{item.text}' (error: {err})",
            hash_digest=item.alert_id,
        )

    @Slot(object, str)
    def _on_voice_alert_dropped(self, item: VoiceAlertItem, reason: str) -> None:
        self.event_panel.add_event(
            severity="INFO",
            source="VOICE",
            description=f"Alert dropped: '{item.text}' ({reason})",
            hash_digest=item.alert_id,
        )

    def closeEvent(self, event: QCloseEvent) -> None:
        """Ensure capture worker thread, hand tracking worker, and recording are stopped cleanly."""
        if hasattr(self, "alert_manager") and self.alert_manager is not None:
            logger.info("Stopping alert manager during application shutdown")
            self.alert_manager.stop()

        if self._capture_worker is not None:
            logger.info("Stopping capture worker during application shutdown")
            self._capture_worker.stop_capture()
            self._capture_worker = None

        if hasattr(self, "hand_worker") and self.hand_worker is not None:
            logger.info("Stopping hand tracking worker during application shutdown")
            self.hand_worker.stop()

        if hasattr(self, "obb_worker") and self.obb_worker is not None:
            logger.info("Stopping YOLOv8-OBB detection worker during application shutdown")
            self.obb_worker.stop()

        if hasattr(self, "interaction_worker") and self.interaction_worker is not None:
            logger.info("Stopping interaction worker during application shutdown")
            self.interaction_worker.stop()

        event.accept()
