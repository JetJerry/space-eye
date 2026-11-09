"""VYOM Application runtime coordinator and desktop launcher."""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any

from vyom import __version__
from vyom.core.config import ConfigManager, get_config_manager
from vyom.core.logging_setup import get_logger, setup_logging

logger = get_logger("app")


class VyomApplication:
    """Manages application lifecycle, configuration, hardware inspection, and desktop launch."""

    def __init__(self, project_root: Path | None = None) -> None:
        self.config_manager: ConfigManager = get_config_manager(project_root=project_root)
        self.config = self.config_manager.config

        # Initialize structured logging
        setup_logging(
            log_level=self.config.logging.log_level,
            log_dir=self.config.logging.log_dir,
            log_file=self.config.logging.log_file,
            console_output=self.config.logging.console_output,
            json_format=self.config.logging.json_format,
            max_bytes=self.config.logging.max_bytes,
            backup_count=self.config.logging.backup_count,
        )

    def get_system_status(self) -> Dict[str, Any]:
        """Perform comprehensive system readiness check distinguishing GPU detection from execution providers."""
        hw_report = self.config_manager.get_hardware_report()
        enigma_info = self.config_manager.validate_enigma360_dataset()

        # Check default protocol
        protocol_status = "NOT_FOUND"
        protocol_steps = 0
        try:
            protocol = self.config_manager.load_protocol(self.config.default_protocol)
            protocol_status = f"VALID ({protocol.protocol_id} v{protocol.protocol_version})"
            protocol_steps = len(protocol.steps)
        except Exception as e:
            protocol_status = f"ERROR ({e})"

        # Check model files status
        model_status = {
            "object_detector": "NOT_CONFIGURED" if not (self.config.models.object_detector_path and self.config.models.object_detector_path.exists()) else "FOUND",
            "yolo_pt": "NOT_CONFIGURED" if not (self.config.models.yolo_pt_path and self.config.models.yolo_pt_path.exists()) else "FOUND",
            "hand_landmarker": "NOT_CONFIGURED" if not (self.config.models.hand_landmarker_path and self.config.models.hand_landmarker_path.exists()) else "FOUND",
        }

        return {
            "app_name": self.config.app_name,
            "version": self.config.version,
            "environment": self.config.environment,
            "offline_mode": self.config.offline_mode,
            "hardware_report": {
                "physical_gpu_detected": hw_report.gpu_hardware_detected,
                "gpu_name": hw_report.gpu_name,
                "gpu_vram_mb": hw_report.gpu_vram_mb,
                "inference_runtime_installed": hw_report.inference_runtime_installed,
                "gpu_execution_provider_available": hw_report.gpu_execution_provider_available,
                "gpu_execution_provider_name": hw_report.gpu_execution_provider_name,
                "active_device_selected": hw_report.active_device_selected,
                "device_selection_reason": hw_report.device_selection_reason,
            },
            "dataset": enigma_info,
            "models": model_status,
            "protocol": {
                "file": self.config.default_protocol,
                "status": protocol_status,
                "step_count": protocol_steps,
            },
            "rig": {
                "dictionary": self.config.rig.dictionary_name,
                "marker_size_m": self.config.rig.marker_size_m,
                "marker_ids": self.config.rig.marker_ids,
                "reference_frame": self.config.rig.reference_frame,
            },
            "camera": {
                "source_type": self.config.camera.source_type,
                "device_index": self.config.camera.device_index,
                "resolution": f"{self.config.camera.width}x{self.config.camera.height}",
                "target_fps": self.config.camera.target_fps,
            },
        }

    def print_diagnostic_banner(self) -> None:
        """Print structured startup information and 4-tier hardware diagnostics."""
        status = self.get_system_status()
        hw = status["hardware_report"]

        print("=" * 78)
        print(f" VYOM — Offline Human Activity Recognition & Compliance Monitor (v{status['version']})")
        print("=" * 78)
        print(f" • Mode: {'OFFLINE' if status['offline_mode'] else 'ONLINE'} | Environment: {status['environment']}")
        print(f" • Project Root: {self.config_manager.project_root}")
        print("-" * 78)
        print(" [COMPUTE HARDWARE & INFERENCE RUNTIME]")
        gpu_label = f"{hw['gpu_name']} ({hw['gpu_vram_mb']} MiB)" if hw['physical_gpu_detected'] else "Not detected"
        print(f"   [1] Physical GPU Detected:   {gpu_label}")
        print(f"   [2] Inference Runtime:       {hw['inference_runtime_installed']}")
        prov_label = hw['gpu_execution_provider_name'] if hw['gpu_execution_provider_available'] else "Not available (CPU only)"
        print(f"   [3] GPU Execution Provider:  {prov_label}")
        print(f"   [4] Active Device Selected:  {hw['active_device_selected']}")
        print(f"       Reason: {hw['device_selection_reason']}")
        print("-" * 78)
        print(" [DATASET CONFIGURATION]")
        ds = status["dataset"]
        print(f"   ENIGMA-360 Path:  {ds['path']}")
        print(f"   Status:           {ds['status']} (Read-only: {ds['read_only']})")
        print(f"   Annotations:      {'Present' if ds['annotations_found'] else 'Not yet unpacked/downloaded'}")
        print(f"   Video Directory:  {'Present' if ds['video_dir_found'] else 'Not yet unpacked/downloaded'}")
        print(f"   Selected View:    {ds['subset']} (exocentric fixed camera)")
        print("-" * 78)
        print(" [MODEL ARTIFACTS]")
        for model_name, state in status["models"].items():
            print(f"   {model_name:<18}: {state}")
        print("-" * 78)
        print(" [PROTOCOL & RIG]")
        print(f"   Default Protocol: {status['protocol']['file']} -> {status['protocol']['status']}")
        print(f"   Rig Reference:    {status['rig']['dictionary']} (Markers: {status['rig']['marker_ids']})")
        print("=" * 78)

    def run_desktop_app(self) -> int:
        """Launch the PySide6 desktop GUI workstation."""
        from PySide6.QtWidgets import QApplication
        from vyom.ui.main_window import VyomMainWindow

        logger.info("Starting VYOM PySide6 desktop application")

        qt_app = QApplication.instance()
        if qt_app is None:
            qt_app = QApplication(sys.argv)

        qt_app.setApplicationName("VYOM")
        qt_app.setApplicationDisplayName("VYOM — Experiment Monitoring Workstation")
        qt_app.setApplicationVersion(__version__)

        main_window = VyomMainWindow(self.config_manager)
        main_window.show()

        return qt_app.exec()


def main() -> int:
    """Entry point supporting desktop launch and diagnostic commands."""
    parser = argparse.ArgumentParser(
        description="VYOM: Offline Experiment Procedure Compliance & Activity Monitoring"
    )
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Run in headless CLI mode instead of launching the desktop GUI",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print system configuration status and hardware diagnostics",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=None,
        help="Custom configuration directory",
    )
    parser.add_argument(
        "--validate-protocol",
        type=str,
        default=None,
        help="Path to an experiment protocol JSON to validate against schema",
    )
    args = parser.parse_args()

    app = VyomApplication(project_root=args.config_dir)

    if args.validate_protocol:
        try:
            proto = app.config_manager.load_protocol(args.validate_protocol)
            print(f"SUCCESS: Protocol '{proto.experiment_name}' (ID: {proto.protocol_id}) is valid.")
            print(f"Found {len(proto.steps)} defined procedural steps.")
            return 0
        except Exception as e:
            print(f"ERROR: Protocol validation failed: {e}", file=sys.stderr)
            return 1

    if args.status or args.cli:
        app.print_diagnostic_banner()
        return 0

    # Default action: launch desktop application
    try:
        return app.run_desktop_app()
    except Exception as e:
        logger.error("GUI launch failed: %s. Falling back to diagnostic status.", e)
        print(f"GUI launch failed ({e}). Outputting system diagnostics:", file=sys.stderr)
        app.print_diagnostic_banner()
        return 1


if __name__ == "__main__":
    sys.exit(main())
