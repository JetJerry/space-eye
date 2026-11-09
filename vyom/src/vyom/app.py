"""VYOM Application runtime coordinator and foundation manager."""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any

from vyom import __version__
from vyom.core.config import ConfigManager, get_config_manager
from vyom.core.logging_setup import get_logger, setup_logging

logger = get_logger("app")


class VyomApplication:
    """Manages application lifecycle, configuration, hardware inspection, and subsystem coordination."""

    def __init__(self, project_root: Path | None = None) -> None:
        self.config_manager: ConfigManager = get_config_manager(project_root=project_root)
        self.config = self.config_manager.config

        # Initialize logging according to configuration
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
        """Perform comprehensive system readiness check for the VYOM foundation."""
        active_device, provider_desc = self.config_manager.resolve_inference_device()
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
            "inference": {
                "preference": self.config.inference.device_preference.value,
                "active_device": active_device,
                "provider": provider_desc,
                "gpu_backend": self.config.inference.gpu_backend.value,
                "cpu_fallback_allowed": self.config.inference.allow_cpu_fallback,
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
        """Print structured startup information and component status."""
        status = self.get_system_status()

        print("=" * 78)
        print(f" VYOM — Offline Human Activity Recognition & Compliance Monitor (v{status['version']})")
        print("=" * 78)
        print(f" • Mode: {'OFFLINE' if status['offline_mode'] else 'ONLINE'} | Environment: {status['environment']}")
        print(f" • Project Root: {self.config_manager.project_root}")
        print("-" * 78)
        print(" [COMPUTE & HARDWARE ACCELERATION]")
        print(f"   Preference:       {status['inference']['preference']}")
        print(f"   Active Device:    {status['inference']['active_device']}")
        print(f"   Provider/Status:  {status['inference']['provider']}")
        print(f"   CPU Fallback:     {'Enabled' if status['inference']['cpu_fallback_allowed'] else 'Disabled'}")
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
        print(" Repository Foundation Status: READY (Prompt 1 Complete)")
        print(" Next Phase: Desktop GUI Shell (Prompt 2)")
        print("=" * 78)


def main() -> int:
    """CLI entry point for VYOM."""
    parser = argparse.ArgumentParser(
        description="VYOM: Offline Experiment Procedure Compliance & Activity Monitoring"
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

    app.print_diagnostic_banner()
    return 0


if __name__ == "__main__":
    sys.exit(main())

