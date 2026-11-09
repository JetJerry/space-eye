"""System performance, latency profiler, and resource diagnostics engine.

Part of VYOM Prompt 12: Performance Profiling, Resource Monitoring, and Runtime Diagnostics.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
from datetime import datetime, timezone
import os
import time
from typing import Any, Deque, Dict, List, Optional
import psutil
import torch

from vyom.core.logging_setup import get_logger

logger = get_logger("core.profiler")


@dataclass
class SubsystemLatencyMetrics:
    """Latency rolling metrics for an individual pipeline stage."""
    name: str
    active_provider: str
    samples: Deque[float] = field(default_factory=lambda: collections.deque(maxlen=30))

    def record(self, latency_ms: float) -> None:
        """Add latency sample in milliseconds."""
        self.samples.append(latency_ms)

    @property
    def current_ms(self) -> float:
        return self.samples[-1] if self.samples else 0.0

    @property
    def mean_ms(self) -> float:
        return float(sum(self.samples) / len(self.samples)) if self.samples else 0.0

    @property
    def min_ms(self) -> float:
        return float(min(self.samples)) if self.samples else 0.0

    @property
    def max_ms(self) -> float:
        return float(max(self.samples)) if self.samples else 0.0


class SystemPerformanceProfiler:
    """Monitors live end-to-end pipeline latencies, frame rates, CPU, RAM, and GPU/VRAM."""

    def __init__(self) -> None:
        self._proc = psutil.Process(os.getpid())

        # Subsystem latency trackers
        self.yolo = SubsystemLatencyMetrics("YOLOv8-OBB", active_provider="CPU")
        self.mediapipe = SubsystemLatencyMetrics("MediaPipe Hands", active_provider="CPU (Windows CPU-only)")
        self.aruco = SubsystemLatencyMetrics("ArUco Tracking", active_provider="CPU (OpenCV)")
        self.interaction = SubsystemLatencyMetrics("Interaction Engine", active_provider="CPU")
        self.procedure = SubsystemLatencyMetrics("Procedure Compliance", active_provider="CPU")

        # Capture FPS tracking
        self._capture_timestamps: Deque[float] = collections.deque(maxlen=30)
        self._measured_capture_fps: float = 0.0

        # End-to-end freshness
        self._e2e_latencies: Deque[float] = collections.deque(maxlen=30)

        # Total frames processed
        self._total_frames_profiled: int = 0
        self._start_time: float = time.monotonic()

    def record_capture_frame(self, timestamp_s: float) -> None:
        """Record frame arrival time to compute moving capture FPS."""
        now = time.monotonic()
        self._capture_timestamps.append(now)
        self._total_frames_profiled += 1
        if len(self._capture_timestamps) >= 2:
            dt = self._capture_timestamps[-1] - self._capture_timestamps[0]
            if dt > 0.001:
                self._measured_capture_fps = (len(self._capture_timestamps) - 1) / dt

    def record_subsystem_latency(
        self,
        subsystem: str,
        latency_ms: float,
        provider: Optional[str] = None,
    ) -> None:
        """Record execution latency for a named pipeline stage."""
        sub = subsystem.lower()
        if "yolo" in sub or "obb" in sub:
            self.yolo.record(latency_ms)
            if provider:
                self.yolo.active_provider = provider
        elif "hand" in sub or "mediapipe" in sub:
            self.mediapipe.record(latency_ms)
        elif "aruco" in sub or "rig" in sub:
            self.aruco.record(latency_ms)
        elif "interaction" in sub:
            self.interaction.record(latency_ms)
        elif "procedure" in sub or "compliance" in sub:
            self.procedure.record(latency_ms)

    def record_end_to_end_freshness(self, capture_timestamp_s: float) -> None:
        """Record end-to-end observation lag from camera capture to compliance output."""
        now = time.time()
        lag_ms = max(0.0, (now - capture_timestamp_s) * 1000.0)
        self._e2e_latencies.append(lag_ms)

    def get_system_resource_snapshot(self) -> Dict[str, Any]:
        """Capture live CPU, RAM, and NVIDIA GPU/VRAM statistics."""
        # CPU & Host RAM
        cpu_pct = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        proc_mem = self._proc.memory_info().rss / (1024 * 1024)

        # GPU metrics if CUDA is available
        gpu_info: Dict[str, Any] = {
            "cuda_available": False,
            "gpu_name": None,
            "vram_allocated_mb": 0.0,
            "vram_reserved_mb": 0.0,
            "vram_total_mb": 0.0,
        }
        try:
            if torch.cuda.is_available():
                gpu_info["cuda_available"] = True
                gpu_info["gpu_name"] = torch.cuda.get_device_name(0)
                gpu_info["vram_allocated_mb"] = round(torch.cuda.memory_allocated(0) / (1024 * 1024), 1)
                gpu_info["vram_reserved_mb"] = round(torch.cuda.memory_reserved(0) / (1024 * 1024), 1)
                dev_props = torch.cuda.get_device_properties(0)
                gpu_info["vram_total_mb"] = round(dev_props.total_memory / (1024 * 1024), 1)
        except Exception as e:
            logger.debug("Failed querying GPU telemetry: %s", e)

        return {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "cpu_percent": round(cpu_pct, 1),
            "ram_used_mb": round((mem.total - mem.available) / (1024 * 1024), 1),
            "ram_total_mb": round(mem.total / (1024 * 1024), 1),
            "ram_percent": round(mem.percent, 1),
            "process_rss_mb": round(proc_mem, 1),
            "gpu": gpu_info,
        }

    def get_performance_summary(self) -> Dict[str, Any]:
        """Generate structured diagnostic dictionary covering all subsystems."""
        sys_res = self.get_system_resource_snapshot()
        mean_e2e = float(sum(self._e2e_latencies) / len(self._e2e_latencies)) if self._e2e_latencies else 0.0

        return {
            "profiler_elapsed_s": round(time.monotonic() - self._start_time, 1),
            "total_frames_profiled": self._total_frames_profiled,
            "camera_capture_fps": round(self._measured_capture_fps, 1),
            "end_to_end_freshness_ms": round(mean_e2e, 1),
            "subsystems": {
                "yolo_obb": {
                    "mean_latency_ms": round(self.yolo.mean_ms, 1),
                    "active_provider": self.yolo.active_provider,
                },
                "mediapipe_hands": {
                    "mean_latency_ms": round(self.mediapipe.mean_ms, 1),
                    "active_provider": self.mediapipe.active_provider,
                },
                "aruco_tracking": {
                    "mean_latency_ms": round(self.aruco.mean_ms, 1),
                    "active_provider": self.aruco.active_provider,
                },
                "interaction_engine": {
                    "mean_latency_ms": round(self.interaction.mean_ms, 1),
                    "active_provider": self.interaction.active_provider,
                },
                "procedure_engine": {
                    "mean_latency_ms": round(self.procedure.mean_ms, 1),
                    "active_provider": self.procedure.active_provider,
                },
            },
            "resources": sys_res,
        }

    def format_diagnostics_report(self) -> str:
        """Render multi-line formatted diagnostics text for logging and UI display."""
        summary = self.get_performance_summary()
        res = summary["resources"]
        gpu = res["gpu"]
        subs = summary["subsystems"]

        gpu_line = (
            f"GPU: {gpu['gpu_name']} | VRAM: {gpu['vram_allocated_mb']:.0f}MB / {gpu['vram_total_mb']:.0f}MB"
            if gpu["cuda_available"]
            else "GPU: None / Disabled (CPU Execution Only)"
        )

        lines = [
            "=== VYOM RUNTIME PERFORMANCE & SUBSYSTEM DIAGNOSTICS ===",
            f"Elapsed: {summary['profiler_elapsed_s']}s | Total Frames: {summary['total_frames_profiled']}",
            f"Camera Capture:  {summary['camera_capture_fps']} FPS",
            f"End-to-End Lag:  {summary['end_to_end_freshness_ms']:.1f} ms",
            f"System CPU:      {res['cpu_percent']}% | RAM: {res['ram_used_mb']:.0f}MB / {res['ram_total_mb']:.0f}MB ({res['ram_percent']}%)",
            f"Process Memory:  {res['process_rss_mb']:.0f} MB",
            gpu_line,
            "",
            "--- Subsystem Execution Latencies ---",
            f"  YOLOv8-OBB:    {subs['yolo_obb']['mean_latency_ms']:.1f} ms [{subs['yolo_obb']['active_provider']}]",
            f"  MediaPipe:     {subs['mediapipe_hands']['mean_latency_ms']:.1f} ms [{subs['mediapipe_hands']['active_provider']}]",
            f"  ArUco Rig:     {subs['aruco_tracking']['mean_latency_ms']:.1f} ms [{subs['aruco_tracking']['active_provider']}]",
            f"  Interactions:  {subs['interaction_engine']['mean_latency_ms']:.1f} ms [{subs['interaction_engine']['active_provider']}]",
            f"  Compliance:    {subs['procedure_engine']['mean_latency_ms']:.1f} ms [{subs['procedure_engine']['active_provider']}]",
            "=========================================================",
        ]
        return "\n".join(lines)


# Global singleton profiler instance
_profiler_instance: Optional[SystemPerformanceProfiler] = None


def get_performance_profiler() -> SystemPerformanceProfiler:
    """Access the shared performance profiler singleton."""
    global _profiler_instance
    if _profiler_instance is None:
        _profiler_instance = SystemPerformanceProfiler()
    return _profiler_instance

