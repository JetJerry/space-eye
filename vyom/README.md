# VYOM — Offline Human Activity Recognition & Experiment Procedure Compliance Workstation

**Smart India Hackathon (SIH 26174)**  
**Target Environment:** Ground-based microgravity simulation, controlled gloveboxes, and laboratory workstations  
**Operational Status:** Research & Demonstration Prototype (Not Flight Qualified)

---

## 1. System Overview

VYOM is an offline-first Windows desktop monitoring system engineered for real-time human activity recognition, procedure compliance monitoring, and tamper-evident audit logging in controlled experiment racks.

The system combines simultaneous multi-modal perception (YOLOv8-OBB oriented object detection, Google MediaPipe 3D hand tracking, and OpenCV ArUco rig coordinate tracking) with a deterministic, explainable finite-state machine. **No external cloud APIs or black-box large language models make procedure compliance decisions.**

> [!IMPORTANT]
> **Prototype Scope Statement:** VYOM is a technological prototype developed for SIH 26174. It is not DO-178C or NASA NPR 7150.2 flight-certified software and must not be used as the sole authority in life-critical or hazardous flight operations.

---

## 2. Key Architectural Pillars

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        VYOM WORKSTATION PIPELINE                       │
└────────────────────────────────────────────────────────────────────────┘

  [ Video Ingestion ] ──────> 30 FPS BGR Frames (Webcam / Replay / LAN)
           │
           ▼
  [ Multi-Modal Perception ]
   ├── ArUco Tracking ──────> 3D Metric Rig Coordinate Frame (DICT_4X4_50)
   ├── MediaPipe Hands ────> 21 3D Landmarks per Hand (CPU-only, no fake GPU)
   └── YOLOv8n-OBB ─────────> Oriented 4-Corner Bounding Boxes (CUDA / CPU)
           │
           ▼
  [ Spatial-Temporal Engine ]
   ├── Hungarian Tracker ───> Persistent Object Track IDs & 2D Velocities
   ├── Geometric Fusion ────> Finger-to-BBox Proximity & Distance History
   └── Temporal State ──────> Approach -> Contact -> Pickup -> Placement -> Release
           │
           ▼
  [ Deterministic Rule Engine ]
   ├── JSON Protocols ──────> Strict ordering, preconditions, 3D target zones
   └── Inviolable Logic ────> Touch does not satisfy completion; out-of-order alarms
           │
           ▼
  [ Auditing & Offline Alerts ]
   ├── Windows SAPI5 TTS ───> Spoken instructions & prioritized voice alarms
   ├── SQLite Database ─────> Relational tables with WAL & foreign keys
   ├── SHA-256 Hash Chain ──> Cryptographic genesis-to-root audit trail
   └── Multi-Format Reports ─> Certified HTML, JSON, CSV, and detached manifest
```

---

## 3. Implemented Capabilities (Prompts 1 – 12 Complete)

1. **Foundations & Architecture (Prompt 1):** Robust configuration management via Pydantic schemas, unified logging, and project structure.
2. **PySide6 Desktop Workstation (Prompt 2):** High-DPI dark-theme workstation interface with multi-panel layout, live video viewport, procedure timeline, and telemetry.
3. **Capture Pipeline & Replay (Prompt 3):** Threaded video acquisition supporting USB webcams, exocentric MP4/AVI replay (with FPS synchronization), and lossless local MP4 recording.
4. **ArUco Rig Calibration (Prompt 4):** Real-time fiducial marker tracking (`DICT_4X4_50`), metric rig-coordinate transformation, and 3D target zone bounding boxes.
5. **MediaPipe Hand Tracking (Prompt 5):** 21 3D landmarks per hand with palm tracking, handedness classification, and CPU execution.
6. **Dataset Curation & Smoke Training (Prompt 6):** Integration of ENIGMA-360, LabOS, and ChemEq25 datasets; verified 10-class taxonomy; initial YOLOv8n-OBB training.
7. **Real-Time OBB Inference (Prompt 7):** Custom YOLOv8n-OBB loader, four-corner polygon extraction, and automatic CUDA GPU to CPU fallback.
8. **Spatial-Temporal Interaction Reasoning (Prompt 8):** Explainable event recognition (`PICKUP_CANDIDATE`, `OBJECT_MOVEMENT`, `PLACEMENT_CANDIDATE`, `RELEASE_CANDIDATE`) with spatial uncertainty tracking.
9. **Deterministic JSON Procedure Engine (Prompt 9):** Versioned JSON protocol schema, strict ordering, precondition checks, timeout detection, and out-of-order alarms.
10. **Offline Voice Alerts (Prompt 10):** Windows SAPI5 offline text-to-speech engine running on a dedicated worker thread with alert prioritization and cooldown throttling.
11. **Persistent Storage & Cryptographic Audit (Prompt 11):** Relational SQLite database, append-only SHA-256 hash chaining, replay verification tool, and automated multi-format experiment reports (HTML, JSON, CSV, manifest).
12. **Complete System Integration & Windows Packaging (Prompt 12):** Multi-threaded orchestration, CPU/GPU profiling and telemetry dialog, pre-session readiness interlocks, interrupted session auto-recovery, ONNX parity verification, and PyInstaller onedir distribution.

---

## 4. Hardware & Inference Strategy

VYOM supports dynamic hardware detection with universal fallback:

- **NVIDIA CUDA GPU Acceleration:** Supported via PyTorch CUDA (`torch.cuda`) on compatible NVIDIA GPUs (e.g., RTX 3050 Laptop GPU, RTX 3060, RTX 40-series). Inference latency: **~14–25 ms**.
- **Universal CPU Fallback:** If GPU hardware is unavailable or VRAM is exhausted, the engine falls back automatically to CPU execution without crashing. Inference latency: **~140–280 ms**.
- **ONNX Runtime Export:** An exported ONNX model (`vyom_yolov8n_obb_smoke.onnx`) is bundled with verified parity (mean displacement < 0.25 px, confidence delta < 1.5%).
- **Strictly Offline Enforcement:** `os.environ["YOLO_AUTOINSTALL"] = "0"` is enforced to prohibit unauthorized background network requests.

---

## 5. Quick Start (Windows PowerShell)

### Prerequisites
- Windows 10/11 (64-bit)
- Python 3.10 or 3.11 (or use the standalone pre-packaged `.exe`)
- Virtual environment with dependencies installed

### Launching the Desktop Workstation

```powershell
# Set Python path to src directory
$env:PYTHONPATH="src"

# Launch VYOM Desktop Workstation GUI
python main.py
```

### Validating a Procedure Protocol

```powershell
python main.py --validate-protocol config/protocols/sample_experiment.json
```

### Running All Test Suites

```powershell
pytest tests/unit/ -v
```

### Running the End-to-End Acceptance Verification Script

```powershell
python scripts/verify_prompt12_integration.py
```

---

## 6. Project Directory Layout

```text
vyom/
├── main.py                     # Workstation entry point (CLI and GUI launch)
├── pyproject.toml              # PEP 621 package metadata and dependency groups
├── README.md                   # Complete system documentation
├── vyom.spec                   # PyInstaller onedir packaging specification
├── config/
│   ├── app.json                # Core application, device preference, offline guard
│   ├── camera.json             # Frame resolution, capture FPS, recording codec
│   ├── rig.json                # ArUco marker dictionary and rig target zones
│   └── protocols/
│       └── sample_experiment.json  # Synthetic specimen transfer experiment protocol
├── models/
│   ├── vyom_yolov8n_obb_smoke.pt   # Custom-trained YOLOv8n-OBB checkpoint (PyTorch)
│   ├── vyom_yolov8n_obb_smoke.onnx # Exported ONNX model
│   ├── hand_landmarker.task        # MediaPipe 21-landmark hand task bundle
│   ├── smoke_model_metadata.json   # Model provenance and class mapping
│   └── onnx_parity_report.json     # Mathematical parity benchmark report
├── src/
│   └── vyom/
│       ├── core/               # Configuration, profiler, and logging
│       ├── ui/                 # PySide6 GUI (main window, readiness, diagnostics)
│       ├── capture/            # Camera and video replay pipelines
│       ├── vision/             # YOLOv8-OBB, MediaPipe Hands, ArUco tracker
│       ├── interaction/        # Multi-object tracker, spatial-temporal fusion
│       ├── protocol/           # JSON protocol schema, rule engine, compliance engine
│       ├── alerts/             # SAPI5 voice engine and alert manager
│       ├── storage/            # SQLite database manager and session manager
│       └── audit/              # SHA-256 hash chaining, replay verifier, reporter
├── docs/
│   └── DEPLOYMENT_GUIDE.md     # In-depth operator deployment & setup guide
├── scripts/
│   ├── build_windows_dist.py   # Automated PyInstaller onedir packaging
│   ├── export_and_verify_onnx.py # ONNX export and parity validator
│   ├── replay_audit_session.py # Cryptographic hash-chain replayer
│   └── verify_prompt12_integration.py # 14-criterion final acceptance suite
└── tests/
    └── unit/                   # 60+ isolated unit test modules
```

---

## 7. Pre-Session Readiness & Safety Interlocks

Prior to initiating procedure compliance tracking, VYOM executes a mandatory pre-session checklist (`SessionReadinessChecker`):

- **Protocol Definition:** Confirms a valid experiment protocol is loaded. *(Mandatory)*
- **Video Source:** Confirms active webcam or valid exocentric video replay. *(Mandatory)*
- **YOLOv8-OBB Model:** Verifies offline model checkpoint file exists. *(Mandatory)*
- **MediaPipe Hand Model:** Verifies offline `hand_landmarker.task` file exists. *(Mandatory)*
- **ArUco Rig Alignment:** Checks if Origin Marker ID 0 is in tracking lock. *(Advisory / Metric Warning)*
- **Disk Storage Margin:** Verifies > 1.0 GB storage margin for recording and database. *(Mandatory)*
- **Offline Voice Engine:** Verifies Windows SAPI5 audio backend is operational. *(Advisory)*

If any mandatory prerequisite fails, the system blocks session start to prevent unmonitored or ungrounded operation.

---

## 8. Cryptographic Audit Logging & Reports

Every session maintains a tamper-evident audit log:
- **Canonical JSON:** Deterministic key ordering and compact JSON formatting.
- **SHA-256 Hash Chaining:** Each event record incorporates the cryptographic hash of the preceding record (`prev_record_hash`), anchored to a fixed 64-character genesis block.
- **Independent Verification:** The `scripts/replay_audit_session.py` utility recalculates all record hashes from the event log and compares them against the SQLite database.
- **Multi-Format Export:** Every completed session automatically exports:
  - `*_report.html`: Self-contained interactive compliance certificate.
  - `*_summary.json`: Machine-readable protocol results and timings.
  - `*_steps.csv`: Step-by-step tabular event log.
  - `*_manifest.json`: Cryptographic detached manifest containing SHA-256 hashes of all exported files and recordings.

---

## 9. Windows Standalone Distribution Packaging

To generate the standalone Windows onedir package:

```powershell
python scripts/build_windows_dist.py
```

The resulting distribution folder (`dist/VYOM/`) contains:
- `VYOM.exe` — Standalone executable
- Bundled models (`models/`) and configurations (`config/`)
- `checksums.json` — Cryptographic SHA-256 release manifest

---

## 10. License & Declarations

- **Project:** VYOM (SIH 26174)
- **Status:** Development and demonstration prototype
- **Zero-Network Guarantee:** 100% offline; no remote telemetry, cloud APIs, or external data transmission.
