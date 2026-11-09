# VYOM Deployment & Operations Guide

**System:** VYOM — Offline Human Activity Recognition & Experiment Procedure Compliance Workstation  
**Problem Statement:** Smart India Hackathon (SIH 26174)  
**Target Environment:** Ground-based microgravity simulation, controlled gloveboxes, and laboratory workstations  
**Operational Status:** Research & Demonstration Prototype (Not Flight Qualified)

---

## 1. System Overview & Architecture

VYOM is an offline-first Windows desktop application that combines real-time computer vision, geometric reasoning, and deterministic finite-state validation to monitor human procedures and verify step compliance without relying on external cloud APIs or black-box large language models.

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        VYOM WORKSTATION PIPELINE                       │
└────────────────────────────────────────────────────────────────────────┘

  [ Video Input ]
   ├── USB HD Webcam (1080p/720p @ 30 FPS)
   ├── Local Video Replay (MP4 / AVI Container)
   └── LAN RTSP / IP Camera Stream
          │
          ▼
  [ Multi-Modal Perception Layer ] (Concurrent off-thread workers)
   ├── OpenCV ArUco Tracking (DICT_4X4_50 fiducials -> 3D Metric Rig Frame)
   ├── MediaPipe Hand Landmarker (21 3D landmarks per hand, CPU-only)
   └── YOLOv8n-OBB Detector (Oriented 4-corner bounding boxes, GPU CUDA / CPU)
          │
          ▼
  [ Spatial-Temporal Interaction Engine ]
   ├── Hungarian Multi-Object Tracker (Persistent tracking IDs)
   ├── Finger-to-BBox Metric Proximity & Directional Velocity
   └── Temporal Evidence Accumulation (Approach -> Contact -> Pickup -> Placement -> Release)
          │
          ▼
  [ Deterministic Procedure Compliance Engine ]
   ├── JSON Protocol Schema (Strict ordering, preconditions, target zones)
   ├── Rule Engine (Inviolable constraints: touch does not satisfy completion)
   └── Compliance State (COMPLETED, OUT_OF_ORDER, TIMEOUT, UNCERTAIN)
          │
          ▼
  [ Notification & Auditing Layer ]
   ├── Offline SAPI5 Voice Engine (Non-blocking spoken instructions & alarms)
   ├── SQLite Relational Database (WAL mode, foreign key relational integrity)
   ├── SHA-256 Append-Only Hash Chain (Cryptographic genesis-to-root audit trail)
   └── Multi-Format Certification Package (HTML, JSON, CSV, detached manifest)
```

---

## 2. Hardware & Operating System Prerequisites

### Supported Platforms
- **Operating System:** Microsoft Windows 10 (64-bit) or Windows 11 (64-bit).
- **Architecture:** x86_64.

### Minimum Hardware Specifications (CPU Mode)
- **CPU:** Intel Core i5 (8th Gen or newer) / AMD Ryzen 5 (3000 series or newer) &mdash; 4 cores / 8 threads minimum.
- **RAM:** 8 GB system memory.
- **Storage:** 5 GB free disk space (NTFS filesystem recommended for SQLite WAL).
- **Camera:** 720p USB 2.0 webcam (30 FPS capable).

### Recommended Hardware Specifications (GPU Acceleration)
- **CPU:** Intel Core i7 / AMD Ryzen 7 (6+ physical cores).
- **RAM:** 16 GB system memory.
- **GPU:** Dedicated NVIDIA GeForce RTX 3050 (or higher) with at least 4 GB GDDR6 VRAM.
- **Drivers:** NVIDIA Studio or Game Ready Driver version 525.60+ with CUDA 12 support.
- **Camera:** 1080p USB 3.0 wide-angle webcam or overhead rig-mounted industrial camera.

---

## 3. Zero-Network Offline Operation & Security

VYOM is engineered to operate in air-gapped, zero-network environments:
- **No Cloud Telemetry:** No outbound network sockets, telemetry tracking, or analytics pings.
- **No Remote Dependencies:** All neural network weights, configuration files, and speech synthesis run locally.
- **Ultralytics Offline Enforcement:** Environment variable `YOLO_AUTOINSTALL=0` is strictly enforced to prevent background download attempts.
- **Offline TTS:** Windows SAPI5 driver utilizes pre-installed system voices (`Microsoft David Desktop`, `Microsoft Zira Desktop`) without external cloud TTS engines.

---

## 4. Software Setup & Installation

### Option A: Running from Packaged Distribution (No Python Required)
1. Extract the release archive `VYOM-Windows-x64-v0.1.0.zip` to a local folder (e.g., `C:\VYOM` or `D:\space_eye\vyom\dist\VYOM`).
2. Verify package integrity against `checksums.json`:
   ```powershell
   Get-FileHash -Algorithm SHA256 .\VYOM.exe
   ```
3. Run `VYOM.exe`. The workstation GUI will initialize.

### Option B: Running from Source / Virtual Environment
1. Ensure Python 3.10 or 3.11 is installed.
2. Clone the repository and navigate to `vyom`:
   ```powershell
   cd d:\space_eye\vyom
   ```
3. Create and activate a virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
4. Install dependencies:
   ```powershell
   pip install -e ".[desktop,vision,dev]"
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```
5. Launch the application:
   ```powershell
   $env:PYTHONPATH="src"
   python main.py
   ```

---

## 5. Model Assets & Weights Placement

Place the required offline model files in the `models/` directory:

| Filename | Purpose | Size | Format |
| :--- | :--- | :--- | :--- |
| `vyom_yolov8n_obb_smoke.pt` | Custom-trained YOLOv8n-OBB checkpoint (PyTorch) | ~6.4 MB | PyTorch (`.pt`) |
| `vyom_yolov8n_obb_smoke.onnx` | Exported ONNX model for CPU inference | ~12.5 MB | ONNX (`.onnx`) |
| `hand_landmarker.task` | Google MediaPipe 21 3D hand landmarker | ~7.8 MB | TFLite Task (`.task`) |
| `smoke_model_metadata.json` | Dataset provenance, class mapping, and metrics | ~2.5 KB | JSON |
| `onnx_parity_report.json` | Mathematical parity report between PT and ONNX | ~2.4 KB | JSON |

---

## 6. ArUco Marker Setup & Rig Coordinate Frame

VYOM establishes a rig-centric 3D metric coordinate system using OpenCV ArUco markers:
- **Marker Dictionary:** `cv2.aruco.DICT_4X4_50`
- **Origin Marker ID:** `Marker ID 0` (placed at rig zero: `[0.0, 0.0, 0.0]` meters).
- **Physical Size:** 80.0 mm side length (configurable in `config/rig.json`).
- **Target Zones:** Defined in metric bounding boxes (e.g., `rack_zone_a`: `[-0.15, -0.15, 0.0]` to `[0.15, 0.15, 0.25]` meters).

### Printing and Mounting Fiducials
1. Print ArUco marker ID 0 at exact 1:1 scale (80 mm x 80 mm).
2. Adhere the marker to a rigid, non-reflective surface at the base of your experiment glovebox or workstation.
3. Ensure the camera has an unobstructed line of sight with adequate, non-glaring lighting (> 300 lux).

---

## 7. Pre-Session Readiness Checklist

When an operator clicks **"Start Session"**, VYOM executes a mandatory safety checklist (`SessionReadinessChecker`):

```text
┌──────────────────────────────────────────────────────────────┐
│                PRE-SESSION READINESS CHECKLIST               │
├──────────────────────────────────────────────────────────────┤
│  [✓] Protocol Definition:       sample-specimen-transfer     │
│  [✓] Video Acquisition Source:  Ready (USB Camera / Replay)  │
│  [✓] YOLOv8-OBB Detector:       vyom_yolov8n_obb_smoke.pt    │
│  [✓] MediaPipe Hand Landmarker: hand_landmarker.task (CPU)   │
│  [✓] ArUco Rig Frame:           Marker ID 0 tracking         │
│  [✓] Hardware Backend:          NVIDIA RTX 3050 (CUDA)       │
│  [✓] Disk Storage Margin:       > 1.0 GB free on drive D:\   │
│  [✓] SAPI5 Voice Alert Engine:  Microsoft David Desktop      │
└──────────────────────────────────────────────────────────────┘
```

> [!IMPORTANT]
> **Safety Interlock:** If any mandatory prerequisite (Protocol or Video Source) is missing, the application **strictly blocks session startup**. If ArUco calibration is uncalibrated, the operator receives an explicit warning that 3D metric zone validation will remain uncalibrated.

---

## 8. Operator Workflow

1. **Select Protocol:** In the top bar, select a validated JSON protocol from `config/protocols/` (e.g., `sample_experiment.json`).
2. **Select Video Source:** Choose either a connected webcam or a recorded exocentric video file.
3. **Verify Video Feed:** Verify that the glovebox and hands are visible in the central live viewport.
4. **Click "Start Session":** Complete the readiness checklist.
5. **Perform Procedure:**
   - Follow instructions announced via voice and shown in the procedure panel.
   - Live visual cues: Oriented bounding boxes (cyan), Hand landmarks (green), Active interaction vectors (yellow).
   - If a step is performed out of order or with the wrong apparatus, an immediate voice warning is dispatched.
6. **Complete Session:**
   - When the final step is verified, VYOM automatically marks the session as `COMPLETED`.
   - The SHA-256 hash chain is sealed and certified.
   - Comprehensive reports (HTML, JSON, CSV, Manifest) are generated in `sessions/<session_id>/reports/`.

---

## 9. Cryptographic Audit Verification

To independently verify the tamper-evident audit log of any past session:

```powershell
$env:PYTHONPATH="src"
python scripts/replay_audit_session.py --session-dir sessions/SES_20261109_...
```

The verifier will:
1. Parse the canonical JSONL event log.
2. Re-compute every SHA-256 record hash from the genesis block (`0` * 64) to the final root.
3. Verify that the SQLite database records match the hash chain verbatim.
4. Confirm detached manifest cryptographic integrity.

---

## 10. Operational Boundaries & Safety Declarations

- **Research and Demonstration Prototype:** VYOM is developed as an academic and technological prototype under SIH 26174.
- **No Spaceflight Certification:** This software has not undergone DO-178C or NASA NPR 7150.2 flight-software qualification.
- **Advisory Role Only:** All alerts and procedural progression decisions are advisory; human operators retain operational authority.
- **Lighting & Occlusion:** Extreme lighting conditions (glare, deep shadows) or severe hand-object occlusions may reduce detection confidence. In such cases, the engine emits `UNCERTAIN_OBSERVATION` warnings rather than assuming successful step completion.
