# VYOM — Offline Human Activity Recognition & Experiment Procedure Compliance

VYOM is an offline-first Windows desktop monitoring system designed for camera-based procedure compliance and human activity recognition in controlled laboratory and space payload environments (SIH 26174).

> **Important Scope Statement:** VYOM is a research and demonstration prototype. It is not a flight-certified safety system and must not be used as the sole authority for safety-critical operations.

---

## Key Capabilities & Architecture

1. **Video Ingestion:** Live webcam, local video file replay, or LAN network camera input with monotonic frame timestamps.
2. **Oriented Object Detection:** YOLOv8-OBB model detecting experiment tools and specimen modules with rotated bounding boxes.
3. **Hand Tracking:** Google MediaPipe Tasks Hand Landmarker extracting 21 3D landmarks per detected hand.
4. **Rig Reference Frame:** OpenCV ArUco fiducial marker tracking establishing a rig-relative coordinate system independent of gravity.
5. **Temporal Interaction Reasoning:** Fusion layer converting raw detections and landmarks into discrete, timestamped interaction events (`OBJECT_PICKED_UP`, `OBJECT_PLACED_IN_TARGET`).
6. **Deterministic Rule Engine:** JSON-driven finite-state machine validating procedures step-by-step. **No cloud APIs or LLMs make compliance decisions.**
7. **Offline Local Voice Alerts:** SAPI5 / `pyttsx3` text-to-speech engine issuing spoken guidance for skipped, out-of-order, or timed-out steps.
8. **Tamper-Evident Audit Logging:** Cryptographic append-only JSONL log with SHA-256 hash chaining and SQLite session persistence.
9. **Universal Compute Support:** Universal CPU fallback with optional NVIDIA CUDA GPU acceleration.

---

## Repository Structure

```text
vyom/
├── main.py                     # Application foundation entry point
├── pyproject.toml              # PEP 621 package metadata and dependency groups
├── README.md                   # Comprehensive project documentation
├── .gitignore                  # Git exclusions for models, datasets, caches, sessions
├── .env.example                # Non-secret local configuration overrides
├── config/
│   ├── app.json                # Core app, inference preferences, and dataset paths
│   ├── camera.json             # Video source resolution, FPS, and recording settings
│   ├── rig.json                # ArUco marker dictionary, physical dimensions, rig layout
│   └── protocols/
│       └── sample_experiment.json  # Synthetic sample experiment procedure
├── models/
│   └── README.md               # Asset placement instructions (weights are not committed)
├── data/
│   ├── external/               # Downloaded public proxy datasets (ignored by Git)
│   ├── manifests/              # Source URLs, licenses, and checksums (dataset_sources.yaml)
│   ├── raw/                    # Original uncompressed video/frames (ignored by Git)
│   ├── annotations/            # Original raw annotations (ignored by Git)
│   ├── yolo_obb/               # Normalized YOLOv8-OBB train/val/test splits
│   ├── README.md               # Data layout and external dataset documentation
│   └── SOURCES.md              # Public dataset provenance and licensing catalog
├── runs/                       # Training logs and YOLO checkpoints (ignored by Git)
├── sessions/                   # SQLite database, recordings, audit logs (ignored by Git)
├── logs/                       # Application logs (vyom.log with rotation)
├── src/
│   └── vyom/
│       ├── __init__.py         # Package version and metadata
│       ├── app.py              # Application lifecycle and foundation coordinator
│       ├── core/               # Configuration manager, Pydantic schemas, logging, exceptions
│       ├── ui/                 # PySide6 Qt Widgets desktop interface (Phase 2)
│       ├── capture/            # VideoCapture and recording workers (Phase 3)
│       ├── vision/             # YOLOv8-OBB, MediaPipe Hands, ArUco trackers (Phases 4-7)
│       ├── interaction/        # Spatial-temporal interaction event extractor (Phase 8)
│       ├── protocol/           # Deterministic finite-state machine & rule engine (Phase 9)
│       ├── alerts/             # Local offline speech and alert queue (Phase 10)
│       ├── audit/              # SHA-256 hash-chaining and report generator (Phase 11)
│       └── storage/            # SQLite session repository (Phase 11)
├── scripts/
│   └── README.md               # Utility scripts for dataset inspection and training
└── tests/
    ├── unit/                   # Fast isolated unit tests
    ├── integration/            # Multi-component integration tests
    ├── fixtures/               # Test fixtures and schemas
    └── test_data/              # Minimal synthetic fixtures (no fake training data)
```

---

## Dependency Organization

Dependencies in `pyproject.toml` are strictly separated into functional groups:

- **Core Runtime (`dependencies`):**
  Minimal lightweight dependencies required for configuration parsing, schema validation, and logging (`pydantic`, `pyyaml`, `python-dateutil`).
- **Desktop (`[project.optional-dependencies] desktop`):**
  PySide6 (Qt Widgets) desktop GUI and local text-to-speech engine (`pyttsx3`).
- **Computer Vision (`[project.optional-dependencies] vision`):**
  `opencv-contrib-python` (with `cv2.aruco`), `mediapipe` (Hand Landmarker), `onnxruntime` (CPU execution provider), and `numpy`.
- **GPU Acceleration (`[project.optional-dependencies] gpu`):**
  `onnxruntime-gpu` for NVIDIA CUDA acceleration on supported hardware.
- **Model Training (`[project.optional-dependencies] training`):**
  `torch`, `torchvision`, and `ultralytics`. **Kept completely separate from the final desktop application runtime.**
- **Development (`[project.optional-dependencies] dev`):**
  Testing and linting tools (`pytest`, `pytest-cov`, `pytest-qt`, `black`, `ruff`, `mypy`).

---

## Dataset Configuration

### External ENIGMA-360 Dataset

VYOM uses public exocentric procedural data to train and evaluate procedural activity recognition. The ENIGMA-360 repository is configured as an external directory:

```text
D:\space_eye\datasets\ENIGMA-360
```

- **Selected Subset:** Exocentric (`exo`) training and validation splits. Fixed-payload camera perspective directly simulates space experiment rack monitoring.
- **Data Protection:** This external path is configured as **strictly read-only** (`enigma360_read_only: true`). Files in this directory are never moved, deleted, or modified.

To configure an alternate path, set the environment variable:
```powershell
$env:VYOM_ENIGMA360_PATH = "D:\space_eye\datasets\ENIGMA-360"
```

---

## Windows Setup Guide (PowerShell)

### Prerequisites

- Windows 10/11 (64-bit)
- Python 3.11 (64-bit) installed (or managed via `uv`)
- Optional: NVIDIA GPU (e.g., RTX 3050 Laptop GPU) with compatible NVIDIA drivers

### 1. Activate Environment

If using the existing `.venv` created in the workspace:

```powershell
# From the repository root (d:\space_eye\vyom)
Set-Location D:\space_eye\vyom

# Activate the 64-bit Python 3.11 virtual environment
& "..\.venv\Scripts\Activate.ps1"
```

Or create a fresh virtual environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 2. Install Dependencies

Install the core foundation dependencies:

```powershell
uv pip install -e .
```

To install desktop and computer vision dependencies:

```powershell
uv pip install -e ".[desktop,vision]"
```

For development and testing:

```powershell
uv pip install -e ".[dev]"
```

---

## Running the Application Foundation

### 1. Launch Diagnostics & Configuration Status

Execute `main.py` to inspect the system readiness, hardware detection, dataset paths, and protocol status:

```powershell
python main.py
```

Expected diagnostic output:
- App mode: `OFFLINE`
- Active Compute Device: `CPU` or `GPU` (with fallback status)
- ENIGMA-360 Dataset: Verified at configured path
- Model Artifacts: `NOT_CONFIGURED` (graceful initial state; no fake AI)
- Default Protocol: Validated `sample_experiment.json`

### 2. Validate Experiment Protocols

Validate any protocol JSON file against the formal Pydantic schema:

```powershell
python main.py --validate-protocol config/protocols/sample_experiment.json
```

### 3. Run Focused Unit Verification

```powershell
pytest tests/unit/test_config.py
```

---

## Inference Device & Hardware Strategy

VYOM supports three device policies in `config/app.json`:

1. **`AUTO` (Default):** Inspects the environment for verified CUDA availability and available VRAM. If operational, selects GPU; otherwise safely falls back to CPU.
2. **`CPU`:** Enforces pure CPU inference using ONNX Runtime CPU provider.
3. **`GPU`:** Targets NVIDIA CUDA GPU. If initialization fails and `allow_cpu_fallback: true` is set, issues a clear status notification and switches to CPU without crashing.

---

## Implementation Roadmap

- [x] **Prompt 1: Repository Foundation & Configuration** (Complete)
- [ ] **Prompt 2: PySide6 Desktop GUI Shell**
- [ ] **Prompt 3: Video Input & Local Recording**
- [ ] **Prompt 4: ArUco Marker Detection & Rig Calibration**
- [ ] **Prompt 5: MediaPipe Hand Tracking (21 Landmarks)**
- [ ] **Prompt 6: Dataset Ingestion & YOLOv8-OBB Training Pipeline**
- [ ] **Prompt 7: ONNX Runtime CPU/CUDA Inference Integration**
- [ ] **Prompt 8: Temporal Interaction Event Extractor**
- [ ] **Prompt 9: JSON Deterministic Protocol & State Machine**
- [ ] **Prompt 10: Local Windows Voice Alerts**
- [ ] **Prompt 11: Cryptographic Hash-Chained Audit Logging & Reports**
- [ ] **Prompt 12: End-to-End System Integration & Acceptance**

