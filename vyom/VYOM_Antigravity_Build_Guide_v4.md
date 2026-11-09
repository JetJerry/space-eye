# VYOM — Antigravity Build Guide (Public Dataset Sourcing + CPU/GPU)

## What we are building

Build a Windows desktop prototype named **VYOM**: an offline-first, camera-based experiment monitoring system for human activity recognition and procedure compliance.

The application must:

1. Read a live camera, recorded video, or configured network camera feed.
2. Detect experiment tools/specimen modules with a custom YOLOv8-OBB model.
3. Track hands using MediaPipe Hand Landmarker (21 landmarks per detected hand).
4. Detect ArUco fiducial markers on the experiment rig and use them to establish a rig-relative reference frame.
5. Convert observations into explicit, timestamped interaction events (for example, hand approaches tool, tool picked up, item placed in module).
6. Validate those events against a user-editable JSON experiment protocol with a deterministic finite-state/rule engine.
7. Give local/offline voice alerts for skipped, out-of-order, failed, or timed-out steps.
8. Record session metadata and an append-only, hash-chained event log; verify the chain and generate a report.
9. Store video locally. Add configurable streaming to a specific IP as a later integration phase.
10. Run fully offline on Windows, using CPU inference as a universal fallback and supported local GPUs as optional accelerators. The desktop application must not depend on a cloud API or an LLM.

**Important scope statement:** This is a research/demo prototype, not a flight-certified or safety-certified system. It must not be presented as suitable for real astronaut operations or as the sole authority for safety-critical decisions.

---

# 1. Technology choices

| Area | Technology | Why |
|---|---|---|
| Desktop UI | Python + PySide6 / Qt Widgets | Native Windows desktop interface and Python integration |
| Video | OpenCV | Read camera/video frames, draw overlays, record locally |
| Object detection | Ultralytics YOLOv8-OBB | Detect oriented experiment objects and tools |
| Hand tracking | MediaPipe Tasks — Hand Landmarker | 21 hand landmarks; VIDEO/LIVE_STREAM modes |
| Rig reference | OpenCV ArUco | Detect printed markers and derive a stable reference frame |
| Inference runtime | Ultralytics/PyTorch for training and initial validation; ONNX Runtime with CPU and optional GPU execution providers for deployment | Support CPU fallback and compatible local GPU acceleration |
| Voice alerts | pyttsx3 / Windows SAPI5 | Local text-to-speech; verify the needed voice is installed |
| Protocol | JSON + Pydantic validation | Editable, validated experiment procedure |
| Decision logic | Python deterministic state machine/rule engine | Explainable, repeatable step validation; no LLM decisions |
| Audit | JSONL + `hashlib.sha256` hash chain | Timestamped event history and integrity verification |
| Session metadata | SQLite | Store experiment/session information and report references |
| Testing | pytest; pytest-qt where appropriate | Unit, integration and GUI tests |
| Packaging | PyInstaller | Create a Windows distributable after the app is stable |

### Model deployment and hardware-acceleration strategy

Keep **model training** separate from the **desktop runtime**, and support CPU fallback plus compatible GPU acceleration:

- Training environment: Ultralytics + PyTorch, dataset tools and training scripts. Use CUDA when a supported NVIDIA GPU is available; otherwise allow CPU training.
- Initial inference: validate the real YOLOv8-OBB `.pt` checkpoint through Ultralytics on CPU and, where available, NVIDIA CUDA. This proves the pipeline before export.
- Deployment inference: export to ONNX only after the `.pt` model works. Support ONNX Runtime CPU as the universal fallback and CUDA Execution Provider for compatible NVIDIA GPUs. A Windows GPU provider such as DirectML/WinML may be added only when the exact runtime/model path is verified on the target Windows version; do not promise universal AMD/Intel GPU support without testing.
- At startup, detect available execution providers and expose `AUTO`, `CPU`, and supported `GPU` choices. `AUTO` should select a verified provider, not merely assume a GPU exists. If GPU initialization fails, fall back to CPU with a visible status message. Record the selected provider and device in each session.
- Preserve OBB output geometry and implement the required rotated-box decoding/post-processing. Do not assume ordinary axis-aligned NMS is correct. Compare `.pt` and ONNX outputs on the same saved images before enabling ONNX as the default.
- MediaPipe Hand Landmarker may run on CPU on Windows even when YOLO uses the GPU. Do not force an unsupported MediaPipe GPU delegate; keep each component's device choice independent.
- Do not bundle model weights inside the executable. Store model files under a separate `models/` directory and record their version/checksum in session metadata.

### Python version and environment

Use **64-bit Python 3.11** as the initial compatibility target for the computer-vision environment. Check installed package compatibility before pinning versions. Do not install multiple OpenCV distributions together. If `cv2.aruco` is unavailable, use one compatible `opencv-contrib-python` distribution and remove conflicting `opencv-python` packages.

The user's development laptop has an NVIDIA RTX 3050 Laptop GPU with 4 GB VRAM and 24 GB RAM. Design for entry-level hardware rather than requiring a high-end GPU:

- Start with a small YOLOv8-OBB model and conservative image size/batch settings. Use mixed precision only when the selected GPU/runtime supports it and it is stable.
- Check actual CUDA availability and GPU memory before selecting CUDA. Never assume that installing PyTorch means CUDA is working.
- Support three practical modes: CPU, compatible NVIDIA CUDA GPU, and AUTO with safe fallback. Consider DirectML/WinML for other Windows GPUs only as an optional, explicitly verified backend.
- Measure performance on CPU and on the available GPU when present. Report device/provider, model size, input resolution, FPS, latency and memory use separately.
- MediaPipe on Windows may remain CPU-based; GPU acceleration of YOLO still makes the overall pipeline faster.
- If training is too slow or memory-constrained, keep the training script reproducible so training can be run on another machine. Do not make the final app depend on that machine.


---

# 1A. Dataset sourcing plan — the user has no dataset or sample footage

**Critical update:** The user currently has no private dataset, no sample experiment video and no labelled frames. Do not assume the 1,032 hand-labelled frames mentioned in the VYOM video are available. Treat them as unavailable unless a real archive is supplied and inspected.

The SIH 26174 problem statement itself says dataset generation is part of the challenge. It asks for data to train object detection, pose estimation and hand-object interaction for the experiment. The public SIH problem statement describes fixed-payload cameras and on-device processing, but does not provide the full experiment apparatus specification or its labelled training dataset in the accessible statement. The archive's "View attached dataset" link did not resolve during research. Do not claim that an exact official VYOM/ISRO dataset has been acquired.

## Recommended public datasets

Use public data to bootstrap and benchmark individual capabilities. Do not merge all datasets blindly: they have different camera viewpoints, object taxonomies, annotation formats and licenses. Keep source datasets separated and maintain a provenance manifest.

| Priority | Dataset | What it contributes | Important limits / license |
|---|---|---|---|
| 1 — procedural fixed-camera data | **ENIGMA-360** — https://fpv-iplab.github.io/ENIGMA-360/ and https://github.com/fpv-iplab/ENIGMA-360 | 180 exocentric (fixed/outside) and 180 egocentric procedural videos, synchronized pairs, temporal/spatial annotations, hand-object interaction, action segmentation and key-step recognition. Use the `exo` view first because VYOM is intended for fixed-payload cameras. Official downloader supports videos, frames, masks, features and HOI data with train/val/test splits. | CC BY-NC 4.0 in the repository license; non-commercial research/demo use only unless separate permission is obtained. It is industrial, not the ISRO payload. |
| 2 — industrial tool/hand interaction | **ENIGMA-51** — https://fpv-iplab.github.io/ENIGMA-51/ | 51 egocentric videos, 22 hours, 45,505 RGB frames, two procedures, temporal interaction labels, 275,135 object boxes, 56,473 hand boxes, 25 object classes and tool-use actions involving electrical-board repair. Download page exposes videos, frames, annotations, textual procedures, 3D models and additional hand keypoints/masks. | Very relevant actions and tools, but egocentric rather than fixed-rig camera. The project page's visible material does not clearly state a dataset license; inspect the dataset terms before use or redistribution. |
| 3 — scientific object detection | **Annotated Chemical Apparatus Image Dataset** — https://data.mendeley.com/datasets/8p2hvgdvpn/1 (DOI: 10.17632/8p2hvgdvpn.1) | 5,078 real lab images; 7 classes (hand plus six chemistry apparatus types); train/validation/test folders; YOLO TXT bounding-box annotations; 4.50 GB archive. Useful to start the object-detection data pipeline and learn from hands overlapping apparatus. | CC BY 4.0 according to the dataset record. Labels are ordinary axis-aligned boxes, **not OBB**. Apparatus are chemistry items, not spacecraft modules. |
| 4 — broader physics-lab objects | **Physics Lab Equipment Image Dataset** — https://doi.org/10.6084/m9.figshare.30984658.v1 | Catalog record describes 5,121 images, 27 physics-lab instrument classes, about 8,700 instances, YOLO-compatible TXT annotations, 640×640 images, train/validation/test split and bounding-box/polygon labels. Could broaden equipment coverage after archive inspection. | CC BY 4.0 according to the catalog record. Verify the actual Figshare archive, class list and exact annotation schema before integrating. Do not assume the polygons are valid OBB labels. |
| 5 — procedural action/mistake benchmark | **Assembly101** — https://assembly-101.github.io/ and https://huggingface.co/datasets/ | 4,321 videos of assembling/disassembling 101 toy vehicles, static and egocentric views, over 100K coarse and 1M fine-grained action segments, 18M 3D hand poses, natural order variations, mistakes and corrections. Useful for action-sequence/mistake-detection research. | CC BY-NC 4.0. Toys are not experiment apparatus. Dataset was migrated to Hugging Face in May 2026; use selected subsets only. |
| 6 — rich hand/object interaction | **HOI4D** — https://hoi4d.github.io/ | 2.4M RGB-D frames over 4,000 sequences, 800 object instances in 16 categories, hand pose/action, object pose, segmentation and motion labels. Useful for research on hand-object interaction and temporal geometry. | CC BY-NC 4.0; large and potentially expensive to download/process. Not the target rig. Use only if a specific model-development need justifies it. |
| Optional — interactive procedural assistance | **HoloAssist** — https://holoassist.github.io/ | 169 hours of collaborative physical tasks, action/conversation annotations, mistake detection and intervention benchmarks. | Dataset is large: official page lists 144.62 GB compressed videos, 219.24 GB hand pose and 560.46 GB depth. Do not download wholesale for this laptop. |

### How to use these sources

1. **Start with ENIGMA-360 exocentric videos and annotations** for fixed-camera procedural sequencing and temporal activity research. Use a small train/validation subset first. Do not download both camera views or all modalities unless needed.
2. **Use ENIGMA-51** as a second source for hand/tool interaction and procedure-event ideas. Start with annotations, textual procedures and a small video/frame subset. Verify its license before reusing or redistributing any data.
3. **Use the Mendeley chemical-apparatus dataset** as the first manageable, labelled object-detection dataset. It is 4.50 GB and has existing YOLO axis-aligned boxes. Use it to validate ingestion/training code and as a transfer-learning source, not as VYOM's final detector.
4. **Inspect the Physics Lab Equipment dataset** before download/use; if its categories are more relevant than the chemical apparatus set, add selected classes. Preserve source annotations and licensing.
5. Consider Assembly101 or HOI4D only when the interaction/action-learning phase needs additional examples. Avoid bulk downloading HoloAssist initially because of its very large storage footprint.
6. Keep each source's raw files, original annotations, license and citation separate. Create a normalized derivative dataset only through explicit, reproducible conversion scripts.

## What these public datasets do NOT solve

None of the above sources is verified as containing the exact ISRO/BAS experimental rack, specimen modules, fixtures, tools or experiment protocol described in the VYOM demonstration. Generic public data cannot make a detector recognize an unseen target apparatus reliably.

Therefore VYOM must have two dataset tracks:

- **Public proxy-data track:** learn and benchmark general object detection, hand-object interaction, temporal action segmentation and mistake detection from the sources above.
- **VYOM-specific data track:** create a small synthetic/illustrative dataset for generic demo objects and, when reliable reference geometry becomes available, create target-specific synthetic images from CAD/3D models plus accurately labelled representative images/video. Do not describe generic demo objects as the real ISRO apparatus.

If no physical sample, reference photo, video or CAD model of the actual payload objects is available, build only a clearly-labelled generic prototype. Use procedural Blender/BlenderProc rendering to create synthetic variations (camera angle, lighting, background, occlusion and object placement) for the configurable demo classes. Synthetic geometry and labels must be generated together and checked visually. Synthetic data is useful to prove the pipeline, but it does not close the real-world domain gap.

For YOLOv8-OBB specifically:
- The Mendeley dataset's labels are axis-aligned YOLO boxes. Do not rename or pretend them to be OBB labels.
- A rectangular box can be represented as a four-corner polygon with zero rotation for format-conversion/debugging, but this does not create genuine oriented-box training diversity.
- Preserve source labels; convert only with a tested converter and document that converted labels remain horizontal boxes.
- Train/validate a true OBB model only on valid four-corner OBB labels (from an appropriate OBB dataset, synthetic rendering, or target-specific annotation).

### MediaPipe and ArUco do not need the same custom dataset

- **MediaPipe Hand Landmarker:** use Google's published pretrained Hand Landmarker task model to obtain 21 landmarks. The official model documentation states it was trained on roughly 30K real-world images plus rendered synthetic hands. Do not retrain it for the first prototype. Download the task asset from the official MediaPipe documentation/model link and record its version/checksum.
- **ArUco:** no supervised image dataset is required for the basic marker detector. Generate printable marker IDs/boards with OpenCV, print them, measure the physical marker size, and calibrate the camera/rig. Without measured geometry and calibration, report marker detections but do not claim metric rig coordinates.

## Dataset engineering requirements for Antigravity

- Create `data/SOURCES.md` or `data/manifests/dataset_sources.yaml` recording source name, official URL, download date, license, citation, selected splits/views/modalities, archive size if verified, original classes, annotation format, checksum and local path.
- Create a downloader/manifest workflow that supports selective downloads and resumable transfers where the official source permits. Never use unofficial mirrors when an official source is available.
- Before any multi-gigabyte download, display exact archive size/storage impact and wait for the user's approval. Never silently download the full HOI4D/HoloAssist/Assembly101 corpus.
- Do not scrape arbitrary Google images or YouTube frames as labelled training data. Use only sources with clear licensing/permission and preserve attribution.
- Inspect every downloaded archive before extraction; protect against path traversal, preserve original files, verify archive integrity and keep raw sources read-only.
- Keep source-specific train/validation/test splits. Do not merge different datasets and random-split adjacent video frames, which causes leakage.
- Include a dataset report with counts, classes, image dimensions, annotation validity and duplicate/near-duplicate checks. No training starts until the chosen labels are understood.
- If a dataset is unavailable, gated, corrupted or its license is unclear, report that accurately and move to another verified source. Never claim it has been downloaded when it has not.


---

# 2. Critical engineering principles

1. **No fake AI.** Never show mock detections as if they came from a real model. If a model or dataset is missing, show an explicit `MODEL NOT CONFIGURED` / `DATASET NOT FOUND` state.
2. **No invented dataset or weights.** The user has no dataset/sample footage. Do not assume the VYOM video's claimed 1,032 labelled frames are available. Source public datasets only from the verified sources in Section 1A, preserve licenses/provenance, and clearly separate proxy data from VYOM-specific data. Do not silently create fake images, annotations, trained checkpoints, accuracy scores or demo results. Synthetic demo assets/protocols are acceptable only when clearly labelled as synthetic.
3. **Object detection is not action recognition.** YOLO detects objects; MediaPipe provides hand landmarks; ArUco provides a reference frame. A separate temporal interaction/event layer must infer actions from these observations.
4. **Unknown means unknown.** If the evidence is insufficient, emit `UNKNOWN` or `NEEDS_REVIEW`; do not mark a step complete.
5. **Deterministic compliance.** The protocol engine must be explicit Python rules/state transitions. Do not use an LLM, generative model or opaque classifier to decide whether a procedure step is compliant.
6. **Temporal stability.** Use confidence thresholds, consecutive-frame confirmation, minimum-duration rules and alert cooldowns to avoid flickering detections and repeated voice warnings.
7. **Rig-relative coordinates.** Use ArUco detections and a documented calibration to compute positions relative to the experiment rig. Do not assume the floor, gravity or screen-up direction is meaningful in microgravity.
8. **Thread safety.** Camera capture, inference, audio and file writing must not freeze the Qt GUI thread. Use worker objects/threads and Qt signals.
9. **Offline by design.** No cloud APIs, external telemetry or mandatory network access at runtime. Package model assets locally. The optional IP video stream is a configurable local-network output, not a cloud dependency.
10. **Auditable claims.** Record timestamps, model versions, selected inference provider/device, protocol version, event confidence/evidence and the final hash. Never claim a hash chain is mathematically tamper-proof: it is tamper-evident if the trusted final hash is preserved separately.
11. **Selective verification.** Do not run a full test suite after every prompt. Use focused checks for critical components (OBB output parity, event extraction, procedure-state transitions and hash-chain verification), then perform broader integration and end-to-end validation in Prompt 12.
12. **Explain the code.** The agent must explain the important files, design decisions and commands in beginner-friendly language.

---

# 3. Proposed repository structure

```text
VYOM/
├── main.py
├── pyproject.toml
├── README.md
├── .gitignore
├── .env.example                 # only non-secret local settings; no API keys needed
├── config/
│   ├── app.json
│   ├── camera.json
│   ├── rig.json                 # ArUco IDs, marker size, rig coordinates/calibration
│   └── protocols/
│       └── sample_experiment.json
├── models/
│   ├── README.md                # where to place model files; weights are not committed
│   ├── object_detector.onnx     # added only after a real model is trained/exported
│   └── hand_landmarker.task     # official task asset, if used by MediaPipe Tasks
├── data/
│   ├── external/                # downloaded public datasets; source-specific folders; ignored by Git
│   ├── manifests/               # source URLs, licenses, checksums, split selections
│   ├── raw/                     # original images/videos; ignored by Git
│   ├── annotations/             # source annotation exports; ignored by Git
│   ├── yolo_obb/
│   │   ├── images/train/
│   │   ├── images/val/
│   │   ├── images/test/
│   │   ├── labels/train/
│   │   ├── labels/val/
│   │   └── labels/test/
│   └── README.md                # dataset manifest and provenance
├── runs/                        # training outputs; ignored by Git
├── sessions/                    # local recordings, reports and event logs; ignored by Git
├── src/
│   └── vyom/
│       ├── __init__.py
│       ├── app.py
│       ├── core/
│       │   ├── config.py
│       │   ├── logging_setup.py
│       │   ├── schemas.py
│       │   └── exceptions.py
│       ├── ui/
│       │   ├── main_window.py
│       │   ├── live_monitor.py
│       │   ├── protocol_panel.py
│       │   ├── event_panel.py
│       │   ├── session_panel.py
│       │   └── settings_dialog.py
│       ├── capture/
│       │   ├── frame_source.py
│       │   ├── camera_source.py
│       │   ├── video_file_source.py
│       │   └── recorder.py
│       ├── vision/
│       │   ├── pipeline.py
│       │   ├── object_detector.py
│       │   ├── hand_tracker.py
│       │   ├── aruco_tracker.py
│       │   ├── calibration.py
│       │   └── drawing.py
│       ├── interaction/
│       │   ├── observations.py
│       │   ├── event_extractor.py
│       │   ├── temporal_filter.py
│       │   └── geometry.py
│       ├── protocol/
│       │   ├── protocol_schema.py
│       │   ├── protocol_loader.py
│       │   ├── rule_engine.py
│       │   └── state_machine.py
│       ├── alerts/
│       │   ├── alert_manager.py
│       │   └── speech_service.py
│       ├── audit/
│       │   ├── hash_chain.py
│       │   ├── event_writer.py
│       │   ├── verifier.py
│       │   └── report_generator.py
│       └── storage/
│           ├── database.py
│           └── session_repository.py
├── scripts/
│   ├── inspect_dataset.py
│   ├── validate_obb_labels.py
│   ├── split_dataset.py
│   ├── train_obb.py
│   ├── export_onnx.py
│   ├── compare_pt_onnx.py
│   └── generate_aruco_markers.py
└── tests/
    ├── unit/
    ├── integration/
    ├── fixtures/
    └── test_data/               # tiny synthetic test fixtures only; not claimed as real training data
```

The agent may refine this tree, but it must preserve the separation between UI, capture, perception, interaction events, protocol rules, alerts and audit.

---

# 4. Build sequence — use one prompt at a time

**Workflow:** paste Prompt 0 first if the architecture/dataset have not already been reviewed. Then provide Prompt 1, Prompt 2, and so on, one at a time. Do not paste all prompts together. Avoid repetitive full test runs after every prompt. Run only the checks that are important for the capability being introduced; perform the main integration and end-to-end verification in Prompt 12. Use Git commits at meaningful milestones so you can recover from a broken change.

## Prompt 0 — Architecture and public dataset sourcing plan (no code yet)

```text
We are building VYOM, an offline-first Windows desktop Human Activity Recognition and Experiment Procedure Compliance application for SIH 26174.

Read the complete VYOM build guide, especially Section 1A (Dataset sourcing plan). The user has NO dataset, NO sample experiment video and NO private labelled frames. Do not assume the 1,032 frames mentioned in the demonstration video are available.

Core requirements:
- PySide6 desktop GUI.
- OpenCV live camera/local video and local recording.
- YOLOv8-OBB for configurable experiment objects/tools.
- MediaPipe Hand Landmarker (21 landmarks).
- OpenCV ArUco rig reference and calibration.
- A temporal hand-object interaction event extractor.
- JSON-defined deterministic procedure/rule engine. No LLM makes compliance decisions.
- Offline voice alerts, SQLite sessions, JSONL SHA-256 hash-chain logs and report generation.
- CPU fallback plus supported local GPU acceleration; optional configured-IP video streaming later.

DATASET SOURCING IS PART OF THE PROJECT:
1. Inspect the official public sources listed in Section 1A.
2. Prioritize ENIGMA-360 exocentric/fixed-camera data for temporal procedure research, ENIGMA-51 for industrial tool/hand interactions, and the Mendeley annotated chemical-apparatus dataset for an initial object-detection pipeline.
3. Inspect the Physics Lab Equipment dataset as an additional source only after validating its archive and annotations.
4. Consider Assembly101/HOI4D only for a defined interaction/action research need. Do not download very large corpora such as HoloAssist wholesale.
5. Use the official MediaPipe Hand Landmarker task asset; no custom MediaPipe training dataset is required initially. Generate ArUco markers procedurally; no ArUco training dataset is required.
6. Keep all proxy datasets separate, preserve source splits/licenses/attribution, and distinguish them from any VYOM-specific data.
7. No source listed is confirmed to contain the exact ISRO/BAS apparatus. Do not present generic laboratory tools as those exact objects.

First inspect the workspace and produce:
- proposed architecture and data flow;
- complete repository structure, including `data/external`, `data/manifests` and `data/SOURCES.md`;
- runtime vs training dependency plan;
- a source-by-source acquisition manifest with official URLs, licenses, expected contents/annotation formats and recommended subset;
- estimated download/storage impact where verified;
- a plan for obtaining the smallest useful ENIGMA-360 exo subset, ENIGMA-51 annotation/video subset and the 4.50 GB Mendeley dataset;
- a plan for synthetic VYOM demo objects if exact rig photos/CAD/video are unavailable;
- Windows/Python/OpenCV/MediaPipe/CUDA compatibility risks.

Do not download multi-gigabyte archives or write application code yet. First show the exact source list, license notes, sizes and local storage impact, then wait for approval to download. Do not invent source links, file sizes or data contents.
```

## Prompt 1 — Repository foundation and configuration

```text
Implement Phase 1: create the VYOM repository foundation only.

Requirements:
- Use the approved folder architecture.
- Set up a Python package under src/vyom and a clear application entry point.
- Add pyproject.toml with compatible dependency groups for runtime, development and optional training. Do not indiscriminately install PyTorch/Ultralytics into the final runtime if they are not yet needed.
- Add a config loader with validation and sensible local defaults.
- Add structured logging to a local logs directory.
- Add .gitignore entries for .venv, __pycache__, data/raw, data/annotations, runs, models/*.pt, models/*.onnx, sessions and generated recordings/reports. Do not ignore README files or source code.
- Add a README with Windows PowerShell setup and run instructions.

Do not implement computer vision yet. Do not create fake model files, fake dataset files or fake accuracy values. Do not run a broad test suite in this foundation step; only resolve basic import/configuration issues that prevent the project from starting. Explain every new top-level folder.
```

## Prompt 2 — PySide6 desktop shell

```text
Implement Phase 2: create the VYOM desktop GUI shell using PySide6 Qt Widgets.

Design a professional engineering/mission-monitoring interface with:
- Top bar: VYOM name, Offline status, current session status and settings.
- Left panel: camera/video source, Start, Pause, Stop, protocol selector and session controls.
- Main area: large video preview area with a clear placeholder until a real frame arrives.
- Right panel: protocol step list, current step, next-step instruction and progress.
- Bottom panel: timestamped event/alert timeline and FPS/system status. Include a clear compute status field that can later show AUTO/CPU/GPU and the active provider.

Requirements:
- Use reusable UI components and layouts; do not put the entire UI in main.py.
- Add working navigation and button states, but do not pretend that AI is running.
- Display explicit statuses such as CAMERA NOT CONNECTED, MODEL NOT LOADED and PROTOCOL NOT SELECTED.
- Keep all long-running work out of the GUI thread.
Do not add fake detections or fake events. Perform only the basic launch check needed to confirm that the window opens; do not run a broad test suite.
```

## Prompt 3 — Video input and local recording

```text
Implement Phase 3: real video-source support.

Support:
1. Webcam by device index.
2. Local video file (MP4/AVI supported by the installed OpenCV build).
3. Design the interface so RTSP/network camera input can be added without rewriting the pipeline.

Requirements:
- Use OpenCV VideoCapture in a worker thread.
- Send frames to the UI through Qt signals or a bounded latest-frame queue; never update widgets directly from a worker thread.
- Add start/pause/resume/stop and clean resource release.
- Show source name, frame size, measured FPS and connection errors.
- Add optional local session recording using VideoWriter with a user-selected output path.
- Record capture timestamps and preserve monotonic frame timestamps for downstream MediaPipe video processing.
- Handle unavailable cameras, invalid file paths, end-of-file and camera disconnection gracefully.
Do not add detection yet. Demonstrate that a real camera or selected video appears in the UI. Do not claim live-camera behaviour was verified if no camera is available.
```

## Prompt 4 — ArUco marker detection and rig reference

```text
Implement Phase 4: ArUco marker detection and calibration utilities.

Use OpenCV's ArUco module. Make sure the selected OpenCV package exposes cv2.aruco; do not install opencv-python and opencv-contrib-python side by side.

Requirements:
- Add a script to generate printable markers for the configured dictionary and marker IDs.
- Read marker dictionary, marker IDs, physical marker size and rig-coordinate definitions from config/rig.json.
- Detect markers on each processed frame and draw IDs/corners in the preview.
- Create a documented calibration/reference-frame abstraction. If required calibration data is missing, show UNCALIBRATED rather than inventing physical coordinates.
- Expose marker visibility and calibration state to the UI.
Important: explain the assumptions for the camera/rig geometry. Do not claim real-world metric coordinates unless camera calibration and physical marker measurements are supplied. Verify the coordinate-transform math with a small known-point check, but do not run the full project test suite.
```

## Prompt 5 — MediaPipe hand tracking

```text
Implement Phase 5: hand tracking using the MediaPipe Tasks Hand Landmarker API.

Requirements:
- Use the current MediaPipe Tasks API, not an undocumented or deprecated wrapper.
- Support VIDEO or LIVE_STREAM mode for camera/video processing and provide monotonically increasing frame timestamps.
- Expose all 21 landmarks for every detected hand, including handedness and confidence when available.
- Draw landmark connections on the preview and show the number of detected hands.
- Run tracking in a worker/pipeline thread and keep the GUI responsive.
- Make the task model asset path configurable. If it is missing, show HAND MODEL NOT INSTALLED and give the user setup instructions; do not fabricate landmarks.
Do not infer a completed experiment action from hand landmarks alone. This phase only produces hand observations. Use a short sample image/video to confirm the integration; defer broader tests.
```

## Prompt 6 — Public dataset acquisition, inspection and object-detector training

```text
Implement Phase 6: acquire and prepare verified public datasets, then establish a reproducible object-detection training pipeline.

The user has no private dataset, sample video or labelled images. Do not expect a user-provided 1,032-frame dataset. Use the official public sources in Section 1A of VYOM_Antigravity_Build_Guide_v4.md.

DATA ACQUISITION:
1. Create `data/SOURCES.md` and `data/manifests/dataset_sources.yaml` with source URL, source name, license, citation, date accessed, download selection, classes, annotation format, archive size (only if verified), checksum and local extraction path.
2. Prioritize these sources:
   - ENIGMA-360 official repo/site: start with a small train/validation subset of the EXOCENTRIC (`exo`) fixed-camera videos and matching annotations/HOI material. Use the official downloader and its documented arguments; inspect `annotations/train.txt`, `val.txt`, `test.txt` and downloader before invoking it.
   - ENIGMA-51 official project page: retrieve annotations/textual procedures and a small selected video/frame subset for industrial tool/hand interactions. Verify license terms before use or redistribution; if unclear, mark as research-only pending confirmation.
   - Mendeley Annotated Chemical Apparatus Image Dataset: 5,078 images, 7 classes (hand + six apparatus), 4.50 GB ZIP, standard YOLO axis-aligned box labels, CC BY 4.0. Use as the initial labelled object-detection source if its download and archive are accessible.
   - Inspect the Physics Lab Equipment dataset (DOI 10.6084/m9.figshare.30984658.v1) before deciding whether to include it. Verify the actual file archive, classes and labels; catalog metadata is not enough.
3. Do not download full Assembly101, HOI4D or HoloAssist unless a later model-development need justifies the storage. Use only selected subsets and obey their licenses.
4. Before downloading any multi-gigabyte archive, report the verified size and required disk space and wait for user approval. Prefer official downloaders and resumable transfer. Never claim success unless the files are actually present and archive checksums/integrity are verified.
5. Keep the original datasets untouched in source-specific folders under `data/external/`. Do not merge labels/classes without an explicit mapping and provenance.

DATASET INSPECTION:
- Create scripts/inspect_dataset.py to report image/video count, class distribution, dimensions, missing/corrupt files, annotation count, invalid labels, duplicate/near-duplicate frames and split sizes.
- Create format-specific parsers/validators for the original datasets. Preserve original annotations.
- Split by source video/session where possible. Retain official train/val/test partitions. Do not randomly split adjacent video frames.
- Write an explicit class mapping for any normalized derivative dataset.
- Do not treat the chemical-apparatus dataset's axis-aligned YOLO labels as OBB. Do not claim they are OBB after merely renaming files. If converting axis-aligned boxes to four-corner rectangles for tooling tests, label them as zero-rotation boxes and explain that this does not provide genuine oriented-box variety.

OBB TRAINING:
- Build a YOLOv8-OBB pipeline only when valid OBB labels are available.
- For the first public lab-object baseline, train a regular YOLO detection model if using axis-aligned labels; keep this baseline separate from the final OBB model.
- Create a synthetic VYOM demo-object pipeline using Blender/BlenderProc only for clearly named generic demo classes if no real rig geometry/photos/CAD exist. Generate image and OBB labels from the same scene geometry; vary camera pose, lighting, background, scale and occlusion. Do not label these as real ISRO equipment.
- A target-specific OBB model for the exact experiment objects cannot be considered complete until those object's geometry or reference images/video are available and a representative labelled set is made.
- Configure training device (`auto`, `cpu`, selected CUDA device), image size, batch, workers, epochs, patience and output paths. Check actual CUDA and VRAM; start conservatively on a 4 GB GPU. Provide a reproducible CPU route or instructions for external training if local resources are insufficient.
- Save training configuration, metrics, class-wise results, confusion matrix and best checkpoint. Never invent scores. Evaluate on a holdout split that is not used for tuning.

DELIVERABLE:
A verified dataset manifest and inspection report, source-specific prepared datasets, a reproducible axis-aligned detection baseline where suitable labels exist, and an OBB training pipeline that runs only on valid OBB data. Clearly report which parts use real public data and which parts use synthetic demo data. Do not claim VYOM-specific recognition accuracy for proxy datasets.

Keep all training dependencies separate from the production desktop runtime.
```

## Prompt 7 — CPU/GPU inference and model-artifact management

```text
Implement Phase 7: integrate the trained object detector into the VYOM vision pipeline.

Requirements:
- Load only a real user-trained checkpoint or configured model artifact.
- Support the validated Ultralytics `.pt` model on CPU and on a compatible NVIDIA CUDA GPU when available.
- Add an export script for ONNX after the checkpoint is validated.
- Support ONNX Runtime CPU as the universal deployment path and CUDA Execution Provider for compatible NVIDIA GPUs. Add DirectML/WinML only as an optional backend after validating the exact provider on the target Windows environment.
- Implement provider detection and a runtime selection policy (`AUTO`, `CPU`, supported `GPU`). If GPU initialization fails, report the reason and fall back to CPU.
- Preserve correct OBB output decoding and rotated-box post-processing. Do not assume ordinary axis-aligned NMS is correct for oriented boxes.
- Add one focused parity check: run the same saved images through `.pt` and ONNX and compare class IDs, confidence and polygon/box geometry within documented tolerances.
- Add model metadata (filename, SHA-256 checksum, class names, input size, model version and training run identifier).
- Show model status, selected provider/device, GPU availability and inference latency in the GUI.
- If no trained model exists, keep detection disabled and display MODEL NOT CONFIGURED.
- Record the inference provider/device in each session.

Measure CPU performance and, when available, GPU performance on the target laptop. Do not promise real-time performance before measurement. Do not force MediaPipe to use a Windows GPU delegate if that delegate is unsupported; YOLO and hand tracking may use different devices.
```

## Prompt 8 — Interaction observations and temporal event extraction

```text
Implement Phase 8: convert visual observations into explicit interaction events.

This is a critical phase. YOLO object detections and MediaPipe landmarks are not actions by themselves.

Create typed observation objects containing:
- frame timestamp;
- detected OBB objects (class, confidence, polygon corners);
- hand landmarks and handedness;
- ArUco marker detections and calibration status;
- transformed positions in rig coordinates when calibration is valid.

Implement a modular event extractor for configurable events such as:
- HAND_NEAR_OBJECT;
- OBJECT_CONTACT_CANDIDATE;
- OBJECT_PICKED_UP;
- OBJECT_MOVED;
- OBJECT_PLACED_IN_TARGET;
- OBJECT_REMOVED_FROM_TARGET;
- UNKNOWN / NEEDS_REVIEW.

Requirements:
- Use hand-to-object geometric distance, object polygon geometry, object movement across consecutive frames and rig-relative coordinates where available.
- Configure thresholds in a file; do not bury arbitrary distances in code.
- Add temporal smoothing, consecutive-frame confirmation, minimum duration and hysteresis.
- Every event must include timestamp, source frame range, evidence, confidence and any uncertainty.
- Never mark a step complete from a single noisy frame.
- If the camera/marker calibration is invalid or the object is occluded, emit UNKNOWN rather than guessing.
- Unit-test geometry, event transitions, boundary cases, missing landmarks and occlusion handling with synthetic observations.

Do not implement procedure compliance in this phase. Produce a reliable stream of low-level events first.
```

## Prompt 9 — JSON protocol and deterministic procedure engine

```text
Implement Phase 9: the JSON-driven deterministic experiment protocol engine.

Requirements:
- Define and validate a versioned JSON protocol schema.
- A protocol step should support: step_id, order, title, human instruction, expected_event, required_objects, preconditions, minimum evidence duration, timeout, completion criteria, severity and voice message.
- Support protocol loading, schema validation and a clear UI preview.
- Implement a finite-state machine with states such as NOT_STARTED, IN_PROGRESS, COMPLETED, SKIPPED, OUT_OF_ORDER, TIMED_OUT, PAUSED and NEEDS_REVIEW.
- Consume the event stream from Phase 8. Do not read camera frames or call YOLO/MediaPipe directly from the rule engine.
- Detect a later-step event before the expected step and mark the correct violation. Do not declare a violation based on one weak/low-confidence observation.
- After each completed step, present the next expected step in the UI.
- Add alert de-duplication/cooldown identifiers so the same violation does not generate voice output on every frame.
- Add comprehensive unit tests for correct order, skipped step, out-of-order event, missing evidence, timeout, pause/resume and invalid protocol.
- Include a clearly labelled sample protocol for testing. Do not claim the sample describes the real experiment unless the user supplies the procedure.

All compliance decisions must be explainable by protocol rules and recorded evidence. No LLM or generative AI is allowed in this decision path.
```

## Prompt 10 — Offline voice alerts

```text
Implement Phase 10: local voice alerts.

Use pyttsx3 with the Windows SAPI5 engine or another verified local Windows voice engine. No cloud TTS service, API key or internet call is allowed.

Requirements:
- Speak only alerts emitted by the protocol engine.
- Run speech playback on a dedicated worker so the GUI and vision loop never block.
- Add configurable voice, rate and volume where supported.
- Add priority handling (critical violation before informational next-step prompt), a queue limit and de-duplication.
- Provide a mute/test voice control in settings.
- Log alert text, reason, event ID and timestamp. Do not log private content unrelated to the experiment.
- Verify that an offline Windows voice is installed. If no voice is available, show a visible audio error while keeping visual alerts active.
- Test that rapid repeated events do not produce repeated or overlapping warnings.
```

## Prompt 11 — Hash-chain audit log, session storage and reports

```text
Implement Phase 11: session persistence and auditable event reports.

Requirements:
- Store session metadata in SQLite: session ID, protocol ID/version, start/end times, camera source label, model versions/checksums, calibration status and final status.
- Write an append-only JSONL event log for observations selected for audit, protocol transitions, alerts, operator actions and session lifecycle.
- Canonicalize each event payload deterministically (stable key ordering and stable serialization).
- Use SHA-256 chaining: each record includes sequence number, timestamp, event payload, previous_hash and current_hash. Compute current_hash from previous_hash plus canonical event bytes using one documented algorithm.
- The first record must use a documented genesis hash.
- Add a verifier that replays the entire file and reports the first invalid/missing/reordered record.
- Add tests that verify a valid log and detect modified, deleted, reordered and appended records.
- Explain clearly that a hash chain detects tampering only relative to a trusted/anchored final hash. Offer export of the final digest so it can be stored separately; do not claim local hashes alone make files impossible to alter.
- Generate a human-readable session report and machine-readable JSON/CSV summary with step outcomes, timestamps, violations, alerts, evidence references, model/protocol version and hash-verification result.
- Do not mark a session audit-valid if verification fails.
```

## Prompt 12 — Integration, offline acceptance and optional IP streaming

```text
Implement Phase 12: full VYOM integration and offline deployment hardening.

Acceptance requirements:
1. Launch the desktop app without internet access.
2. Load a local protocol JSON.
3. Open a local video or attached webcam.
4. Load the actual trained object detector and local MediaPipe task asset.
5. Show real object, hand and ArUco overlays when detected; otherwise clearly report the missing component.
6. Produce interaction events and validate them using the rule engine.
7. Show correct next-step guidance and voice alerts for an intentionally tested skipped/out-of-order sequence.
8. Save local video and timestamped audit log.
9. Verify the hash chain and generate a report.
10. Measure CPU FPS/latency and, when available, GPU FPS/latency, memory use and event-to-alert latency on the target laptop. Report each execution provider separately.
11. Verify that the application still works with Wi-Fi/internet disconnected.
12. Verify safe fallback to CPU if the selected GPU/provider is unavailable or fails to initialize.
13. Verify the UI can select AUTO, CPU or a supported GPU provider and accurately reports which provider each component actually uses. Do not imply MediaPipe is GPU-accelerated on Windows if its installed Python backend is CPU-only.

Optional IP streaming:
- Add configurable streaming to a specified LAN IP/port using a documented FFmpeg or GStreamer pipeline.
- Keep recording and inference working if streaming fails.
- Do not expose a network stream by default; require explicit configuration and show the destination/status in the UI.

Packaging:
- Package the GUI with PyInstaller only after the integration tests pass.
- Keep model weights/task assets in a separate models folder beside the app or in a documented user-selected model directory.
- Include all required Qt plugins and the selected ONNX Runtime package/provider dependencies. Avoid bundling conflicting CPU/GPU runtime distributions; test CUDA/cuDNN compatibility on target NVIDIA systems and provide the CPU fallback package/path. Test the packaged build on a clean Windows environment if possible.
- Include an offline installation checklist and model checksum manifest.

Produce a final verification report. Clearly separate completed features, tested features, untested assumptions and known limitations. Do not claim flight readiness or certified safety.
```

---

# 5. Example experiment protocol JSON

This is a **synthetic schema example** to show the shape of a protocol. Replace the names, events, thresholds and order with the actual experiment supplied by the user. It is not a real ISRO procedure.

```json
{
  "protocol_id": "sample-specimen-transfer",
  "protocol_version": "1.0.0",
  "experiment_name": "Sample Specimen Transfer - DEMO",
  "description": "Synthetic demo protocol for software testing only",
  "steps": [
    {
      "step_id": "STEP_01",
      "order": 1,
      "title": "Pick up specimen container",
      "instruction": "Pick up the specimen container from the holder.",
      "expected_event": "OBJECT_PICKED_UP",
      "required_objects": ["specimen_container", "holder"],
      "min_confidence": 0.80,
      "stable_duration_ms": 600,
      "timeout_seconds": 30,
      "severity": "warning",
      "voice_message": "Please pick up the specimen container."
    },
    {
      "step_id": "STEP_02",
      "order": 2,
      "title": "Place specimen in module",
      "instruction": "Place the specimen container inside the module.",
      "expected_event": "OBJECT_PLACED_IN_TARGET",
      "required_objects": ["specimen_container", "experiment_module"],
      "min_confidence": 0.85,
      "stable_duration_ms": 800,
      "timeout_seconds": 45,
      "severity": "critical",
      "voice_message": "Warning: place the specimen inside the experiment module."
    }
  ]
}
```

The agent must validate this JSON against a schema and must not assume that an event such as `OBJECT_PICKED_UP` is already detectable until the perception/event-extraction phase has been implemented and tested.

---

# 6. Dataset and model notes you must understand

## YOLOv8-OBB labels

Ultralytics OBB labels typically contain a class index followed by four normalized corner points:

```text
class_index x1 y1 x2 y2 x3 y3 x4 y4
```

That is different from standard horizontal YOLO boxes (`class x_center y_center width height`) and different from segmentation polygons with an arbitrary number of points. Inspect the original annotation export before converting anything. Preserve the originals and test the converter on hand-checked examples.

## Frame count alone does not establish dataset quality

The number of frames alone does not establish dataset quality. Check:

- number of classes and examples per class;
- variation in distance, orientation, lighting and occlusion;
- whether images come from multiple video sessions;
- annotation consistency;
- whether train/validation/test contain near-duplicate neighbouring frames.

For a small dataset, start with pretrained weights and fine-tune a small model, use conservative augmentation, and report uncertainty. Evaluate on whole held-out sessions where possible.

## YOLO does not validate the procedure

YOLOv8-OBB detects oriented objects. It does not, by itself, know that a person has picked up an object, inserted it into another object, or completed a sequence. That requires temporal reasoning over object movement, hand landmarks and the rig reference frame. The resulting event is then passed to the deterministic protocol engine.

## MediaPipe landmarks

Use the MediaPipe Tasks Hand Landmarker API. For live video, use its live-stream pattern/callback or a carefully managed video mode with monotonically increasing timestamps. Do not run a blocking detector call on the Qt GUI thread.

## ArUco reference frame

A marker gives an identifiable visual reference and, with known marker geometry/camera calibration, supports coordinate estimation. A printed marker alone does not magically provide accurate 3D metric coordinates. Ask for marker size, camera calibration and rig-coordinate definitions before reporting physical distances.

## CPU and entry-level GPU support must be measured

The application should run offline on CPU and use a compatible local GPU when available. Test the actual target laptop in both CPU and GPU modes where possible. Report the model, input resolution, execution provider, FPS, latency and CPU/RAM/VRAM use separately. Do not write “real-time” in the README until you measure and define the target frame rate. MediaPipe Hand Landmarker on Windows Python may use CPU even when YOLO object detection uses CUDA; this is acceptable and should be shown accurately in diagnostics.

---

# 7. Recommended first-day checklist

1. Create `D:\AI_Projects\VYOM`.
2. Open that folder in Antigravity.
3. Create a Git repository and make the initial commit.
4. Paste **Prompt 0 only**.
5. Review the architecture and dataset audit. Do not let the agent skip dataset verification.
6. Paste Prompt 1 only after approving the plan.
7. Launch the desktop shell before proceeding. Save broader testing for the critical perception/rule/audit milestones and the final integration phase.

Suggested PowerShell setup:

```powershell
mkdir D:\AI_Projects\VYOM
cd D:\AI_Projects\VYOM
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If Python 3.11 is not installed, install the 64-bit release first. Keep GPU acceleration optional: verify the NVIDIA driver and PyTorch CUDA availability separately; the app must still run on CPU when CUDA or a compatible GPU provider is unavailable. Do not install arbitrary CUDA/cuDNN versions; follow the matching PyTorch/ONNX Runtime compatibility instructions.

---

# 8. Official references

- SIH 26174 problem statement: https://sih2026.vuce.in/ps/SIH26174
- Ultralytics YOLOv8 model overview: https://docs.ultralytics.com/models/yolov8
- Ultralytics OBB dataset format: https://docs.ultralytics.com/datasets/obb
- Ultralytics OBB task and export: https://docs.ultralytics.com/tasks/obb
- MediaPipe Hand Landmarker for Python: https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python
- OpenCV ArUco detection: https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html
- ONNX Runtime execution providers: https://onnxruntime.ai/docs/execution-providers/
- Qt for Python threads/signals: https://doc.qt.io/qtforpython-6/PySide6/QtCore/QThread.html
- Python hashlib: https://docs.python.org/3/library/hashlib.html
- pyttsx3: https://pyttsx3.readthedocs.io/en/latest/

---

# 9. Final instruction to the developer

Do not optimise for the number of files generated. Optimise for a working, explainable and testable system. Implement one phase at a time and explain it. Use focused checks only where they protect a critical capability; perform broad integration testing at the final phase. The most important engineering work is the reliable conversion of noisy visual observations into verified interaction events and then deterministic protocol decisions. The GUI is only the surface of that system.


## Hardware acceleration notes (verified documentation)

- Ultralytics training supports specifying `device` and can use an available GPU or CPU; always inspect actual CUDA availability and available VRAM before training: https://docs.ultralytics.com/modes/train
- PyTorch's Windows installation guidance provides CUDA-enabled builds when the Windows/NVIDIA environment is compatible: https://pytorch.org/get-started/locally/
- ONNX Runtime exposes CPU, CUDA and other execution providers. CUDA requires matching CUDA/cuDNN dependencies: https://onnxruntime.ai/docs/execution-providers/ and https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
- ONNX Runtime's Windows guidance notes that Windows ML is the preferred route for broad hardware support on Windows 11 24H2+, while DirectML is in sustained engineering: https://onnxruntime.ai/docs/get-started/with-windows.html
- MediaPipe Hand Landmarker supports Python on Windows, but the Python GPU delegate is platform/model constrained; do not assume GPU acceleration for Hand Landmarker on Windows: https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python and https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/BaseOptions

The implementation agent must check current compatibility and install instructions at implementation time. Do not mix arbitrary PyTorch, CUDA, cuDNN and ONNX Runtime versions.
