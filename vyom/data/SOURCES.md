# Public Dataset Sourcing Manifest

This document records the provenance, licensing, citation, and characteristics of public datasets evaluated and utilized for VYOM research.

---

## 1. ENIGMA-360 (Primary Procedural & Interaction Source)

- **Source Name:** ENIGMA-360 (Egocentric and Exocentric Procedural Video Dataset)
- **Official URL:** https://fpv-iplab.github.io/ENIGMA-360/
- **Repository:** https://github.com/fpv-iplab/ENIGMA-360
- **Local Configured Path:** `D:\space_eye\datasets\ENIGMA-360`
- **License:** CC BY-NC 4.0 (Non-commercial research / demo only)
- **Modality:** Synchronized Exocentric (`exo`) and Egocentric (`ego`) video pairs, spatial/temporal annotations, hand-object interaction (HOI), action segmentation.
- **Target Selection:** Exocentric (`exo`) training and validation splits. Fixed-payload camera perspective matches the VYOM hardware setup.
- **Safety Directive:** Read-only external reference. Do not modify or delete files during downloads or ingestion.

---

## 2. ENIGMA-51 (Tool & Hand Interaction Reference)

- **Source Name:** ENIGMA-51
- **Official URL:** https://fpv-iplab.github.io/ENIGMA-51/
- **License:** Inspect dataset terms before redistribution (research use only).
- **Modality:** Egocentric electrical-board repair, 51 videos, 45,505 frames, 25 object classes, tool manipulation.
- **Target Selection:** Textual procedures, temporal interaction labels, and tool bounding boxes.

---

## 3. Annotated Chemical Apparatus Image Dataset (Object Detection Baseline)

- **Source Name:** Annotated Chemical Apparatus Image Dataset
- **Official URL:** https://data.mendeley.com/datasets/8p2hvgdvpn/1 (DOI: 10.17632/8p2hvgdvpn.1)
- **License:** CC BY 4.0
- **Size:** 4.50 GB archive
- **Modality:** 5,078 lab images across 7 classes (hands + 6 chemistry apparatus items).
- **Annotation Format:** Axis-aligned YOLO bounding boxes (`class x_center y_center width height`).
- **Important Note:** These labels are standard axis-aligned rectangles, **not OBB**. They serve as an initial pipeline validator, not as true oriented bounding-box ground truth.

---

## 4. Physics Lab Equipment Image Dataset

- **Source Name:** Physics Lab Equipment Image Dataset
- **Official URL:** https://doi.org/10.6084/m9.figshare.30984658.v1
- **License:** CC BY 4.0
- **Modality:** 5,121 images, 27 instrument classes, YOLO-compatible TXT annotations.

---

## 5. MediaPipe Hand Landmarker Task Asset

- **Source:** Google Edge AI / MediaPipe Solutions
- **Official URL:** https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python
- **License:** Apache 2.0
- **Asset Name:** `hand_landmarker.task`
- **Output:** 21 3D hand keypoints with handedness scores. Pre-trained on ~30,000 real images + synthetic hands. No custom re-training required for baseline hand tracking.

---

## 6. Procedural ArUco Fiducials

- **Source:** OpenCV `cv2.aruco` procedural marker generator
- **Dictionary:** `DICT_4X4_50`
- **Markers:** Generated algorithmically without requiring training data.

