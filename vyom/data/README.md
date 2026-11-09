# VYOM Data Architecture & Provenance

This directory contains configuration, manifests, and local storage structures for public proxy datasets, synthetic demo assets, and derivative annotations.

## Directory Layout

- `external/`: Downloaded public datasets stored in source-specific subdirectories (ignored by Git).
- `manifests/`: Dataset manifests with URLs, licenses, checksums, splits, and class mappings (`dataset_sources.yaml`).
- `raw/`: Unaltered raw extracted frames and video clips (ignored by Git).
- `annotations/`: Original exported annotations before format normalization (ignored by Git).
- `yolo_obb/`: Formatted YOLOv8-OBB dataset for oriented bounding-box training (`train`, `val`, `test` splits for `images/` and `labels/`).
- `SOURCES.md`: Detailed catalog of public dataset sources, licenses, citations, and integration status.

## Configured External Dataset

The primary exocentric procedural dataset is located outside this repository at:
```text
D:\space_eye\datasets\ENIGMA-360
```
- **Selected Split:** Exocentric (`exo`) training and validation subset.
- **Safety Rule:** This path is kept as an external reference in configuration (`VYOM_ENIGMA360_PATH`). Never move, copy, delete, or modify files in this directory while downloads or processes are active.

