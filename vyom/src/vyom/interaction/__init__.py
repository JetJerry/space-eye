"""Visual observation fusion and temporal interaction event extraction.

Planned components (Prompt 8):
- observations.py: Typed observation dataclasses (OBB, hands, ArUco).
- event_extractor.py: Converts raw spatial detections into typed events (e.g., OBJECT_PICKED_UP).
- temporal_filter.py: Consecutive-frame smoothing, debounce, and hysteresis.
- geometry.py: Hand-to-object spatial distance and rotated bounding-box geometry.
"""

