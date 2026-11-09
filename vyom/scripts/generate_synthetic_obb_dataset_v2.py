"""Synthetic procedural OBB demo dataset generator v2 for VYOM apparatus detection.

Implements version 2 improvements over data/yolo_obb/:
1. Strict background separation between train and val using official ENIGMA-360 splits:
   - Train backgrounds derived exclusively from videos in annotations/train.txt.
   - Validation backgrounds derived exclusively from videos in annotations/val.txt.
   - Zero source video overlap; background diversity across 16 distinct videos.
2. Background quality filtering:
   - Laboratory workbench focus with brightness, variance, and edge-density gating.
   - Rejection of chaotic or ambiguous cluttered frames.
3. Enhanced procedural visual realism:
   - Multi-layered RGBA rendering with cylindrical shading gradients.
   - Volumetric liquid columns with meniscus arcs and reagent color variety.
   - Translucent glass/plastic alpha-blending onto workbench background.
   - Specular highlight reflection streaks from laboratory overhead lighting.
   - Precision features: graduations, cap knurling, ANSI A1 well plate notch, rig mounting recesses.
   - Soft directional contact drop shadows grounded on the workbench surface.
4. Mathematical OBB accuracy:
   - Four-corner rotated bounding boxes derived directly from rendered object geometry.
   - Non-overlapping object placement via convex polygon intersection collision checks.
5. Comprehensive dataset manifest and preview generation.

DEMO APPARATUS TAXONOMY:
    0: sample_vial
    1: pipette
    2: multiwell_plate
    3: centrifuge_tube
    4: rig_slot
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

# Class Taxonomy
DEMO_CLASSES = [
    "sample_vial",
    "pipette",
    "multiwell_plate",
    "centrifuge_tube",
    "rig_slot",
]


# ==============================================================================
# Object Patch Renderers (Local RGBA Patches)
# ==============================================================================


def create_sample_vial_patch(
    w: int,
    h: int,
    amber: bool = True,
    liquid_color: Optional[Tuple[int, int, int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Render realistic sample vial on transparent RGBA patch.

    Returns: (patch_rgba, local_corners)
    """
    pad = 30
    pw, ph = w + 2 * pad, h + 2 * pad
    patch = np.zeros((ph, pw, 4), dtype=np.uint8)
    cx, cy = pw // 2, ph // 2
    hw, hh = w // 2, h // 2

    corners = np.array([
        [cx - hw, cy - hh],
        [cx + hw, cy - hh],
        [cx + hw, cy + hh],
        [cx - hw, cy + hh],
    ], dtype=np.float32)

    # 1. Directional soft drop shadow
    shadow = np.zeros((ph, pw), dtype=np.uint8)
    cv2.ellipse(shadow, (cx + 5, cy + 6), (hw + 3, hh + 2), 0, 0, 360, 110, -1)
    shadow = cv2.GaussianBlur(shadow, (15, 15), 0)
    patch[:, :, 3] = (shadow * 0.50).astype(np.uint8)
    patch[:, :, 0] = 15
    patch[:, :, 1] = 18
    patch[:, :, 2] = 22

    cap_h = int(h * 0.22)
    body_y1 = cy - hh + cap_h
    body_y2 = cy + hh

    glass_layer = np.zeros((ph, pw, 4), dtype=np.uint8)
    # 2. Glass cylinder body with cylindrical cosine shading
    for xi in range(-hw, hw + 1):
        x_norm = xi / float(hw)
        intensity = np.sqrt(max(0.05, 1.0 - 0.7 * (x_norm ** 2)))
        if amber:
            # Amber borosilicate glass
            b = int(np.clip(25 * intensity, 0, 255))
            g = int(np.clip(85 * intensity, 0, 255))
            r = int(np.clip(175 * intensity, 0, 255))
            alpha = int(np.clip(175 + 55 * abs(x_norm), 0, 240))
        else:
            # Clear laboratory glass
            b = int(np.clip(200 * intensity, 0, 255))
            g = int(np.clip(210 * intensity, 0, 255))
            r = int(np.clip(215 * intensity, 0, 255))
            alpha = int(np.clip(130 + 70 * abs(x_norm), 0, 220))

        col_x = cx + xi
        glass_layer[body_y1:body_y2, col_x, 0] = b
        glass_layer[body_y1:body_y2, col_x, 1] = g
        glass_layer[body_y1:body_y2, col_x, 2] = r
        glass_layer[body_y1:body_y2, col_x, 3] = alpha

    # 3. Liquid column inside (bottom 40-75% of body)
    if liquid_color is None:
        liquid_color = (10, 120, 220) if amber else (210, 110, 30)  # golden or blue
    fill_ratio = random.uniform(0.40, 0.75)
    liquid_top = int(body_y2 - (body_y2 - body_y1) * fill_ratio)

    lb, lg, lr = liquid_color
    for yi in range(liquid_top, body_y2 - 3):
        for xi in range(-hw + 2, hw - 1):
            x_norm = xi / float(hw)
            factor = 1.0 - 0.3 * (x_norm ** 2)
            glass_layer[yi, cx + xi, 0] = int(lb * factor)
            glass_layer[yi, cx + xi, 1] = int(lg * factor)
            glass_layer[yi, cx + xi, 2] = int(lr * factor)
            glass_layer[yi, cx + xi, 3] = 230

    # Curved meniscus
    cv2.ellipse(glass_layer, (cx, liquid_top), (hw - 3, 2), 0, 0, 360, (lb // 2, lg // 2, lr // 2, 255), 1)

    # 4. White graduation tick marks
    for tick_y in np.linspace(body_y1 + int((body_y2 - body_y1) * 0.25), body_y2 - 8, 4):
        ty = int(tick_y)
        cv2.line(glass_layer, (cx + hw // 4, ty), (cx + hw - 3, ty), (235, 240, 245, 210), 1)

    # 5. Specular highlight streak
    streak_x = cx - int(hw * 0.45)
    for yi in range(body_y1 + 2, body_y2 - 2):
        for sx in range(streak_x - 1, streak_x + 2):
            glass_layer[yi, sx, :3] = 245
            glass_layer[yi, sx, 3] = min(255, glass_layer[yi, sx, 3] + 90)

    # Thickened glass base
    cv2.rectangle(glass_layer, (cx - hw, body_y2 - 4), (cx + hw, body_y2), (20, 65, 130, 240), -1)

    # 6. Molded screw cap
    cap_layer = np.zeros((ph, pw, 4), dtype=np.uint8)
    cap_y1 = cy - hh
    cap_y2 = body_y1
    cap_hw = hw + 1
    cv2.rectangle(cap_layer, (cx - cap_hw, cap_y1), (cx + cap_hw, cap_y2), (215, 220, 225, 255), -1)
    for xi in range(-cap_hw, cap_hw + 1):
        x_norm = xi / float(cap_hw)
        val = int(220 - 45 * (x_norm ** 2))
        cap_layer[cap_y1:cap_y2, cx + xi, :3] = val
    # Knurling ridges
    for kx in range(cx - cap_hw + 3, cx + cap_hw - 2, 4):
        cv2.line(cap_layer, (kx, cap_y1 + 2), (kx, cap_y2 - 2), (160, 165, 170, 255), 1)
        cv2.line(cap_layer, (kx + 1, cap_y1 + 2), (kx + 1, cap_y2 - 2), (245, 250, 255, 255), 1)
    cv2.line(cap_layer, (cx - cap_hw, cap_y1), (cx + cap_hw, cap_y1), (250, 255, 255, 255), 1)
    cv2.line(cap_layer, (cx - cap_hw, cap_y2), (cx + cap_hw, cap_y2), (130, 135, 140, 255), 1)

    # Alpha composite glass and cap onto patch
    for layer in [glass_layer, cap_layer]:
        l_alpha = layer[:, :, 3].astype(np.float32) / 255.0
        p_alpha = patch[:, :, 3].astype(np.float32) / 255.0
        out_alpha = l_alpha + p_alpha * (1.0 - l_alpha)
        for c in range(3):
            patch[:, :, c] = np.where(
                out_alpha > 0,
                (layer[:, :, c] * l_alpha + patch[:, :, c] * p_alpha * (1.0 - l_alpha)) / np.maximum(out_alpha, 1e-4),
                0,
            ).astype(np.uint8)
        patch[:, :, 3] = (out_alpha * 255).astype(np.uint8)

    return patch, corners


def create_pipette_patch(
    w: int,
    h: int,
    button_color: Optional[Tuple[int, int, int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Render realistic ergonomic micropipette on transparent RGBA patch.

    Returns: (patch_rgba, local_corners)
    """
    pad = 30
    pw, ph = w + 2 * pad, h + 2 * pad
    patch = np.zeros((ph, pw, 4), dtype=np.uint8)
    cx, cy = pw // 2, ph // 2
    hw, hh = w // 2, h // 2

    corners = np.array([
        [cx - hw, cy - hh],
        [cx + hw, cy - hh],
        [cx + hw, cy + hh],
        [cx - hw, cy + hh],
    ], dtype=np.float32)

    # 1. Drop shadow
    shadow = np.zeros((ph, pw), dtype=np.uint8)
    cv2.ellipse(shadow, (cx + 5, cy + 6), (hw + 2, hh + 2), 0, 0, 360, 110, -1)
    shadow = cv2.GaussianBlur(shadow, (15, 15), 0)
    patch[:, :, 3] = (shadow * 0.48).astype(np.uint8)
    patch[:, :, 0] = 15
    patch[:, :, 1] = 18
    patch[:, :, 2] = 22

    y_plunger_top = cy - hh
    y_plunger_bot = y_plunger_top + int(h * 0.08)
    y_hook_bot = y_plunger_top + int(h * 0.14)
    y_handle_bot = y_plunger_top + int(h * 0.60)
    y_shaft_bot = y_plunger_top + int(h * 0.82)
    y_tip_bot = cy + hh

    fg = np.zeros((ph, pw, 4), dtype=np.uint8)

    # Plunger button (blue P1000, yellow P200, or orange P20)
    if button_color is None:
        button_color = random.choice([
            (220, 130, 20),   # Blue
            (30, 200, 240),   # Yellow
            (20, 110, 235),   # Orange
        ])
    plunger_hw = max(3, int(hw * 0.65))
    cv2.rectangle(fg, (cx - plunger_hw, y_plunger_top), (cx + plunger_hw, y_plunger_bot), (*button_color, 255), -1)
    cv2.line(fg, (cx - plunger_hw, y_plunger_top), (cx + plunger_hw, y_plunger_top), (250, 250, 255, 255), 1)

    # Finger hook
    hook_hw = hw
    hook_pts = np.array([
        [cx - plunger_hw, y_plunger_bot],
        [cx + hook_hw, y_plunger_bot],
        [cx + hook_hw, y_hook_bot],
        [cx - plunger_hw, y_hook_bot],
    ], dtype=np.int32)
    cv2.fillPoly(fg, [hook_pts], (200, 205, 210, 255))

    # Main handle body (light polymer with longitudinal shading)
    handle_hw = max(4, int(hw * 0.85))
    for xi in range(-handle_hw, handle_hw + 1):
        x_norm = xi / float(handle_hw)
        val = int(210 - 40 * (x_norm ** 2))
        fg[y_hook_bot:y_handle_bot, cx + xi, :3] = val
        fg[y_hook_bot:y_handle_bot, cx + xi, 3] = 255

    # Volume digital readout window
    win_w = max(4, int(handle_hw * 0.9))
    win_h = max(8, int(h * 0.12))
    win_y = y_hook_bot + int((y_handle_bot - y_hook_bot) * 0.25)
    cv2.rectangle(fg, (cx - win_w, win_y), (cx + win_w, win_y + win_h), (35, 40, 45, 255), -1)
    cv2.rectangle(fg, (cx - win_w + 1, win_y + 1), (cx + win_w - 1, win_y + win_h - 1), (240, 245, 250, 255), -1)
    cv2.line(fg, (cx - win_w + 2, win_y + win_h // 3), (cx + win_w - 2, win_y + win_h // 3), (20, 25, 30, 255), 1)
    cv2.line(fg, (cx - win_w + 2, win_y + 2 * win_h // 3), (cx + win_w - 2, win_y + 2 * win_h // 3), (20, 25, 30, 255), 1)

    # Stainless steel nose cone shaft
    shaft_hw = max(3, int(hw * 0.45))
    for xi in range(-shaft_hw, shaft_hw + 1):
        x_norm = xi / float(shaft_hw)
        val = int(175 - 50 * (x_norm ** 2))
        fg[y_handle_bot:y_shaft_bot, cx + xi, :3] = val
        fg[y_handle_bot:y_shaft_bot, cx + xi, 3] = 255
    fg[y_handle_bot:y_shaft_bot, cx - shaft_hw // 2, :3] = 230

    # Disposable translucent pipette tip (tapered conical nozzle)
    tip_pts = np.array([
        [cx - shaft_hw, y_shaft_bot],
        [cx + shaft_hw, y_shaft_bot],
        [cx + 2, y_tip_bot],
        [cx - 2, y_tip_bot],
    ], dtype=np.int32)
    cv2.fillPoly(fg, [tip_pts], (180, 225, 240, 215))
    cv2.polylines(fg, [tip_pts], isClosed=True, color=(140, 185, 200, 240), thickness=1)

    # Combine with shadow
    l_alpha = fg[:, :, 3].astype(np.float32) / 255.0
    p_alpha = patch[:, :, 3].astype(np.float32) / 255.0
    out_alpha = l_alpha + p_alpha * (1.0 - l_alpha)
    for c in range(3):
        patch[:, :, c] = np.where(
            out_alpha > 0,
            (fg[:, :, c] * l_alpha + patch[:, :, c] * p_alpha * (1.0 - l_alpha)) / np.maximum(out_alpha, 1e-4),
            0,
        ).astype(np.uint8)
    patch[:, :, 3] = (out_alpha * 255).astype(np.uint8)

    return patch, corners


def create_multiwell_plate_patch(w: int, h: int) -> Tuple[np.ndarray, np.ndarray]:
    """Render realistic microtiter plate with well grid and skirt details.

    Returns: (patch_rgba, local_corners)
    """
    pad = 30
    pw, ph = w + 2 * pad, h + 2 * pad
    patch = np.zeros((ph, pw, 4), dtype=np.uint8)
    cx, cy = pw // 2, ph // 2
    hw, hh = w // 2, h // 2

    corners = np.array([
        [cx - hw, cy - hh],
        [cx + hw, cy - hh],
        [cx + hw, cy + hh],
        [cx - hw, cy + hh],
    ], dtype=np.float32)

    # 1. Ambient drop shadow
    shadow = np.zeros((ph, pw), dtype=np.uint8)
    cv2.rectangle(shadow, (cx - hw + 4, cy - hh + 4), (cx + hw + 6, cy + hh + 6), 120, -1)
    shadow = cv2.GaussianBlur(shadow, (19, 19), 0)
    patch[:, :, 3] = (shadow * 0.48).astype(np.uint8)
    patch[:, :, 0] = 15
    patch[:, :, 1] = 18
    patch[:, :, 2] = 22

    fg = np.zeros((ph, pw, 4), dtype=np.uint8)

    # 2. Outer skirt
    cv2.rectangle(fg, (cx - hw, cy - hh), (cx + hw, cy + hh), (70, 75, 80, 240), -1)

    # A1 corner index chamfer notch (standard ANSI/SBS feature)
    notch_sz = 8
    notch_pts = np.array([
        [cx - hw, cy - hh + notch_sz],
        [cx - hw + notch_sz, cy - hh],
        [cx - hw, cy - hh],
    ], dtype=np.int32)
    cv2.fillPoly(fg, [notch_pts], (0, 0, 0, 0))

    cv2.rectangle(fg, (cx - hw, cy - hh), (cx + hw, cy + hh), (130, 138, 145, 255), 2)
    cv2.rectangle(fg, (cx - hw + 4, cy - hh + 4), (cx + hw - 4, cy + hh - 4), (55, 60, 65, 255), 1)

    # 3. Well grid (6 rows x 8 cols)
    cols = 8
    rows = 6
    margin_x = int(w * 0.12)
    margin_y = int(h * 0.14)
    grid_w = w - 2 * margin_x
    grid_h = h - 2 * margin_y

    xs = np.linspace(cx - hw + margin_x, cx + hw - margin_x, cols)
    ys = np.linspace(cy - hh + margin_y, cy + hh - margin_y, rows)
    well_rad = max(2, int(min(grid_w / cols, grid_h / rows) * 0.38))

    for y in ys:
        for x in xs:
            pt = (int(x), int(y))
            cv2.circle(fg, pt, well_rad, (160, 168, 175, 255), 1)
            cv2.circle(fg, pt, well_rad - 1, (30, 32, 35, 255), -1)
            cv2.ellipse(fg, pt, (well_rad - 1, well_rad - 1), 0, 200, 280, (220, 225, 230, 255), 1)

    # Combine with shadow
    l_alpha = fg[:, :, 3].astype(np.float32) / 255.0
    p_alpha = patch[:, :, 3].astype(np.float32) / 255.0
    out_alpha = l_alpha + p_alpha * (1.0 - l_alpha)
    for c in range(3):
        patch[:, :, c] = np.where(
            out_alpha > 0,
            (fg[:, :, c] * l_alpha + patch[:, :, c] * p_alpha * (1.0 - l_alpha)) / np.maximum(out_alpha, 1e-4),
            0,
        ).astype(np.uint8)
    patch[:, :, 3] = (out_alpha * 255).astype(np.uint8)

    return patch, corners


def create_centrifuge_tube_patch(
    w: int,
    h: int,
    cap_color: Optional[Tuple[int, int, int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Render realistic conical centrifuge tube with screw cap.

    Returns: (patch_rgba, local_corners)
    """
    pad = 30
    pw, ph = w + 2 * pad, h + 2 * pad
    patch = np.zeros((ph, pw, 4), dtype=np.uint8)
    cx, cy = pw // 2, ph // 2
    hw, hh = w // 2, h // 2

    corners = np.array([
        [cx - hw, cy - hh],
        [cx + hw, cy - hh],
        [cx + hw, cy + hh],
        [cx - hw, cy + hh],
    ], dtype=np.float32)

    # 1. Drop shadow
    shadow = np.zeros((ph, pw), dtype=np.uint8)
    cv2.ellipse(shadow, (cx + 5, cy + 6), (hw + 3, hh + 2), 0, 0, 360, 110, -1)
    shadow = cv2.GaussianBlur(shadow, (15, 15), 0)
    patch[:, :, 3] = (shadow * 0.50).astype(np.uint8)
    patch[:, :, 0] = 15
    patch[:, :, 1] = 18
    patch[:, :, 2] = 22

    fg = np.zeros((ph, pw, 4), dtype=np.uint8)

    cap_h = int(h * 0.18)
    y_cap_top = cy - hh
    y_cap_bot = y_cap_top + cap_h
    y_taper_start = y_cap_top + int(h * 0.72)
    y_cone_tip = cy + hh

    # 2. Conical tube polygon
    body_pts = np.array([
        [cx - hw, y_cap_bot],
        [cx + hw, y_cap_bot],
        [cx + hw, y_taper_start],
        [cx + 3, y_cone_tip],
        [cx - 3, y_cone_tip],
        [cx - hw, y_taper_start],
    ], dtype=np.int32)

    cv2.fillPoly(fg, [body_pts], (205, 215, 225, 200))

    # Cylindrical shading
    for xi in range(-hw + 1, hw):
        x_norm = xi / float(hw)
        val = int(220 - 45 * (x_norm ** 2))
        for yi in range(y_cap_bot, y_taper_start):
            fg[yi, cx + xi, :3] = val

    # 3. Liquid filling
    liquid_top = int(y_cap_bot + (y_cone_tip - y_cap_bot) * random.uniform(0.40, 0.65))
    liquid_mask = np.zeros((ph, pw), dtype=np.uint8)
    cv2.fillPoly(liquid_mask, [body_pts], 255)
    liquid_mask[:liquid_top, :] = 0
    # Pink culture media / reagent
    fg[liquid_mask > 0, 0] = 180
    fg[liquid_mask > 0, 1] = 50
    fg[liquid_mask > 0, 2] = 220
    fg[liquid_mask > 0, 3] = 230

    cv2.ellipse(fg, (cx, liquid_top), (hw - 2, 2), 0, 0, 360, (140, 30, 180, 255), 1)

    # 4. White frosted labeling patch
    label_w = max(4, int(hw * 0.7))
    label_h = int((y_taper_start - y_cap_bot) * 0.40)
    label_y = y_cap_bot + int((y_taper_start - y_cap_bot) * 0.15)
    cv2.rectangle(fg, (cx - label_w, label_y), (cx + label_w, label_y + label_h), (245, 250, 255, 230), -1)

    # 5. Volumetric graduation tick marks
    for tick_y in np.linspace(label_y + 4, y_taper_start, 4):
        ty = int(tick_y)
        cv2.line(fg, (cx + hw // 3, ty), (cx + hw - 2, ty), (40, 45, 50, 240), 1)

    fg[y_cap_bot + 2:y_taper_start, cx - int(hw * 0.4), :3] = 250
    fg[y_cap_bot + 2:y_taper_start, cx - int(hw * 0.4), 3] = 245
    cv2.polylines(fg, [body_pts], isClosed=True, color=(140, 150, 160, 255), thickness=1)

    # 6. Flanged screw cap
    if cap_color is None:
        cap_color = random.choice([
            (20, 25, 220),   # Vibrant Red
            (20, 130, 245),  # Orange
            (220, 110, 20),  # Blue
        ])
    cap_hw = hw + 2
    cap_pts = np.array([
        [cx - cap_hw, y_cap_top],
        [cx + cap_hw, y_cap_top],
        [cx + cap_hw, y_cap_bot],
        [cx - cap_hw, y_cap_bot],
    ], dtype=np.int32)
    cv2.fillPoly(fg, [cap_pts], (*cap_color, 255))
    for kx in range(cx - cap_hw + 2, cx + cap_hw - 1, 4):
        cv2.line(fg, (kx, y_cap_top + 1), (kx, y_cap_bot - 1), (10, 15, 160, 255), 1)
        cv2.line(fg, (kx + 1, y_cap_top + 1), (kx + 1, y_cap_bot - 1), (80, 85, 255, 255), 1)
    cv2.line(fg, (cx - cap_hw, y_cap_top), (cx + cap_hw, y_cap_top), (120, 130, 255, 255), 1)
    cv2.line(fg, (cx - cap_hw, y_cap_bot), (cx + cap_hw, y_cap_bot), (10, 10, 130, 255), 1)

    # Combine with shadow
    l_alpha = fg[:, :, 3].astype(np.float32) / 255.0
    p_alpha = patch[:, :, 3].astype(np.float32) / 255.0
    out_alpha = l_alpha + p_alpha * (1.0 - l_alpha)
    for c in range(3):
        patch[:, :, c] = np.where(
            out_alpha > 0,
            (fg[:, :, c] * l_alpha + patch[:, :, c] * p_alpha * (1.0 - l_alpha)) / np.maximum(out_alpha, 1e-4),
            0,
        ).astype(np.uint8)
    patch[:, :, 3] = (out_alpha * 255).astype(np.uint8)

    return patch, corners


def create_rig_slot_patch(w: int, h: int) -> Tuple[np.ndarray, np.ndarray]:
    """Render precision machined experiment rig mounting receptacle.

    Returns: (patch_rgba, local_corners)
    """
    pad = 30
    pw, ph = w + 2 * pad, h + 2 * pad
    patch = np.zeros((ph, pw, 4), dtype=np.uint8)
    cx, cy = pw // 2, ph // 2
    hw, hh = w // 2, h // 2

    corners = np.array([
        [cx - hw, cy - hh],
        [cx + hw, cy - hh],
        [cx + hw, cy + hh],
        [cx - hw, cy + hh],
    ], dtype=np.float32)

    fg = np.zeros((ph, pw, 4), dtype=np.uint8)

    # 1. Machined aluminum / anodized plate
    cv2.rectangle(fg, (cx - hw, cy - hh), (cx + hw, cy + hh), (60, 65, 72, 255), -1)

    cv2.line(fg, (cx - hw, cy - hh), (cx + hw, cy - hh), (170, 178, 185, 255), 2)
    cv2.line(fg, (cx - hw, cy - hh), (cx - hw, cy + hh), (140, 148, 155, 255), 2)
    cv2.line(fg, (cx + hw, cy - hh), (cx + hw, cy + hh), (35, 38, 42, 255), 2)
    cv2.line(fg, (cx - hw, cy + hh), (cx + hw, cy + hh), (30, 32, 35, 255), 2)

    # 2. Deep recessed pocket cavity
    pocket_inset = max(6, int(w * 0.14))
    px1, py1 = cx - hw + pocket_inset, cy - hh + pocket_inset
    px2, py2 = cx + hw - pocket_inset, cy + hh - pocket_inset
    cv2.rectangle(fg, (px1, py1), (px2, py2), (18, 20, 24, 255), -1)

    cv2.line(fg, (px1, py1), (px2, py1), (10, 12, 14, 255), 2)
    cv2.line(fg, (px1, py1), (px1, py2), (12, 14, 16, 255), 2)
    cv2.line(fg, (px2, py1), (px2, py2), (80, 85, 92, 255), 1)
    cv2.line(fg, (px1, py2), (px2, py2), (90, 95, 102, 255), 1)

    # 3. Corner socket head cap screws
    screw_offset = pocket_inset // 2 + 1
    screw_pts = [
        (cx - hw + screw_offset, cy - hh + screw_offset),
        (cx + hw - screw_offset, cy - hh + screw_offset),
        (cx + hw - screw_offset, cy + hh - screw_offset),
        (cx - hw + screw_offset, cy + hh - screw_offset),
    ]
    for pt in screw_pts:
        cv2.circle(fg, pt, 3, (25, 28, 30, 255), -1)
        cv2.circle(fg, pt, 3, (120, 128, 135, 255), 1)
        cv2.circle(fg, pt, 1, (160, 170, 180, 255), -1)

    patch = fg
    return patch, corners


# ==============================================================================
# Transformation, Collision & Compositing
# ==============================================================================


def rotate_and_composite(
    canvas: np.ndarray,
    patch: np.ndarray,
    local_corners: np.ndarray,
    center: Tuple[int, int],
    angle_deg: float,
) -> Optional[np.ndarray]:
    """Rotate local RGBA patch and composite onto BGR canvas.

    Returns: rotated 4 corners in canvas coordinates, or None if invalid.
    """
    ph, pw = patch.shape[:2]
    patch_cx, patch_cy = pw // 2, ph // 2
    cx, cy = center

    # Rotate patch using OpenCV
    rot_mat = cv2.getRotationMatrix2D((patch_cx, patch_cy), angle_deg, 1.0)
    rotated_patch = cv2.warpAffine(
        patch,
        rot_mat,
        (pw, ph),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )

    # Transform 4 local corners using same rotation matrix
    rad = np.radians(angle_deg)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    rel_corners = local_corners - np.array([patch_cx, patch_cy], dtype=np.float32)

    rot_c = np.zeros_like(rel_corners)
    rot_c[:, 0] = cos_a * rel_corners[:, 0] + sin_a * rel_corners[:, 1]
    rot_c[:, 1] = -sin_a * rel_corners[:, 0] + cos_a * rel_corners[:, 1]

    canvas_corners = rot_c + np.array([cx, cy], dtype=np.float32)

    ch, cw = canvas.shape[:2]
    # Check boundaries (margin of 6px from edge)
    if np.any(canvas_corners < 6) or np.any(canvas_corners[:, 0] > cw - 6) or np.any(canvas_corners[:, 1] > ch - 6):
        return None

    # Composite onto canvas
    top_left_x = cx - patch_cx
    top_left_y = cy - patch_cy

    x1, y1 = max(0, top_left_x), max(0, top_left_y)
    x2, y2 = min(cw, top_left_x + pw), min(ch, top_left_y + ph)

    if x1 >= x2 or y1 >= y2:
        return None

    fg_x1 = x1 - top_left_x
    fg_y1 = y1 - top_left_y
    fg_x2 = fg_x1 + (x2 - x1)
    fg_y2 = fg_y1 + (y2 - y1)

    fg_crop = rotated_patch[fg_y1:fg_y2, fg_x1:fg_x2]
    bg_crop = canvas[y1:y2, x1:x2]

    alpha = fg_crop[:, :, 3].astype(np.float32) / 255.0
    alpha = np.expand_dims(alpha, axis=2)

    fg_rgb = fg_crop[:, :, :3].astype(np.float32)
    bg_rgb = bg_crop.astype(np.float32)

    blended = fg_rgb * alpha + bg_rgb * (1.0 - alpha)
    canvas[y1:y2, x1:x2] = blended.astype(np.uint8)

    return canvas_corners


def check_collision(new_corners: np.ndarray, existing_corners_list: List[np.ndarray]) -> bool:
    """Check if new oriented bounding box collides/severely overlaps with existing objects.

    Returns: True if collision detected, False otherwise.
    """
    new_pts = new_corners.astype(np.float32)
    new_area = cv2.contourArea(new_pts)
    if new_area <= 0:
        return True

    for ex_corners in existing_corners_list:
        ex_pts = ex_corners.astype(np.float32)
        ex_area = cv2.contourArea(ex_pts)
        if ex_area <= 0:
            continue

        ret, _ = cv2.intersectConvexConvex(new_pts, ex_pts)
        # Reject if overlap area exceeds 5% of either box (prevent severe occlusion)
        if ret > 0.05 * min(new_area, ex_area):
            return True

    return False


# ==============================================================================
# Authentic Background Extraction & Quality Gating
# ==============================================================================


def filter_background_quality(crop: np.ndarray) -> Tuple[bool, str]:
    """Quality filter to accept only realistic, clean laboratory workbench regions."""
    m = float(crop.mean())
    s = float(crop.std())
    if m < 70.0 or m > 145.0:
        return False, f"Abnormal brightness (mean={m:.1f})"
    if s < 25.0 or s > 60.0:
        return False, f"Abnormal contrast (std={s:.1f})"

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = float(np.count_nonzero(edges)) / float(edges.size)
    if edge_ratio < 0.015 or edge_ratio > 0.060:
        return False, f"Excessive clutter or flat image (edge_ratio={edge_ratio:.3f})"

    return True, f"OK (m={m:.1f}, s={s:.1f}, e={edge_ratio:.3f})"


def extract_split_backgrounds(
    enigma_dir: Path,
    video_ids: List[str],
    split_name: str,
    target_count: int,
    img_size: int = 640,
) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
    """Extract diversified, quality-filtered workbench backgrounds from specified video IDs."""
    video_dir = enigma_dir / "data" / "videos"
    backgrounds: List[np.ndarray] = []
    manifest_entries: List[Dict[str, Any]] = []

    print(f"Extracting {split_name} backgrounds from {len(video_ids)} videos: {video_ids}")

    # Determine frames per video
    per_video = max(3, target_count // len(video_ids) + 1)

    for vid in video_ids:
        vid_path = video_dir / vid / f"{vid}_third_person.mp4"
        if not vid_path.exists():
            print(f"  Warning: Video {vid_path} not found. Skipping.")
            continue

        cap = cv2.VideoCapture(str(vid_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0

        if total_frames <= 0:
            cap.release()
            continue

        # Sample at diverse points (avoid first and last 5% where recording starts/ends)
        sample_fractions = np.linspace(0.08, 0.90, per_video + 4)
        for pct in sample_fractions:
            if len(backgrounds) >= target_count and len([m for m in manifest_entries if m["video_id"] == vid]) >= per_video:
                break

            fn = int(total_frames * pct)
            cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            h, w = frame.shape[:2]
            # Center-workbench crop: 360x360 from middle workbench area
            crop_sz = 360
            y1 = 10
            y2 = min(h, y1 + crop_sz)
            x1 = max(0, (w - crop_sz) // 2)
            x2 = min(w, x1 + crop_sz)

            crop = frame[y1:y2, x1:x2]
            ok, reason = filter_background_quality(crop)
            if not ok:
                continue

            # High-quality resize to 640x640
            resized = cv2.resize(crop, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
            backgrounds.append(resized)

            manifest_entries.append({
                "video_id": vid,
                "frame_number": fn,
                "timestamp_sec": round(fn / fps, 2),
                "split": split_name,
                "crop_rect": [x1, y1, x2, y2],
                "mean_brightness": round(float(crop.mean()), 1),
                "std_contrast": round(float(crop.std()), 1),
            })

        cap.release()

    print(f"Extracted {len(backgrounds)} valid backgrounds for {split_name} split.")
    return backgrounds, manifest_entries


# ==============================================================================
# Dataset Generation Pipeline
# ==============================================================================


def generate_dataset_v2(
    output_dir: Path,
    enigma_dir: Path,
    num_train: int = 240,
    num_val: int = 60,
    img_size: int = 640,
    seed: int = 42,
) -> None:
    """Generate complete improved synthetic dataset v2 with strict background separation."""
    output_dir = output_dir.resolve()
    images_train = output_dir / "images" / "train"
    images_val = output_dir / "images" / "val"
    labels_train = output_dir / "labels" / "train"
    labels_val = output_dir / "labels" / "val"

    for d in [images_train, images_val, labels_train, labels_val]:
        d.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    np.random.seed(seed)

    # 1. Read official train.txt and val.txt
    train_txt = enigma_dir / "annotations" / "train.txt"
    val_txt = enigma_dir / "annotations" / "val.txt"

    train_all = train_txt.read_text().splitlines() if train_txt.exists() else []
    val_all = val_txt.read_text().splitlines() if val_txt.exists() else []

    # Selected diverse videos from official partitions
    selected_train_vids = ["103", "104", "105", "109", "110", "111", "112", "113", "114", "124"]
    selected_val_vids = ["100", "101", "102", "115", "116", "127"]

    # Verify partitions
    for v in selected_train_vids:
        assert v in train_all, f"Violation: video {v} is not in train.txt!"
    for v in selected_val_vids:
        assert v in val_all, f"Violation: video {v} is not in val.txt!"
    overlap = set(selected_train_vids).intersection(set(selected_val_vids))
    assert len(overlap) == 0, f"Violation: overlap between train and val videos: {overlap}"

    # 2. Extract authentic backgrounds
    train_bgs, train_bg_manifest = extract_split_backgrounds(
        enigma_dir, selected_train_vids, "train", target_count=70, img_size=img_size
    )
    val_bgs, val_bg_manifest = extract_split_backgrounds(
        enigma_dir, selected_val_vids, "val", target_count=35, img_size=img_size
    )

    all_bg_manifest = train_bg_manifest + val_bg_manifest

    # Size configurations per class (w, h)
    class_sizes = {
        0: (34, 85),    # sample_vial
        1: (26, 140),   # pipette
        2: (120, 85),   # multiwell_plate
        3: (36, 100),   # centrifuge_tube
        4: (75, 75),    # rig_slot
    }

    # Tracking metrics
    class_counts: Dict[str, Dict[int, int]] = {
        "train": {i: 0 for i in range(len(DEMO_CLASSES))},
        "val": {i: 0 for i in range(len(DEMO_CLASSES))},
    }

    splits = [
        ("train", num_train, train_bgs, images_train, labels_train),
        ("val", num_val, val_bgs, images_val, labels_val),
    ]

    total_instances = 0
    generated_images_meta: List[Dict[str, Any]] = []

    # Global class rotator to maintain balanced distribution
    global_class_idx = 0

    for split_name, count, bg_pool, img_dir, lbl_dir in splits:
        print(f"\nGenerating {count} images for {split_name} split...")

        for i in range(count):
            bg = random.choice(bg_pool).copy()

            # Random subtle brightness/contrast adjustment
            alpha = random.uniform(0.92, 1.08)
            beta = random.randint(-10, 10)
            canvas = cv2.convertScaleAbs(bg, alpha=alpha, beta=beta)

            # Number of objects: 2 to 4 objects per image
            num_objects = random.randint(2, 4)
            placed_corners: List[np.ndarray] = []
            labels_lines: List[str] = []
            img_objects: List[Dict[str, Any]] = []

            for _ in range(num_objects):
                # Round-robin selection with slight randomization to maintain balance
                class_id = global_class_idx % len(DEMO_CLASSES)
                global_class_idx += 1

                base_w, base_h = class_sizes[class_id]
                scale = random.uniform(0.85, 1.15)
                w = int(base_w * scale)
                h = int(base_h * scale)

                # Create class-specific patch
                if class_id == 0:
                    patch, local_corners = create_sample_vial_patch(w, h, amber=random.choice([True, False]))
                elif class_id == 1:
                    patch, local_corners = create_pipette_patch(w, h)
                elif class_id == 2:
                    patch, local_corners = create_multiwell_plate_patch(w, h)
                elif class_id == 3:
                    patch, local_corners = create_centrifuge_tube_patch(w, h)
                else:
                    patch, local_corners = create_rig_slot_patch(w, h)

                # Attempt non-overlapping placement
                placed = False
                for _attempt in range(25):
                    angle = random.uniform(0.0, 360.0)
                    cx = random.randint(65, img_size - 65)
                    cy = random.randint(65, img_size - 65)

                    # Compute candidate rotated corners
                    rad = np.radians(angle)
                    cos_a, sin_a = np.cos(rad), np.sin(rad)
                    ph, pw = patch.shape[:2]
                    rel_c = local_corners - np.array([pw // 2, ph // 2], dtype=np.float32)
                    cand_c = np.zeros_like(rel_c)
                    cand_c[:, 0] = cos_a * rel_c[:, 0] + sin_a * rel_c[:, 1] + cx
                    cand_c[:, 1] = -sin_a * rel_c[:, 0] + cos_a * rel_c[:, 1] + cy

                    # Check collision with already placed objects
                    if check_collision(cand_c, placed_corners):
                        continue

                    # Safe to render and composite
                    final_corners = rotate_and_composite(canvas, patch, local_corners, (cx, cy), angle)
                    if final_corners is not None:
                        placed_corners.append(final_corners)
                        c_norm = final_corners.copy()
                        c_norm[:, 0] = np.clip(c_norm[:, 0] / img_size, 0.0, 1.0)
                        c_norm[:, 1] = np.clip(c_norm[:, 1] / img_size, 0.0, 1.0)

                        line = f"{class_id} " + " ".join(f"{coord:.6f}" for pt in c_norm for coord in pt)
                        labels_lines.append(line)
                        class_counts[split_name][class_id] += 1
                        total_instances += 1
                        img_objects.append({
                            "class_id": class_id,
                            "class_name": DEMO_CLASSES[class_id],
                            "center": [cx, cy],
                            "angle_deg": round(angle, 1),
                            "corners_px": final_corners.tolist(),
                        })
                        placed = True
                        break

            # Save image and label file
            stem = f"syn_v2_{split_name}_{i:04d}"
            img_file = img_dir / f"{stem}.jpg"
            lbl_file = lbl_dir / f"{stem}.txt"

            cv2.imwrite(str(img_file), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            lbl_file.write_text("\n".join(labels_lines) + "\n", encoding="utf-8")

            generated_images_meta.append({
                "split": split_name,
                "image_file": img_file.name,
                "label_file": lbl_file.name,
                "object_count": len(labels_lines),
                "objects": img_objects,
            })

    # 3. Generate dataset.yaml
    yaml_content = f"""# Ultralytics YOLOv8-OBB dataset configuration for VYOM demo apparatus v2
path: {output_dir.as_posix()}
train: images/train
val: images/val
test: images/val

names:
  0: sample_vial
  1: pipette
  2: multiwell_plate
  3: centrifuge_tube
  4: rig_slot
"""
    yaml_path = output_dir / "dataset.yaml"
    yaml_path.write_text(yaml_content, encoding="utf-8")
    print(f"\nGenerated dataset YAML: {yaml_path}")

    # 4. Save comprehensive manifest JSON
    manifest_data = {
        "dataset_name": "VYOM Procedural Apparatus Demo Dataset v2 (OBB)",
        "version": "2.0.0",
        "taxonomy": DEMO_CLASSES,
        "splits": {
            "train": {
                "image_count": num_train,
                "source_videos": selected_train_vids,
                "class_counts": {DEMO_CLASSES[k]: v for k, v in class_counts["train"].items()},
            },
            "val": {
                "image_count": num_val,
                "source_videos": selected_val_vids,
                "class_counts": {DEMO_CLASSES[k]: v for k, v in class_counts["val"].items()},
            },
        },
        "background_separation": {
            "train_source_videos": selected_train_vids,
            "val_source_videos": selected_val_vids,
            "overlap_video_count": 0,
            "total_extracted_backgrounds": len(all_bg_manifest),
            "backgrounds": all_bg_manifest,
        },
        "total_images": num_train + num_val,
        "total_instances": total_instances,
    }

    manifest_json_path = output_dir / "manifest.json"
    manifest_json_path.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")
    print(f"Generated dataset manifest: {manifest_json_path}")

    # 5. Save Human-readable DATASET_MANIFEST.md
    md_content = f"""# VYOM YOLOv8-OBB Dataset Manifest (v2)

## 1. Overview
- **Dataset Name:** VYOM Procedural Demo Apparatus Dataset v2
- **Format:** YOLOv8-OBB (`class_id x1 y1 x2 y2 x3 y3 x4 y4`, normalized $[0, 1]$)
- **Target Size:** {img_size} $\\times$ {img_size}
- **Total Images:** {num_train + num_val} ({num_train} train, {num_val} val)
- **Total Annotated Instances:** {total_instances}

## 2. Background Source Video Separation
| Split | Image Count | Background Source Videos (ENIGMA-360) | Video Partition Origin | Overlap Check |
|---|---|---|---|---|
| **Train** | {num_train} | {', '.join(selected_train_vids)} | `annotations/train.txt` | **0 overlap (Disjoint)** |
| **Val** | {num_val} | {', '.join(selected_val_vids)} | `annotations/val.txt` | **0 overlap (Disjoint)** |

> [!IMPORTANT]
> Train and validation splits use completely disjoint source videos. No background frame or video from the training partition appears in the validation partition.

## 3. Class Taxonomy & Distribution
| Class ID | Class Name | Train Instances | Val Instances | Total Instances | Percentage |
|---|---|---|---|---|---|
"""
    for cid, cname in enumerate(DEMO_CLASSES):
        tr_c = class_counts["train"][cid]
        val_c = class_counts["val"][cid]
        tot = tr_c + val_c
        pct = (tot / total_instances * 100.0) if total_instances > 0 else 0.0
        md_content += f"| {cid} | `{cname}` | {tr_c} | {val_c} | {tot} | {pct:.1f}% |\n"

    md_content += f"""| **Total** | **All 5 Classes** | **{sum(class_counts['train'].values())}** | **{sum(class_counts['val'].values())}** | **{total_instances}** | **100.0%** |

## 4. Visual Improvements in Version 2
1. **Multi-layer RGBA Procedural Rendering:**
   - Shading gradients (cylindrical lighting).
   - Glass transparency and refraction alpha-blending onto real workbench wood/laminate textures.
   - Reagent liquid columns with curved menisci.
   - Volumetric graduations and molded knurled cap ribs.
   - ANSI/SBS microplate A1 corner notch and well bottom shadows.
   - Machined aluminum rig receptacle with counterbored socket screws.
2. **Contact Drop Shadows:**
   - Directional soft contact shadows beneath objects, realistically grounding them onto the workbench.
3. **Collision Detection & Occlusion Prevention:**
   - Non-overlapping object placement using `cv2.intersectConvexConvex`.
   - Prevents unannotated mutual occlusions and ensures 100% boundary tightness.
"""
    manifest_md_path = output_dir / "DATASET_MANIFEST.md"
    manifest_md_path.write_text(md_content, encoding="utf-8")
    print(f"Generated manifest markdown: {manifest_md_path}")


# ==============================================================================
# Visual Preview Generator (PART 6)
# ==============================================================================


def generate_visual_previews(dataset_dir: Path, max_previews_per_split: int = 15) -> None:
    """Generate annotated visual preview images with OBB polygons, direction vectors, and class pills."""
    previews_dir = dataset_dir / "previews"
    prev_train = previews_dir / "train"
    prev_val = previews_dir / "val"
    prev_train.mkdir(parents=True, exist_ok=True)
    prev_val.mkdir(parents=True, exist_ok=True)

    palette = [
        (0, 230, 115),   # Green - sample_vial
        (255, 140, 0),   # Cyan - pipette
        (0, 165, 255),   # Orange - multiwell_plate
        (220, 60, 220),  # Violet - centrifuge_tube
        (0, 215, 255),   # Yellow - rig_slot
    ]

    for split_name, out_dir in [("train", prev_train), ("val", prev_val)]:
        img_dir = dataset_dir / "images" / split_name
        lbl_dir = dataset_dir / "labels" / split_name
        img_files = sorted(img_dir.glob("*.jpg"))[:max_previews_per_split]

        for img_p in img_files:
            lbl_p = lbl_dir / f"{img_p.stem}.txt"
            if not lbl_p.exists():
                continue

            img = cv2.imread(str(img_p))
            if img is None:
                continue

            h, w = img.shape[:2]
            lines = lbl_p.read_text(encoding="utf-8").splitlines()

            for line in lines:
                parts = line.strip().split()
                if len(parts) != 9:
                    continue
                cid = int(parts[0])
                coords = [float(x) for x in parts[1:]]
                pts = np.array(coords, dtype=np.float32).reshape(4, 2)
                pts[:, 0] *= w
                pts[:, 1] *= h
                pts_int = pts.astype(np.int32)

                color = palette[cid % len(palette)]

                # Draw polygon outline
                cv2.polylines(img, [pts_int], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)

                # Draw corner vertices
                for idx, pt in enumerate(pts_int):
                    cv2.circle(img, tuple(pt), 3, color, -1)

                # Draw orientation vector (p0 -> p1)
                mid_top = ((pts[0] + pts[1]) / 2.0).astype(np.int32)
                center = pts.mean(axis=0).astype(np.int32)
                cv2.arrowedLine(img, tuple(center), tuple(mid_top), (255, 255, 255), 1, tipLength=0.25)

                # Label text pill
                cname = DEMO_CLASSES[cid]
                text = f"{cid}:{cname}"
                cv2.putText(
                    img,
                    text,
                    (pts_int[0, 0] + 3, max(15, pts_int[0, 1] - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (0, 0, 0),
                    2,
                    lineType=cv2.LINE_AA,
                )
                cv2.putText(
                    img,
                    text,
                    (pts_int[0, 0] + 3, max(15, pts_int[0, 1] - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (255, 255, 255),
                    1,
                    lineType=cv2.LINE_AA,
                )

            preview_path = out_dir / f"{img_p.stem}_preview.jpg"
            cv2.imwrite(str(preview_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    print(f"Generated visual previews under {previews_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic YOLOv8-OBB demo dataset v2.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="d:/space_eye/vyom/data/yolo_obb_v2",
        help="Target output directory for dataset v2.",
    )
    parser.add_argument(
        "--enigma-dir",
        type=str,
        default="D:/space_eye/datasets/ENIGMA-360",
        help="Root path to ENIGMA-360 dataset.",
    )
    parser.add_argument("--num-train", type=int, default=240, help="Number of training images.")
    parser.add_argument("--num-val", type=int, default=60, help="Number of validation images.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    enigma_dir = Path(args.enigma_dir)

    generate_dataset_v2(
        output_dir=out_dir,
        enigma_dir=enigma_dir,
        num_train=args.num_train,
        num_val=args.num_val,
        seed=args.seed,
    )

    generate_visual_previews(out_dir, max_previews_per_split=15)


if __name__ == "__main__":
    main()

