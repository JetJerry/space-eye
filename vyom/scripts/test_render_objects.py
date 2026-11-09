"""Test and visual verification script for improved procedural apparatus rendering.
Renders all five classes with realistic shading, transparency, highlights, shadows,
and verifies OBB geometric correspondence.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np


def create_sample_vial_patch(w: int, h: int) -> Tuple[np.ndarray, np.ndarray]:
    """Render realistic sample vial on transparent RGBA patch.
    Returns: (patch_rgba, local_corners)
    """
    pad = 30
    pw, ph = w + 2 * pad, h + 2 * pad
    patch = np.zeros((ph, pw, 4), dtype=np.uint8)
    cx, cy = pw // 2, ph // 2
    hw, hh = w // 2, h // 2

    # Local bounding box corners (unrotated)
    corners = np.array([
        [cx - hw, cy - hh],
        [cx + hw, cy - hh],
        [cx + hw, cy + hh],
        [cx - hw, cy + hh],
    ], dtype=np.float32)

    # 1. Soft contact drop shadow (projected to bottom-right)
    shadow = np.zeros((ph, pw), dtype=np.uint8)
    cv2.ellipse(shadow, (cx + 5, cy + 6), (hw + 3, hh + 2), 0, 0, 360, 110, -1)
    shadow = cv2.GaussianBlur(shadow, (15, 15), 0)
    patch[:, :, 3] = np.maximum(patch[:, :, 3], (shadow * 0.55).astype(np.uint8))
    patch[:, :, 0] = np.where(shadow > 0, 15, patch[:, :, 0])
    patch[:, :, 1] = np.where(shadow > 0, 18, patch[:, :, 1])
    patch[:, :, 2] = np.where(shadow > 0, 22, patch[:, :, 2])

    cap_h = int(h * 0.22)
    body_y1 = cy - hh + cap_h
    body_y2 = cy + hh

    # 2. Glass cylinder body with cylindrical shading and amber/clear tint
    glass_layer = np.zeros((ph, pw, 4), dtype=np.uint8)
    for xi in range(-hw, hw + 1):
        x_norm = xi / float(hw)
        # Cosine cylindrical shading factor
        intensity = np.sqrt(max(0.05, 1.0 - 0.7 * (x_norm ** 2)))
        # Amber glass base color: BGR (25, 85, 175)
        b = int(np.clip(25 * intensity, 0, 255))
        g = int(np.clip(85 * intensity, 0, 255))
        r = int(np.clip(175 * intensity, 0, 255))
        alpha = int(np.clip(180 + 50 * abs(x_norm), 0, 240))  # higher opacity at edges (fresnel)

        col_x = cx + xi
        glass_layer[body_y1:body_y2, col_x, 0] = b
        glass_layer[body_y1:body_y2, col_x, 1] = g
        glass_layer[body_y1:body_y2, col_x, 2] = r
        glass_layer[body_y1:body_y2, col_x, 3] = alpha

    # 3. Liquid column inside (bottom 60% of body)
    liquid_top = int(body_y2 - (body_y2 - body_y1) * 0.65)
    for yi in range(liquid_top, body_y2 - 3):
        for xi in range(-hw + 2, hw - 1):
            x_norm = xi / float(hw)
            factor = 1.0 - 0.3 * (x_norm ** 2)
            # Golden amber liquid: BGR (10, 120, 220)
            glass_layer[yi, cx + xi, 0] = int(10 * factor)
            glass_layer[yi, cx + xi, 1] = int(120 * factor)
            glass_layer[yi, cx + xi, 2] = int(220 * factor)
            glass_layer[yi, cx + xi, 3] = 235

    # Curved meniscus line at liquid surface
    cv2.ellipse(glass_layer, (cx, liquid_top), (hw - 3, 2), 0, 0, 360, (5, 90, 180, 255), 1)

    # 4. White graduation tick marks
    for tick_y in np.linspace(body_y1 + int((body_y2 - body_y1) * 0.25), body_y2 - 10, 4):
        ty = int(tick_y)
        cv2.line(glass_layer, (cx + hw // 4, ty), (cx + hw - 3, ty), (230, 235, 240, 220), 1)

    # 5. Specular highlight streak (reflection of lab fluorescent light)
    streak_x = cx - int(hw * 0.45)
    for yi in range(body_y1 + 2, body_y2 - 2):
        for sx in range(streak_x - 1, streak_x + 2):
            glass_layer[yi, sx, :3] = 245
            glass_layer[yi, sx, 3] = min(255, glass_layer[yi, sx, 3] + 90)

    # Thickened glass base
    cv2.rectangle(glass_layer, (cx - hw, body_y2 - 5), (cx + hw, body_y2), (20, 70, 140, 245), -1)

    # 6. Molded screw cap with knurling ridges
    cap_layer = np.zeros((ph, pw, 4), dtype=np.uint8)
    cap_y1 = cy - hh
    cap_y2 = body_y1
    cap_hw = hw + 1  # slightly flanged
    cv2.rectangle(cap_layer, (cx - cap_hw, cap_y1), (cx + cap_hw, cap_y2), (215, 220, 225, 255), -1)
    # Cap shading
    for xi in range(-cap_hw, cap_hw + 1):
        x_norm = xi / float(cap_hw)
        val = int(220 - 45 * (x_norm ** 2))
        cap_layer[cap_y1:cap_y2, cx + xi, :3] = val
    # Knurling vertical ridges
    for kx in range(cx - cap_hw + 3, cx + cap_hw - 2, 4):
        cv2.line(cap_layer, (kx, cap_y1 + 2), (kx, cap_y2 - 2), (160, 165, 170, 255), 1)
        cv2.line(cap_layer, (kx + 1, cap_y1 + 2), (kx + 1, cap_y2 - 2), (245, 250, 255, 255), 1)
    # Cap top bevel
    cv2.line(cap_layer, (cx - cap_hw, cap_y1), (cx + cap_hw, cap_y1), (250, 255, 255, 255), 1)
    cv2.line(cap_layer, (cx - cap_hw, cap_y2), (cx + cap_hw, cap_y2), (130, 135, 140, 255), 1)

    # Alpha composite glass and cap onto patch
    # Combine layers
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


def create_pipette_patch(w: int, h: int) -> Tuple[np.ndarray, np.ndarray]:
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
    patch[:, :, 3] = (shadow * 0.50).astype(np.uint8)
    patch[:, :, 0] = 15
    patch[:, :, 1] = 18
    patch[:, :, 2] = 22

    # Proportions along height:
    # Plunger button: 0.00 to 0.08
    # Finger hook: 0.08 to 0.14
    # Main handle: 0.14 to 0.60
    # Shaft (metal/dark): 0.60 to 0.82
    # Disposable tip (tapered plastic): 0.82 to 1.00
    y_plunger_top = cy - hh
    y_plunger_bot = y_plunger_top + int(h * 0.08)
    y_hook_bot = y_plunger_top + int(h * 0.14)
    y_handle_bot = y_plunger_top + int(h * 0.60)
    y_shaft_bot = y_plunger_top + int(h * 0.82)
    y_tip_bot = cy + hh

    fg = np.zeros((ph, pw, 4), dtype=np.uint8)

    # Plunger button (blue P1000/orange P200)
    plunger_hw = max(3, int(hw * 0.65))
    cv2.rectangle(fg, (cx - plunger_hw, y_plunger_top), (cx + plunger_hw, y_plunger_bot), (220, 130, 20, 255), -1)
    cv2.line(fg, (cx - plunger_hw, y_plunger_top), (cx + plunger_hw, y_plunger_top), (255, 180, 80, 255), 1)

    # Finger hook / thumb rest
    hook_hw = hw
    hook_pts = np.array([
        [cx - plunger_hw, y_plunger_bot],
        [cx + hook_hw, y_plunger_bot],
        [cx + hook_hw, y_hook_bot],
        [cx - plunger_hw, y_hook_bot],
    ], dtype=np.int32)
    cv2.fillPoly(fg, [hook_pts], (200, 205, 210, 255))

    # Main handle body (light gray ergonomic polymer with longitudinal gradient)
    handle_hw = max(4, int(hw * 0.85))
    for xi in range(-handle_hw, handle_hw + 1):
        x_norm = xi / float(handle_hw)
        val = int(210 - 40 * (x_norm ** 2))
        fg[y_hook_bot:y_handle_bot, cx + xi, :3] = val
        fg[y_hook_bot:y_handle_bot, cx + xi, 3] = 255

    # Volume digital readout window (black frame with white background and digital numbers)
    win_w = max(4, int(handle_hw * 0.9))
    win_h = max(8, int(h * 0.12))
    win_y = y_hook_bot + int((y_handle_bot - y_hook_bot) * 0.25)
    cv2.rectangle(fg, (cx - win_w, win_y), (cx + win_w, win_y + win_h), (35, 40, 45, 255), -1)
    cv2.rectangle(fg, (cx - win_w + 1, win_y + 1), (cx + win_w - 1, win_y + win_h - 1), (240, 245, 250, 255), -1)
    # Digits display lines
    cv2.line(fg, (cx - win_w + 2, win_y + win_h // 3), (cx + win_w - 2, win_y + win_h // 3), (20, 25, 30, 255), 1)
    cv2.line(fg, (cx - win_w + 2, win_y + 2 * win_h // 3), (cx + win_w - 2, win_y + 2 * win_h // 3), (20, 25, 30, 255), 1)

    # Stainless steel nose cone shaft
    shaft_hw = max(3, int(hw * 0.45))
    for xi in range(-shaft_hw, shaft_hw + 1):
        x_norm = xi / float(shaft_hw)
        val = int(175 - 50 * (x_norm ** 2))
        fg[y_handle_bot:y_shaft_bot, cx + xi, :3] = val
        fg[y_handle_bot:y_shaft_bot, cx + xi, 3] = 255
    # Shaft highlight line
    fg[y_handle_bot:y_shaft_bot, cx - shaft_hw // 2, :3] = 230

    # Disposable translucent pipette tip (tapered conical nozzle)
    tip_pts = np.array([
        [cx - shaft_hw, y_shaft_bot],
        [cx + shaft_hw, y_shaft_bot],
        [cx + 2, y_tip_bot],
        [cx - 2, y_tip_bot],
    ], dtype=np.int32)
    # Clear / subtle yellow polypropylene: BGR (180, 225, 240, 215)
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
    patch[:, :, 3] = (shadow * 0.50).astype(np.uint8)
    patch[:, :, 0] = 15
    patch[:, :, 1] = 18
    patch[:, :, 2] = 22

    fg = np.zeros((ph, pw, 4), dtype=np.uint8)

    # 2. Outer skirt (translucent / opaque polystyrene lab plastic)
    skirt_rect = (cx - hw, cy - hh, w, h)
    cv2.rectangle(fg, (cx - hw, cy - hh), (cx + hw, cy + hh), (70, 75, 80, 240), -1)

    # A1 corner index chamfer notch (standard ANSI/SBS feature)
    notch_sz = 8
    notch_pts = np.array([
        [cx - hw, cy - hh + notch_sz],
        [cx - hw + notch_sz, cy - hh],
        [cx - hw, cy - hh],
    ], dtype=np.int32)
    # Clear the notch
    cv2.fillPoly(fg, [notch_pts], (0, 0, 0, 0))

    # Outer skirt bevel lines
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
            # Outer well rim
            cv2.circle(fg, pt, well_rad, (160, 168, 175, 255), 1)
            # Dark interior cavity
            cv2.circle(fg, pt, well_rad - 1, (30, 32, 35, 255), -1)
            # Specular glint on rim (top-left)
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


def create_centrifuge_tube_patch(w: int, h: int) -> Tuple[np.ndarray, np.ndarray]:
    """Render realistic 15mL/50mL conical centrifuge tube with red screw cap.
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

    # Proportions:
    # Cap: top to 0.18h
    # Cylindrical body: 0.18h to 0.72h
    # Conical taper: 0.72h to 1.00h (terminating in rounded apex)
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

    # Fill semi-transparent polypropylene plastic: BGR (205, 215, 225, 200)
    cv2.fillPoly(fg, [body_pts], (205, 215, 225, 200))

    # Cylindrical shading
    for xi in range(-hw + 1, hw):
        x_norm = xi / float(hw)
        val = int(220 - 45 * (x_norm ** 2))
        for yi in range(y_cap_bot, y_taper_start):
            fg[yi, cx + xi, :3] = val

    # 3. Liquid filling inside (bottom 50% of tube)
    liquid_top = int(y_cap_bot + (y_cone_tip - y_cap_bot) * 0.50)
    liquid_mask = np.zeros((ph, pw), dtype=np.uint8)
    cv2.fillPoly(liquid_mask, [body_pts], 255)
    liquid_mask[:liquid_top, :] = 0
    # Pink culture media / blue dye: BGR (180, 50, 220, 220)
    fg[liquid_mask > 0, 0] = 180
    fg[liquid_mask > 0, 1] = 50
    fg[liquid_mask > 0, 2] = 220
    fg[liquid_mask > 0, 3] = 230

    # Meniscus
    cv2.ellipse(fg, (cx, liquid_top), (hw - 2, 2), 0, 0, 360, (140, 30, 180, 255), 1)

    # 4. White frosted labeling patch for writing
    label_w = max(4, int(hw * 0.7))
    label_h = int((y_taper_start - y_cap_bot) * 0.40)
    label_y = y_cap_bot + int((y_taper_start - y_cap_bot) * 0.15)
    cv2.rectangle(fg, (cx - label_w, label_y), (cx + label_w, label_y + label_h), (245, 250, 255, 230), -1)

    # 5. Volumetric graduation tick marks (5, 10, 15 mL scale lines)
    for tick_y in np.linspace(label_y + 4, y_taper_start, 4):
        ty = int(tick_y)
        cv2.line(fg, (cx + hw // 3, ty), (cx + hw - 2, ty), (40, 45, 50, 240), 1)

    # Specular reflection streak
    fg[y_cap_bot + 2:y_taper_start, cx - int(hw * 0.4), :3] = 250
    fg[y_cap_bot + 2:y_taper_start, cx - int(hw * 0.4), 3] = 245

    # Outline
    cv2.polylines(fg, [body_pts], isClosed=True, color=(140, 150, 160, 255), thickness=1)

    # 6. Flanged screw cap (vibrant laboratory red/orange)
    cap_hw = hw + 2
    cap_pts = np.array([
        [cx - cap_hw, y_cap_top],
        [cx + cap_hw, y_cap_top],
        [cx + cap_hw, y_cap_bot],
        [cx - cap_hw, y_cap_bot],
    ], dtype=np.int32)
    # Red cap: BGR (20, 25, 220, 255)
    cv2.fillPoly(fg, [cap_pts], (20, 25, 220, 255))
    # Cap vertical grip ribs
    for kx in range(cx - cap_hw + 2, cx + cap_hw - 1, 4):
        cv2.line(fg, (kx, y_cap_top + 1), (kx, y_cap_bot - 1), (10, 15, 160, 255), 1)
        cv2.line(fg, (kx + 1, y_cap_top + 1), (kx + 1, y_cap_bot - 1), (80, 85, 255, 255), 1)
    # Cap rim highlight
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

    # 1. Machined aluminum / anodized outer plate
    cv2.rectangle(fg, (cx - hw, cy - hh), (cx + hw, cy + hh), (60, 65, 72, 255), -1)

    # Outer metallic border highlight & shadow
    cv2.line(fg, (cx - hw, cy - hh), (cx + hw, cy - hh), (170, 178, 185, 255), 2)  # top highlight
    cv2.line(fg, (cx - hw, cy - hh), (cx - hw, cy + hh), (140, 148, 155, 255), 2)  # left highlight
    cv2.line(fg, (cx + hw, cy - hh), (cx + hw, cy + hh), (35, 38, 42, 255), 2)   # right shadow
    cv2.line(fg, (cx - hw, cy + hh), (cx + hw, cy + hh), (30, 32, 35, 255), 2)   # bottom shadow

    # 2. Deep recessed pocket cavity (depth gradient)
    pocket_inset = max(6, int(w * 0.14))
    px1, py1 = cx - hw + pocket_inset, cy - hh + pocket_inset
    px2, py2 = cx + hw - pocket_inset, cy + hh - pocket_inset
    cv2.rectangle(fg, (px1, py1), (px2, py2), (18, 20, 24, 255), -1)

    # Inner bevel / pocket shadow lip
    cv2.line(fg, (px1, py1), (px2, py1), (10, 12, 14, 255), 2)  # deep shadow under top lip
    cv2.line(fg, (px1, py1), (px1, py2), (12, 14, 16, 255), 2)
    cv2.line(fg, (px2, py1), (px2, py2), (80, 85, 92, 255), 1)  # subtle lip reflection
    cv2.line(fg, (px1, py2), (px2, py2), (90, 95, 102, 255), 1)

    # 3. Corner socket head cap screws (mounting holes)
    screw_offset = pocket_inset // 2 + 1
    screw_pts = [
        (cx - hw + screw_offset, cy - hh + screw_offset),
        (cx + hw - screw_offset, cy - hh + screw_offset),
        (cx + hw - screw_offset, cy + hh - screw_offset),
        (cx - hw + screw_offset, cy + hh - screw_offset),
    ]
    for pt in screw_pts:
        cv2.circle(fg, pt, 3, (25, 28, 30, 255), -1)  # dark hole
        cv2.circle(fg, pt, 3, (120, 128, 135, 255), 1)  # metallic washer rim
        cv2.circle(fg, pt, 1, (160, 170, 180, 255), -1) # hex socket center

    patch = fg
    return patch, corners


def rotate_and_composite(
    canvas: np.ndarray,
    patch: np.ndarray,
    local_corners: np.ndarray,
    center: Tuple[int, int],
    angle_deg: float,
) -> Optional[np.ndarray]:
    """Rotate local RGBA patch and composite onto BGR canvas.
    Returns: rotated OBB corners in canvas coordinates, or None if out of bounds.
    """
    ph, pw = patch.shape[:2]
    patch_cx, patch_cy = pw // 2, ph // 2
    cx, cy = center

    # 1. Rotate the patch image
    rot_mat = cv2.getRotationMatrix2D((patch_cx, patch_cy), angle_deg, 1.0)
    rotated_patch = cv2.warpAffine(
        patch,
        rot_mat,
        (pw, ph),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )

    # 2. Transform the 4 local corners
    # Point rotation: (x - pcx, y - pcy) rotated by -angle (in standard math coords)
    # cv2 rotation angle is clockwise in screen coordinates
    rad = np.radians(angle_deg)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    rot_2d = np.array([[cos_a, sin_a], [-sin_a, cos_a]]) # matches cv2 warpAffine convention

    # Transform corners relative to patch center
    rel_corners = local_corners - np.array([patch_cx, patch_cy], dtype=np.float32)
    # cv2 getRotationMatrix2D applies:
    # x' = alpha * (x - cx) + beta * (y - cy) + cx
    # y' = -beta * (x - cx) + alpha * (y - cy) + cy
    # where alpha = cos(angle), beta = sin(angle)
    alpha = cos_a
    beta = sin_a
    rot_c = np.zeros_like(rel_corners)
    rot_c[:, 0] = alpha * rel_corners[:, 0] + beta * rel_corners[:, 1]
    rot_c[:, 1] = -beta * rel_corners[:, 0] + alpha * rel_corners[:, 1]

    # Translate to canvas center
    canvas_corners = rot_c + np.array([cx, cy], dtype=np.float32)

    # Check bounds
    ch, cw = canvas.shape[:2]
    if np.any(canvas_corners < 5) or np.any(canvas_corners[:, 0] > cw - 5) or np.any(canvas_corners[:, 1] > ch - 5):
        return None

    # Composite rotated patch onto canvas
    top_left_x = cx - patch_cx
    top_left_y = cy - patch_cy

    # Alpha composite
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


def main() -> None:
    # Test on a realistic lab workbench frame
    bg_path = Path("d:/space_eye/vyom/data/debug_frames/train_103_frame500.jpg")
    if bg_path.exists():
        raw_bg = cv2.imread(str(bg_path))
        h, w = raw_bg.shape[:2]
        crop = raw_bg[8:368, 156:516]
        canvas = cv2.resize(crop, (640, 640), interpolation=cv2.INTER_CUBIC)
    else:
        canvas = np.full((640, 640, 3), (85, 95, 105), dtype=np.uint8)

    test_canvas = canvas.copy()

    # Render each class
    # Class 0: sample_vial
    patch0, c0 = create_sample_vial_patch(34, 85)
    corners0 = rotate_and_composite(test_canvas, patch0, c0, (120, 180), 25.0)

    # Class 1: pipette
    patch1, c1 = create_pipette_patch(26, 140)
    corners1 = rotate_and_composite(test_canvas, patch1, c1, (260, 220), -45.0)

    # Class 2: multiwell_plate
    patch2, c2 = create_multiwell_plate_patch(120, 85)
    corners2 = rotate_and_composite(test_canvas, patch2, c2, (480, 200), 15.0)

    # Class 3: centrifuge_tube
    patch3, c3 = create_centrifuge_tube_patch(36, 100)
    corners3 = rotate_and_composite(test_canvas, patch3, c3, (180, 450), 70.0)

    # Class 4: rig_slot
    patch4, c4 = create_rig_slot_patch(75, 75)
    corners4 = rotate_and_composite(test_canvas, patch4, c4, (450, 450), -30.0)

    out_p = Path("d:/space_eye/vyom/data/debug_frames/test_rendered_all5.jpg")
    cv2.imwrite(str(out_p), test_canvas)
    print(f"Saved test composite: {out_p}")

    # Draw OBB overlay on a copy to verify geometry
    vis = test_canvas.copy()
    items = [
        (0, corners0, "sample_vial"),
        (1, corners1, "pipette"),
        (2, corners2, "multiwell_plate"),
        (3, corners3, "centrifuge_tube"),
        (4, corners4, "rig_slot"),
    ]
    for cid, corners, name in items:
        if corners is not None:
            pts = corners.astype(np.int32)
            cv2.polylines(vis, [pts], isClosed=True, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
            for idx, p in enumerate(pts):
                cv2.circle(vis, tuple(p), 3, (0, 0, 255), -1)
                cv2.putText(vis, f"p{idx}", (p[0] + 4, p[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    out_vis = Path("d:/space_eye/vyom/data/debug_frames/test_rendered_all5_obb.jpg")
    cv2.imwrite(str(out_vis), vis)
    print(f"Saved test OBB verification: {out_vis}")


if __name__ == "__main__":
    main()

