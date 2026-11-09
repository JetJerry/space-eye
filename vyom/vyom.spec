# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller specification file for VYOM (Offline Human Activity Recognition Workstation).

Builds a standalone onedir Windows distribution with bundled model assets, configurations,
protocols, and PySide6 Qt runtime.
"""

from pathlib import Path
import sys

block_cipher = None

# SPECPATH is automatically injected by PyInstaller with the spec file directory
ROOT_DIR = Path(SPECPATH).resolve() if "SPECPATH" in globals() else Path.cwd().resolve()

# Collect data files
datas = [
    (str(ROOT_DIR / "config"), "config"),
    (str(ROOT_DIR / "models" / "hand_landmarker.task"), "models"),
    (str(ROOT_DIR / "models" / "vyom_yolov8n_obb_smoke.pt"), "models"),
    (str(ROOT_DIR / "models" / "vyom_yolov8n_obb_smoke.onnx"), "models"),
    (str(ROOT_DIR / "models" / "smoke_model_metadata.json"), "models"),
    (str(ROOT_DIR / "models" / "onnx_parity_report.json"), "models"),
]

# Hidden imports needed for dynamic drivers / subpackages
hiddenimports = [
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
    "cv2",
    "mediapipe",
    "mediapipe.tasks",
    "mediapipe.tasks.python",
    "mediapipe.tasks.python.vision",
    "ultralytics",
    "ultralytics.models.yolo.obb",
    "torch",
    "torchvision",
    "pyttsx3",
    "pyttsx3.drivers",
    "pyttsx3.drivers.sapi5",
    "comtypes",
    "onnxruntime",
    "psutil",
    "sqlite3",
]

a = Analysis(
    [str(ROOT_DIR / "main.py")],
    pathex=[str(ROOT_DIR / "src"), str(ROOT_DIR)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tkinter",
        "matplotlib.tests",
        "scipy.spatial.tests",
        "pytest",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="VYOM",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,  # Console enabled for development & diagnostics inspection
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="VYOM",
)
