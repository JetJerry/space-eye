"""Automated Windows onedir build and distribution packager for VYOM.

Part of VYOM Prompt 12: Windows Application Packaging and Asset Bundling.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT_DIR = Path(__file__).resolve().parent.parent
SPEC_PATH = ROOT_DIR / "vyom.spec"
DIST_DIR = ROOT_DIR / "dist" / "VYOM"


def compute_file_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def build_distribution() -> bool:
    print("=" * 70)
    print("  VYOM WINDOWS APPLICATION PACKAGING (PyInstaller onedir)")
    print("=" * 70)

    # 1. Verify source assets
    print("\n[Step 1/4] Verifying required runtime assets...")
    required_assets = [
        ROOT_DIR / "models" / "hand_landmarker.task",
        ROOT_DIR / "models" / "vyom_yolov8n_obb_smoke.pt",
        ROOT_DIR / "config" / "app.json",
        ROOT_DIR / "config" / "protocols" / "sample_experiment.json",
    ]
    for p in required_assets:
        if not p.is_file():
            print(f"  ✕ Missing required asset: {p}")
            return False
        print(f"  ✓ Verified: {p.name} ({p.stat().st_size:,} bytes)")

    # 2. Run PyInstaller
    skip_build = "--verify-only" in sys.argv or "--skip-build" in sys.argv
    if skip_build and (DIST_DIR / "VYOM.exe").is_file():
        print("\n[Step 2/4] Skipping PyInstaller compilation (--verify-only specified and VYOM.exe exists)...")
    else:
        print("\n[Step 2/4] Executing PyInstaller build...")
        t0 = time.time()
        cmd = [
            sys.executable,
            "-m",
            "PyInstaller",
            str(SPEC_PATH),
            "--noconfirm",
            "--clean",
        ]
        print(f"Running command: {' '.join(cmd)}")
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT_DIR / "src")
        res = subprocess.run(cmd, cwd=str(ROOT_DIR), env=env)

        if res.returncode != 0:
            print(f"\n✕ PyInstaller build failed with returncode {res.returncode}")
            return False
        print(f"  ✓ Build finished in {time.time() - t0:.1f}s")

    # 3. Verify packaged distribution
    print("\n[Step 3/4] Verifying packaged output directory...")
    exe_path = DIST_DIR / "VYOM.exe"
    if not exe_path.is_file():
        print(f"  ✕ Executable not found at: {exe_path}")
        return False
    print(f"  ✓ Standalone executable: {exe_path.name} ({exe_path.stat().st_size:,} bytes)")

    # Ensure models and config exist in dist (top-level and _internal)
    internal_models = DIST_DIR / "_internal" / "models"
    internal_config = DIST_DIR / "_internal" / "config"
    dist_models = DIST_DIR / "models"
    dist_config = DIST_DIR / "config"

    # In PyInstaller 6+, datas are placed in _internal/. Also mirror to top level for user ergonomics
    if not dist_models.exists() and internal_models.exists():
        shutil.copytree(internal_models, dist_models)
    if not dist_config.exists() and internal_config.exists():
        shutil.copytree(internal_config, dist_config)

    assert dist_models.exists() or internal_models.exists(), "Packaged models directory missing"
    assert dist_config.exists() or internal_config.exists(), "Packaged config directory missing"
    print(f"  ✓ Packaged assets verified: {dist_models.name}/ and {dist_config.name}/")

    # 4. Generate Checksum Manifest for packaged release
    print("\n[Step 4/4] Generating deployment checksum manifest...")
    manifest = {
        "build_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "application_name": "VYOM",
        "executable": exe_path.name,
        "executable_sha256": compute_file_sha256(exe_path),
        "packaged_models": {},
    }
    for m in dist_models.glob("*.*"):
        if m.is_file():
            manifest["packaged_models"][m.name] = {
                "size_bytes": m.stat().st_size,
                "sha256": compute_file_sha256(m),
            }

    checksum_file = DIST_DIR / "checksums.json"
    with open(checksum_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"  ✓ Checksum manifest generated: {checksum_file.name}")

    print("\n" + "=" * 70)
    print(f"  ✓ VYOM WINDOWS ONEDIR DISTRIBUTION READY AT: {DIST_DIR}")
    print("=" * 70)
    return True


if __name__ == "__main__":
    success = build_distribution()
    sys.exit(0 if success else 1)
