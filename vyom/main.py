#!/usr/bin/env python3
"""VYOM Application Entry Point.

Offline-first experiment monitoring system for human activity recognition
and procedure compliance.
"""

import sys
from pathlib import Path

# Ensure src/ directory is on sys.path for direct execution
CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vyom.app import main

if __name__ == "__main__":
    sys.exit(main())

