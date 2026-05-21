"""
Pytest configuration: ensure project root is on sys.path so that
`from src.api.main import app` works during test collection.

This file is auto-discovered by pytest. No imports of this file are needed.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
