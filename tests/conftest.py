"""Shared pytest configuration.

Scripts under ``scripts/`` start with ``import _repo_path`` so that running
them directly measures the checkout they live in. Tests load those scripts
either as ``scripts.<name>`` or via ``importlib`` from a file path; neither puts
``scripts/`` on ``sys.path``, so make ``_repo_path`` importable here.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(_SCRIPTS_DIR))
