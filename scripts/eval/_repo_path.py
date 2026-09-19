"""Put the repository root first on ``sys.path``.

Import this module before ``mlx_qwen3_asr`` in any script under ``scripts/``.
When the package is installed editable, ``python scripts/foo.py`` otherwise
resolves ``mlx_qwen3_asr`` through the install's path hook, which points at
whatever checkout was ``pip install -e``'d. Run from a worktree, a benchmark or
eval script would then silently measure another checkout's code. Inserting the
script's own repository root ahead of everything else makes the script measure
the code it sits next to.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

if str(REPO_ROOT) not in sys.path[:1]:
    sys.path.insert(0, str(REPO_ROOT))
