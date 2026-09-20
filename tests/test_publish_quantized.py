"""Token resolution for scripts/publish_quantized.py."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]


def _load_module():
    scripts_dir = _REPO / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    spec = importlib.util.spec_from_file_location(
        "publish_quantized_script", scripts_dir / "publish_quantized.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_env_wins_over_files(tmp_path: Path):
    mod = _load_module()
    local = tmp_path / "hf_token"
    local.write_text("from-file\n", encoding="utf-8")
    assert mod.resolve_hf_token(env={"HF_TOKEN": " from-env "}, local_file=local) == "from-env"
    assert (
        mod.resolve_hf_token(env={"HUGGINGFACE_HUB_TOKEN": "hub-env"}, local_file=local)
        == "hub-env"
    )


def test_local_gitignored_file_then_hub_login_file(tmp_path: Path):
    mod = _load_module()
    local = tmp_path / ".secrets" / "hf_token"
    home = tmp_path / "home"
    (home / ".cache" / "huggingface").mkdir(parents=True)
    (home / ".cache" / "huggingface" / "token").write_text("from-login\n", encoding="utf-8")

    assert mod.resolve_hf_token(env={}, local_file=local, home=home) == "from-login"
    local.parent.mkdir()
    local.write_text("from-local\n", encoding="utf-8")
    assert mod.resolve_hf_token(env={}, local_file=local, home=home) == "from-local"
    # Empty file is not a token.
    local.write_text("\n", encoding="utf-8")
    assert mod.resolve_hf_token(env={}, local_file=local, home=home) == "from-login"
    # Explicit HF_TOKEN_PATH overrides the default login location.
    other = tmp_path / "elsewhere"
    other.write_text("from-path", encoding="utf-8")
    assert (
        mod.resolve_hf_token(env={"HF_TOKEN_PATH": str(other)}, local_file=local, home=home)
        == "from-path"
    )


def test_missing_everything_returns_none(tmp_path: Path):
    mod = _load_module()
    assert mod.resolve_hf_token(env={}, local_file=tmp_path / "nope", home=tmp_path) is None


def test_local_token_file_is_gitignored():
    mod = _load_module()
    rel = mod.LOCAL_TOKEN_FILE.relative_to(_REPO)
    proc = subprocess.run(
        ["git", "check-ignore", "-q", str(rel)],
        cwd=str(_REPO),
        check=False,
        capture_output=True,
    )
    assert proc.returncode == 0, f"{rel} must be ignored by .gitignore"
