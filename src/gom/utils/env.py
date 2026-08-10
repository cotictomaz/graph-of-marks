# gom/utils/env.py
"""
Minimal .env loader (no external dependency).

Loads KEY=VALUE pairs from a .env file into ``os.environ`` without
overriding variables that are already set in the environment. Call
:func:`load_dotenv` as early as possible in entry points — before importing
torch / transformers / vllm — so variables like ``HF_HOME`` take effect.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def load_dotenv(path: Optional[str | Path] = None, *, override: bool = False) -> dict:
    """
    Load environment variables from a .env file.

    Args:
        path: Path to the .env file. When None, searches the current working
              directory and then each parent directory for a ``.env`` file.
        override: When True, values in the file replace existing environment
                  variables; by default existing variables win.

    Returns:
        Dict of the variables read from the file (empty if no file found).
    """
    if path is None:
        for candidate_dir in [Path.cwd(), *Path.cwd().parents]:
            candidate = candidate_dir / ".env"
            if candidate.is_file():
                path = candidate
                break
        else:
            return {}
    path = Path(path)
    if not path.is_file():
        return {}

    loaded: dict = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        loaded[key] = value
        if override or key not in os.environ:
            os.environ[key] = value
    return loaded
