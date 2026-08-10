#!/usr/bin/env python3
"""Verify every preprocessing weight against the reproduction registry."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from common import REPRODUCTION_ROOT, ROOT, load_yaml, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path("/model-cache"))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def candidate_roots(cache: Path) -> list[Path]:
    roots = [ROOT, cache]
    for variable in ("TORCH_HOME", "HF_HOME"):
        value = os.environ.get(variable)
        if value:
            roots.append(Path(value))
    unique: list[Path] = []
    for root in roots:
        resolved = root.expanduser().resolve()
        if resolved.is_dir() and resolved not in unique:
            unique.append(resolved)
    return unique


def main() -> int:
    args = parse_args()
    expected = load_yaml(REPRODUCTION_ROOT / "weights.yaml")["weights"]
    roots = candidate_roots(args.cache)
    report: dict[str, dict[str, str]] = {}
    failures: list[str] = []
    for filename, wanted_hash in expected.items():
        paths = sorted(
            {path.resolve() for root in roots for path in root.rglob(filename)}
        )
        if not paths:
            failures.append(f"{filename}: not found below {roots}")
            continue
        observed = [(path, sha256_file(path)) for path in paths]
        match = next((item for item in observed if item[1] == wanted_hash), None)
        if match is None:
            details = ", ".join(f"{path}={digest}" for path, digest in observed)
            failures.append(f"{filename}: expected {wanted_hash}; found {details}")
            continue
        report[filename] = {"path": str(match[0]), "sha256": match[1]}
    if failures:
        raise RuntimeError("Weight verification failed:\n  - " + "\n  - ".join(failures))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"verified {len(report)} preprocessing weights -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
