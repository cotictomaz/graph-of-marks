#!/usr/bin/env python3
"""Install and verify the exact image subsets used by the GoM paper."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import MANIFEST_REGISTRY, REPRODUCTION_ROOT, load_yaml, sha256_file


ARCHIVE_SHA256 = "a9c0f446ed4d99bcb7e00cbc3cd686d9fe19149ad3a1015a379e05569992f404"
ARCHIVE_LAYOUT = {
    "gqa": (
        "Preprocessing/GQA/original_GQA/",
        "gqa/images",
    ),
    "vqav1": (
        "Preprocessing/VQAV1/original_VQAV1/vqav1_images/",
        "vqav1/train2014",
    ),
    "vqav2": (
        "Preprocessing/VQAV2/original_VQAV2/vqav2_imgs_1000/",
        "vqav2/train2014",
    ),
    "refcocog": (
        "Preprocessing/RefCOCOg/original_RefCOCOg/",
        "refcocog/images",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install only the exact 1,000-image author split per dataset"
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path(os.environ.get("GOM_DATASET_ARCHIVE", "data_paper/gom_datasets.zip")),
    )
    parser.add_argument(
        "--archive-url",
        default=os.environ.get("GOM_DATASET_ARCHIVE_URL"),
        help="Download URL used only when --archive is absent",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("GOM_PAPER_DATA", REPRODUCTION_ROOT / "data")),
    )
    parser.add_argument(
        "--datasets",
        default=",".join(ARCHIVE_LAYOUT),
        help="Comma-separated subset of gqa,vqav1,vqav2,refcocog",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def download_archive(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(url) as response, temporary.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    temporary.replace(destination)


def manifest_image_names(dataset: str, entry: dict[str, Any]) -> set[str]:
    path = REPRODUCTION_ROOT / str(entry["manifest"])
    source = json.loads(path.read_text(encoding="utf-8"))
    return {Path(str(row["image_path"])).name for row in source}


def collection_sha256(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(directory.iterdir(), key=lambda item: item.name):
        if not path.is_file():
            continue
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def archive_members(archive: zipfile.ZipFile, prefix: str) -> dict[str, str]:
    members: dict[str, str] = {}
    for info in archive.infolist():
        if info.is_dir() or not info.filename.startswith(prefix):
            continue
        name = Path(info.filename).name
        if not name.lower().endswith((".jpg", ".jpeg", ".png")):
            raise RuntimeError(f"Unexpected non-image member in paper split: {info.filename}")
        if name in members:
            raise RuntimeError(f"Duplicate image basename in paper split: {name}")
        members[name] = info.filename
    return members


def install_dataset(
    archive: zipfile.ZipFile,
    data_root: Path,
    dataset: str,
    entry: dict[str, Any],
    force: bool,
) -> dict[str, Any]:
    prefix, relative_output = ARCHIVE_LAYOUT[dataset]
    destination = data_root / "images" / relative_output
    expected = manifest_image_names(dataset, entry)
    members = archive_members(archive, prefix)
    if set(members) != expected:
        missing = sorted(expected - set(members))[:10]
        extra = sorted(set(members) - expected)[:10]
        raise RuntimeError(
            f"{dataset}: archive/manifest image mismatch; missing={missing}, extra={extra}"
        )
    if len(expected) != int(entry["unique_images"]):
        raise RuntimeError(
            f"{dataset}: manifest has {len(expected)} images, expected {entry['unique_images']}"
        )

    destination.mkdir(parents=True, exist_ok=True)
    existing = {path.name for path in destination.iterdir() if path.is_file()}
    unexpected = existing - expected
    if unexpected and not force:
        raise RuntimeError(
            f"{dataset}: output contains files outside the paper split: {sorted(unexpected)[:10]}"
        )
    if force:
        for name in unexpected:
            (destination / name).unlink()

    for name in sorted(expected):
        output = destination / name
        if output.is_file() and not force:
            continue
        temporary = output.with_suffix(output.suffix + ".part")
        with archive.open(members[name]) as source, temporary.open("wb") as target:
            shutil.copyfileobj(source, target)
        temporary.replace(output)

    actual = {path.name for path in destination.iterdir() if path.is_file()}
    if actual != expected:
        raise RuntimeError(f"{dataset}: installed image set differs from the author manifest")
    return {
        "dataset": dataset,
        "image_count": len(actual),
        "output": destination.resolve().as_posix(),
        "collection_sha256": collection_sha256(destination),
        "manifest": str(entry["manifest"]),
        "manifest_sha256": str(entry["sha256"]),
        "image_split_status": str(entry["status"]),
        "query_status": str(entry["query_status"]),
    }


def main() -> int:
    args = parse_args()
    datasets = tuple(value.strip() for value in args.datasets.split(",") if value.strip())
    unknown = sorted(set(datasets) - set(ARCHIVE_LAYOUT))
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}")

    archive_path = args.archive.expanduser().resolve()
    if not archive_path.is_file():
        if not args.archive_url:
            raise FileNotFoundError(
                f"Missing {archive_path}; pass --archive-url or GOM_DATASET_ARCHIVE_URL"
            )
        download_archive(args.archive_url, archive_path)
    actual_archive_hash = sha256_file(archive_path)
    if actual_archive_hash != ARCHIVE_SHA256:
        raise RuntimeError(
            f"Dataset archive hash {actual_archive_hash} != expected {ARCHIVE_SHA256}"
        )

    registry = load_yaml(MANIFEST_REGISTRY)["datasets"]
    installed = []
    with zipfile.ZipFile(archive_path) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise RuntimeError(f"Corrupt dataset archive member: {bad_member}")
        for dataset in datasets:
            entry = registry[dataset]
            if entry.get("status") != "exact_author_image_split":
                raise RuntimeError(
                    f"{dataset}: image split is not marked exact_author_image_split"
                )
            installed.append(
                install_dataset(archive, args.data_root.resolve(), dataset, entry, args.force)
            )

    provenance = {
        "schema_version": 1,
        "archive": archive_path.as_posix(),
        "archive_sha256": actual_archive_hash,
        "datasets": installed,
    }
    output = args.data_root.resolve() / "dataset_provenance.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    for item in installed:
        print(
            f"installed {item['dataset']}: {item['image_count']} images "
            f"({item['collection_sha256']})"
        )
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError, zipfile.BadZipFile) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
