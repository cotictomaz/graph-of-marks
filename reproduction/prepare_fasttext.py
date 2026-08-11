#!/usr/bin/env python3
"""Convert the paper's cc.en.300.vec into mmap-loadable KeyedVectors."""
from __future__ import annotations

import argparse
import gzip
import shutil
import urllib.request
from pathlib import Path

VECTORS_URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.vec.gz"


def download_vectors(destination: Path) -> None:
    """Fetch and decompress the official cc.en.300 vectors (~4.3 GB expanded)."""
    archive = destination.with_suffix(destination.suffix + ".gz")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not archive.is_file():
        print(f"downloading {VECTORS_URL}", flush=True)
        partial = archive.with_suffix(archive.suffix + ".part")
        with urllib.request.urlopen(VECTORS_URL) as response, partial.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        partial.replace(archive)
    print(f"decompressing {archive}", flush=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    with gzip.open(archive, "rb") as source, partial.open("wb") as target:
        shutil.copyfileobj(source, target)
    partial.replace(destination)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="Decompressed cc.en.300.vec")
    parser.add_argument("output", type=Path, help="Output .kv path")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Fetch cc.en.300.vec.gz from fastText when the source is absent",
    )
    args = parser.parse_args()
    if not args.source.is_file():
        if not args.download:
            raise FileNotFoundError(args.source)
        download_vectors(args.source)
    from gensim.models import KeyedVectors

    vectors = KeyedVectors.load_word2vec_format(
        str(args.source), binary=False, unicode_errors="replace"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    vectors.save(str(args.output), separately=["vectors"])
    print(f"saved {len(vectors)} vectors to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
