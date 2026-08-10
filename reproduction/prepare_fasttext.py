#!/usr/bin/env python3
"""Convert the paper's cc.en.300.vec into mmap-loadable KeyedVectors."""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="Decompressed cc.en.300.vec")
    parser.add_argument("output", type=Path, help="Output .kv path")
    args = parser.parse_args()
    if not args.source.is_file():
        raise FileNotFoundError(args.source)
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
