#!/usr/bin/env python3
"""Download every model the reproduction needs into the mounted model cache.

Run before the long unattended job so auth/network failures surface immediately
instead of hours into preprocessing.  Every fetch is idempotent.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import PAPER_SPEC, load_yaml

OWLV2 = "google/owlv2-base-patch16"
MIDAS_REPO = "isl-org/MiDaS:1645b7e1675301fdfac03640738fe5a6531e17d6"
MIDAS_MODEL = "DPT_Large"
DETECTRON2_CONFIG = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"


def fetch_preprocess() -> None:
    import torch
    from transformers import Owlv2ForObjectDetection, Owlv2Processor

    print(f"-> OWLv2 {OWLV2}", flush=True)
    Owlv2Processor.from_pretrained(OWLV2)
    Owlv2ForObjectDetection.from_pretrained(OWLV2)

    print(f"-> MiDaS {MIDAS_MODEL} from {MIDAS_REPO}", flush=True)
    torch.hub.load(MIDAS_REPO, MIDAS_MODEL)
    torch.hub.load(MIDAS_REPO, "transforms")

    from detectron2 import model_zoo
    from detectron2.utils.file_io import PathManager

    url = model_zoo.get_checkpoint_url(DETECTRON2_CONFIG)
    print(f"-> Detectron2 {url}", flush=True)
    print(PathManager.get_local_path(url), flush=True)


def fetch_inference() -> None:
    from huggingface_hub import snapshot_download

    for key, model in load_yaml(PAPER_SPEC)["models"].items():
        print(f"-> {key}: {model['id']}@{model['revision']}", flush=True)
        path = snapshot_download(model["id"], revision=model["revision"])
        print(path, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("preprocess", "inference"), required=True)
    args = parser.parse_args()
    if args.stage == "preprocess":
        fetch_preprocess()
    else:
        fetch_inference()
    print(f"{args.stage} models cached", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
