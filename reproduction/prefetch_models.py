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

from common import PAPER_SPEC, ROOT, load_yaml

OWLV2 = "google/owlv2-base-patch16"
MIDAS_REPO = "isl-org/MiDaS:1645b7e1675301fdfac03640738fe5a6531e17d6"
MIDAS_MODEL = "DPT_Large"
DETECTRON2_CONFIG = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
SAM_HQ_TYPE = "vit_h"
YOLO_WEIGHTS = "yolov8x.pt"


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

    # SAM-HQ and YOLOv8 normally self-download on first use, but verify_weights.py
    # runs before preprocessing, so a fresh machine would fail there instead.  Both
    # land under the working directory, which every stage sets to the repo root --
    # the same place verify_weights.py searches.
    from gom.segmentation.samhq import _SAM_HQ_URLS

    checkpoint = ROOT / "checkpoints" / f"sam_hq_{SAM_HQ_TYPE}.pth"
    if checkpoint.is_file():
        print(f"-> SAM-HQ cached {checkpoint}", flush=True)
    else:
        url = _SAM_HQ_URLS[SAM_HQ_TYPE]
        print(f"-> SAM-HQ {url}", flush=True)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(url, str(checkpoint), progress=False)

    from ultralytics import YOLO

    print(f"-> YOLOv8 {YOLO_WEIGHTS}", flush=True)
    YOLO(str(ROOT / YOLO_WEIGHTS))


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
