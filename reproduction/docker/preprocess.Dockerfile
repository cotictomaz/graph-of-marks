FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

ARG DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git libgl1 libglib2.0-0 ninja-build && \
    rm -rf /var/lib/apt/lists/*

COPY reproduction/docker/preprocess-requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --upgrade pip==25.1.1 setuptools==80.9.0 wheel==0.45.1 && \
    python3 -m pip install -r /tmp/requirements.txt && \
    python3 -m pip install --no-build-isolation \
      'git+https://github.com/facebookresearch/detectron2.git@02b5c4e295e990042a714712c21dc79b731e8833'

WORKDIR /workdir
COPY pyproject.toml README.md LICENSE /workdir/
COPY src /workdir/src
COPY reproduction /workdir/reproduction
RUN python3 -m pip install --no-deps -e /workdir && \
    python3 -m nltk.downloader -d /usr/local/share/nltk_data wordnet omw-1.4

LABEL org.graph-of-marks.profile="paper_aaai26" \
      org.graph-of-marks.torch="2.7.1+cu128" \
      org.graph-of-marks.detectron2="02b5c4e295e990042a714712c21dc79b731e8833"

ENTRYPOINT []
