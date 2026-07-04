# Standard image for the SLURM cluster's RTX 3090 / Titan Xp nodes
# (faretra, moro43, moro232, deeplearn2). For the RTX 5090 node, build
# build/Dockerfile.rtx5090 instead (see that file and README.md for details).
#
# Runs the modern VLM stack (torch 2.8.0+cu126 / vllm 0.11.0 / transformers
# 4.57.x) so the cutting-edge ablation models — Qwen3-VL, InternVL3.5, Gemma-4 —
# load; the older pinned cu124/vllm-0.8.5 set only supported LlamaV-o1. See the
# strip-and-reinstall block below (and build/requirements.txt's header note).
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04
LABEL maintainer="disi-unibo-nlp"

# Zero interaction (default answers to all questions)
ENV DEBIAN_FRONTEND=noninteractive

# Set work directory (do not rename: standardized convention on the cluster)
WORKDIR /workspace
ENV APP_PATH=/workspace

# Install general-purpose + OpenCV runtime dependencies
RUN apt-get update -y && \
    apt-get install -y curl \
                        git \
                        bash \
                        nano \
                        python3.11 \
                        python3-pip \
                        libgl1 \
                        libglib2.0-0 && \
    apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install wrapt --upgrade --ignore-installed
RUN pip install gdown

# Strip the six pins that requirements.txt fixes for the *original* cu124 stack
# (torch 2.6.0 + vllm 0.8.5 + transformers 4.51.3 + tokenizers 0.21.4) and
# reinstall the modern stack below. Rationale: vllm 0.8.5 / transformers 4.51.3
# do NOT register the Qwen3-VL, InternVL3.5 or Gemma-4 architectures these
# ablations use (verified: those config keys are simply ABSENT from the 0.8.5
# vLLM registry / 4.51.3 transformers CONFIG_MAPPING), so the old image can only
# load LlamaV-o1. vllm 0.11.0 + transformers 4.57.x add them. This mirrors the
# same override in build/Dockerfile.rtx5090, but targets Ampere (RTX 3090,
# sm_86) instead of Blackwell:
#   * torch 2.8.0 is what vllm 0.11.0 requires, and it ships only as cu126/cu128
#     wheels (cu124 was dropped). We use the cu126 channel here — sm_86 is fully
#     supported on cu126 and it is the closest channel to these nodes' driver
#     (the 5090 image needs cu128 for sm_120; the 3090 does not).
#   * TORCH_CUDA_ARCH_LIST is left UNSET (unlike the 5090's "12.0"): sm_86 is in
#     torch's default arch list, so detectron2 builds for it without an override,
#     and leaving it unset keeps the Titan Xp (sm_61) preprocessing path working.
# The non-torch/vllm pins are untouched, so the rest of the set stays identical
# to the verified cu128 resolution (only the torch cu tag differs).
COPY build/requirements.txt .
RUN grep -v -iE '^(torch|torchvision|torchaudio|vllm|transformers|tokenizers)[>=<]' \
        requirements.txt > requirements.cu126.txt

# Install PyTorch (CUDA 12.6, Ampere/sm_86) FIRST so the vllm + rest-of-
# requirements resolve below sees torch already satisfied and never pulls a
# different-CUDA wheel over it.
RUN pip install --no-cache-dir \
        torch==2.8.0+cu126 \
        torchvision==0.23.0+cu126 \
        torchaudio==2.8.0+cu126 \
        --index-url https://download.pytorch.org/whl/cu126

# Resolve vllm 0.11.0 together with the remaining pinned deps in one pass (torch
# is already satisfied by the cu126 wheels above; vllm pulls transformers 4.57.x
# / tokenizers 0.22.x, kept <5 so the rest of the pinned stack is unaffected).
RUN pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cu126 \
        vllm==0.11.0 \
        -r requirements.cu126.txt

# detectron2 is not distributed on PyPI; build it against the torch version
# installed above (--no-deps avoids pulling a second, unpinned torch).
RUN pip install --no-cache-dir --no-build-isolation --no-deps \
        "git+https://github.com/facebookresearch/detectron2.git@main"

# Model assets used by gom.relations (spaCy pipeline, NLTK wordnet)
RUN python3 -m spacy download en_core_web_md && \
    python3 -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"

# Register the gom package + console scripts (gom-preprocess, gom-vqa) as an
# editable install. The project directory is normally bind-mounted over
# /workspace at `docker run` time (see run_docker.sh); since the mount keeps
# the same src/ layout, this editable link keeps working against the live
# code without needing a rebuild on every code change.
COPY setup.py pyproject.toml MANIFEST.in README.md LICENSE ./
COPY src/ ./src/
RUN pip install --no-cache-dir --no-deps -e .

# Back to default frontend
ENV DEBIAN_FRONTEND=dialog

CMD ["/bin/bash"]
