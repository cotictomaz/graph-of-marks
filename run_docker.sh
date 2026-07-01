#!/bin/bash
set -euo pipefail

# Launches a container from the GoM Docker image, submitted to SLURM via
# sbatch (see sbatch_train.sh). Mounts the project directory at /workspace
# and the cluster's shared model cache at /llms (see the "SLURM Web Guide",
# section 6, "HELP US SAVE SOME MEMORY DISK"), so Hugging Face downloads
# (e.g. the Qwen2.5-VL checkpoints used by gom.ablations) already present on
# the node are reused instead of re-downloaded per user.
#
# Usage: sbatch -N 1 --gpus=nvidia_geforce_rtx_3090:1 run_docker.sh <path/to/config.yaml>
# e.g.:  sbatch -N 1 --gpus=nvidia_geforce_rtx_3090:1 run_docker.sh configs/ablation_experiments.yaml
#
# Override the image name/tag with the GOM_IMAGE_NAME env var if you built a
# custom tag (default matches the README's `docker build ... -t gom:latest`).

IMAGE_NAME="${GOM_IMAGE_NAME:-gom:latest}"
PHYS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLM_CACHE_DIR="/llms"
DOCKER_INTERNAL_CACHE_DIR="/llms"

docker run \
    -v "$PHYS_DIR":/workspace \
    -v "$LLM_CACHE_DIR":"$DOCKER_INTERNAL_CACHE_DIR" \
    -e HF_HOME="$DOCKER_INTERNAL_CACHE_DIR" \
    --rm \
    --memory="30g" \
    --gpus '"device='"$CUDA_VISIBLE_DEVICES"'"' \
    "$IMAGE_NAME" \
    /workspace/train.sh "$@"
