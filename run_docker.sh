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
#
# Gated Hugging Face repos (e.g. google/gemma-3-12b-it) need an access token.
# SET IT ONCE PER MACHINE and forget it: drop your token into ~/.hf_token, e.g.
#     umask 077; echo hf_xxx > ~/.hf_token; chmod 600 ~/.hf_token
# The block below auto-loads that file (independently of your shell/SLURM env),
# and `docker run -e HF_TOKEN` forwards it into the container. Precedence:
#   1. HF_TOKEN already in the environment (e.g. `export HF_TOKEN=...`) wins;
#   2. else ~/.hf_token;  3. else $PHYS_DIR/.hf_token.local (git-ignored).
# If none is set it is simply a no-op (fine for ungated models / cached weights).
# The file may hold either the raw token (`hf_...`) or an `export HF_TOKEN=...`
# line. Alternatively pre-stage the weights once into /llms so no token is needed.

IMAGE_NAME="${GOM_IMAGE_NAME:-gom:latest}"
PHYS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLM_CACHE_DIR="/llms"
DOCKER_INTERNAL_CACHE_DIR="/llms"

# Load the Hugging Face token from a machine-local file unless already set.
if [ -z "${HF_TOKEN:-}" ]; then
    for _hf_file in "$HOME/.hf_token" "$PHYS_DIR/.hf_token.local"; do
        [ -r "$_hf_file" ] || continue
        # Accept an `export HF_TOKEN=...` file (source it) or a raw-token file.
        # shellcheck disable=SC1090
        . "$_hf_file" 2>/dev/null || true
        [ -n "${HF_TOKEN:-}" ] || HF_TOKEN="$(tr -d ' \t\r\n' < "$_hf_file")"
        export HF_TOKEN
        break
    done
fi

docker run \
    -v "$PHYS_DIR":/workspace \
    -v "$LLM_CACHE_DIR":"$DOCKER_INTERNAL_CACHE_DIR" \
    -e HF_HOME="$DOCKER_INTERNAL_CACHE_DIR" \
    -e HF_TOKEN \
    --rm \
    --memory="30g" \
    --gpus '"device='"$CUDA_VISIBLE_DEVICES"'"' \
    "$IMAGE_NAME" \
    /workspace/train.sh "$@"
