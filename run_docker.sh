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
# Resolve the project directory (bind-mounted at /workspace). Under `sbatch`,
# SLURM copies this script to its spool dir and runs it from there, so
# ${BASH_SOURCE[0]} points at /var/spool/slurmd/job*/slurm_script — NOT the
# repo — and mounting its dirname yields a /workspace with no train.sh (exit
# 127). Precedence: GOM_PROJECT_DIR override → SLURM_SUBMIT_DIR (the dir sbatch
# was launched from = the repo) → the script's own dir (direct srun/bash run).
if [ -n "${GOM_PROJECT_DIR:-}" ]; then
    PHYS_DIR="$GOM_PROJECT_DIR"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    PHYS_DIR="$SLURM_SUBMIT_DIR"
else
    PHYS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
if [ ! -f "$PHYS_DIR/train.sh" ]; then
    echo "ERROR: train.sh not found in project dir '$PHYS_DIR'." >&2
    echo "       Submit sbatch from the repo root, or set GOM_PROJECT_DIR:" >&2
    echo "       export GOM_PROJECT_DIR=/home/cotic/graph-of-marks" >&2
    exit 1
fi
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

# Host path to the source image dataset (node 40 / faretra only). The images
# live OUTSIDE the repo and /llms, in a dedicated dataset directory, so they
# must be bind-mounted explicitly here. This is what makes the config's
# `images_dir: /images` resolve on node 40; on any other node the path won't
# exist, the mount is skipped, and the pipeline fetches images over HTTP
# (images_base_url) instead. Set it ONCE PER NODE (node 40), e.g.:
#     echo /datasets/VisualQA_Datasets/Preprocessing/VQAV1/original_VQAV1/vqav1_images > ~/.gom_images_dir
# Precedence: 1. GOM_IMAGES_DIR in the environment;  2. ~/.gom_images_dir;
#             3. $PHYS_DIR/.gom_images_dir.local (git-ignored).
if [ -z "${GOM_IMAGES_DIR:-}" ]; then
    for _img_file in "$HOME/.gom_images_dir" "$PHYS_DIR/.gom_images_dir.local"; do
        [ -r "$_img_file" ] || continue
        GOM_IMAGES_DIR="$(tr -d ' \t\r\n' < "$_img_file")"
        break
    done
fi

# Mount the images read-only at a FIXED container path (/images) so the YAML's
# images_dir is identical on every node regardless of the host location. Only
# add the mount when the directory actually exists on this node — mounting a
# nonexistent host path would make Docker create an empty dir and silently
# re-break image resolution (node 40 would then look local, find nothing, and
# never fall back to HTTP).
IMAGES_MOUNT=()
if [ -n "${GOM_IMAGES_DIR:-}" ]; then
    if [ -d "$GOM_IMAGES_DIR" ]; then
        IMAGES_MOUNT=(-v "$GOM_IMAGES_DIR":/images:ro)
        echo "🖼️  Mounting images (ro): $GOM_IMAGES_DIR -> /images"
    else
        echo "⚠️  GOM_IMAGES_DIR=$GOM_IMAGES_DIR is not a directory on this node; skipping the images mount (will rely on images_base_url)." >&2
    fi
fi

docker run \
    -v "$PHYS_DIR":/workspace \
    -v "$LLM_CACHE_DIR":"$DOCKER_INTERNAL_CACHE_DIR" \
    "${IMAGES_MOUNT[@]}" \
    -e HF_HOME="$DOCKER_INTERNAL_CACHE_DIR" \
    -e HF_TOKEN \
    --rm \
    --memory="30g" \
    --gpus '"device='"$CUDA_VISIBLE_DEVICES"'"' \
    "$IMAGE_NAME" \
    /workspace/train.sh "$@"
