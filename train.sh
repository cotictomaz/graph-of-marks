#!/bin/bash
set -euo pipefail

# Entry point executed *inside* the Docker container (WORKDIR=/workspace,
# see build/Dockerfile). Invoked by run_docker.sh, which is itself submitted
# to SLURM via sbatch (see sbatch_train.sh).
#
# Runs the ablation studies pipeline (src/gom/ablations/main.py). Everything
# about *what* runs (which experiments, models, datasets) is controlled by
# the YAML file passed in, not by this script — see slurm_configs/*.yaml for the
# three ready-to-edit templates (ablation experiments, VLM comparison,
# prompting experiments) and README_SLURM.md for how to customize them.
#
# Usage: train.sh <path/to/config.yaml>
# Example: train.sh slurm_configs/ablation_experiments.yaml

CONFIG_PATH="${1:?Usage: train.sh <path/to/config.yaml>}"

exec python3 -m gom.ablations.main --config "$CONFIG_PATH"
