#!/bin/bash
set -euo pipefail

# Example SLURM job submissions for the gom.ablations pipeline. Each sbatch
# call below queues one independent job; add/remove/edit lines to change
# which config files run. To change *what* an experiment does (models,
# datasets, ablation grids, prompting strategies, ...) edit the referenced
# YAML file under slurm_configs/ instead of this script — see README_SLURM.md.
#
#   ./sbatch_train.sh
#
# Note: titan_xp and nvidia_geforce_rtx_5090 can be requested instead of
# nvidia_geforce_rtx_3090; don't request more GPU than a job actually needs.

# Ablation grid experiments (edge thickness, relation caps, edge color, ...)
sbatch -N 1 --gpus=nvidia_geforce_rtx_3090:1 run_docker.sh slurm_configs/ablation_experiments.yaml

# VLM comparison across models on a fixed preprocessing configuration
sbatch -N 1 --gpus=nvidia_geforce_rtx_3090:1 run_docker.sh slurm_configs/vlm_comparison.yaml

# Prompting strategy comparison (baseline, few-shot, CoT, graph-guided, ...)
sbatch -N 1 --gpus=nvidia_geforce_rtx_3090:1 run_docker.sh slurm_configs/prompting_experiments.yaml
