# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Graph of Marks (GoM)** is a visual prompting framework (AAAI 2026) that transforms images into structured semantic graphs. It chains object detection → fusion → segmentation → depth estimation → relationship extraction → scene graph generation → visualization into a single pipeline, producing annotated images and structured scene graph text for multimodal LLMs.

## Commands

### Installation
```bash
pip install -e ".[all]"     # editable install with all extras
make install_deps           # install system-level dependencies
```

### Preprocessing
```bash
# Single image
make preprocess INPUT_PATH=path/to/image.jpg [QUESTION='What is on the table?']

# Batch from JSON
make preprocess JSON_FILE=data.json

# Fast mode (<10s per image)
make fast_preprocess INPUT_PATH=path/to/image.jpg

# Detector-specific shortcuts
make preprocess_owlvit INPUT_PATH=image.jpg
make preprocess_yolo INPUT_PATH=image.jpg
make preprocess_detectron2 INPUT_PATH=image.jpg
```

### VQA
```bash
make run_vqa VQA_INPUT_FILE=data.json MODEL_NAME=llava-hf/llava-1.5-7b-hf
make run_vqa_folder IMAGE_FOLDER=path/to/images [FIXED_PROMPT='Describe this']
```

### CLI Entry Points
```bash
gom-preprocess --input_file data.json --image_dir images/ --output_folder output/
gom-vqa --input_file vqa_data.json --model_name llava-hf/llava-1.5-7b-hf
```

### Ablation Studies
```bash
python -m gom.ablations.main --config src/gom/ablations/config.yaml   # local/Colab
python -m gom.ablations.main --config slurm_configs/ablation_experiments.yaml

# On the SLURM cluster (see README_SLURM.md for the full workflow):
sbatch -N 1 --gpus=nvidia_geforce_rtx_3090:1 run_docker.sh slurm_configs/vlm_comparison.yaml
```

### Other
```bash
make help     # list all make targets with parameters
make clean    # remove output files
```

## Architecture

The package lives under `src/gom/`. The pipeline is orchestrated by `ImageGraphPreprocessor` in `src/gom/pipeline/preprocessor.py`, which owns a `PreprocessorConfig` (70+ parameters) and drives all stages in order:

```
Detectors → Fusion → Segmentation → Depth → Relations → Scene Graph → Visualization
```

### Key modules

| Path | Role |
|------|------|
| `src/gom/api.py` | High-level `GoM` class; recommended entry point |
| `src/gom/pipeline/preprocessor.py` | `ImageGraphPreprocessor` — pipeline orchestrator |
| `src/gom/config.py` | `PreprocessorConfig` and related config dataclasses |
| `src/gom/types.py` | Core types: `Detection`, `Relationship`, `Box` |
| `src/gom/detectors/manager.py` | Multi-detector orchestration (YOLO, OWL-ViT, Detectron2, GroundingDINO) |
| `src/gom/fusion/` | Weighted Box Fusion (WBF) and NMS strategies |
| `src/gom/segmentation/` | SAM v1/v2/HQ/FastSAM wrappers with a common base interface |
| `src/gom/relations/inference.py` | Relationship extractor combining geometry, CLIP, physics, 3D, LLM-guided |
| `src/gom/graph/scene_graph.py` | NetworkX `DiGraph` builder from detections + relations |
| `src/gom/graph/prompt.py` | Converts scene graph to text prompts for LLMs |
| `src/gom/viz/visualizer.py` | Renders annotated images (boxes, masks, relation arrows) |
| `src/gom/vqa/runner.py` | VQA inference pipeline |
| `src/gom/utils/cache.py` | Detection caching (~60-80% speedup on repeated runs) |

### Visual prompting styles

The `style` parameter in `ProcessingConfig` selects one of 9 configurations that control label mode (text/numeric/alphabetic), whether relation arrows are drawn, and whether relation labels are shown. Examples: `som_text`, `gom_text_labeled` (main GoM style for VQA), `gom_numeric_labeled` (for REC tasks).

### Pipeline return value

All processing methods return a dict with keys: `boxes`, `labels`, `scores`, `masks`, `depth`, `relationships`, `scene_graph` (NetworkX DiGraph), `scene_graph_text`, `scene_graph_prompt`, `output_image` (PIL), `processing_time`.

### High-level API usage

```python
from gom import GoM, ProcessingConfig

pipeline = GoM(device="cuda")
result = pipeline.process("scene.jpg", config=ProcessingConfig(
    style="gom_text_labeled",
    question="What objects are in the room?",
    apply_question_filter=True
))
```

## Optional Dependency Groups (pyproject.toml)

- `segmentation` — sam-hq
- `detection` — ultralytics (YOLOv8)
- `vqa` — vllm, ollama, qwen-vl-utils, HuggingFace inference (`setup.py`'s `vqa` extra
  installs both vllm and ollama unconditionally — see note under Ablations below)
- `dev` — testing and linting tools
- `all` — everything above

## Ablations (`src/gom/ablations/`)

A YAML-driven experiment runner built on top of the core pipeline, used to
compare preprocessing configurations, VLMs, and prompting strategies for the
paper's ablation studies. Entry point: `python -m gom.ablations.main --config
<file.yaml>` (see `src/gom/ablations/config.yaml` for the full schema, or
`slurm_configs/*.yaml` at the repo root for cluster-ready templates).

| Path | Role |
|------|------|
| `src/gom/ablations/main.py` | CLI entry point; loads YAML, builds the VQA dataset, dispatches to the three experiment types below |
| `src/gom/ablations/ablate_preprocessing.py` | `generate_ablated_dataset` / `generate_default_dataset` — pre-renders GoM images once per config, cached on disk (skipped on rerun unless `force_reprocess: true`) |
| `src/gom/ablations/run_experiments.py` | `run_ablation_experiments`, `run_vlm_comparison`, `run_prompting_experiments` — loads each model once, runs `n_runs` repeats, writes `summary_metrics.json` (mean/std) per config |
| `src/gom/ablations/models.py` | `OllamaVLM` / `VllmVLM` wrappers; **imports both `ollama` and `vllm` unconditionally at module level**, so both packages must be installed regardless of which `backend` a config selects |
| `src/gom/ablations/prompts.py` | `build_prompt_template` — the four prompting strategies (`baseline`, `few_shot`, `chain_of_thought`, `graph_guided`); add a new strategy here first, then reference it by name in a config's `strategies:` block |
| `src/gom/ablations/utils.py` | `update_cfg_correct` (in-place `PreprocessorConfig` patching to avoid reloading models), `run_preprocessing` |

### Config structure

A single YAML config controls three independent experiment types, toggled via
`enabled: true/false` under `ablations:` / `vlm_comparison:` / `prompting:`
(more than one can be enabled in the same run — `main.py` executes them in
that order against the same loaded dataset):

- **`ablations`** — for each entry under `experiments:`, preprocesses images
  across a grid of parameter values (`ablation_grid:`) and evaluates every
  model in `models:` against each grid point. New experiment names must also
  get a branch in `main.py`'s `apply_experiment_config()` (maps an experiment
  name to `PreprocessorConfig` overrides applied before preprocessing).
- **`vlm_comparison`** — preprocesses images once with a fixed config, then
  compares every model in `models:` on that same set.
- **`prompting`** — preprocesses images once, then evaluates every strategy
  enabled under `strategies:` against every model in `models:` (model loaded
  once, reused across strategies).

Results always land under `{base_dir}/results/...` as `summary_metrics.json`
(aggregate) plus one `raw_results.json` per run; preprocessed images live
under `{base_dir}/preprocessed_images/...` and are reused across reruns
unless `force_reprocess: true`.

## Docker / SLURM Cluster Setup

The repo ships a Docker + SLURM workflow (per the university's "SLURM Web
Guide") wired specifically to run `gom.ablations.main`. Full walkthrough in
`README_SLURM.md`; summary:

| Path | Role |
|------|------|
| `build/Dockerfile` | Standard image (CUDA 12.2, Ubuntu 22.04) for RTX 3090 / Titan Xp nodes. Installs `build/requirements.txt`, builds `detectron2` from git (not on PyPI), downloads spaCy/NLTK model data, then `pip install --no-deps -e .` to register `gom` as editable against `/workspace` |
| `build/Dockerfile.rtx5090` | CUDA 12.8 / Ubuntu 24.04 variant for the cluster's RTX 5090 node; installs everything except torch from `build/requirements.txt`, then installs `torch==2.7.1+cu128` wheels separately |
| `build/requirements.txt` | Exact-pinned dependency set (verified via `pip install --dry-run` to resolve with no conflicts) — deliberately pins `torch==2.6.0`/cu124 rather than letting `torch>=2.4.0` float to whatever's newest, since an unconstrained resolve drifts to cu13 wheels that need a newer NVIDIA driver than older cluster nodes may have |
| `train.sh` | Container entry point; runs `python3 -m gom.ablations.main --config "$1"` |
| `run_docker.sh` | Host-side script SLURM's `sbatch` invokes; bind-mounts the project dir to `/workspace` and the cluster's shared model cache (`/llms`) with `HF_HOME` set, then runs `train.sh` inside the container |
| `sbatch_train.sh` | Example `sbatch` submissions, one per `slurm_configs/*.yaml` |
| `slurm_configs/*.yaml` | Ready-to-edit configs (`ablation_experiments.yaml`, `vlm_comparison.yaml`, `prompting_experiments.yaml`) — copy one to define a new experiment set; no script or image rebuild needed to change what runs |

Key design point: the image does **not** bake in the live source tree — it
installs `gom` as an editable package pointing at `/workspace/src`, and
`run_docker.sh` bind-mounts the actual project directory over `/workspace` at
run time. So a rebuild is only needed for actual dependency/environment
changes (edits to `build/Dockerfile*` or `build/requirements.txt`), never for
ordinary `src/gom/**` or `slurm_configs/*.yaml` edits — `git pull` + `sbatch`
is enough.

## Legacy / Reference Files

- `src/all_in_one_gom.py` — monolithic prototype; not part of the packaged API
- `src/image_preprocessor.py`, `src/vqa.py` — thin CLI wrappers around the package
- `examples/demo_gom.ipynb` — Jupyter notebook walkthrough
