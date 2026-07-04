# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**The user's primary focus in this repository is `src/gom/ablations/` — the ablation studies experiments (grid ablations, VLM comparison, prompting strategies) and the SLURM/Docker infrastructure used to run them on the cluster.** Prioritize that directory and its supporting `slurm_configs/*.yaml`, `build/`, and SLURM scripts when interpreting ambiguous requests; see the dedicated sections below for both.

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
| `src/gom/ablations/run_experiments.py` | `run_ablation_experiments`, `run_vlm_comparison`, `run_prompting_experiments` — loads each model once, runs `n_runs` repeats, writes `summary_metrics.json` (mean/std) per config. Scores with `evaluate_vqa` from `evaluation.py` (**not** the old `runner.evaluate`) — see "VQA evaluation & answer scoring" below. All three iterate **models as the outer loop** and call `models.release_model(current_model)` at the end of each model's iteration to free its VRAM before the next model loads — see "VRAM lifecycle across models" below |
| `src/gom/ablations/evaluation.py` | Correct VQA scorer used by `run_experiments.py`: `extract_final_answer` (strips reasoning traces), `normalize_vqa_answer` (official VQA normalization), `vqa_soft_accuracy` (official 10-answer soft metric), `evaluate_vqa` (drop-in replacement for `runner.evaluate`). Added to fix reasoning-model scoring without touching `gom.vqa.runner`; see the dedicated section below |
| `src/gom/ablations/models.py` | `OllamaVLM` / `VllmVLM` wrappers + `parse_model_entry`; **imports both `ollama` and `vllm` unconditionally at module level**, so both packages must be installed regardless of which `backend` a config selects. `VllmVLM(quantize_fp8=True)` passes `quantization="fp8"` to vLLM (on-the-fly FP8, ~halves the weight footprint; bf16 path unchanged when `False`). `parse_model_entry` normalizes each `models:` list item to `(name, quantize_fp8)` — see per-model FP8 below. **Auto-raises the generation cap for reasoning models** (`is_reasoning_model` / `resolve_max_tokens`: 2048 tokens vs 512 default; `OllamaVLM` also widens `num_ctx` to 16384) so a `<think>` trace + answer is not truncated — see VQA evaluation below. Both wrappers expose a `shutdown()`, and the module offers `release_model(model)` — the explicit VRAM teardown used between models; see "VRAM lifecycle across models" below |
| `src/gom/ablations/prompts.py` | `build_prompt_template` — the four prompting strategies (`baseline`, `few_shot`, `chain_of_thought`, `graph_guided`); add a new strategy here first, then reference it by name in a config's `strategies:` block. Every template ends with an `Answer:` marker so the answer extractor has a reliable split point; none forbids reasoning (see VQA evaluation below) |
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

**Per-model FP8 (`models:` entry schema).** Each item in any `models:` list may
be **either** a plain string (`"repo/id"`, loaded in bf16) **or** a mapping
`{name: "repo/id", fp8: true}` that FP8-quantizes *that model* on load
(`model` is an alias for `name`, `quantize_fp8` for `fp8`). This lets a single
config keep small models in bf16 while quantizing a large one to fit 24 GB —
e.g. `google/gemma-3-12b-it` is ~24 GB in bf16 (too tight for a 3090 with KV
cache) but ~13 GB in FP8. FP8 is a no-op on the `ollama` backend (warned and
ignored). Parsed by `gom.ablations.models.parse_model_entry`.

**Model selection (this study's choices; VQA only, no REC).** The paper used
Qwen-2.5-VL-7B / Gemma-3-4B / LlamaV-o1-11B. The current configs extend along
generation, scale, architecture, and reasoning axes:
- **`vlm_comparison`** — `LlamaV-o1` (retained paper anchor) + newest-gen
  `Qwen3-VL-8B-Instruct` and `google/gemma-3-12b-it` (FP8) + `InternVL3_5-8B`
  (distinct ViT–MLP–LLM lineage). Tests whether GoM still helps two-generations-newer and different-architecture models. (Gemma-4 / `gemma4_unified` is not yet supported by transformers 4.57.6 / vllm 0.11.0, so the 12B Gemma-3 fills this slot — the newest supported Gemma.)
- **`ablation_experiments`** — `LlamaV-o1` anchor + `Qwen3-VL-4B` and
  `Qwen3-VL-8B` (same family, two sizes → is drawn-mark sensitivity
  scale-dependent?). Kept bf16-safe to keep the big grid cheap.
- **`prompting_experiments`** — `Qwen3-VL-8B-Instruct` vs `Qwen3-VL-8B-Thinking`
  (reasoning off/on, identical weights) + `LlamaV-o1`, to isolate how
  CoT/graph_guided prompting interacts with native reasoning.

Note: there is **no** `Qwen3.5-VL` (Qwen3.5 is text-only); `Qwen3-VL` is the
current multimodal Qwen. **Reasoning/Thinking models (LlamaV-o1, any
`*-Thinking`) emit reasoning tokens before the answer.** This *was* a
correctness risk because `gom.vqa.runner.evaluate` does a bare string
exact-match, which scores the reasoning trace verbatim and understates those
models. It is now handled by `ablations/evaluation.py` (answer extraction +
official VQA scoring) and the reasoning-aware `max_tokens` / prompt changes —
see "VQA evaluation & answer scoring" below. `runner.evaluate` itself is left
unchanged; the ablations code no longer calls it.

### VRAM lifecycle across models (`run_experiments.py` → `models.release_model`)

All three runners load **one model at a time** — the `models:` list is the
**outermost** loop, so a model's weights are loaded once (`VllmVLM.__init__` /
`OllamaVLM`) and reused across every ablation grid point / strategy / `n_runs`
repeat, then the loop advances to the next model. Models are **never**
co-resident in VRAM by design; per-model FP8 sizing targets a single model +
KV cache on one 24 GB card, not the sum of all models.

The catch is that reassigning `current_model` to the next model does **not**
free the previous one: vLLM keeps the CUDA context, KV-cache blocks and NCCL
state alive until the process exits, regardless of Python GC — so without an
explicit teardown two models can briefly sit in VRAM at a model transition and
OOM. Each outer-loop iteration therefore ends with
`release_model(current_model); current_model = None`:

- `release_model(model)` (in `models.py`) calls the wrapper's `shutdown()`,
  then `gc.collect()` + `torch.cuda.empty_cache()` **in that order** (flushing
  the cache only reclaims blocks the engine has already released). It is
  best-effort and exception-safe — cleanup never aborts a run.
- `VllmVLM.shutdown()` drops the engine's executor refs (`llm_engine.model_executor`
  for V0 / `llm_engine.engine_core` for V1 — best-effort, version-dependent),
  nulls `self.llm`, then calls `destroy_model_parallel()` /
  `destroy_distributed_environment()`. Idempotent via a `_is_shutdown` guard.
- `OllamaVLM.shutdown()` is a daemon nudge, not an in-process free: Ollama
  serves models from its own process, so the wrapper holds no GPU memory; it
  just issues `ollama.generate(..., keep_alive=0)` to unload the model now
  instead of after its keep-alive window.

Because cleanup runs at the end of **every** model iteration (including the
last), it also frees the final model between the experiment types `main.py`
runs back-to-back in one process (ablations → vlm_comparison → prompting).

### Data loading — how the VQA examples are built (`main.py::build_vqa_examples`)

The dataset is a **single flat JSON list** (`dataset_path`, e.g.
`vqav1_limited_1000.json`): each record is
`{"image_path": "<basename>.jpg", "question": "...", "answers": [<10 strings>]}`.
The current file holds 3000 records = 1000 unique images × 3 questions each,
with `COCO_train2014_*` basenames. There is **no** separate
questions/annotations file and no `question_id`/`image_id` — this replaced an
older two-file VQAv2-style format, so ignore any lingering references to
`questions_path`/`annotations_path`/`multiple_choice_answer`.

`build_vqa_examples(dataset_path, images_dir, images_base_url, image_cache_dir)`
turns each record into a `gom.vqa.types.VQAExample`:

- **`answer`** = VQA-style **majority vote** over the 10 `answers`
  (`_majority_answer`, ties → first occurrence). The full list is preserved in
  `metadata["answers"]` and is what the scorer actually uses: `evaluate_vqa`
  computes the official soft VQA accuracy over all 10, and falls back to the
  majority only for the strict exact-match number (see VQA evaluation below).
- **`image_id`** = filename stem (e.g. `COCO_train2014_000000487025`); used to
  count unique images and to apply `num_examples` (which subsamples by unique
  image, keeping all questions of the chosen images).
- **`metadata`** = `{"answers": [...], "image_file": <basename>, "dataset": "vqav1"}`.
- **`image_path`** = a **resolved local absolute path** (see below), so every
  downstream stage (basename+question-hash preprocessing caches, image
  grouping, `evaluate`) is oblivious to where the image came from.

**Image resolution / the node-40 problem** (`_resolve_local_image`): the JSON
stores only basenames, and the actual `.jpg` files live **only on node 40
(faretra)** with no shared filesystem across cluster nodes. For each unique
image (resolved once per run and memoized), it tries in order: (1)
`images_dir/<basename>` if it exists locally (the node-40 case → no download);
(2) `image_cache_dir/<basename>` if already fetched; (3) download
`images_base_url/<basename>` into `image_cache_dir` (atomic `.part` rename,
retries with exponential backoff, via `requests`). Missing/undownloadable
images are skipped with a warning; an all-missing run raises. Config knobs:
`images_dir` (local dir, used on node 40), `images_base_url` (empty on node 40;
`http://137.204.107.40:8000` elsewhere), `image_cache_dir` (default
`{base_dir}/image_cache`, must stay under the bind-mounted `/workspace`). This
is what lets the same config run on **any** SLURM-selected node — node 40 reads
local files, every other node fetches over HTTP from node 40's
`python3 -m http.server`. Full operational writeup in `README_SLURM.md` §2.1.
No Docker change is needed (default bridge networking reaches node 40;
`requests` is already installed).

### VQA evaluation & answer scoring (`main.py::build_vqa_examples` → `evaluation.py`)

The ablations score VQA answers with `gom.ablations.evaluation.evaluate_vqa`,
**not** the legacy `gom.vqa.runner.evaluate`. All three runners in
`run_experiments.py` import and call `evaluate_vqa`; `runner.evaluate` (a bare
case-insensitive exact match against the single majority answer) is left in
place but unused by the ablations. This exists to fix two correctness problems
without editing anything outside `src/gom/ablations/`:

1. **Reasoning-model output vs exact match.** `LlamaV-o1` and any `*-Thinking`
   model emit a reasoning trace (often inside `<think>...</think>`) before the
   answer. `run_vqa` only trims output when the literal token `Answer:` is
   present, so the trace was being scored verbatim → those models were
   systematically understated.
2. **The official VQA metric is not exact string match.** The paper evaluates
   VQAv1/VQAv2 with the *official protocol* (Antol 2015 / Goyal 2017): canonical
   answer normalization + a **soft** score `min(1, #humans_who_said_it / 3)`
   averaged over the ten leave-one-out subsets of the 10 human answers — not a
   single-string comparison.

**How it works (all in `evaluation.py`):**
- `extract_final_answer(raw)` — strips `<think>...</think>` blocks (closed or
  dangling), then keeps the text after the **last** answer marker (`Answer:` /
  `Final answer:` / `The answer is`), else falls back to the last non-empty
  line; trims quotes/markdown/trailing punctuation. **Idempotent and a no-op on
  ordinary short answers** (`red`→`red`), so non-reasoning models are unaffected.
  The raw, un-extracted model output is still stored in `raw_results.json` for
  debugging the traces.
- `normalize_vqa_answer(s)` — the official VQA `processPunctuation` /
  `processDigitArticle` normalization (lowercase, punctuation, articles a/an/the,
  number-words→digits, contractions), ported from the VQA eval toolkit.
- `vqa_soft_accuracy(pred, human_answers)` — the official leave-one-out soft
  accuracy over the 10 answers (falls back to normalized exact match when <10
  answers are present).
- `evaluate_vqa(results)` — drop-in for `runner.evaluate`. Returns
  `vqa_accuracy` (official soft, 0–100, **the headline metric** — surfaced as
  `acc` / `mean_accuracy` in `summary_metrics.json`), plus `exact`/`exact_percent`
  (stricter normalized exact-match vs the majority answer, kept as a lower bound)
  and `avg_time`. Reads the 10 answers from each record's
  `metadata["answers"]` (put there by `build_vqa_examples`).

**Supporting changes (also in `src/gom/ablations/`):**
- **`max_tokens` is reasoning-aware** (`models.py`). `is_reasoning_model(name)`
  (name substrings: `thinking`, `-o1`, `reasoning`, `r1`, `qvq`, `cot`, …) and
  `resolve_max_tokens` give reasoning models **2048** generation tokens vs the
  **512** default, so a long `<think>` trace plus the final answer is not cut
  off (which would otherwise be scored as empty). `OllamaVLM` also widens
  `num_ctx` to 16384 for those models. Both wrappers accept an explicit
  `max_tokens=` override. Detection is name-based, so a thinking model whose
  repo id lacks an obvious keyword defaults to 512 — use the override for that.
  Note: `google/gemma-3-12b-it` is an instruction-tuned model, **not** a
  reasoning model, so it correctly stays at 512.
- **Prompts request a concise, parseable answer without forbidding reasoning.**
  The shared `multimodal_prompt` / `system_prompt` in `main.py` and the four
  templates in `prompts.py` ask the model to *conclude* with
  `Answer: <one word or a short phrase>` and end each template with an `Answer:`
  marker (the extractor's split point). They deliberately **do not** forbid
  explanation and **do not** end the whole prompt with a trailing `Answer:` that
  would pre-empt a reasoning model's thinking — that would cripple LlamaV-o1 /
  `*-Thinking` and contradict `chain_of_thought` prompting. This aligns every
  run with the official metric's short-answer assumption while leaving reasoning
  intact.

**Known limitation (by design).** Extraction recovers a concise answer when the
model ends with a marker or a short final line (which the prompts now request).
It intentionally does **not** try to distil a short answer out of an arbitrary
free-form sentence (e.g. "The bowl is in the top part of the image." is *not*
matched to gold `top`) — that would need an LLM judge and could silently
inflate/deflate scores. The mitigation is prompting for terse answers, not
fuzzy matching. If a future model still answers in sentences, a conservative,
negation-aware containment fallback could be added to `evaluate_vqa`, but it is
not implemented.

## Docker / SLURM Cluster Setup

The repo ships a Docker + SLURM workflow (per the university's "SLURM Web
Guide") wired specifically to run `gom.ablations.main`. Full walkthrough in
`README_SLURM.md`; summary:

| Path | Role |
|------|------|
| `build/Dockerfile` | Standard image (CUDA 12.2 base, Ubuntu 22.04, python3.11) for RTX 3090 / Titan Xp nodes. Like the 5090 variant, it **strips** the six legacy pins from `requirements.txt` (`torch`/`torchvision`/`torchaudio` + `vllm`/`transformers`/`tokenizers`) and installs the modern stack — `torch==2.8.0+cu126` first, then `vllm==0.11.0` + the remaining pins in one pass (pulling `transformers==4.57.x` / `tokenizers==0.22.x`). This is what makes the cutting-edge ablation models (Qwen3-VL, InternVL3.5, Gemma-3) loadable; the old cu124/`vllm==0.8.5` set only registered LlamaV-o1. (Gemma-**4** / `gemma4_unified` is still unsupported by transformers 4.57.6 — the configs use `gemma-3-12b-it`, the newest supported Gemma.) Differs from the 5090 file only where Ampere requires: **cu126** wheels (not cu128 — sm_86 is fully supported on cu126, the closest channel to these nodes' 12.5 driver) and **no `TORCH_CUDA_ARCH_LIST` override** (sm_86 is in torch's default arch list, and leaving it unset keeps the Titan Xp / sm_61 preprocessing path). Then builds `detectron2` from git (not on PyPI), downloads spaCy/NLTK model data, and `pip install --no-deps -e .` to register `gom` as editable against `/workspace` |
| `build/Dockerfile.rtx5090` | CUDA 12.8 / Ubuntu 24.04 variant for the cluster's RTX 5090 (Blackwell / sm_120) node. It **cannot reuse** `requirements.txt`'s cu124 stack: it strips out `torch`/`torchvision`/`torchaudio` **and** `vllm`/`transformers`/`tokenizers`, installs `torch==2.8.0+cu128` first, then resolves `vllm==0.11.0` together with the remaining pins in one pass (which pulls `transformers==4.57.x` / `tokenizers==0.22.x`, kept <5 so the rest of the pinned stack is unaffected). **Why the vllm bump is mandatory, not cosmetic:** `vllm==0.8.5` is compiled against torch 2.6.0/cu124 and has no sm_120 kernels — its extensions won't even import against a cu128 torch, so an RTX 5090 image built on the cu124 pins produces a vLLM that fails at `import`. `vllm==0.11.0` (torch 2.8.0, Blackwell kernels) also adds support for the Qwen3-VL / InternVL3.5-era models the configs use. Everything except torch is `--no-cache-dir` installed; detectron2 is built from source with `TORCH_CUDA_ARCH_LIST="12.0"` |
| `build/requirements.txt` | Exact-pinned dependency set (verified via `pip install --dry-run` to resolve with no conflicts). The `torch==2.6.0`/cu124 and `vllm==0.8.5` pins here are now the **legacy baseline**, overridden by **both** Docker images and kept only as reference + for the non-Docker `make install_deps` path. `build/Dockerfile` (cu126) and `build/Dockerfile.rtx5090` (cu128) each strip the same six pins (the three torch packages + `vllm`/`transformers`/`tokenizers`) and install `torch==2.8.0` + `vllm==0.11.0`, differing only in the torch CUDA channel. **Do not assume a pin here is what either GPU image actually runs** — see each Dockerfile's header + strip block |
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

### Last verified build state

As of 2026-07-04, `build/Dockerfile` (the standard RTX 3090 / Titan Xp image)
was upgraded from the legacy cu124/`vllm==0.8.5` stack to the modern
`torch==2.8.0+cu126` / `vllm==0.11.0` / `transformers==4.57.6` stack (see the
table row above) and built + verified **on an RTX 3090 node**. The cu126 set
was pre-checked with `pip install --dry-run` (torch 2.8.0+cu126 + `vllm==0.11.0`
+ the stripped `requirements.txt`), resolving to the same package set as the
5090's cu128 build (`transformers==4.57.6`, `tokenizers==0.22.2`,
`numpy==2.1.1`, `sentence-transformers==3.4.1`, `xformers==0.0.32.post1`) —
only the torch CUDA channel differs. The actual build matched it, compiled
`detectron2` 0.6 from source, downloaded spaCy/NLTK data, and registered
`graph-of-mark 1.1.0` editable. Verified on-GPU (`docker run --gpus all`):
`torch.cuda.is_available()` is `True`, device `NVIDIA GeForce RTX 3090
(sm_86)`, torch's arch list includes `sm_86`, and `from vllm import LLM,
SamplingParams` imports cleanly. Crucially, `AutoConfig.from_pretrained` now
resolves every VLM the configs use **except one**: `Qwen/Qwen3-VL-{4B,8B}-*`
(`qwen3_vl`), `OpenGVLab/InternVL3_5-8B` (`internvl_chat`), and
`omkarthawakar/LlamaV-o1` (`mllama`) all load — these were the architectures
**ABSENT** on the old `vllm==0.8.5` image. The **image was retagged
`gom:latest`** so `run_docker.sh` (default `GOM_IMAGE_NAME=gom:latest`) uses it
unchanged. **Note — `google/gemma-4-12B-it` (`model_type=gemma4_unified`) is
unsupported on BOTH images** (`transformers==4.57.6` raises `ValueError: does
not recognize this architecture`; a fix needs a transformers bump `vllm==0.11.0`
does not yet allow — this is a shared limitation, not a cu126 regression). The
`vlm_comparison.yaml` config was therefore switched to
`google/gemma-3-12b-it` (`model_type=gemma3`, `Gemma3ForConditionalGeneration`,
supported by `vllm==0.11.0`), kept at `fp8: true` so it fits a 24 GB card. It is
a **gated** repo, so the run needs an `HF_TOKEN` — see README_SLURM / the
`HF_TOKEN` note in `run_docker.sh`.

As of 2026-07-02, the earlier cu124 `build/Dockerfile` (torch 2.6.0 / vllm
0.8.5) also built cleanly end to end and passed `python3 -m gom.ablations.main
--help`; that stack only registered the LlamaV-o1 (`mllama`) architecture,
which is why it was superseded by the cu126 build above.

As of 2026-07-04, `build/Dockerfile.rtx5090` was reworked (see its table row
above) and built + verified **on the RTX 5090 node (server 43)**. The cu128
set was pre-checked with a `pip install --dry-run` (torch 2.8.0+cu128 +
`vllm==0.11.0` + the stripped `requirements.txt`) resolving to 215 packages
with no conflicts, and the actual build matched it exactly:
`torch==2.8.0+cu128`, `vllm==0.11.0`, `transformers==4.57.6`,
`tokenizers==0.22.2`, `numpy==2.1.1`, `sentence-transformers==3.4.1` (kept),
`xformers==0.0.32.post1`. detectron2 compiled from source, spaCy/NLTK data
downloaded, and `graph-of-mark 1.1.0` registered editable. Verified on-GPU
(`docker run --gpus all`): `torch.cuda.is_available()` is `True`, the device
is `NVIDIA GeForce RTX 5090 (sm_120)`, torch's arch list includes `sm_120`,
and `from vllm import LLM, SamplingParams` (the exact API `VllmVLM` uses)
imports cleanly — i.e. the old cu124 `vllm==0.8.5` import failure is gone.
This was the fix for that failure; the earlier `Dockerfile.rtx5090` (which
kept `vllm==0.8.5` and only swapped torch to 2.7.1+cu128) would have produced
a vLLM that could not import against the cu128 torch.

## Legacy / Reference Files

- `src/all_in_one_gom.py` — monolithic prototype; not part of the packaged API
- `src/image_preprocessor.py`, `src/vqa.py` — thin CLI wrappers around the package
- `examples/demo_gom.ipynb` — Jupyter notebook walkthrough
