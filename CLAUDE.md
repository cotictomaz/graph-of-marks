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

> **⚠️ 2026-07-06: on-the-fly FP8 does NOT work on the RTX 3090 (Ampere / sm_86).**
> Ampere has no hardware FP8, so vLLM emulates it via Marlin, which requires layer
> dims divisible by 64 — Qwen3-VL fails with `RuntimeError: size_n = 4304 is not
> divisible by tile_n_size = 64`. FP8 is only usable on the 5090 (or other
> sm_89+/dim-aligned models); **on the 3090 use bf16.** All `fp8: true` was removed
> from `vlm_comparison.yaml`. See the "Ablations end-to-end on faretra" session
> section below for the full VRAM/token-budget picture.

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

> **⚠️ 2026-07-06 reality check, UPDATED 2026-07-14 (RTX 3090):** `InternVL3_5-8B`
> and `Qwen2.5-VL-7B` run end-to-end on the 3090, and **`Qwen3-VL-8B-Instruct` also
> runs** — but only via "config D" (`limit_mm video:0` + `max_num_batched_tokens=4096`
> + `gpu_memory_utilization=0.96`; see "## VLM VRAM fitting on the 3090 — measured
> (2026-07-14)"). Its blocker was **vision-encoder *video* profiling at engine init**,
> not the KV/token budget as this line originally claimed. `LlamaV-o1` (mllama) still
> won't serve on `vllm==0.11.0`, and `gemma-3-12b` OOMs loading bf16 weights (FP8 dead
> on Ampere). See the 2026-07-14 section for numbers.

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
  count unique images and to apply the two subsampling knobs `num_images` /
  `questions_per_image` (see below).
- **`metadata`** = `{"answers": [...], "image_file": <basename>, "dataset": "vqav1"}`.
- **`image_path`** = a **resolved local absolute path** (see below), so every
  downstream stage (basename+question-hash preprocessing caches, image
  grouping, `evaluate`) is oblivious to where the image came from.

**Dataset subsampling (`num_images` + `questions_per_image`).** After the full
dataset is built, `main.py` subsamples it with two **independent** global knobs:
- **`num_images`** — how many **unique images** to keep, in first-seen order
  (`-1` = all images). This is the renamed `num_examples`; the old key is still
  read as a fallback (`cfg.get("num_images", cfg.get("num_examples", -1))`) so
  existing configs don't break, but new configs should use `num_images`.
- **`questions_per_image`** — how many **questions to keep per kept image**, in
  first-seen order (`-1` = all questions for that image; default).

The two compose: `num_images: 5, questions_per_image: 2` keeps 5 images × up to
2 questions = up to 10 examples. Setting `num_images: -1, questions_per_image: 1`
keeps one question for **every** image (the old `num_examples`'s accidental
behaviour, now opt-in). The selection scans the whole flat list (no early
`break`) because a selected image's questions are not guaranteed contiguous.
Note this replaced the previous behaviour where `num_examples` silently kept
only the **first** question of each selected image — `questions_per_image: -1`
(the default) now keeps them all.

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

### runc/kernel incompatibility on old-kernel nodes (5.4)

On nodes still running **kernel 5.4** (e.g. faretra / node 40), rootless
Docker's bundled **runc 1.3.x** (`~/bin/runc`) cannot start *any* container —
both `docker run` and every `docker build` `RUN` step die with
`runc create failed: ... can't mask dir "/proc/acpi": mount src=tmpfs ...
nr_blocks=1,nr_inodes=1: invalid argument`. runc 1.2+ masks sensitive `/proc`
paths with a size-limited read-only tmpfs the 5.4 kernel rejects (`EINVAL`).
This is a **runc-newer-than-kernel mismatch**, not a Docker/image/config bug,
and the tutors' `docker_rootless_fix.sh` does **not** fix it (it reinstalls the
*same* latest runc from `get.docker.com/rootless`). **Fix:** shadow the
rootless runc with the node's system runc **1.1.7** (`/usr/sbin/runc`, whose
older masking scheme the 5.4 kernel accepts) and restart the per-user daemon —
`cp ~/bin/runc ~/bin/runc.1.3.6.bak; cp /usr/sbin/runc ~/bin/runc; systemctl
--user restart docker`. No sudo, no impact on other users (only `$HOME` files +
your own rootless daemon are touched; `/usr/sbin/runc` is read-only copied).
**Keep 1.1.7 permanently** — runc runs on every container start (build *and*
the SLURM `docker run`), and the image doesn't bake it in, so reverting
re-breaks the built image at run time. A Docker reinstall/upgrade (or re-running
`docker_rootless_fix.sh`) overwrites `~/bin/runc` back to 1.3.x and reintroduces
the bug. Full writeup + a `--security-opt systempaths=unconfined` per-command
band-aid: README_SLURM.md §7. (Verified 2026-07-04 on faretra: after the swap,
`docker run hello-world` and a trivial `docker build` both succeed.)

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

## Ablations end-to-end on faretra — session findings & fixes (2026-07-06)

Getting `slurm_configs/vlm_comparison.yaml` to actually run VLM inference on the
RTX 3090 node (faretra) surfaced a chain of submission, code, and VRAM problems.
This section records the final code state, the fixes, and the hard numbers so the
next session doesn't re-derive them.

### Submission / infrastructure

- **`run_docker.sh` mounted the wrong dir under `sbatch` (exit 127).** It derived
  the project dir from `${BASH_SOURCE[0]}`, but `sbatch` runs a *copy* of the
  script from `/var/spool/slurmd/job*/slurm_script`, so `-v "$PHYS_DIR":/workspace`
  bind-mounted the spool dir and the container died with
  `/opt/nvidia/nvidia_entrypoint.sh: line 67: /workspace/train.sh: No such file`.
  **Fix:** `PHYS_DIR` now resolves as `GOM_PROJECT_DIR` (env override) →
  `SLURM_SUBMIT_DIR` (the dir `sbatch` was launched from) → the script's own dir,
  with a fail-fast check that errors clearly if `$PHYS_DIR/train.sh` is missing.
- **No shared filesystem across nodes.** The repo *and* the VQA images exist only
  on faretra; a job scheduled elsewhere can't bind-mount `/workspace` or `/images`
  (a probe job on `deeplearn2` couldn't even read `/home/cotic`). **Always submit
  from the repo root and pin the node:**
  `sbatch -N 1 -w faretra --gpus=nvidia_geforce_rtx_3090:1 run_docker.sh <cfg>`.
  Running off-faretra needs the repo cloned there (point `GOM_PROJECT_DIR` at it)
  + `images_base_url` set to faretra's `http.server`.
- **`scancel` leaks the container.** SLURM kills the batch script but the rootless
  `docker run` container keeps running (not in the job's cgroup) and holds GPU
  VRAM — a cancelled run left an orphaned `gom:latest` container pinning ~6 GB on
  a card. After cancelling, `docker ps` + `docker rm -f <id>` to reclaim it. A
  *failed* job (python raises) tears its container down on its own; only `scancel`
  leaks. Check with `squeue -u $USER` (empty = nothing pending/running) and
  `sacct -u $USER --starttime now-1days` for finished/failed jobs.
- **The image-mount sanity check is a false alarm, not a broken mount.**
  `docker run ... gom:latest ls /images | head` prints the CUDA base-image banner
  first; `head` then closes the pipe and the container dies with
  `write /dev/stdout: broken pipe`. Pipe *inside* the container instead:
  `docker run ... gom:latest bash -c 'ls /images | wc -l'` → `1000`. (README_SLURM
  one-time-setup block updated to match.)

### Plain code bugs fixed
- `main.py` called `update_cfg_correct()` with no argument (it requires
  `cfg_updates`) → `TypeError` at startup. Now `update_cfg_correct(None)` (builds
  the default-config preprocessor).
- `nx.node_link_data(G)` emitted a `FutureWarning` per graph (NetworkX 3.6 will
  change the `edges` default). Silenced with the recommended `edges="links"` in
  `pipeline/preprocessor.py` and `graph/scene_graph.py` — no behaviour change
  (`"links"` is the current default).

### The preprocessor-vs-VLM VRAM conflict — `release_preprocessor` (main fix)

`vlm_comparison` / `prompting` (and ablations phase 2) run preprocessing and VLM
inference **in one process**. The GoM preprocessor (YOLO / OWL-ViT / Detectron2 /
SAM / depth / CLIP) holds **~6 GB** of GPU models, and `main.py` kept it resident
during the VLM loop even though inference reads preprocessed images from disk
(`run_vqa(..., skip_preproc=True)`) and never uses it — indeed `main.py` does not
pass the preprocessor to the inference runners at all. Resident, it left only
`18.19 / 23.68 GiB` free, so vLLM's default `gpu_memory_utilization=0.9` (which
demands `0.9 × 23.68 = 21.32 GiB` **free** at startup) failed immediately.

**Fix:** `gom.ablations.utils.release_preprocessor(preproc)` nulls the heavy
GPU-resident submodules in place (so the memory frees even though `main.py` still
holds the object), then `gc.collect()` + `torch.cuda.empty_cache()`. `main.py`
imports it and calls it after each preprocessing phase and before the inference
runner in all three experiment blocks, re-creating the preprocessor
(`if preprocessor is None: update_cfg_correct(None)`) before a later experiment's
preprocessing phase. Net effect: the VLM gets the whole card. (Alternative
considered and rejected: `skip_preprocessing: true` — valid only when images are
already cached, and the goal was to run the whole pipeline in-process.)

(The "demands 21.32 GiB **free** at startup" above is vLLM's *startup gate* — the
secondary role of `gpu_memory_utilization`; its primary role is sizing the KV
cache. See the corrected explanation in "vLLM load knobs" below. This is why
`release_preprocessor` is the right fix — free the card, then a normal 0.90 util
just works — rather than the auto-util sizer, which tries to squeeze into the
memory the preprocessor left behind.)

### vLLM load knobs for a 24GB card — `VllmVLM` new params (`models.py`)

> **⚠️ 2026-07-07 correction + fix applied — the earlier writeup of these knobs
> (and the comments in `models.py`) encoded a wrong model of
> `gpu_memory_utilization`.** The section below is the corrected understanding,
> and the code has since been fixed to match it (see "What was changed in code"
> at the end). The **measured numbers are still valid**; what changed is their
> *interpretation*, which knob is the right lever, and the now-applied defaults.

**How `gpu_memory_utilization` really works.** It is **not** "the memory vLLM
demands be free at startup." It is the **total-budget cap**: vLLM sizes the KV
cache as `KV = gpu_memory_utilization × total − weights − activation_peak −
overhead`. So **higher util → *more* KV cache, not less.** There is *also* a
startup gate (it won't reserve more than is currently free), which is the
`free < desired` error — but that gate is the secondary role, not the main one.
Getting this backwards is what drove the fragile auto-sizer below.

`VllmVLM.__init__` currently sets these (all overridable):

- **`gpu_memory_utilization` auto (`_auto_gpu_mem_util`, headroom 0.02 / cap 0.96).**
  Auto-sizes to `min(cap, free/total − headroom)` from *currently-free* VRAM.
  Measured on the 3090 (8B bf16): util 0.85 → KV **−2.22 GiB** (`No available
  memory for the cache blocks`); 0.94 → **+0.22 GiB**; 0.95 → **+0.46 GiB**
  (works); 0.96 → startup gate fails (`free 22.71 < desired 22.74 GiB`). Viable
  band ≈ 0.945–0.959; auto lands ~0.95. **The correct reading of these numbers:**
  they show KV is a *thin sliver* only because ~16 GB of weights + reserved KV +
  ~1 GB CUDA-context overhead already fill the card — the low numbers are not a
  law of nature, they are the consequence of running an 8B *at util ~0.95 on a
  card that is never 100% clean*. **This auto-sizer is a mistake now that
  `release_preprocessor` frees the card before load:** on a clean card it
  computes ~0.96 and lands in the fail band, so it duplicates and *fights* the
  preprocessor-release fix. A fixed **0.90** (vLLM's own default) is the robust
  choice once the card is clean; raise it only to *buy* KV when weights already
  fit. (The auto path is only genuinely useful if the preprocessor is *not*
  released — which it now always is.)
- **`enforce_eager=True`** — skips CUDA-graph capture (vLLM otherwise captures ~70
  batch sizes, each reserving VRAM). Costs throughput; fine for a small run.
- **`max_num_batched_tokens=2048`** — bounds the startup *profiling* activation
  (allocated on top of the weights, **not** capped by `gpu_memory_utilization`);
  the default 8192 peaked at **22.87 GiB** and OOM'd during profiling. **Footgun:
  for multimodal, a single image's vision tokens generally can't be split across
  prefill chunks, so a value below one image's token count fails with "multimodal
  item cannot fit into max_num_batched_tokens" — it is coupled to `max_model_len`
  and to the vision-token cap, and can't be raised/lowered in isolation.**
- **`max_num_seqs=8`** — vLLM warms up the sampler with `max_num_seqs` (default
  **256**) dummy requests at once, which OOM'd; inference here is `batch_size=1`.
- **`max_model_len=2048`** — caps context (native windows are absurd for VQA,
  Qwen3-VL = 262144) and sets the KV block size + the per-sequence KV minimum.
  **This is a self-inflicted wound, not just a "limitation": 2048 hard-truncates
  GoM prompts (below) and is the direct cause of Qwen3-VL's `decoder prompt ...
  longer than the maximum model length` error and of InternVL only fitting 6/15
  examples.** It should be sized to the *measured* max prompt length and become
  **per-model**; guessing 2048 is what breaks both models.
- **`trust_remote_code=True`** — required to load `OpenGVLab/InternVL3_5-8B`
  (`InternVLChatModel`); harmless for natively-supported models.

**What was changed in code (2026-07-07).** The three problems below were fixed
in `models.py` (and threaded through `run_experiments.py`):
1. **The false "8B eats ~22 GB in weights+activation" claim was corrected.** 8B
   bf16 weights ≈ **16 GB**; single-sequence activation is small. A process
   sitting at ~22 GB is there because util was ~0.95 and vLLM *reserved KV to
   fill that budget* — that memory is mostly **desirable, controllable KV**, not
   weights. The misleading comments (which had justified squeezing util to 0.96)
   are rewritten to say so.
2. **The auto-util sizer (`_auto_gpu_mem_util`) was removed** and
   `gpu_memory_utilization` now defaults to a fixed **0.90**. It duplicated
   `release_preprocessor` (which already frees the card before the VLM loads) and
   drifted into the ~0.96 startup-fail band. Raise toward ~0.95 only to buy KV.
3. **A multimodal-token control surface was added.** `VllmVLM` now accepts
   `max_pixels` / `min_pixels` / `mm_processor_kwargs` (Qwen-style vision-token
   caps, the real lever for oversized image prompts) and defaults
   `limit_mm_per_prompt={"image": 1}` for single-image VQA. `max_num_batched_tokens`
   now defaults to `max_model_len` (so a whole image always fits one prefill
   chunk) instead of a fixed 2048 that could reject a large image.

**Per-model sizing (also applied).** `parse_model_entry` now returns a
`ModelSpec` dataclass, and a `models:` entry may carry per-model
`max_model_len`, `max_pixels`, and `max_tokens` (in addition to `fp8`), threaded
into `VllmVLM` by all three runners. `max_model_len` now defaults to **8192**
(was 2048) — enough for InternVL's full GoM prompts — and should be set
per-model to each model's *measured* max prompt length; Qwen3-VL additionally
needs a `max_pixels` cap to bring its ~16k prompt under that window.

### Model reality on the 3090 (measured)

- **FP8 is dead on Ampere.** The 3090 (sm_86) has no hardware FP8, so vLLM emulates
  it via Marlin, which needs layer dims divisible by 64 —
  `RuntimeError: size_n = 4304 is not divisible by tile_n_size = 64` on Qwen3-VL.
  **All `fp8: true` was removed from `vlm_comparison.yaml`; use bf16 on the 3090.**
  The Per-model FP8 machinery still works where hardware/dims allow (e.g. the 5090).
- **`omkarthawakar/LlamaV-o1` (mllama) is not servable** by `vllm==0.11.0` +
  `transformers==4.57.6`: no native vLLM impl, and the Transformers fallback calls
  `MllamaProcessor._get_num_multimodal_tokens`, absent in 4.57.6 → `AttributeError`
  at engine init. (The earlier "LlamaV-o1 loads" note only meant `AutoConfig`
  resolves it, **not** that vLLM serves it.) **Disabled in the config.**
- **`google/gemma-3-12b-it` doesn't fit the 3090 in bf16** (~24 GB) and FP8 is
  unreliable (above). **Disabled on the 3090; run it on the 5090 in bf16.**
> **⚠️ 2026-07-14 CORRECTION — the "token wall / max_pixels fixes it" story below
> is WRONG for Qwen3-VL.** A dedicated VRAM-probe run (see "## VLM VRAM fitting on
> the 3090 — measured (2026-07-14)" below) showed Qwen3-VL's real blocker is that
> vLLM profiles its **vision encoder with a *video* item at max feature size**
> (~151k-token budget), which fills the card *at engine init, before any prompt* —
> `max_pixels` barely dents it. The fix is `limit_mm_per_prompt={"image":1,"video":0}`
> **plus** `max_num_batched_tokens=4096` **plus** `gpu_memory_utilization=0.96`
> (all three — "config D"), after which the full 16384-ctx GoM prompt runs. The
> paragraph below is retained for history; read the 2026-07-14 section for truth.

- **The GoM-prompt token wall — the current blocker (this is a *token-length*
  problem, largely *not* a memory problem).** GoM prompts are an annotated image +
  scene-graph text, and the token count is **dominated by the image's vision
  tokens**, which differ ~4× between the two VLMs' vision encoders. The `2048`
  `max_model_len` default is what turns this into a hard failure:
  - **`Qwen/Qwen3-VL-8B-Instruct`: ~16,000 tokens/prompt** (overwhelmingly vision
    tokens, not scene-graph text), so **every** request errors with `decoder
    prompt (length 16081) is longer than the maximum model length` and it scores
    0.00%. The reported "**0.46 GiB** KV (~1,600 tokens)" is a *consequence of
    util ~0.95*, not the root cause — even with more KV the 16k prompt would still
    blow past a sane `max_model_len`. **The fix is to cut the image's visual-token
    count** (`mm_processor_kwargs={"max_pixels": ...}` / `limit_mm_per_prompt` /
    downscale the GoM image), *then* set `max_model_len` to the resulting length —
    a GoM-fidelity trade-off, but the only way to make Qwen3-VL usable. Raising KV
    alone does nothing.
  - **`OpenGVLab/InternVL3_5-8B`: ~3,464–4,287 tokens/prompt**, and it leaves
    **6.19 GiB** KV. It **works**: loads, runs, completes (`State COMPLETED`),
    scores real answers. At `max_model_len=2048` only 6/15 examples fit (13.33%)
    **purely because 2048 truncates the longer prompts** — raising `max_model_len`
    to ~5120–8192 (well within its 6.19 GiB KV) should let all examples run.
    InternVL is the practical GoM model on the 3090.

### Current config state & open items

- `slurm_configs/vlm_comparison.yaml` (as of **2026-07-14**) runs `num_images: 100`
  and three 3090-viable models — `Qwen2.5-VL-7B`, `InternVL3.5-8B` (len 8192), and
  `Qwen3-VL-8B-Instruct` as **config D** — with FP8 removed and LlamaV-o1 /
  gemma-3-12b commented out. See "## VLM VRAM fitting on the 3090 — measured
  (2026-07-14)" below for how config D was derived and why the others are out.
- **Done (2026-07-07, see "What was changed in code" above):** per-model
  `max_model_len` / `max_pixels` / `max_tokens` via `ModelSpec`; `max_model_len`
  default 2048→8192; vision-token cap (`max_pixels`/`mm_processor_kwargs`) +
  `limit_mm_per_prompt={"image":1}`; `max_num_batched_tokens` defaults to
  `max_model_len`; auto-util sizer removed, `gpu_memory_utilization` fixed at 0.90;
  false "22 GB weights" comments corrected.
- **Open (still to verify on the 3090):** (1) **empirically tune per-model values**
  — confirm InternVL runs all examples at `max_model_len: 8192`, and find the
  `max_pixels` that brings Qwen3-VL's prompt under its window without wrecking GoM
  fidelity (the values in `vlm_comparison.yaml` are first guesses, not measured).
  (2) Confirm the image-vs-text token split of a GoM prompt (trimming verbose
  scene-graph text is lossless; downscaling the image is a fidelity trade-off).
  (3) Preprocessing is slow — **~100–135 s/image, ~28–37 h for the full
  1000-image set**.

## VLM VRAM fitting on the 3090 — measured (2026-07-14)

A dedicated VRAM-probe run (each model loaded in an **isolated subprocess** via the
production `gom.ablations.models.VllmVLM`, on a **clean** 24GB card — ~23.4GB free —
running 6 real GoM prompts) settled which VLMs actually fit and **corrected** the
earlier "token-wall" story above. The probe harness lived in `vram_probe.py` /
`run_probe_docker.sh` at the repo root (temporary; may be gone). Key operational
note: the probe deliberately did **not** go through `main.py` — `run_vlm_comparison`
does **not** catch a per-model load failure, so one OOM there aborts the whole loop;
the probe isolates each model so one failure can't kill the batch (and a child's
exit fully frees VRAM, which vLLM's in-process teardown does not reliably do).

### Round 1 — which of the candidates load/run at all (all bf16 unless noted)

| Model / config | Result | Root cause |
|---|---|---|
| **InternVL3.5-8B**, len 8192 | ✅ **WORKS**, 6/6 answers | weights 16.8GB → 21.2GB loaded, **32,944-token KV** (4.0× concurrency) — comfortable |
| Qwen3-VL-8B-Instruct, len 16384, full image | ❌ load fail | **OOM in `profile_run` dummy-forward** — weights load (17.0GB) then vision-encoder profiling pushes PyTorch to 22.4GB and OOMs (+768MB) |
| Qwen3-VL-8B-Instruct, len 8192, `max_pixels=200704` | ❌ load fail | `No available memory for the cache blocks` — weights + encoder-profiling budget leave **0 KV** at util 0.90 |
| Qwen3-VL-8B-**Thinking**, len 8192, `max_pixels=200704` | ❌ load fail | identical to Instruct |
| **gemma-3-12b-it**, len 4096 | ❌ load fail | **OOM while *loading weights*** — hits 22.8GB placing shards, can't fit the last (112MB free). ~24GB bf16 weights don't fit a 24GB card at all. HF gated auth worked. |
| Qwen3-VL-8B FP8 (on-the-fly) | ❌ load fail | **Marlin dim error** `size_n = 4304 is not divisible by tile_n_size = 64` — confirms Ampere FP8 is dead |

**The dominant, previously-undocumented culprit for Qwen3-VL:** at engine init vLLM
logs *"Encoder cache will be initialized with a budget of **151,250 tokens**, and
profiled with **1 video item of the maximum feature size**"* and runs that dummy
multimodal forward. That single video-profiling forward is what fills the card on
top of the 16.8GB weights — **not** the prompt length and **not** the KV cache.
`max_pixels` caps a *real image's* prompt tokens but does **not** shrink this
encoder-profiling budget (it only moved 153,600 → 151,250), which is why it "didn't
help". So the earlier "cut `max_pixels`, then set `max_model_len`" advice was wrong.

### Round 2 — rescuing Qwen3-VL-8B-Instruct at `max_model_len=16384`

Three levers tested **separately then together** (base = bf16, `max_pixels=200704`;
weights always load fine at 16.7–17.0GB):

| Cfg | Lever(s) | Result | Why |
|---|---|---|---|
| A | `limit_mm_per_prompt={image:1, video:0}` | ❌ | **Necessary, not sufficient.** Encoder budget collapses **151,250 → 16,384 tokens** (now profiles "1 *image*"). But at util 0.90 only 0.57GB KV is left; a 16384 seq needs 2.25GB → "estimated max len 4176". |
| B | `gpu_memory_utilization=0.96` | ❌ | Useless alone — video profiling still on, so it OOMs in the dummy-forward (same wall as round 1). Higher util can't help a transient profiling *peak*. |
| C | `max_num_batched_tokens=4096` | ❌ | Dodges the dummy-forward OOM, but video profiling still eats the card → `No available memory for the cache blocks`. |
| **D** | **all three together** | ✅ **WORKS** | video-off frees the encoder budget, batch=4096 tames the profiling peak, util 0.96 buys back KV. |

**Config D (the working recipe):** loads in ~43s (weights 16.7GB), **GPU KV cache
16,560 tokens** (1.01× concurrency — exactly one 16384-token request, fine for the
ablations' `batch_size=1`), after-load 19.7GB, **after-inference 23.4GB used /
0.28GB free** — works but razor-thin. Ran 6/6 real answers (66.7% on the tiny set,
e.g. `curved`↔gold `curved`). **All three knobs are required — no single lever
loads.** For a 1000-image run, drop `max_model_len` to ~12288 if you want KV
headroom. *Thinking* was not re-tested with D; it emits ~2048 reasoning tokens/answer
against the same KV pool, so it will have even less slack — check before relying on it.

### Baked into config + code (2026-07-14)

- `ModelSpec` / `parse_model_entry` (`ablations/models.py`) now also carry per-model
  **`gpu_memory_utilization`**, **`max_num_batched_tokens`**, and
  **`limit_mm_per_prompt`** (a dict, e.g. `{image: 1, video: 0}`), on top of the
  existing `max_model_len` / `max_pixels` / `max_tokens` / `fp8`. All are surfaced by
  the new `ModelSpec.vllm_overrides()` helper, and the three runners in
  `run_experiments.py` build the model with `vllm_kwargs.update(spec.vllm_overrides())`
  (replacing three copies of the old per-field `if spec.x is not None` block — one
  place to add future knobs). `VllmVLM` already accepted all of these.
- `slurm_configs/vlm_comparison.yaml`'s `models:` now runs the three 3090-viable
  models: `Qwen2.5-VL-7B` (anchor), `InternVL3.5-8B` (len 8192), and
  `Qwen3-VL-8B-Instruct` as **config D** (the full 5-knob entry). LlamaV-o1 /
  gemma-3-12b stay commented out with their failure reasons.

## Why preprocessing is slow & GPU-idle — the matplotlib viz bottleneck (2026-07-07)

Symptom that prompted this: two ablation jobs (`ablation_experiments.yaml` on
`moro232`, `vlm_comparison.yaml` on `faretra`) showed **~0% GPU utilization** in
the cluster web UI while in the preprocessing phase. **This is expected, not a
hang.** Sampling `nvidia-smi` on the job nodes (`srun --jobid=<id> --overlap
nvidia-smi`) shows GPU util *fluctuating* (e.g. faretra GPU0 51→56→66→0→0%, ~6–7 GB
resident for the preprocessor) and the tqdm `Preprocessing` bar advancing — the
job is progressing. The flat-0% snapshots are the **CPU-bound matplotlib
rendering** of each scene, during which the GPU (detectors/SAM/depth/CLIP) is
idle. On the crowded VQA scenes rendering dominates the per-image time.

**Where the time goes (all in `src/gom/viz/visualizer.py`).** Render entry is
`draw()` (`visualizer.py:299`), which runs `_create_canvas` → `_draw_objects` →
`_draw_labels` → `_draw_relationships` → `_draw_legend` → `_finalize_figure`.
Two dominant cost centers (profiled on a 640×425, ~75-relation scene):

- **`draw(dpi=800)` default (`visualizer.py:310`).** `_create_canvas`
  (`:507`) builds `plt.subplots(figsize=(W/100, H/100))` (`:528`), so a 640×425
  image renders on a ~5120×3400 canvas — the final `savefig`/image resample
  alone was **~11.7 s**.
- **`_draw_relationships()` (`visualizer.py:838`) — the ~58.5 s hotspot.** Per
  relation it builds a curved `FancyArrowPatch` (`connectionstyle="arc3"`,
  `:918`) + a rounded-bbox `ax.text()` label (`:954`) — linear in relation count.
  The real killer is **label-overlap resolution** (`:986–1059`): it calls
  `fig.canvas.draw()` (`:988`, and again in helpers at `:1126`, `:1362`) — a full
  canvas rasterization *just to measure text boxes* via `get_window_extent()`
  (~22 s cumulative, forcing `text._get_layout` ~15 s + repeated font-manager
  hashing) — then runs the adjustText-style iterative solver
  (`_resolve_relation_vs_relation_overlaps` `:2047`, `_resolve_overlaps` `:2319`)
  **up to four times per image**. That "draw-canvas → measure → nudge → repeat"
  loop is what pins one CPU core and produces the flat-0%-GPU profile. Object
  labels have the same pattern in `_draw_labels()` (`:1064`, canvas-draw at
  `:1126`).

**Why it can't be cached / re-runs per grid point.** The whole path is gated by
`cfg.display_relationships` / `cfg.display_relation_labels` / `cfg.resolve_overlaps`
(`:875`, `:932`, `:986`), and the ablation grids toggle exactly these viz knobs,
so `_draw_relationships` re-runs for every grid point.

**Cheapest high-impact levers, in order:** (1) drop `dpi=800`→~150–200 in
`draw()` (`:310`) — near-linear speedup; a `RENDERING_OPT_AVAILABLE` fast path
(`visualizer.py:65/68`, used at `:709`/`:1120`) is worth checking. (2) Gate
`resolve_overlaps` or cap its iterations — the four solver passes (each doing a
full `fig.canvas.draw()`) are the single biggest cost. (3) Longer term: replace
the `FancyArrowPatch`/text path with PIL/cv2 drawing to eliminate the
canvas-redraw-to-measure loop entirely. None of these has been applied yet.

**Op note:** logs/stdout for a job live on the **local disk of its node** (no
shared FS), so `slurm-<id>.out` is only readable from the node it ran on — read a
remote job's log via `srun --jobid=<id> --overlap tail .../slurm-<id>.out`, not
from the submit host.

## Paper-mismatch: preprocessing diverged from the upstream default (2026-07-08)

Qwen2.5-VL VQA numbers didn't match the paper. Root cause: this fork's
**preprocessing/rendering** diverged from the upstream default
(`disi-unibo-nlp/graph-of-marks`), **not** the inference wiring. Diffing the two
package trees (only `src/gom/ablations/` is fork-only; the rest is a shared
lineage) showed the inference path already matches the paper — `run_vqa` runs in
the equivalent of upstream's default **`visual_textual`** mode: the annotated
image **plus** the textual scene-graph triples are sent, with the same
spatial-reasoning system prompt (`vqa/runner.py` prepends `scene_graph_text` when
`include_scene_graph=True`, which all three ablation runners pass). The
divergences are all in how the marked image is produced:

- **`aggressive_pruning` — the real regression.** Upstream's VQA driver *forces*
  `aggressive_pruning=True` for every preprocess call (both the
  `vqa/preproc.py::preprocess_for_qa` default **and** the `vqa/runner.py` call
  site). This fork **commented those out** (in `ablations/utils.py::preprocess_for_qa`'s
  `cfg_updates` block and at `vqa/runner.py:377-378`), so it falls back to the
  config default `aggressive_pruning=False` (`preprocessor.py:427`). Effect:
  without question-relevant pruning, **every** detection (up to
  `max_detections_total=80`) is marked, so the VLM sees a far busier image than
  the paper's clean, question-focused one. (Originated from the "Disable question
  based filtering for ablations experiments" commit — intended for the *grid*
  ablations, but it leaked into `vlm_comparison` / `prompting` too.) Note
  `apply_question_filter` stays effectively `True` in both; only the
  distance-based aggressive pruning changed.
- **`sam_version`** default `"1"` (SAM v1) upstream → `"hq"` (SAM-HQ / `vit_h`)
  here (`preprocessor.py`), so the drawn masks differ.
- **`viz/visualizer.py VisualizerConfig`**: `max_relations_per_object` 1→3,
  `auto_scale_styles` `True`→`False` **in the RenderConfig default** — but
  `preprocessor.py` (`auto_scale_styles=True`, passed through to the visualizer)
  overrides that at runtime, so the effective value is ≈`True` in current code;
  the 2026-07-06 `ImagePreprocessing1.log` showing `False` / `max_relations_per_object=3`
  simply predates those code changes (the log is **not** authoritative for
  current defaults — read the dataclasses).
- **Not relevant to these VQA runs** (ruled out): depth model `vitl`→`vits`
  (`utils/depth.py`; depth is skipped, `enable_spatial_3d=False`), and
  `api.py`'s `sam_hq_model_type` `vit_h`→`vit_b` (only the `GoM` API path; the
  ablations use the config path, still `vit_h`).
- **Live-but-latent bug:** `config.py:157 detectors_to_use = ("yolov8")` is a
  **string, not a tuple** (would iterate per-character). It only lives in the
  *fallback* `PreprocessorConfig` (used if the `from gom.pipeline.preprocessor
  import PreprocessorConfig` at `config.py:97` fails); the authoritative
  `preprocessor.py:448` still lists all three detectors. Worth fixing to
  `("yolov8",)` so a fallback import can't silently gut detection.

### Which config is authoritative

`gom.config.default_config()` returns
`gom.pipeline.preprocessor.PreprocessorConfig` — `config.py` imports it and only
falls back to its own lightweight class on `ImportError`. So the **real**
defaults live in `preprocessor.py`, not in the `config.py` dataclass (they drift:
e.g. `config.py` says `sam_version="1"`, `preprocessor.py` says `"hq"` — the run
uses `"hq"`).

### The fix — per-experiment `preprocessing_overrides` (vlm_comparison + prompting only)

`main.py` reads a `preprocessing_overrides:` map per experiment type and forwards
it as `cfg_overrides` into `generate_default_dataset` → `run_preprocessing` →
`preprocess_for_qa` → `update_cfg_correct` (applied to `cfg` and propagated to the
`visualizer` / `relations_inferencer` sub-configs). Added to
`slurm_configs/vlm_comparison.yaml` and `slurm_configs/prompting_experiments.yaml`
**only**:

```yaml
preprocessing_overrides:
  aggressive_pruning: true   # the real fix (code default is False)
  auto_scale_styles: true    # already ≈True in current code; pinned for drift-safety
  max_relations: 10          # already the default (RelationsConfig/preprocessor); pinned
```

> **2026-07-17: both blocks gained three more keys** (`enforce_max_global`,
> `enforce_max_per_object`, `max_relations_per_object: 999`) so that `max_relations:
> 10` above is *actually enforced* — on its own it did nothing. See "## Relation caps
> were never enforced" below for why, and read the live YAML for the current block.

`ablation_experiments.yaml` was intentionally **untouched** at the time — but this
is now **stale**: the grid path has since gained its own `preprocessing_overrides`
(`main.py:493` → `generate_ablated_dataset(..., preprocessing_overrides=...)`), and
that file pins `aggressive_pruning: true` / `auto_scale_styles: false` across every
grid point (the swept grid params still win on a key collision). `max_relations_per_object`
was deliberately **not** changed here (still `5` in `preprocessor.py` vs the paper's
`1` — open item) — **superseded 2026-07-17**: vlm_comparison/prompting now pin
`max_relations_per_object: 999` to disable per-object capping (see "## Relation caps
were never enforced" below). Overrides only take effect on **regeneration**: set
`skip_preprocessing: false` + `force_reprocess: true` to rebuild the stale images
(inference otherwise reads them from disk). The vlm_comparison and prompting
preprocessing outputs are content-identical **only while the two configs'
`preprocessing_overrides` blocks actually match** (same dataset/subsample, same
overrides, deterministic pipeline); when they do, they land in different
`{base_dir}` trees, so one can be generated once and **symlinked** into the other's
`preprocessed_images/<exp>/default/` to avoid preprocessing twice. **Diff the two
blocks before symlinking** — they have drifted before (`auto_scale_styles`), and a
symlink silently serves the wrong images.

### Preprocess-only run pattern

To (re)generate the corrected image set without loading a VLM, set `models: []`
(no model → no vLLM load) with `skip_preprocessing: false` + `force_reprocess:
true`; `main.py` runs the preprocessing phase then the inference runner as a
0-model no-op. Used 2026-07-08 to rebuild `vlm_comparison`'s images with the
pruning fix before the real comparison run. Remember to flip
`skip_preprocessing`/`force_reprocess`/`models` back afterward.

## Relation caps were never enforced — audit, fix & rename (2026-07-17)

An audit of the `ablate_max_relations_global` / `ablate_max_relations_per_object`
grids (counting edges in the saved `*_graph.json`, i.e. links carrying a
`relation` attribute — the structural `scene -> object` edges have none and are
excluded; this count matches `*_graph_triples.txt` exactly) found **both caps were
silently ignored in the output**:

| grid | violating images | worst case |
|---|---|---|
| `max_relations` X ∈ {0,4,8,12,16,24,32} | **52% → 18%** (falls as X rises) | 60 relations at **X=0**; 88 at X=32 |
| `max_relations_per_object` Y ∈ {1,3,5,7,9} | **68% → 16%** | out-degree 15 at Y=9 (48.7% of objects violate at Y=1) |

`max_relations_0/` containing images with 60 relations is the giveaway. Violations
*fall* as the cap rises because the caps never bound anything — scene geometry did.

### Root cause — `build_scene_graph` never knew about the caps

`limit_relationships_per_object` (`inference.py`) correctly produces the capped
`rels_all` (`preprocessor.py:3961`), and `drop_inverse_duplicates` only removes.
Then `build_scene_graph` (`preprocessor.py:361`, called at `:4036`) **rebuilds its
own edge set from proximity alone** — `_candidate_neighbors` (`scene_graph.py:607`)
keeps every neighbour within `max_dist_norm=0.4` up to `max_neighbors=32`, and
`_maybe_add_edge` (`:626`) only drops pairs with `iou < min_iou_keep=0.01` **and**
`clip_sim < 0.20` (so in crowded, overlapping COCO scenes almost nothing is
dropped). The `rels_all` loop at `:4047-4057` then merely *labels or adds* edges —
it can never remove one — and the fallback at `:4089` labels every leftover
geometric edge via `_infer_relation_from_attrs`, promoting it to a full relation.

This was never only a `graph.json` problem: `rels_for_viz` (`:4114`) is re-extracted
**from the scene graph**, so the surplus edges reached the drawn arrows too, and
`graph_to_triples_text` (`prompt.py:445-447`) **re-infers a relation for any edge
lacking one** — so unlabelling an edge is useless, it must be **deleted**.

### The fix — prune the graph down to `rels_all`

A prune block (`preprocessor.py`, right after the labeling loop) deletes every
object-object edge not in `rels_all`. **Order matters**: it runs *after* the
labeling loop, which first adds any capped relation geometry never proposed, so
`rels_all ⊆ edges` and the result is exactly the capped set. Gated:

```python
if need_rel and (self.cfg.enforce_max_global or self.cfg.enforce_max_per_object):
```

`need_rel` matters — when relations were never inferred `rels_all` is `[]` meaning
"unknown", not "capped to zero"; pruning there would strip a legitimately
un-capped graph. (`:4132` already gates the viz extraction the same way.) The
`enforce_*` gate keeps the **default pipeline unchanged** unless a config opts in.

Alternative rejected: capping inside `build_scene_graph`. It selects by *nearest*,
not by importance; it has no global-budget concept; and it wouldn't work anyway —
the labeling loop's `add_edge` can push a node back over the cap regardless.

### Global cap: off-by-one + it kept the *nearest* N, not the most important

Two separate bugs in the global filter (`inference.py`):

- **Off-by-one**: `sorted(...)[:global_budget - 1]` → `max_relations=1` yielded **0**
  relations, `=4` yielded 3. Now `[:global_budget]`.
- **Wrong ranking**: the old `global_sort_key` sorted on raw score only, **dropping
  both** question-relevance and relation priority — the two things the per-object
  stage had just prioritised. Worse, that single axis mixes **incomparable scales**,
  because relation dicts carry different fields: geometric relations have only
  `{src_idx, tgt_idx, relation, distance}` → `_get_relation_confidence` falls back to
  inverse distance `1/(1+d/100)` (≈1.0 when close); physics/3D carry `confidence`
  0.75–0.8; CLIP relations carry `clip_sim` ≈0.2–0.3. Measured: a `touching` pair at
  d=5px scores **0.952**, beating a priority-4 `on_top_of` (0.800) and `holding`
  (0.280). Inverse distance beats `confidence=0.8` whenever d<25px and `clip_sim=0.28`
  whenever d<~300px — i.e. **the global cap approximated "keep the N nearest pairs"**
  and CLIP relations were almost always ranked last.

Now it reuses the per-object `rel_sort_key`, so both axes agree on "most important":
`(q_priority, -rel_priority, -score, distance)` → question-relevant first, then
relation priority (`_get_relation_priority`: semantic 4 > contact 3 > proximity/
directional 2 > other 0), then confidence/proximity. Budgets nest properly (each
larger N extends the smaller). **Residual**: the 3rd term still mixes scales, so
within one priority tier a close geometric relation outranks a CLIP one.

### The small-object guard (`area_ratio`) was never wired up

`if i < len(areas): rel_cap = min(rel_cap, 2)` — the comments (`# Compute area-based
caps to avoid overcrowding small objects.` / `# Shrink cap for small objects`) say it
should only hit **small** objects, but `i < len(areas)` is a **bounds check that is
always true**. Introduced in `d229a79` already broken: it computed
`area_ratio = areas[i]/max_area` and **never tested it**; `6436842` then deleted the
`area_ratio` line entirely, leaving `max_area` as dead code. So a small-object cap
applied to **every** object, making `max_relations_per_object` a no-op above 2.

Fixed: the guard now tests `area_ratio` against **`RelationsConfig.small_object_area_ratio`
(0.25)** and clamps to **`small_object_relation_cap` (2)**; `max_area` is live again.
The ratio is relative to the scene's **largest box** (the original author's design),
not the image. **The 0.25 threshold is a judgement call** — the original never had one.
⚠️ Both new fields live on `RelationsConfig` **only**, so they are **not settable from
YAML**: `update_cfg_correct` skips any key absent from `PreprocessorConfig`
(`ablations/utils.py:53`) — add them there too if they need to be tunable.

### ⚠️ `auto_adjust_relation_cap` makes `max_relations_per_object` inert

`_compute_effective_max_relations_per_object` runs **before** the per-object branch
and, whenever `auto_adjust_relation_cap` is on, returns **1 for any scene size**
(measured for n=2…60): `cap = ceil(min(max_relations, ~n, rel_count) / n)`. It is
**on by default** (`preprocessor.py:474`), so in `vlm_comparison` / `prompting` /
`ablate_edge_thickness` the effective cap is 1 and `max_relations_per_object` does
nothing. `main.py` turns it **off** for `ablate_edge_color` and
`ablate_max_relations_global` (cap stays 3). This is why the `area_ratio` fix is a
no-op for the former group and live for the latter.

Also note the non-ablation branch makes the per-object cap a **soft** limit:
`final.extend(q_sorted + other_sorted[:remaining])` extends `q_sorted` **in full**,
so an object with more question-relevant relations than `rel_cap` exceeds it by
design. The `enforce_max_per_object=True` branch truncates cleanly instead.

### Renamed toggles: `ablate_*` → `enforce_*`

`ablate_max_global` → **`enforce_max_global`**, `ablate_max_per_object` →
**`enforce_max_per_object`** (all 27 occurrences: `inference.py`, `preprocessor.py`,
`main.py`, both YAMLs — including the `getattr(self.config, "...", False)` string
literals). The old names implied "this is ablation-only machinery"; they are now the
pipeline's real cap switches. Semantics:

- **`enforce_max_global`** — apply `max_relations` (and prune the graph to it).
- **`enforce_max_per_object`** — enforce `max_relations_per_object` **literally**,
  bypassing the heuristics (`auto_adjust_relation_cap`, the small-object cap).

**There is no "no per-object cap" switch.** `limit_relationships_per_object` always
runs one of two branches; the flag picks *which*. The heuristic branch (`False`)
cannot be neutralised — auto-adjust pins it to 1. So **"global-only capping" is
expressed as `enforce_max_per_object: true` + `max_relations_per_object: 999`** (a
big number, not `0` — `0` means "drop all relations").

### vlm_comparison / prompting now enforce a global-only cap of 10

`max_relations: 10` had been in both configs since 2026-07-08 doing **nothing**
(the global filter only runs under `enforce_max_global`). Both now carry:

```yaml
preprocessing_overrides:
  aggressive_pruning: true
  auto_scale_styles: false        # true in prompting — the two configs differ here
  max_relations: 10
  enforce_max_global: true        # actually enforce the 10, and prune the graph to it
  enforce_max_per_object: true    # bypass the per-object heuristics
  max_relations_per_object: 999   # => no per-object limit
```

Result: ≤10 relations per image, chosen purely by importance, with `graph.json`, the
triples text and the drawn arrows all agreeing. **This is a large visual change** —
those images previously showed an uncapped geometric graph. Render one image and
look at it before committing to a full run.

### What must be re-run (`force_reprocess: true`)

| run | affected by | re-run? |
|---|---|---|
| `ablate_max_relations_global` | prune + off-by-one + ranking + area guard | **yes** (7 grid pts) |
| `ablate_max_relations_per_object` | prune (takes the `enforce` branch; area guard N/A) | **yes** (5 grid pts) |
| `ablate_edge_color` | area guard only (auto-adjust off → cap 3, so 2→3 on large objects) | **yes** (3 grid pts) |
| `vlm_comparison` / `prompting` | now opt into global-only capping | **yes** |
| `ablate_edge_thickness` | nothing (auto-adjust → cap 1 → area guard unreachable) | no |

~100–135 s/image; at `num_images: 100` budget ~3 h per grid point.

### Verification status (2026-07-17)

Verified **without a GPU** (host has no torch — run tests via
`docker run --rm -v "$PWD":/workspace -w /workspace gom:latest python3 ...`):
the cap function's global budget/ranking; the prune reducing a real
`build_scene_graph` (24 geometric edges → exactly the capped set, scene edges/nodes
intact, X=0 → 0); the area guard discriminating by size; and the renamed toggles
propagating YAML → `PreprocessorConfig` → `RelationsConfig` via the real
`update_cfg_correct` (driven with a stub preprocessor — no models needed).
**Not** verified end-to-end on a real image: the first regenerated grid point is the
real test — re-run the audit against it and expect zero violations.

## Legacy / Reference Files

- `src/all_in_one_gom.py` — monolithic prototype; not part of the packaged API
- `src/image_preprocessor.py`, `src/vqa.py` — thin CLI wrappers around the package
- `examples/demo_gom.ipynb` — Jupyter notebook walkthrough
