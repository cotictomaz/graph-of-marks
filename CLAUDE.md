# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Graph-of-Mark (GoM), the code for the AAAI-26 paper (`paper/`). It's a visual-prompting
pipeline: an image is turned into a marked-up scene graph (segmentation masks + object ID
labels + relation-labeled arrows) that is fed to a multimodal LLM to improve spatial reasoning
on VQA / referring-expression tasks. The pipeline is **preprocessing only** — it produces the
annotated image + a textual scene graph; the VLM inference is a separate step.

## Two code paths (do not confuse them)

- **`src/gom/`** — the refactored, installable package (`graph-of-mark`). This is where all
  development happens. Entry points: the `gom-preprocess` / `gom-vqa` console scripts
  (→ `src/gom/cli/`), the top-level CLI `src/image_preprocessor.py`, and the `gom.api.GoM`
  facade.
- **`src/all_in_one_gom.py`** — the original ~3100-line monolithic script that produced the
  paper's numbers. **Frozen reference — never edit it.** It has known quirks (crashes on some
  images at `rel.split()[1]`, no per-image error handling, ~3–10 min/image). To run it in
  batch fault-tolerantly, use `data_paper/run_original_batch.py`, which imports the class
  unchanged and wraps each image in try/except.

`original_versions/` also holds frozen historical copies; treat as read-only.

## Environment

There is **no system Python ML environment** (no torch on the host). Everything runs in Docker
with the repo bind-mounted at `/workdir`, GPU passthrough, and `PYTHONPATH=/workdir/src`.
Relevant local images: `gom-review` (preprocessing deps: torch 2.7.1+cu128, detectron2, SAM/
SAM-HQ, ultralytics, spaCy+`en_core_web_md`, nltk wordnet), `vllm-serve` (vLLM inference —
must be cu128: torch 2.8.0 + vLLM 0.10.2 + transformers 4.56.2; in-container vLLM needs
`enforce_eager=True`, `-e LIBRARY_PATH=/usr/local/cuda/lib64/stubs`, and `python3-dev`).

`.env` holds `HF_HOME` (→ `/home/molfetta/llms`), `HF_TOKEN` (gated models like Gemma-3), etc.
It is **not auto-loaded** except via `gom.utils.env.load_dotenv`, which is called at the top of
the CLI entry points. When passing the HF token to a container, strip the surrounding quotes
from the `.env` value (`--env-file` passes them literally).

Typical run pattern:
```
docker run --rm --gpus all --env-file .env \
  -v $PWD:/workdir -v /home/molfetta/llms:/llms -v /home/molfetta/llms/torch_cache:/root/.cache/torch \
  -w /workdir -e PYTHONPATH=/workdir/src -e PYTHONUNBUFFERED=1 gom-review:latest \
  python3 src/image_preprocessor.py ...
```

## Commands

- **Tests**: `pytest tests/` (config in `pyproject.toml`; `addopts` enables `--cov=gom`, so
  pass `--override-ini addopts=""` if pytest-cov isn't installed). Run one test:
  `pytest tests/test_integrity.py::TestRelationsConfig::test_relations_config_creation`.
  Tests are import/config/shape smoke tests only — **no test exercises the numeric pipeline**,
  so detection/relation/graph regressions are not caught automatically.
- **Lint/format** (dev extras): `black src tests`, `isort src tests`, `flake8`, `mypy`.
- **Preprocess (package path)**: `python3 src/image_preprocessor.py --json_file <pairs.json>
  --output_folder out/ [render flags]`. Input JSON is a list of `{"image_path", "question"?}`.
  Question filtering is on by default; pass `--disable_question_filter` for a question-agnostic
  render. See `Makefile` (`make preprocess`, `make run_vqa`) for the flag conventions.
- **VQA inference**: `scripts/run_vqa_inference.py` (vLLM) — modes `raw|visual|visual_textual`;
  `scripts/run_ref_inference.py` for referring-expression (REC). The legacy `src/vqa.py` /
  `gom-vqa` path is thin wrappers and less maintained.

## Pipeline architecture

`ImageGraphPreprocessor.process_single_image` (`src/gom/pipeline/preprocessor.py`, ~4600 lines
— the orchestrator) runs, in order:
1. **Detection** — ensemble of OWLv2 + YOLOv8 + Detectron2 (`src/gom/detectors/`, via
   `DetectorManager`), merged with **Weighted Boxes Fusion** (`src/gom/fusion/`).
2. **Question filtering** (Algorithm 3) — prunes to query-relevant objects + neighbors when a
   question is given.
3. **Segmentation** — SAM variants (`src/gom/segmentation/`: `sam1`/`sam2`/`samhq`).
4. **Depth** — MiDaS DPT-Large by default (`src/gom/utils/depth.py`; Depth-Anything-V2 is
   opt-in). Convention: normalized [0,1], **higher = closer**.
5. **Relations** — `RelationInferencer` (`src/gom/relations/inference.py`, ~1900 lines):
   geometric directional (left/right/above/below), depth (in_front_of/behind), proximity;
   CLIP/physics extras exist but are opt-in.
6. **Scene graph** — `SceneGraphBuilder` (`src/gom/graph/scene_graph.py`) → NetworkX DiGraph.
   `build_scene_graph` in the preprocessor wraps it; the relation engine's edges are the
   authority (graph geometric edges are pruned to the filtered relations).
7. **Prompt/triples** — `src/gom/graph/prompt.py` serializes the graph. Paper triple format is
   `head -(relation)-> tail`.
8. **Render** — `src/gom/viz/visualizer.py` draws masks/IDs/arrows.

Config: `PreprocessorConfig` is defined in **`src/gom/pipeline/preprocessor.py`** (the real
one); `src/gom/config.py` has a **fallback duplicate** used only if that import fails — they
have historically diverged, so check which one is active when defaults look wrong. The
`gom.api.GoM` facade sets its own hardcoded model config that differs from the CLI defaults.

## Paper reproduction & known state

`REVIEW.md` (root) is the authoritative record of the replication audit, bug fixes, and the
paper-faithful eval results — read it before doing reproduction work. The July 2026 post-fix
VQAv2 rerun beats raw when every condition is constrained to an officially scorable short
answer (Gemma-3: 59.45 raw → 63.01 GoM visual), but not with unconstrained supplementary-prompt
generations under the explicit phrase-compatibility scorer (64.03 → 53.33). The aggregate
short-answer gain is not a spatial-slice gain; see `REVIEW.md` §10 before interpreting it as
support for the paper's mechanism.

Reproduction infra lives in `data_paper/`: `download_paper_data.py` (HF-streaming 1K-image
samples with official-metric fields), `vqa_metrics.py` (official VQAv2 leave-one-out soft
accuracy plus an explicitly named verbose-response compatibility scorer), and
`eval_paper_faithful.py` (supplementary prompts, optional equal short-answer constraint,
greedy decode, per-slice metrics), and `grid_results.txt`.
`data_paper/` and `data_subsample/` are git-ignored working data.
