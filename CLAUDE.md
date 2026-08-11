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
with GPU passthrough and `PYTHONPATH=<repo>/src`. The two current images are built by
`reproduce.sh` from `reproduction/docker/`:

- `gom-paper-preprocess:1` — preprocessing deps (torch 2.7.1+cu128, detectron2, SAM/SAM-HQ,
  ultralytics, nltk wordnet, gensim for FastText). **spaCy is deliberately not installed**
  — `src/gom/nlp.py:208` imports it lazily and silently falls back to a token heuristic on
  `ImportError`. The paper profile drives Algorithm 3 from FastText, so the fallback never
  runs on that path; do not add spaCy to fix a phantom problem.
- `gom-paper-inference:1` — vLLM inference, must be cu128 (torch 2.8.0 + vLLM 0.10.2 +
  transformers 4.56.2); in-container vLLM needs `enforce_eager=True`,
  `-e LIBRARY_PATH=/usr/local/cuda/lib64/stubs`, and `python3-dev`.

(Older docs/scripts refer to `gom-review` / `vllm-serve`; those images are gone.)

`.env` (template: `.env.example`) holds `HF_TOKEN` (gated models like Gemma-3) and optionally
`HF_HOME`; on the original box that was `/llms`, a shared root-owned dir. The reproduction path
ignores host `HF_HOME` and mounts `--model-cache` at `/model-cache` instead. `.env` is
**not auto-loaded** except via `gom.utils.env.load_dotenv`, which is called at the top of
the CLI entry points. Do not quote values — `--env-file` passes quotes through literally, so
a quoted token arrives inside the container with the quotes attached.

Typical run pattern (mirrors `docker_base()` in `reproduction/reproduce.py:369` — note the
repo is mounted at its *host* path, not `/workdir`, and all model downloads go to one
`/model-cache` mount):
```
docker run --rm --gpus all --env-file .env \
  -v $PWD:$PWD -v ~/.cache/gom-paper:/model-cache \
  -w $PWD -e PYTHONPATH=$PWD/src -e PYTHONUNBUFFERED=1 \
  -e HF_HOME=/model-cache -e TORCH_HOME=/model-cache/torch_cache \
  gom-paper-preprocess:1 python3 src/image_preprocessor.py ...
```

## Commands

- **Tests** (195, all passing): the suite needs torch, so it only runs in the container, and
  the image ships no pytest — install it in the throwaway run:
  ```
  docker run --rm -v $PWD:$PWD -w $PWD -e PYTHONPATH=$PWD/src gom-paper-preprocess:1 \
    bash -c "pip install -q pytest && python3 -m pytest tests/ --override-ini addopts='' -q"
  ```
  `--override-ini addopts=''` is required (`pyproject.toml` `addopts` enables `--cov=gom`;
  pytest-cov isn't installed). On the host, `PYTHONPATH=$PWD/src python3 -m pytest -c /dev/null`
  runs the ~130 torch-free tests; the rest error at collection (host pytest is 6.2.5, below
  the `minversion = 7.0` in `pyproject.toml`, hence `-c /dev/null`).
  Beyond import/config smoke tests there is now real logic coverage — `test_paper_relations.py`,
  `test_relation_enforcement.py`, `test_paper_query_filtering.py` (Algorithm 3),
  `test_reproduction_scoring.py`, `test_preprocessor_dedup.py`. Still **no end-to-end numeric
  run**: detector/segmenter/depth regressions are only caught by `reproduction/audit_relations.py`
  on real artifacts.
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
   `DetectorManager`), merged with **Weighted Boxes Fusion** (`src/gom/fusion/`), then
   per-class NMS at IoU 0.5. That NMS is absent from Algorithm 1's pseudocode but required —
   without it three detectors turn one object into several marks; see `reproduction/README.md`
   §"Published Spec vs Released Artifact".
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

**Profiles** live in `src/gom/config.py`: `default_config(profile)` applies one of
`quality_vqa` (default), `paper_legacy`, or `paper_aaai26`. `paper_aaai26` is the locked
paper-declared setup (SAM-HQ + MiDaS DPT-Large + FastText-driven Algorithm 3 + six render
variants); `validate_paper_config()` raises on any drift from `PAPER_AAAI26_LOCKED_FIELDS`, so
**do not "fix" a paper-profile default without changing that table** — it is called from
`src/image_preprocessor.py` and `gom.api`. Select it with `--profile paper_aaai26`.
It hard-requires `--paper_fasttext_path` (converted `cc.en.300.kv`); Algorithm 3's semantic
half is inert without it.

## Setting this repo up on a new machine

`reproduction/run_afk.sh` is the bootstrapper — it host-preflights, builds both images,
downloads every model, verifies the four pinned weights, installs the image splits,
converts FastText, then runs the pipeline. See `reproduction/README.md` §"Fresh Machine
Setup" for prerequisites and knobs.

**Exactly one artifact cannot be downloaded and must be copied by hand:**

```
data_paper/gom_datasets.zip     605 MB
sha256 a9c0f446ed4d99bcb7e00cbc3cd686d9fe19149ad3a1015a379e05569992f404
```

It contains the four exact 1,000-image author splits and nothing else — no annotations,
which live in the git-tracked `reproduction/manifests/*_author.json`. No URL for it exists
anywhere in the repo. `reproduce.py` and `run_afk.sh` both read it from that path;
`GOM_DATASET_ARCHIVE` overrides. If it is missing, `host_preflight` fails immediately and
says so — do not work around this by substituting other images, `prepare_datasets.py` is
fail-closed on the archive hash and every image basename for good reason.

Everything else is automated: `prefetch_models.py` fetches OWLv2, MiDaS, Detectron2,
SAM-HQ, YOLOv8x and the three revision-pinned VLMs; `prepare_fasttext.py --download`
fetches and converts `cc.en.300.vec`. Model cache defaults to `~/.cache/gom-paper` in both
entry points (`GOM_MODEL_CACHE` to override).

GPU: the cu128 images run unchanged on Ampere/Ada. `GOM_BLACKWELL=1` is only for sm_120,
and the per-model VRAM floors are `GOM_VRAM_<MODEL>`.

## Paper reproduction & known state

Read, in this order, before touching anything reproduction-related:
`reproduction/RESULTS.md` (current verified numbers), `reproduction/README.md` (how to run it,
and every documented paper/artifact conflict), then `REVIEW.md` (the older replication audit
and bug-fix log — its §9/§10 VQAv2 numbers are **superseded** by the Table 2 run).

**`reproduction/` is the canonical path.** One command, resumable at every stage:
```
./reproduce.sh table2 --data-root <dir> --fasttext <cc.en.300.kv> --resume
```
Subcommands: `plan` (prints the matrix, no models loaded — use this to sanity-check first),
`preflight`, `datasets`, `prepare`, `preprocess`, `audit`, `inference`, `score`, `vqav2`,
`table2`. `reproduction/run_afk.sh` is the unattended full-run wrapper. Key files:
`paper_spec.yaml` (pinned model revisions + implementation deltas from the pseudocode),
`manifests.yaml` (exact 1K-image author subsets + provenance), `weights.yaml` (SHA-256 of the
four detector/segmenter/depth weights), `audit_relations.py` (graph/triples/render edge-digest
consistency), `score_table2.py`, `DATASET_AUDIT.md`.
`data_paper/` still holds the earlier one-off eval scripts (`vqa_metrics.py`,
`eval_paper_faithful.py`, `run_original_batch.py`) and the dataset archive; `data_paper/` and
`data_subsample/` are git-ignored working data.

**Two prompt profiles, never pooled.** `paper_declared` reproduces the supplementary prompt
verbatim — it appends "Answer … using a single word or phrase" to the **raw** condition only,
so marked conditions answer in ~20 words and score ~0 by construction under exact match.
`supplementary_concise` adds that same instruction to the marked conditions and is the only
fair comparison. (`released_artifact_bare` is the third, for the released VQAv2 path.)

**Current state (Table 2 run completed 2026-08-10, `supplementary_concise`, 1K images/dataset,
single decode setting):** marks never beat raw on VQA for any of the three models. Gemma-3-4B is
at parity (GQA 48.00 raw vs 48.60 best marked, inside noise); Qwen2.5-VL-7B loses 13–17 points
and LlamaV-o1-11B 34–42. The damage is dominated by the overlay occluding the photo, not by the
scene graph: `segmented` (masks + contours, no IDs and no arrows) already costs Qwen 13–17 points
and is still its *best* marked condition on every VQA dataset. RefCOCOg is the one task where marks are indispensable (raw is 0 by construction);
best mark style there is model-dependent, with a 23-point swing for Qwen between text and
numeric IDs. Do not present these as reproducing the paper's reported gains.

## Known confounds in the Table 2 negative result

Established 2026-08-11 from the existing artifacts (no new compute). Read before interpreting
`RESULTS.md` or designing a follow-up run.

**`segmented` is a filled render, not an outline.** `_PAPER_AAAI26_PROFILE` sets
`fill_segmentation: True, seg_fill_alpha: 0.25` (`src/gom/config.py:399-400`), both locked in
`PAPER_AAAI26_LOCKED_FIELDS`; the plain `PreprocessorConfig` default is the opposite
(`preprocessor.py:634`, commented "Outline-only preserves image evidence for VQA"). All six paper
variants inherit the fill. Verified visually: in `COCO_train2014_000000131127` a **black** t-shirt
renders **blue**. `RESULTS.md:74` and `REVIEW.md:556` still say "contours only" — they are wrong,
and that misdescription is why the run's conclusion stops at "the overlay occludes" without
naming the fill as the cause.

**The fill is applied twice**, so effective opacity is **0.4375, not 0.25**: a vectorized blend at
`src/gom/viz/visualizer.py:744-751`, then a per-object `ax.fill` at `:848-849`. Overlapping masks
accumulate additively and clip to 1.0 (`src/gom/utils/rendering_opt.py:133-158`) — at ≥4
overlapping masks the photo is entirely replaced. Every alpha figure in the paper and in
`REVIEW.md:364` understates real opacity by ~75%.

**The failure is evidence destruction, not label copying.** Models name the actual overlay palette
color only 0–10% of the time, but 48–70% of color answers *change* versus raw.

**Damage concentrates on surface-appearance questions**, not uniformly. Δ = best marked − raw on
VQAv2, by question category:

| category | Gemma | Qwen | LlamaV |
|---|---:|---:|---:|
| color | −28.6 | −38.1 | −78.4 |
| identity ("what kind of") | −4.5 | −20.5 | — |
| count | +5.2 | −11.4 | −46.9 |
| open | −1.0 | −16.3 | −53.2 |
| yes/no | **+7.5** | −8.3 | −10.2 |

Mask coverage correlates only weakly by comparison (Qwen VQAv2: −13.9 in the lowest coverage bin
→ −28.1 in the highest). Category dominates.

**Projected effect of dropping appearance questions** — color, material/texture/pattern, text-in-
image, plus any color word in the question or in ≥50% of gold answers. *Recomputed from the
existing predictions; not yet produced by a scored run.* Kept 606/794/742 of 1000 for
GQA/VQAv1/VQAv2:

| model | | GQA | VQAv1 | VQAv2 |
|---|---|---:|---:|---:|
| gemma3_4b | all | +0.60 | −0.22 | −1.16 |
| gemma3_4b | filtered | **+3.14** | **+3.56** | **+3.15** |
| qwen25_vl_7b | filtered | −8.75 | −13.19 | −12.64 |
| llamav_o1_11b | filtered | −35.15 | −36.74 | −36.39 |

So the appearance confound explains Gemma entirely and **leaves Qwen at −12.6 and LlamaV at −36.4
unexplained**. **Next step: properly re-score all three models before designing any filter** — a
filter that only rescues the smallest model is not yet a sufficient account of the result.

**Two render defects found and not fixed** (out of scope when found, still open):
- the double mask fill above — `visualizer.py:744-751` + `:848-849`.
- `visualizer.py:1350,1357` — `border_y` is never assigned, so the non-batch outside-label path
  raises `NameError`. Unreachable today only because `use_batch_text_renderer=True`.

**If a filtering stage is built later, it must drop whole images, not individual rows.**
`score_table2.py:264-266` applies `first_row_per_image` to `eval.jsonl`, so dropping rows changes
which question is canonical for an affected image and breaks the join to its `_q1` render artifact
and to the existing predictions. Filtering only removes rows, so with an inner join on
`question_id` (replacing the length assert at `score_table2.py:284`) the existing 156K generations
re-score with no GPU cost.
