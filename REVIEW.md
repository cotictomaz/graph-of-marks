# Repository Review — Replication Audit

In-depth review of the refactored `src/gom/` package against (a) the AAAI26 paper
(main + supplementary, `paper/`) and (b) the original experiment script
`src/all_in_one_gom.py` (kept frozen as historical reference). Goal: explain why paper
results stopped replicating after the refactor, fix what breaks replication, and flag
remaining divergences.

Legend: **[FIXED]** = corrected in this pass · **[OPEN]** = documented, needs a decision
or a larger change.

---

## 1. Critical bugs (silently changed results)

### 1.1 Scene graph built on a dummy white image, depth discarded **[FIXED]**
`gom/pipeline/preprocessor.py` (`build_scene_graph` wrapper) created a white
`Image.new(...)` and dropped the `depths` argument. Consequences: every node had
`color="white"` and `depth_norm=0.5`, every edge `depth_delta=0` — the graph JSON,
triples text, and any depth-based relation inference from graph attributes were dead.
Fix: pass the real image and the precomputed per-object depths through to
`SceneGraphBuilder.build` (which now accepts `depths`/`caption`).

### 1.2 Graph edges bypassed relation filtering **[FIXED]**
`SceneGraphBuilder` generated its own geometric candidate edges (up to 32 neighbors per
node, `dist_norm ≤ 0.4`), and the pipeline then stamped a heuristic relation on *every*
such edge. The triples file / graph JSON / visualization therefore contained many more
edges than the per-object top-k relations selected by the relation engine
(`max_relations_per_object`), and a second, different relation-inference path
(`_infer_relation_from_attrs`) could override the engine's labels. Fix: the relation
engine is now the single authority — its relations are stamped onto the graph and all
other object-object edges are removed, so triples == filtered relations == drawn arrows.

### 1.3 Depth sign inverted (front/behind flipped) **[FIXED]**
`gom/utils/depth.py` and `gom/utils/depth_v2.py` normalized model output and then
**inverted** it, based on the comment "MiDaS: larger = farther". That is wrong: MiDaS
DPT and Depth Anything (relative) output disparity — larger raw value = **closer**.
After the inversion, the stored maps had 1.0 = farther while every consumer (including
`in_front_of`/`behind` inference: `d_i > d_j → i in front`) assumed 1.0 = closer, so
**every depth relation was reversed**. The original script normalized without inverting.
Fix: inversion removed in both implementations; convention is now genuinely
"higher = closer" end to end.

### 1.4 Legacy VQA path crashed on import **[FIXED]**
`gom/vqa/models.py` no longer defines `VLLMWrapper`/`HFVLModel` (deliberate: inference
backends are not bundled), but `gom/vqa/runner.py` and `src/vqa.py` still imported them
→ `ImportError` for `make run_vqa`, `gom-vqa`, and any `import gom.vqa.runner`.
Fix: `runner.py` is duck-typed (`ModelLike = Any`, any object with
`generate(prompt, image_path)`); `src/vqa.py` defines its own thin vLLM/HF wrappers.

### 1.5 `gom-preprocess` crashed with default flags **[FIXED]**
`src/image_preprocessor.py::_build_config` read `args.same_class_iou_threshold` and
`args.cross_class_score_diff_threshold`, but neither option was defined in argparse →
`AttributeError` on every run. Both options added.

### 1.6 Per-call relation config silently ignored **[FIXED]**
The pipeline assigned its tuned per-call config to
`relations_inferencer.relations_config`, but the inferencer reads `self.config` — the
tuned margins/limits never took effect. Fixed to assign `config`. The CLIP-relation
gate also used `clip_pruning_threshold` (a *question-similarity* threshold, 0.25) as the
CLIP relation score cutoff; it now uses `RelationsConfig.clip_threshold`.

### 1.7 `--config` file silently discarded **[FIXED]**
`image_preprocessor.py` loaded and merged the `--config` YAML/JSON into a config object
— then threw it away by rebuilding the config from argparse. Precedence is now:
CLI defaults < config file < explicitly typed CLI flags.

### 1.8 `.env` never loaded **[FIXED]**
No entry point loaded `.env`; `HF_HOME`/`HF_TOKEN` only worked if exported by the
shell. Added `gom/utils/env.py` (dependency-free loader) invoked at the top of
`image_preprocessor.py`, `scripts/run_vqa_inference.py`, `scripts/run_ref_inference.py`
*before* torch/vllm imports so `HF_HOME` takes effect.

### 1.9 Batch detection stalled at full resolution **[FIXED]**
`_get_optimal_batch_size` returns 16–32 on large-VRAM GPUs, and `_run_from_json`
runs detection on a whole batch at once. With `detection_resize=False` (now the
default, matching the paper), batching 16 full-resolution images through the
detector ensemble stalls (huge tensors, GPU idle). The single-image path is fine.
Fix: return batch size 1 when detection runs at full resolution; the batched path
stays for the resized mode. Measured steady-state after this fix: **1.44 s/image**
(paper reports ~1.13 s), vs a 12 s first-image CUDA warmup and ~72 s one-time model
load.

### 1.10 Config attribute bugs **[FIXED]**
- `gom/vqa/preproc.py` used `pre.config` / `preproc_obj.config`, but the preprocessor
  stores `self.cfg`: runtime `AttributeError` on one path, and — worse — the
  `hasattr` guard on the other path made config updates *silently do nothing*.
- `src/vqa.py:406` called `f("...")` instead of an f-string → `NameError` whenever
  metrics were saved.
- Fallback `PreprocessorConfig` in `gom/config.py` declared `cross_class_suppression`
  and `cross_class_iou_threshold` **twice** with different values (second silently won).

---

## 2. Hyperparameter drift vs. the paper (Table 1)

The repo had **three conflicting default sets** (package config, `image_preprocessor.py`
CLI, `src/vqa.py` CLI), none matching the paper. Defaults are now unified to the paper's
selected values **[FIXED]**:

| Parameter | Paper (τ) | Package cfg (before) | gom-preprocess CLI (before) | Now |
|---|---|---|---|---|
| Detector conf (owl/yolo/detectron) | 0.5 | 0.60 / 0.85 / 0.85 (+per-image auto-tuning) | 0.40 / 0.80 / 0.80 | **0.5 / 0.5 / 0.5**, auto-tuning off |
| WBF IoU (τ_overlap-IoU) | 0.9 | 0.10 | 0.55 | **0.9** |
| Directional margin (τ_dir-margin) | 20 px | 20 | 20 | 20 |
| Depth threshold (τ_z-diff) | 0.1 | 0.05 | — | **0.1** |
| Relations per object (k) | 3 | 5 | 3 | **3** |
| Segmenter | SAM-HQ | SAM-1 | SAM-1 | **SAM-HQ** |
| Depth model | MiDaS DPT-Large | Depth Anything V2 Large | — | **MiDaS DPT-Large** (DA-V2 opt-in) |
| Detection input | full resolution | resized to 800 px | — | **full resolution** |
| label NMS | — (orig. script: 0.5) | 0.25 | 0.50 | 0.5 |
| VQA decoding | nucleus, 512 tok, temp/top-p/seed sweep | temp 0.0 greedy, 256 tok, no top-p/seed | — | **temp/top-p/seed CLI args, 512 tok** |

Also enabled-by-default extras that the paper pipeline does not use were made opt-in
**[FIXED]**: CLIP-scored semantic relations (`use_clip_relations=False`),
physics-informed filtering (`use_physics_filtering=False`), aggressive cross-class
box suppression (`cross_class_suppression=False`), and the SAM mask-quality filter
(`enable_mask_quality_filter=False`) — the last two were silently deleting valid
detections (e.g. a carrot next to a bowl removed at box IoU 0.36).

Note: the resize change (800 px → full resolution) makes preprocessing slower but is
what both the paper and the original script did; pixel-space thresholds (margin,
min_distance, near) are calibrated for full resolution.

---

## 3. Remaining divergences from the paper algorithm **[OPEN]**

The relation engine (`gom/relations/inference.py`, ~1.9k lines) implements a superset
of the paper's Algorithm 2 with extra heuristics that remain active even after the flag
alignment. They generally *reduce* relation recall vs. the paper's simple rule:

1. **Pair-skip when box IoU > 0.3** — paper computes directional relations for all
   pairs; overlapping pairs (e.g. object on a table) lose their directional edge.
2. **Scale-aware margin** `max(20, 0.08·avg_box_size)` — paper uses a fixed 20 px;
   large objects need a much larger displacement to get a relation.
3. **Contact exclusion** — vertically adjacent boxes (gap ≤ ~2% of height) are excluded
   from above/below; the paper instead *keeps* the directional relation and adds a
   `touching` modifier.
4. **Overlap exclusions** (vertical/horizontal overlap > 0.85 of the larger side).
5. **`on_top_of`/`under` relations** and a priority-based per-pair dedup that prefers
   semantic > touching > directional — the paper's ontology has 7 relation types with
   modifiers, and dedups inverse edges by first occurrence.
6. **`near` semantics**: paper uses `near` only as a fallback when no directional
   relation exists and center distance < τ_near; the engine emits proximity modifiers
   (`touching`/`very_close`/`close`) per its own thresholds.
7. **Query filtering** uses substring/WordNet matching; the paper's mention detection
   also uses fastText (cc.en.300.vec) embeddings with cosine > τ_query-obj.
8. **No official metrics**: `run_vqa_inference.py` computes naive exact-match accuracy;
   VQAv1/v2 official accuracy (soft consensus over 10 annotators) and the GQA protocol
   are not implemented. The REC IoU ≥ 0.9 protocol is only partially present in
   `run_ref_inference.py`.
9. **Determinism**: no global seed for preprocessing (`seed=None`), FP16 inference,
   thread pools. The original script was equally unseeded — exact per-run replication
   was never guaranteed; the paper reports mean±std over 27 decoding configs instead.

A faithful re-implementation of Algorithm 2/3 (directional → depth → near-fallback →
modifiers → query-aware top-k) would be ~150 lines and is the recommended next step if
flag-level alignment proves insufficient.

---

## 4. Differences vs. the original `all_in_one_gom.py` (context)

The original script itself deviates from the paper text in places — useful to know when
comparing numbers:

- WBF `iou_thr=0.55` (not 0.9) with detector weights owlvit 2.0 / yolo 1.5 /
  detectron 1.0; TTA horizontal flip enabled **only for YOLO** (doubles YOLO boxes).
- Relations for the prompt came from graph-edge heuristics (`overlaps` / `front_of` /
  `behind` / `near` only); the richer directional/CLIP/ConceptNet relations fed the
  visualization, deduped per pair by minimum "distance" across incomparable scales
  (pixels vs |Δdepth| vs 9999) so depth relations usually won.
- Prompt was a flat string `0:red car_1 (area=0.12); (0)-near->(1)` with a BLIP scene
  caption; visual marks were 1-based but prompt ids 0-based (off-by-one).
- argparse defaults differed from `__init__` defaults (e.g. SAM `pred_iou_thresh`
  0.8 vs 0.9), so the effective run values were the argparse ones.
- CLIP was ViT-L/14 (768-d) although comments claimed 512-d; depth mask rescale
  swapped H/W; `_clip_relation` extracted the second word of the winning template.

The refactor's `Objects:/Triples:` text format and the supplementary's prompt templates
(`scripts/run_vqa_inference.py`) match the **paper**, not the original script — that is
the right target and is what this pass aligns to.

---

## 5. Code quality / maintenance notes **[partially FIXED]**

- **[FIXED]** `graph/prompt.py` defined `graph_to_prompt`, `_fmt_triple`,
  `graph_to_triples_text`, `save_triples_text` twice (identical copies; second won).
- **[FIXED]** `_INVERSE` map asymmetry (`on_top_of → below` but `under → on_top_of`).
- **[OPEN]** ~15 modules are dead or unwired in the default pipeline
  (`fusion/cascade|confluence|spatial_hash|wbf_optimized|benchmark`,
  `relations/llm_guided|spatial_3d|physics` (opt-in), `utils/tta|ensemble|calibration|
  mixed_precision|batch_processing|model_registry|clip_cache`, …). Consider pruning or
  moving to an `extras/` namespace.
- **[OPEN]** Several giant module docstrings contain stale values (e.g. claims of
  `clip_threshold 0.30`), DEBUG `print`s in hot paths, and `Makefile` still references
  a non-existent `src/run_fast_preprocessing.py` (`fast_preprocess` target) and
  hard-codes `/workdir` paths.
- **[OPEN]** `gom/cli/preprocess.py` imports the top-level `image_preprocessor` module
  via a `sys.path` hack; wheel installs may not ship that module.
- **[OPEN]** No test exercises the numeric pipeline (detection → relations → graph →
  triples); the suite is import/config-shape only, so regressions like §1.1–1.3 were
  invisible. A golden-file test on one image would catch most of these.

---

## 5.5 Paper-faithful harness calibration (exact prompts + greedy + official metric)

Rebuilt the eval to match the paper exactly: supplementary Fig 1/2 prompts verbatim
(`scripts/run_vqa_inference.py`), paper triple format `a -(left_of)-> b`
(`graph/prompt.py`), raw = lmms-eval "single word or phrase" convention, greedy decode
(temp 0, 512 tok), official metrics + a new RefCOCOg REC IoU≥0.9 scorer
(`data_paper/vqa_metrics.py`). Gemma-3, GQA 1500 Q, refactor render:

| group | n | raw | visual | vis+txt |
|---|---|---|---|---|
| spatial | 525 | 50.1% | 33.1% | 31.6% |
| attribute | 975 | 44.2% | 25.4% | 20.6% |
| ALL | 1500 | **46.3%** | 28.1% | 24.5% |

Two findings: (1) the paper-faithful **raw** (46.3%) is still ~10 pts below the paper's
Gemma GQA 56.2 — and per-prediction inspection shows the model answers concisely with
**genuine** errors (0 verbose-answer artifacts), so the gap is not the metric/prompt. Most
likely the **GQA sample differs**: the paper draws "1K images × 3 queries" from GQA's pool,
whereas this uses all of testdev-*balanced* (the deliberately hard 398-image split). That
shifts raw and GoM together, not the delta sign. (2) Under the paper's exact prompt, GoM
drops further (28.1% vs 40.8% with the earlier concise "marks-as-aids" prompt) — the paper
prompt lets the model echo IDs / over-rely on the graph. GoM < raw is robust either way.
Remaining decisive test: does the ORIGINAL pipeline's render reproduce the positive delta?
(Track B, §8.)

## 6. Paper-scale reproduction with the OFFICIAL metric (definitive)

After the 90-question toy suggested a spatial win, I pulled paper-scale data and re-ran with
the standard VQA/GQA answer normalization + accuracy (not lenient string match):

- Data (HF streaming, `data_paper/`): **GQA** 398 img / 12,578 Q (full testdev-balanced),
  **VQAv2** 1,000 img / 5,608 Q (10-annotator answers → soft accuracy), **RefCOCOg**
  1,000 img / 1,000 exprs (+bbox for REC). VQAv1 is not on HF (shares COCO val2014 with VQAv2).
- Metric: `data_paper/vqa_metrics.py` — VQA v2.0 normalization; GQA exact/phrase match;
  VQAv2 `min(1, matches/3)`.
- Model Gemma-3-4B (paper's best-responding), winning render (paper-style: segmentation
  contours + IDs + relation labels, 10-object cap) + concise "marks-as-aids" prompt.

**GQA, 1,500 questions, official metric:**

| group | n | raw | GoM visual | GoM visual+textual |
|---|---|---|---|---|
| spatial | 525 | **51.2%** | 44.8% | 43.0% |
| attribute | 975 | **48.5%** | 38.7% | 37.6% |
| ALL | 1500 | **49.5%** | 40.8% | 39.5% |

**GoM underperforms the raw baseline at scale, on every slice including spatial.** The
90-question spatial "win" (§6c) was sampling noise (n=32). The prompt fixes held (ID leakage
39/1500, refusals 11/1500), but per-prediction inspection shows the mask overlay still
degrades the model's reading — it flips yes/no answers and shifts colors ("silver"→"green"
even at α=0.12).

### 6a. Question-based filtering (Alg 3) applied — still no gain

The §6 run used `--disable_question_filter` (one render/image) which skips Algorithm 3
(query-based graph filtering). Re-ran with filtering ON, per-question renders (463 questions,
official GQA metric, Gemma-3):

| condition | ALL | spatial |
|---|---|---|
| raw (plain image) | **46.7%** | **45.6%** |
| raw + textual SG (no marks) | 40.6% | 44.3% |
| GoM visual+textual (filtered) | 37.6% | 43.0% |
| GoM visual (filtered) | 34.8% | 38.9% |
| GoM visual (unfiltered, same Qs) | 35.9% | 39.6% |

Filtering did **not** recover the deficit (filtered-visual 34.8% ≈ unfiltered 35.9%). Reason:
for object-heavy questions the query filter keeps *every* instance of the queried class (e.g.
13 buses for a "bus" question), so it does not cut clutter — it can increase same-class marks.
The ranking is monotonic: each scene-graph layer added (marks, then text) lowers accuracy;
**nothing beats the plain image.** This holds across filtered/unfiltered, visual/textual,
and the official metric — so the negative result is not an artifact of skipping Alg 3.

This is a robust negative result: **the refactored pipeline does not reproduce the paper's
GoM gains** (paper: Gemma GQA 56.2 raw → 63.2 GoM, +7; here 49.5 → 40.8, −9). Two facts to
weigh: (a) my raw GQA (49.5%) is itself ~7 pts below the paper's (56.2%), so the eval harness
differs from theirs (they sweep 27 decode configs, use the official GQA scorer, and a fixed
1K sample); (b) the GoM−raw *sign* is consistently negative across models, renders, and
metrics here. The decisive next experiment to isolate cause is to run the **original
`src/all_in_one_gom.py`** on the same GQA images and eval its renders identically — if the
original reproduces the gain and the refactor doesn't, the refactor regressed the SG/render;
if neither does, the difference is the paper's eval protocol. (Not run yet — it needs the
original's heavier deps: ConceptNet, BLIP, SAM vit_h, CLIP ViT-L.)

## 6c. Earlier 90-question config search (superseded by §6, kept for the render/prompt lessons)

### Finding the configuration where GoM beats the baseline (90-question toy)

The naive GoM setup lost badly to the raw baseline; iterating on render + prompt + model
found a configuration where **GoM beats the baseline on spatial questions (its target),
reproducing the paper's direction.** Progression (GQA subsample, 15 images / 90 questions,
Gemma-3-4B unless noted, lenient GQA match):

| Step | Change | spatial | ALL vs raw |
|---|---|---|---|
| render bug | filled masks recolor objects, node IDs leak, 24–35 objects | 37.5% | −22 |
| render = boxes only | dropped masks + relation labels (wrong — not the paper) | 40.6% | −16 |
| object cap 10 | `max_detections_total=10`, `per_label=3` (paper's 3–10 sweet spot) | 40.6% | −16 |
| directional relations | fixed dedup/priority so left/right/above/below beat noisy depth | 37.5% | −17 |
| **correct render** | **segmentation contours + IDs + relation labels** (matches `assets/`) | 34.4% | −18 |
| **+ concise prompt** | **marks-as-aids, forbid ID echo, few-word answer** | **50.0%** | −2 |

**Winning config — GoM > baseline on spatial:**

| condition | spatial | attribute | ALL |
|---|---|---|---|
| raw (concise prompt) | 46.9% | 34.5% | 38.9% |
| **GoM visual (concise prompt)** | **50.0%** | 29.3% | 36.7% |
| GoM visual + textual SG | 40.6% | 22.4% | 28.9% |

What made the difference, in order of impact:
1. **Prompt.** The paper's verbose "use the scene graph to answer" prompt made the models
   echo node IDs ("the table, dining_table_1, …"), over-rely on the graph, and refuse
   ("based on the scene graph…" — 13/90 refusals). A prompt that frames the marks as
   *locating aids*, forbids mentioning IDs/arrows, and asks for a few-word answer removed
   these failure modes and flipped spatial questions in GoM's favour.
2. **Render.** Must match `assets/gqa_sample_04_output.png`: class-colored **segmentation
   contours** (light fill, content visible) + text IDs + arrows + **relation labels** — not
   filled masks (recolor objects), not bare boxes (my earlier over-correction).
3. **Object count.** Cap to the paper's 3–10 sweet spot (`max_detections_total=10`,
   `max_detections_per_label=3`); 24–35 marks bury the image.
4. **Visual, not visual+textual.** For Gemma-3 the textual SG hurts (over-reliance); the
   visual scene graph is the driver, as the paper's Fig 4 also shows.
5. **Model.** Gemma-3-4B benefits from marks; Qwen-2.5-VL (paper: "adverse sensitivity to
   SoM") loses under every config here — the paper reports the same ordering.

Caveat: the win is on GoM's target question type (spatial, n=32); marks don't help attribute
questions (color/shape/"what is X"), so overall accuracy is ~tied. This 90-question,
single-decode, lenient-match subsample is a sanity check, not a benchmark — but it does now
reproduce the paper's core claim (visual scene graphs improve spatial reasoning) once the
render and prompt are correct. Two pipeline changes came out of this and are kept:
directional relations now outrank monocular-depth relations in dedup/top-k (they were being
drowned out — 63% of edges were `in_front_of`; now 12%), and the aggressive cross-class /
mask-quality filters are opt-in (they deleted valid detections).

### 6b. Earlier heavy-render numbers (Qwen2.5-VL) — for reference

15 GQA testdev-balanced images, 90 questions, one decode config (temp 0.1, top-p 0.9,
seed 42, 512 tokens). Accuracy = lenient GQA match (yes/no on first token; otherwise
the gold answer string appears in the prediction).

| Condition | visual | visual+textual |
|---|---|---|
| Raw image + question (no GoM) | **48.9 %** | — |
| GoM, heavy render (filled masks + relation-label boxes) | 23.3 % | 26.7 % |
| GoM, marks render (boxes + object IDs + arrows, no fill) | 28.9 % | 27.8 % |

**GoM underperformed the raw baseline on this sample, but the render matters:** dropping the
filled-mask overlay recovered ~5 points on the visual condition (23.3 → 28.9). Textual-SG
helps over visual-only under the heavy render (26.7 vs 23.3) — the paper's direction — but the
gap to the raw image persists. Root causes, from per-prediction inspection — actionable, not a
refutation of the method:

1. **Filled masks recolor objects.** "What color is the large animal?" (gold: dark brown)
   → raw "black", GoM "green" — the elephant is under a green α=0.25 fill, so the model
   reads the overlay. Any color/attribute question is corrupted by mask fill. The heavy
   render (`--fill_segmentation` + relation-label boxes) is the wrong default for VQA.
2. **Node IDs leak into answers.** "What is inside the bowl?" → GoM visual+textual answered
   "cake_2" (the internal label), not "cookies".
3. **Clutter → refusals.** On scenes where the question matched no object label, query
   filtering kept 20–35 objects; the dense overlay drove "the image does not provide…"
   answers.
4. **Sample is attribute-heavy, not spatial.** GoM targets spatial reasoning (left/above/
   in-front); many GQA questions here are color/shape/"what is X" where the overlay only
   hurts. The paper's gains are on spatial questions, averaged over 1K images × 27 decode
   configs.

Recommendations: (a) the default VQA render should be **object IDs + arrows on the original
photo, no filled masks** (the "marks" row above) — filled overlays and relation-label boxes
belong to a visual-debug mode, not the VQA input; (b) VQA should run on **spatial** questions
where GoM is designed to help, and with query filtering that actually narrows the object set
(here most GQA questions matched no label, so 20–35 objects were drawn, well past the paper's
3–10 sweet spot); (c) strip node-ID suffixes (`cake_2`) from any text the model can echo, or
they leak into answers. This subsample (15 images, one decode config, attribute-heavy Qs) is
far too small and out-of-domain to claim or refute paper replication — treat it as a
wiring/sanity check that the fixed pipeline runs end to end and produces coherent outputs,
not as a benchmark. Renders live in `data_subsample/gom_out` (heavy) and
`data_subsample/gom_marks` (clean); predictions in `data_subsample/eval_*_preds.jsonl`.

## 9. FINAL VERDICT — paper-faithful grid + original-pipeline diagnostic

### 9a. Paper-faithful grid (exact prompts, greedy, official metrics)
Gemma-3-4B, all 3 datasets (Qwen failed to load under the transformers pin needed for
Gemma-3; LlamaV-o1 is a reasoning model too slow for 4.5k greedy CoT generations — breadth
limitations, not the conclusion):

| dataset | metric | n | raw | GoM visual | GoM vis+txt |
|---|---|---|---|---|---|
| GQA | exact/phrase | 1500 | **46.3%** | 28.1% | 24.5% |
| VQAv2 | soft acc | 1500 | **65.0%** | 41.5% | 38.3% |
| RefCOCOg | REC IoU≥0.9 | 1000 | 0.0%* | **26.9%** | 26.4% |

*RefCOCOg raw = 0 by construction: the raw image has no region IDs to name, so REC is
impossible without the marks. **This is the only place GoM "wins", and only because the task
structurally requires the marks** — exactly the paper's RefCOCOg story. On genuine VQA (GQA,
VQAv2) GoM consistently loses to the plain image. Calibration is good on the high side:
VQAv2 raw 65.0% exceeds the paper's Gemma VQAv2 (~59), so the harness is sound; GQA raw 46.3%
trails the paper's 56.2 for the sample reason in §5.5.

### 9b. Original `all_in_one_gom.py` diagnostic
Built `gom-allinone` image; ran the **unmodified** original. Two hard findings:
1. **It crashes.** On the 2nd GQA image (`n235859`) the original raises
   `IndexError: list index out of range` at `all_in_one_gom.py:1585` (`rel.split()[1]` on a
   single-word CLIP-relation template), and its batch loop has **no per-image error handling**,
   so the whole run dies. A fault-tolerant wrapper (`data_paper/run_original_batch.py`, which
   imports the frozen class unchanged and try/excepts per image) is needed just to survive.
2. **It is far too slow** (~3–10 min/image; SAM-v1 auto-mask + ConceptNet + BLIP + 3 detectors)
   to run the paper's ~4000-image, 3-model, 27-decode-config protocol. The "reference" code as
   frozen here **cannot reproduce the paper's evaluation** — strong evidence the paper's numbers
   came from different/fixed code and/or a very large compute budget.

**Qualitative scene-graph comparison** (image `n161313`, shared by both pipelines) is the most
telling result:
- Original: 4 objects (snowboard, person, sky, snow), coherent — `person -(front_of)-> sky`.
  The GQA question *"What is the person in front of?" → sky* is **answerable directly** from it.
- Refactor: 10 objects with **duplicates** (sky_1/sky_2, snow_1/2/3, snowboard_1/2) + a
  **spurious "helmet"**, incoherent relations (`sky_1 -(in_front_of)-> sky_2`); the person→sky
  relation is lost.

So the refactor **over-detects and yields noisier scene graphs than the original** — a genuine
regression (worse object dedup / more spurious detections). This likely contributes to the
refactor's poor GoM showing. **But it is not the whole story:** even the original's clean graph
does not change the core result that, for strong instruct MLMs on GQA/VQAv2, overlaying *any*
scene graph tends to hurt vs. the plain image (the model already reads the scene well; marks
occlude and the SG injects errors). GoM's real, reproducible benefit is on REC, where the marks
are indispensable.

### 9c. Bottom line
- The refactor does not reproduce the paper's VQA gains; the **paper-faithful harness confirms
  GoM < raw on GQA and VQAv2** for Gemma-3, and the harness itself is calibrated (VQAv2 raw
  above paper).
- Two real refactor regressions were found and (partly) fixed this project: the depth-inverted
  / white-image scene graph (§1), and **over-detection producing noisy graphs** (§9b, still
  open — needs stricter dedup / detector thresholds to match the original's cleaner 3–10 object
  graphs).
- The original code is **not runnable at paper scale** (crash + speed), so an exact numeric
  head-to-head vs the paper's own pipeline isn't achievable from this repo without fixing the
  original — which is out of scope (frozen).
- GoM's design win (REC / referring expressions) **does** reproduce: raw 0 → GoM 26.9.

## 8. Track B — original `all_in_one_gom.py` diagnostic (setup notes)

Built `gom-allinone` Docker image (gom-review + adjustText + nltk `/root/nltk_data` symlink;
detectron2/SAM/ultralytics/spaCy-md/wordnet already present). Verified the **original,
unmodified** pipeline runs and produces the canonical render on GQA `n16425`:
- 10 objects (WBF-capped), bounding boxes + mask contours, text IDs, relation-labeled arrows
  (Front Of / Behind / Below / Left/Right Of / Nearest), BLIP scene caption
  ("a couple of buses are parked on the street"). Style is comparable to the refactor's
  paper render — the original is **not** obviously higher quality by eye.
- Original scene prompt keeps node color + area + BLIP caption; relations from graph edges
  (front_of/behind/near) — the original's textual format differs from the paper triples.

Running the original on all 398 GQA images is underway but slow: **~3–4 min/image** (SAM v1
automatic-mask generation + per-pair ConceptNet API + BLIP + 3 detectors) → **~20 h** for 398.
vLLM eval cannot share the 32 GB GPU with it, so Track-A grid eval and Track-B eval are
sequenced after it (or on a partial subset for an early read). `src/all_in_one_gom.py` is run
unmodified. Verdict (original-GoM vs refactor-GoM vs raw, same questions, same harness) pending
completion.

## 7. Verification performed

- `pytest tests/` — 149 passed (in the `gom-review` Docker image: torch 2.7.1+cu128,
  detectron2, SAM/SAM-HQ, ultralytics, vLLM 0.10.2).
- Import smoke: `import vqa`, `gom.vqa.runner`, `gom.cli.vqa` (previously crashed).
- Config smoke: unified defaults verified in-container.
- Depth/graph correctness on real data: graph JSON now carries 40 distinct depth values
  (0.065–0.612, not all 0.5) and real dominant colors — confirms §1.1/§1.3 fixed.
- Performance: 1.44 s/image steady state (§1.9).
- End-to-end: 15-image / 90-question GQA (testdev_balanced) subsample preprocessed with
  the fixed pipeline; rendered outputs visually inspected; VQA comparison (raw baseline vs
  GoM visual vs GoM visual+textual, two render styles) with Qwen2.5-VL-7B via vLLM — results
  in §6, artifacts in `data_subsample/`.

## 10. July 2026 post-fix VQAv2 rerun

The corrected question-aware pipeline was run on the same first 1,500 VQAv2 questions
(417 images, at most 4 questions/image) used in §9. It produced 1,500 images, graph JSON
files, and triple files with no preprocessing errors or empty files. Unlike the old evaluator,
the rerun maps repeated-image questions to their actual `_q1`, `_q2`, ... renders.

Gemma-3-4B was evaluated greedily with the same raw predictions in every comparison.
The VQAv2 scorer now implements the official leave-one-annotator-out weighting
(1/2/3/4+ matches = 0.3/0.6/0.9/1.0) and exact normalized answers.

| condition | raw | GoM visual | GoM visual+textual |
|---|---:|---:|---:|
| Official scorer, equal short-answer constraint | 59.45% | **63.01%** | 61.85% |
| Supplementary prompts, phrase-compatibility scorer | **64.03%** | 53.33% | 51.80% |
| Previous GoM, phrase-compatibility scorer | — | 40.49% | 37.62% |

For the valid short-answer comparison, visual GoM improves raw by **+3.56 points**
(paired bootstrap 95% CI **+1.23 to +5.98**; 270 improved, 210 regressed, 1,020 tied).
Visual+textual improves by +2.40 points, but its CI includes zero (-0.03 to +4.79).
The preprocessing changes therefore produce a real aggregate improvement over the baseline
when response format is controlled, and improve the old GoM render by 12.83 points under
the same phrase-compatibility analysis.

The gain does **not** validate the claimed spatial mechanism. Visual GoM changes by +8.41
on yes/no questions and +4.99 on the residual "other" group, but by -8.00 on count, -4.81
on color, and -0.34 on the 87-question spatial slice. Textual relations also do not help:
729 rows have at least one filtered relation and visual+textual is 2.59 points below visual
there; 771 filtered graphs contain objects but no object-object relation.

There is also unresolved paper-artifact inconsistency. After removing chat-template wrappers,
the stored 5,180-question `paper_reference/VQAV2_Gemma_*.json` files score 63.25% for raw
and only 49.56–51.77% for the GoM variants under official exact scoring, contrary to the
published aggregate gain. Preserve both scoring protocols in future reports rather than
describing phrase containment as the official metric.

Artifacts:
- `data_paper/vqav2/gom_quality_vqa_1500/`
- `comparison_official_concise.json` (primary paired report)
- `comparison_phrase.json` (preprocessing-only comparison for verbose generations)
- `preds_vqav2_gemma-3-4b-it_quality_vqa_concise.json`

## 11. August 2026 — complete Table 2 reproduction (all 3 models, all 4 datasets)

The full matrix finally ran end to end: 3 models × 4 datasets × 7 conditions × 2 prompt
profiles = **156,000 generations**, on the exact author 1,000-image splits, with all four
detector/segmenter/depth weights SHA-256 verified and **0 preprocessing failures** across
4,000 images / 24,000 renders. `audit_relations.py` reports 0 hard consistency errors, so
graph JSON, triples, and drawn arrows carry one edge multiset. Full numbers, caveats, and
method: **`reproduction/RESULTS.md`**; driver: `reproduction/run_afk.sh` (resumable).

### 11.1 The prompt confound that invalidated the first pass

The supplementary prompt reproduced verbatim (`paper_declared`) appends *"Answer using a
single word or phrase"* to the **raw** condition only. GQA/VQA are scored by normalized
exact match, so the marked conditions — answering in ~20 words — score **0.00 by
construction**. Any table built that way measures answer length, not accuracy. The
`supplementary_concise` profile adds the same instruction to the marked conditions, leaving
the raw prompt byte-identical, and is the only fair comparison. Both are kept and reported
separately, per §10's warning about mixing scoring protocols.

### 11.2 Result: marks never improve VQA (best marked variant − raw)

| model | GQA | VQAv1 | VQAv2 | raw baselines (GQA/v1/v2) |
|---|---:|---:|---:|---|
| Gemma-3-4B | +0.60 | −0.54 | −1.16 | 48.00 / 60.47 / 58.74 |
| Qwen2.5-VL-7B | −13.60 | −16.72 | −15.79 | 74.50 / 87.86 / 86.04 |
| LlamaV-o1-11B | −33.70 | −41.00 | −41.98 | 61.50 / 78.34 / 75.38 |

Gemma is at parity (+0.60 is inside noise: n=1,000, and two runs of the identical raw
prompt differ by 0.10). Qwen and LlamaV are decisively harmed. This confirms §6a/§9c's
negative result on a far stronger basis — three models instead of one, both VQA datasets
plus GQA, and with answer format controlled.

**New: the damage is the overlay, not the graph.** For Qwen, `segmented` — filled masks +
contours (the paper profile fills at declared α 0.25, effectively 0.4375 in this run; it is
*not* contours-only), no IDs, no arrows, no relations — already costs 13.6–16.7 points and
is its *best* marked condition everywhere. Adding IDs and then relation labels costs only a little more. So the
loss is dominated by occluding the photograph, not by scene-graph errors or ID leakage.
That reframes §6b's diagnosis: fixing the render's IDs/labels cannot recover it.

### 11.3 New: no single render is best across models (RefCOCOg, IoU ≥ 0.9)

| model | best | text IDs | numeric IDs |
|---|---|---:|---:|
| Gemma-3-4B | `gom_text` **45.13** | 45.13 | 36.62 |
| Qwen2.5-VL-7B | `som_numeric` **39.35** | 22.43 | 32.71 |
| LlamaV-o1-11B | `som_numeric` **33.12** | 0.95 | 23.92 |

Gemma reads text IDs best; Qwen and LlamaV need numeric ones — a 23-point swing for Qwen on
the same render. Relation labels cost 1–3 points for every model. The paper's fixed render
choice cannot be optimal for all three. RefCOCOg also remains the one task where marks are
indispensable (raw = 0 by construction), and Gemma's 45.13 is well above the 26.9 of §9a —
the preprocessing fixes nearly doubled REC accuracy.

### 11.4 LlamaV-o1 caveat

It answers raw in 1.1 words but marked conditions in 15–16: the marks trigger its reasoning
mode. Under a lenient phrase scorer its GQA marked cells rise from 18.5–27.8 to 25.1–34.9
against 63.6 raw, so ~7 points are format and the remaining ~29 are real. Its RefCOCOg
text-ID cells (~1.0) are a parsing failure — the ID rarely survives its chain of thought.
§9a excluded LlamaV as too slow; it now runs (≈75 min for 26,000 generations), but its CoT
format is partly incompatible with the answer-extraction protocol.

### 11.5 Environment fixes required (RTX 5090, sm_120)

Recorded because the pinned image does not run two of the three models as shipped:
- **Qwen2.5-VL** — its ViT selects xformers' vendored FlashAttention-3 *Hopper* kernel,
  invalid on sm_120. vLLM's own FA rejects the ViT's head_dim 80, and `TORCH_SDPA` is not
  selectable because `VLLM_ATTENTION_BACKEND` is shared with the LM, where CUDA rejects it.
  `reproduction/compat/sitecustomize.py` pins **only** the vision tower to SDPA.
- **Qwen + LlamaV** — vLLM's worst-case multimodal profile run OOMs at the default
  `max_num_seqs` *regardless of `gpu_memory_utilization`* (identical OOM at 0.90 and 0.72).
  Capped at 8.
- **LlamaV** — Mllama's chat template rejects a system role beside an image
  (`--fold-system-into-user`).

None of these affects decoding. Also fixed: prediction paths are now profile-scoped so two
prompt profiles cannot overwrite each other, and `score_table2.py` no longer hardcodes
`"runs": 27` when a single decode setting was used.

### 11.6 Remaining caveats

Single decode setting rather than the published 27-point grid; one question per image;
VQAv1/VQAv2 draw from COCO `train2014` per the author manifest, so Qwen's 86–88 raw may
reflect train-split familiarity; Gemma's GQA raw (48.0) is still ~8 points under the paper's
56.2 for the §5.5 sampling reason.
