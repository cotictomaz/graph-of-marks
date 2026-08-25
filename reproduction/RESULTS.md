# Verified Results — AAAI-26 Table 2 reproduction

Completed 2026-08-10 on the exact author image splits. This supersedes the earlier
single-experiment record; the incompatible released-artifact scores are kept at the end.

> **A corrected-pipeline rerun completed 2026-08-15** (single 0.25 fill + active
> Algorithm 3) — see §"Corrected-pipeline rerun" below. That run is the
> paper-faithful reference; the tables directly below are the original recorded run,
> kept for the before/after comparison.

## What was run

| | |
|---|---|
| Data | the four exact 1,000-image author subsets (`manifests.yaml`), hash-verified on install |
| Preprocessing | `paper_aaai26` profile, 4,000 images, six render variants each (24,000 renders), **0 failures** |
| Graph audit | `audit_relations.py`: **0 hard consistency errors** on all four datasets — graph JSON, triples, and rendered arrows share one edge multiset |
| Weights | all four SHA-256s in `weights.yaml` verified (`artifacts/preprocessing_weights.json`) |
| Models | the three pinned revisions in `paper_spec.yaml` |
| Decoding | seed 0, temperature 0.2, top_p 0.9, 512 max tokens |
| Sampling | one canonical question per image → 1,000 rows per cell |
| Scale | 3 models × 4 datasets × 7 conditions × 2 prompt profiles = 156,000 generations |

Reproduce with `reproduction/run_afk.sh`, which is resumable at every stage.

## Why there are two prompt profiles

`paper_declared` reproduces the supplementary visual-SG prompt verbatim. That prompt
appends *"Answer the question using a single word or phrase"* to the **raw** condition
only; the marked conditions get no answer-format instruction. Since GQA and VQA are both
scored by normalized exact match, the marked conditions answer in ~20 words and score
**0.00 by construction** — a measurement of answer length, not of accuracy.

`supplementary_concise` is the identical prompt set with that same instruction added to
the marked conditions. The raw prompt is byte-identical between the two profiles, so the
second profile isolates exactly this confound and is **the only fair comparison**.

Never pool the two.

## Table 2 — `supplementary_concise` (equal answer-format constraint)

Primary metric: official VQA consensus (VQAv1/v2), normalized exact match (GQA),
region IoU ≥ 0.9 (RefCOCOg).

| model | dataset | raw | segmented | som_numeric | gom_text | gom_numeric | gom_text_lab | gom_num_lab |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| gemma3_4b | GQA | 48.00 | 45.20 | 46.50 | **48.60** | 46.50 | 46.40 | 44.80 |
| gemma3_4b | VQAv1 | **60.47** | 60.25 | 58.26 | 59.93 | 57.54 | 58.36 | 55.27 |
| gemma3_4b | VQAv2 | **58.74** | 57.58 | 54.60 | 55.77 | 54.50 | 54.22 | 52.08 |
| gemma3_4b | RefCOCOg | — | — | 44.75 | **45.13** | 36.62 | 41.92 | 34.11 |
| qwen25_vl_7b | GQA | **74.50** | 60.90 | 58.80 | 59.00 | 57.60 | 57.10 | 57.00 |
| qwen25_vl_7b | VQAv1 | **87.86** | 71.14 | 70.03 | 68.88 | 66.82 | 65.68 | 63.88 |
| qwen25_vl_7b | VQAv2 | **86.04** | 70.25 | 67.85 | 68.62 | 64.75 | 64.89 | 62.61 |
| qwen25_vl_7b | RefCOCOg | — | — | **39.35** | 22.43 | 32.71 | 21.58 | 34.13 |
| llamav_o1_11b | GQA | **61.50** | 27.80 | 25.50 | 24.20 | 24.70 | 20.60 | 18.50 |
| llamav_o1_11b | VQAv1 | **78.34** | 37.19 | 37.34 | 33.90 | 35.63 | 25.64 | 27.20 |
| llamav_o1_11b | VQAv2 | **75.38** | 33.40 | 33.18 | 29.63 | 32.55 | 24.71 | 24.84 |
| llamav_o1_11b | RefCOCOg | — | — | **33.12** | 0.95 | 23.92 | 1.05 | 13.66 |

RefCOCOg has no raw/segmented condition: the task requires nameable region marks. Its
cells are identical across both profiles because `run_table2.py` uses the REC prompt
regardless of profile.

## Findings

**1. Marks never improve VQA, for any of the three models.** Best marked variant minus raw:

| model | GQA | VQAv1 | VQAv2 |
|---|---:|---:|---:|
| gemma3_4b | +0.60 | −0.22 | −1.16 |
| qwen25_vl_7b | −13.60 | −16.72 | −15.79 |
| llamav_o1_11b | −33.70 | −41.00 | −41.98 |

Gemma is at parity — its +0.60 on GQA is inside noise (two runs of the *identical* raw
prompt differed by 0.10, and n=1,000 gives a standard error near 1.5 points). Qwen and
LlamaV are harmed decisively.

**2. The damage is the overlay itself, not the scene graph.** For Qwen, `segmented` —
filled masks + contours, no IDs, no arrows, no relations — already costs 13.6 to 16.7
points, and it is Qwen's *best* marked condition on every VQA dataset. Adding IDs, then
relation labels, costs a little more. So the loss is dominated by occluding the photo, not
by errors in the graph or by ID leakage into answers. Note the paper profile *fills* the
masks (declared α 0.25; effectively 0.4375 in this run — see Caveats), it does not draw
contours only.

**3. The best mark style is model-dependent** (RefCOCOg, IoU ≥ 0.9):

| model | best | text IDs | numeric IDs |
|---|---|---:|---:|
| gemma3_4b | `gom_text` 45.13 | 45.13 | 36.62 |
| qwen25_vl_7b | `som_numeric` 39.35 | 22.43 | 32.71 |
| llamav_o1_11b | `som_numeric` 33.12 | 0.95 | 23.92 |

Gemma reads text IDs best; Qwen and LlamaV need numeric ones, with a 23-point swing for
Qwen and a near-total failure for LlamaV on the same render. Relation labels cost 1–3
points for every model. A single fixed render therefore cannot be optimal across models.

**4. RefCOCOg is the one place marks are indispensable** — raw is 0 by construction. At
45.13, Gemma is well above the 26.9 this repo recorded for the pre-fix render (§9a), so the
preprocessing corrections nearly doubled REC accuracy.

**5. LlamaV-o1's marked-condition numbers are partly a format artifact.** It answers raw in
1.1 words but marked conditions in 15–16: the marks trigger its reasoning mode. Under a
lenient phrase-compatibility scorer its GQA marked conditions rise from 18.5–27.8 to
25.1–34.9, against 63.6 raw — so format explains roughly 7 points and the remaining ~29 is
real degradation. Its RefCOCOg text-ID scores near 1.0 because the ID string rarely
survives its chain of thought. Read every LlamaV marked cell with this caveat.

## Appearance-filtered re-score (2026-08-14)

The filled overlay recolors objects, so any question about surface appearance (color,
material, texture, text in the image) is answered against a corrupted image.
`reproduction/question_filter.py` drops those questions — color-type questions,
material/texture/pattern and text-in-image keywords, any color word in the question, or a
color word in ≥50% of the gold answers — and `score_table2.py --question-filter appearance`
re-scores the existing predictions on the surviving canonical images (whole images, never
individual rows). Kept 654/832/789 of 1,000 for GQA/VQAv1/VQAv2. Full table:
`data/table2_report.supplementary_concise.appearance_filtered.{json,md}`.

Δ = best marked − raw, all questions vs appearance-filtered:

| model | GQA all | GQA filt | VQAv1 all | VQAv1 filt | VQAv2 all | VQAv2 filt |
|---|---:|---:|---:|---:|---:|---:|
| gemma3_4b | +0.60 | **+3.06** | −0.22 | **+4.42** | −1.16 | **+3.02** |
| qwen25_vl_7b | −13.60 | −8.56 | −16.72 | −12.50 | −15.79 | −12.42 |
| llamav_o1_11b | −33.70 | −35.17 | −41.00 | −35.62 | −41.98 | −35.82 |

The appearance confound explains Gemma entirely: on non-appearance questions its best
marked condition beats raw by 3.0–4.4 points on all three datasets. It does not explain
Qwen (still −8.6 to −12.5) or LlamaV (still ~−35). Marks help only the weakest of the
three models, and only once appearance questions are excluded — do not present this as
reproducing the paper's across-model gains.

## Corrected-pipeline rerun (`data_v2`, 2026-08-15)

The recorded run above had two pipeline defects, both fixed on 2026-08-14: the mask fill
was applied twice (effective opacity 0.4375 instead of the declared 0.25), and its
preprocessing artifacts dated from before commit `039e152`, so Algorithm 3 never pruned —
every detected object was marked (often 20–35 per image). The rerun regenerated everything
into `reproduction/data_v2/` with the corrected renderer and active Algorithm 3
(typically 3–6 query-relevant marks per image), same splits, models, prompts
(`supplementary_concise`), and decoding. Raw reproduced **exactly** (all 9 model×dataset
cells identical to the recorded run — raw does not touch renders), confirming harness
determinism; the graph audit passed with 0 errors.

| model | dataset | raw | segmented | som_numeric | gom_text | gom_numeric | gom_text_lab | gom_num_lab |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| gemma3_4b | GQA | 48.00 | 48.50 | 48.00 | **49.00** | 47.50 | 48.10 | 48.00 |
| gemma3_4b | VQAv1 | 60.47 | **64.06** | 63.55 | 62.74 | 61.55 | 61.33 | 59.50 |
| gemma3_4b | VQAv2 | 58.74 | **60.35** | 58.47 | 58.09 | 57.31 | 56.03 | 56.04 |
| gemma3_4b | RefCOCOg | — | — | 44.44 | **45.37** | 38.19 | 42.79 | 34.58 |
| qwen25_vl_7b | GQA | **74.50** | 63.10 | 60.80 | 61.80 | 60.70 | 59.80 | 60.10 |
| qwen25_vl_7b | VQAv1 | **87.86** | 76.35 | 74.36 | 73.13 | 72.26 | 71.69 | 69.82 |
| qwen25_vl_7b | VQAv2 | **86.04** | 75.23 | 71.53 | 73.03 | 71.11 | 70.76 | 68.75 |
| qwen25_vl_7b | RefCOCOg | — | — | **41.78** | 20.04 | 35.88 | 21.23 | 37.28 |
| llamav_o1_11b | GQA | **61.50** | 29.60 | 26.60 | 26.10 | 26.30 | 22.00 | 21.60 |
| llamav_o1_11b | VQAv1 | **78.34** | 39.96 | 38.63 | 35.71 | 37.69 | 29.49 | 31.93 |
| llamav_o1_11b | VQAv2 | **75.38** | 35.61 | 35.15 | 31.72 | 33.41 | 26.89 | 28.16 |
| llamav_o1_11b | RefCOCOg | — | — | **34.91** | 0.85 | 24.34 | 1.20 | 15.66 |

Δ = best marked − raw, recorded run → corrected run:

| model | dataset | unfiltered | appearance-filtered |
|---|---|---:|---:|
| gemma3_4b | GQA | +0.60 → **+1.00** | +3.06 → **+3.36** |
| gemma3_4b | VQAv1 | −0.22 → **+3.59** | +4.42 → **+6.71** |
| gemma3_4b | VQAv2 | −1.16 → **+1.61** | +3.02 → **+4.92** |
| qwen25_vl_7b | GQA | −13.60 → −11.40 | −8.56 → −7.03 |
| qwen25_vl_7b | VQAv1 | −16.72 → −11.51 | −12.50 → −8.33 |
| qwen25_vl_7b | VQAv2 | −15.79 → −10.81 | −12.42 → −8.94 |
| llamav_o1_11b | GQA | −33.70 → −31.90 | −35.17 → −33.64 |
| llamav_o1_11b | VQAv1 | −41.00 → −38.38 | −35.62 → −33.02 |
| llamav_o1_11b | VQAv2 | −41.98 → −39.77 | −35.82 → −33.51 |

**Verdict.** The corrected pipeline reproduces the paper's *direction* for Gemma-3-4B only:
marks now beat raw on all three VQA datasets even unfiltered, reaching +3.4 to +6.7 with
appearance questions excluded (still short of the paper's ~+7). Every Qwen marked cell
gains 2–6 points and RefCOCOg improves for all three models (best styles unchanged:
Gemma text IDs, Qwen/LlamaV numeric), but Qwen still loses 7–11 points and LlamaV 32–40.
The across-model claim does not reproduce: for models with strong native grounding the
overlay still costs more than the scene graph adds, even at declared opacity with
query-pruned marks and appearance questions removed. Reports:
`data_v2/table2_report.supplementary_concise{,.appearance_filtered}.{json,md}`.

## Best-config run (`data_v3`, 2026-08-15) and the closing verdict

A third full run applied every fix the audit identified, as deliberate deviations from the
paper profile: outline-only marks (`paper_aaai26_outline`, pilot-selected over a true
α=0.10 fill), a final-answer-only prompt on every condition (`direct_concise`), and filter
v2 (appearance ∪ subjective). It also added the zero-occlusion `text_graph` arm (clean
image + textual triples). Full numbers: `data_v3/table2_report.direct_concise*.{json,md}`
and `GENERATION_AUDIT.md` §9–10. Summary (Δ best marked − raw, filtered uniformly):

| model | recorded | corrected | best-config | text_graph (zero occlusion) |
|---|---|---|---|---|
| gemma3_4b | +2.9/+3.9/+2.8 | +3.2/+6.2/+4.7 | −0.2/+1.2/+0.8 | −1.0/+0.5/−0.2 |
| qwen25_vl_7b | −8.6/−12.5/−12.6 | −7.1/−8.3/−9.1 | −2.6/−4.9/−4.5 | −5.8/−4.4/−3.7 |
| llamav_o1_11b | −35.2/−35.8/−36.0 | −33.6/−33.4/−33.6 | −3.4/−6.0/−7.1 | −7.5/−13.2/−13.6 |

Every fix moved every model monotonically toward zero; none crossed it. LlamaV's collapse
was a chat-protocol artifact (plan-only 37–54% → ≤0.1% under direct_concise); Gemma's
apparent gains were answer-format effects that vanish when raw gets the same instruction;
Qwen keeps a 2–4× hurt/help flip ratio under every configuration; and the textual graph
with zero occlusion is neutral-to-harmful — the triples carry less information than the
models extract from clean pixels. Spatial/relational questions, the paper's core claim, are
the worst category for all three models. Marks remain genuinely indispensable for RefCOCOg
grounding. See `GENERATION_AUDIT.md` §10 for the full five-point evidence stack, including
the oracle-ceiling analysis showing the result cannot be inverted even by outcome-based
instance deletion.

The subsequent subsample-rule search (230 a-priori rules, derive-on-VQAv1 /
verify-on-GQA+VQAv2: zero passed even on train) and the final config pilots
(`outline_clean` reached −0.92, the best marked result recorded, still below raw) closed
the investigation — see `GENERATION_AUDIT.md` §11 for the exhaustion record.

**Read `GENERATION_AUDIT.md` before quoting any of these numbers.** The generation-level
audit (2026-08-15) establishes: LlamaV's marked cells are 37–54% answer-less "plan"
statements (a chat-protocol artifact, not a visual result); Qwen's residual deficit is
evidence destruction of the very objects Algorithm 3 selects (the fill makes the queried
objects unrecognizable — existence denial and identity swaps follow); and **Gemma's gains
are mostly answer-format calibration** — under lenient phrase scoring its GQA gain
disappears (+1.99 → −0.92) and VQAv2 drops to +1.51; only VQAv1 keeps a genuine ~+5.
Truncation is ruled out, the off-grid decode setting changes nothing (≤0.7 points), and a
"marks are optional aids" prompt changes nothing (≤0.5) — the deficit is in the pixels,
not the protocol.

## Caveats

- **One decode setting, not the published 27-point grid.** Every cell is a single
  (seed 0, temp 0.2, top_p 0.9) run, so `std/min/max` in the JSON are degenerate and the
  reports carry `"runs": 1`. The grid multiplies cost by 27.
- **One question per image** (`--one-per-image`), matching the paper's one-render-per-image
  contract; GQA in particular has far more questions available per image.
- **VQAv1/VQAv2 images come from COCO `train2014`**, which is what the author manifest
  specifies. Qwen's 86–88 raw may partly reflect train-split familiarity; these are not
  clean held-out scores.
- **Gemma's GQA raw (48.0) remains ~8 points below the paper's 56.2.** Unexplained; §5.5
  attributes it to sample selection rather than the metric or prompt.
- **The run's renders double-applied the mask fill** (`visualizer.py` blended the fill,
  then filled again per object), so effective opacity was 0.4375, not the declared 0.25.
  Fixed 2026-08-14 (fill is now applied once); renders produced after that commit are
  lighter than the recorded artifacts. Scores in this file are from the pre-fix renders.
- **An abandoned third prompt profile, `visual_aid_concise`, exists on disk**
  (`data/predictions/visual_aid_concise/`, 31,000 generations: Gemma complete, Qwen
  GQA-only and partial, LlamaV absent). It was never scored and is excluded from every
  table here. Finish or delete it before using it for anything.
- Three environment fixes were required on this Blackwell (sm_120) GPU and are recorded in
  `run_afk.sh` and `compat/sitecustomize.py`: a `max_num_seqs` cap for the multimodal
  profile run, a vision-tower attention override for Qwen, and folding the system prompt
  into the user turn for Mllama. None affects decoding.

## Released-artifact compatibility check (unchanged, do not pool)

The released Gemma VQAv2 JSON artifacts scored against the same 5,180 rows with VQA
normalization. Single runs, different bare prompt.

| Released artifact | Accuracy |
|---|---:|
| Raw | 63.25 |
| Segmented | 53.77 |
| GoM text IDs | 49.56 |
| GoM numeric IDs | 51.77 |
| GoM text IDs + relation labels | 49.96 |
| GoM numeric IDs + relation labels | 51.04 |

These do not reproduce the published improvement and must not be merged with the runs above.

## gom_v2 run (`data_v5`, 2026-08-16) — defect-free pipeline, curated eval, still no across-model VQA gain

The user-directed overhaul after the FLIP_EXAMPLES_PAPER_GOM audit. Every defect that audit
surfaced was fixed and visually verified on a 49-image control set before launch (see
CLAUDE.md "gom_v2" note): `gom_v2` render profile (outline, targeted open-vocab detection,
caps 15/4, cross-class suppression, stuff-mask filter with the `_N`-suffix bug fixed,
Algorithm 3 zero-match top-6 fallback, 1 relation/head, arrows below labels, active
label-collision avoidance), `gom_v2_concise` prompt (marks explained, demoted to hints,
ID/relation-word/color answers banned, direct answers), curated one-question-per-image eval
(`curate_eval.py`, locks in `manifests/*_curated_v1.txt`: 996/988/991 rows — appearance,
subjective, ambiguous-referent, text-reading questions removed; spatial/relational preferred),
conditions raw + the four `gom_*` only, `seed0/temp0.2/top_p0.9`.

Renders: mean 3.9 marks/image (max 7), max 5 arrows — versus 20–35 marks in the paper-faithful
runs. LlamaV plan-mode cured (4–6 per 1000, was 370–540). ID-leak answers on `gom_text_labeled`:
Gemma 0, Qwen 83/2975, LlamaV 79/2975 (concentrated on "who …?" questions; numeric conditions
immune). Report: `data_v5/table2_report.gom_v2_concise.{json,md}`.

Δ = best gom condition − raw, lenient metric (`gqa_hit` with ≤6-token guard /
`vqa_soft_acc_phrase`); strict in parentheses:

| model | GQA | VQAv1 | VQAv2 | RefCOCOg (best gom, strict) |
|---|---:|---:|---:|---:|
| gemma3_4b | **+1.81** (−0.70) | −2.38 (−2.40) | −0.42 (−0.73) | 51.00 (text) |
| qwen25_vl_7b | −6.43 (−6.22) | −6.72 (−6.88) | −6.66 (−7.17) | 35.04 (numeric+labels) |
| llamav_o1_11b | −3.41 (−6.02) | −6.27 (−8.70) | −7.79 (−10.70) | 28.05 (numeric) |

Break/rescue balance (best condition): Gemma GQA is the only cell where marks net-help
(109 rescues vs 91 breaks); Qwen stays at 3–11× more breaks than rescues. Of the 12 original
flip cases (11 kept after curation), 6–8/11 now answer correctly on the marked image.

**Verdict.** With clutter, label placement, mask fill, prompting, scoring normalization and
question quality all fixed — and verified fixed by direct visual inspection — marks still cost
Qwen ~6.5 and LlamaV ~3–8 points, and leave Gemma at parity. The paper's across-model VQA
gains do not appear under any configuration found in three independent searches (data_v3
§Best-config, the 230-rule subsample search, and this one). The durable positive results
remain: marks are indispensable for RefCOCOg grounding, and mark-induced damage is now small
and characterized (near-miss label leaks like cupcakes→cake, yes→no flips on relational
questions) rather than catastrophic.

## gom_v3 run (`data_v6`, 2026-08-17) — render defects fixed, VQA verdict unchanged

Follow-up to the user's visual audit of the gom_v2 flip gallery, which found five defect
classes still present (`FLIP_AUDIT_GOM_V2.md` classifies all 20 cases). Fixes, all verified
before the run by a 93-row stress-audit set stratified over the failure modes
(`make_audit_set.py`, `check_render_quality.py`, `check_leakage.py`):

1. **Open-vocabulary detector queries** (`question_intent.py`): the closed `_VISUAL_OBJECTS`
   gate meant "van", "cheeseburger", "towel", "skier", "guitar" were never queried. Now any
   content noun plus modifier+head phrases ("teddy bear", "coffee table") reach OWLv2, and bare
   category words are *removed* (querying "animal" produced marks labelled `animal_1`).
   Effect: 228 distinct open-vocabulary classes vs 16, marks only 3.40 → 3.57 per image.
2. **Deterministic label placement** (`visualizer.py`, `deterministic_label_placement`): arrows
   are predicted before labels are placed; a shared registry enforces a hard zero-overlap
   constraint over object labels and relation labels; ranked candidates + spiral
   fallback; every post-hoc mover is bypassed (they were re-introducing overlaps).
   **Result: `label_overlap_count == 0` on all 3,975 images x 6 variants**, recorded per render.
   **Correction (gom_v4, see below): the constraint never covered arrows.** An earlier
   version of this line claimed it did. The registry held label boxes only; arrows sat at
   `zorder=6.5` under every opaque label box, and every arrowhead in this run was painted
   over. `label_overlap_count == 0` was true and said nothing about arrow legibility.
3. **Part-of-object fragment dedup** (mask containment ≥ 0.70, not box containment): removes the
   elephant-leg class of duplicate mark; distinct instances keep disjoint masks and survive.
4. **Prompt v3** — presence assertion plus few-shot exemplars naming the forbidden tokens.

**The prompt was a regression, and the experiment says so.** Running the same gom_v3 renders
under both prompts isolates it (lenient Δ = best gom − raw):

| configuration | gemma GQA | gemma V1 | gemma V2 | qwen GQA | qwen V1 | qwen V2 | llamav GQA | llamav V1 | llamav V2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v2 renders + v2 prompt (`data_v5`) | **+1.81** | −2.38 | −0.42 | −6.43 | −6.72 | −6.66 | −3.41 | −6.27 | −7.79 |
| v3 renders + v3 prompt (`data_v6`) | −1.10 | −2.35 | −2.73 | −6.63 | −7.28 | −7.05 | −11.04 | −8.34 | −10.37 |
| v3 renders + v2 prompt (`data_v6`) | 0.00 | **−1.60** | −1.31 | −6.93 | −7.86 | −8.08 | −4.62 | −6.78 | −8.46 |

Prompt v3 costs LlamaV 6.4 points on GQA and Gemma 1.1: naming a banned token in a prohibition
("never person_1", "never above or below") *raises* its probability — Gemma's bare-relation-word
answers went 1 → 31 and LlamaV's plan-mode returned (0 → 9-14). A positive-only rewrite
(`gom_v3b_concise`) did not beat `gom_v2_concise` on the audit set either. **Use
`gom_v2_concise`.** The v3 profiles stay in the code with this measurement recorded.

**The render fixes are neutral on VQA and slightly positive on grounding.** Comparing the two
`v2 prompt` rows: VQA moves within ±1.8 in both directions; RefCOCOg improves for every model
(51.00 → 51.84 gemma, 35.04 → 35.15 qwen, 28.05 → 28.67 llamav). Clean renders did not convert
into VQA gains.

**Residual, measured, not hidden:** text-tag ID leakage on "who" questions is a model behaviour
no prompt tested removes (full run: 87-90 leaks per condition for Qwen, 152-172 for LlamaV on
text conditions; **4-5 and 18-19 on numeric conditions**). Open-vocabulary labels make it worse
precisely because `boy_1`/`girl_1` *are* the answer. All four gom conditions are evaluated, so
the tables show it.

**Verdict after three overhauls.** With clutter, occlusion, mask fill, label overlap, detection
coverage, fragment marks, prompting, scoring normalisation and question quality all fixed and
each fix verified by an automated gate rather than by eyeballing: Gemma-3-4B reaches VQA parity
(+1.8 to −1.6), Qwen2.5-VL-7B stays at −6 to −8, LlamaV-o1-11B at −3.4 to −8.5, and marks remain
indispensable for RefCOCOg grounding (0 → 28-52). The paper's across-model VQA gains do not
reproduce under any configuration found in four independent searches.

## gom_v4 run (`data_smoke`, 2026-08-17) — the arrowheads were never visible, and the gate said nothing

Driven by the gom_v3 flip gallery (`FLIP_EXAMPLES_PAPER_GOM.md`) plus the user's reading of
those renders (`FLIP_EXAMPLES_PAPER_GOM_ISSUES.txt`), and by a sweep of **all 5,486 flips** in
`data_v6` (3 models x 3 datasets x 4 marked conditions) rather than the 20 cases the gallery
shows. Scope: a 95-row stratified smoke set, not a full run — the numbers below gate defects,
they do not re-measure Table 2.

### What the gom_v3 gate could not see

`label_overlap_count == 0` was true across 3,975 images x 6 variants and said nothing about
arrow legibility, because the registry only ever held label boxes. Measured on the same
scenes after the fact:

| defect | before | gom_v4 |
|---|---:|---:|
| arrowheads painted over by a label box (dense test scene, head reservation disabled) | 5 / 8 | **0 / 8** |
| relation label further than 60 px from its own arc (95 images) | 14 renders, up to 200 px | **0, max 55 px** |
| relation labels with no seat, silently dropped | 2 of 2 on one image | **0** |
| graph/render `edge_digest` mismatch | 27 / 95 once Alg-3 reordered | **0 (order-insensitive)** |

The "before" column is measured on the current code with the specific fix switched off, not on
the `data_v6` binaries — in gom_v3 proper the head was additionally *inside* the target object
(6 px from its centroid), so the real occlusion rate there was higher, not lower.

The head sat 6 px from the *target centroid* (`_shrink_segment_px`), which is exactly where an
inside object label is anchored, under an opaque `alpha=0.95` box at `zorder=7`/`9` while the
arrow was pinned at `6.5`. The zorder was not the bug; it was a symptom of label placement
never seeing the arrows. Arrows are now drawn first, clipped to the target box by matplotlib's
own `patchA`/`patchB`, and their heads reserved in the registry the placers already treat as a
hard constraint. `label_overlap_count` keeps meaning label-vs-label (heads live in a separate
list), so the earlier claim stays comparable.

### Three mechanisms the gallery missed or misattributed

**1. Query-driven marks plant false evidence — the largest fixable mechanism, and it is not a
threshold problem.** GQA existence questions, flip rate by whether the questioned noun is marked:

| model | gold=no, marked | gold=no, unmarked | gold=yes, marked | gold=yes, unmarked |
|---|---:|---:|---:|---:|
| gemma3_4b | **34.5%** | 6.0% | 7.6% | **18.4%** |
| qwen25_vl_7b | **13.0%** | 4.8% | 13.5% | **17.6%** |
| llamav_o1_11b | **34.1%** | 24.2% | 1.6% | 0.0% |

Both directions are one disease: the model reads the mark set as an *existence oracle*.
Gemma/LlamaV over-assert (`no`→`yes` is 21-42% of their flips), Qwen over-denies (`yes`→`no`
is 20-24% of its flips). Detector confidence does **not** separate the false-positive
question-driven marks from the true ones (mean 0.462 vs 0.474; a 0.35 threshold kills 36.6% of
FPs and 35.9% of TPs), so no gating fixes it — hence `gom_v4_concise`'s positive-framed
"the outlines cover only part of the scene, and an outline can be wrong" and nothing else.
The gate deliberately excludes gold=`no` rows from query coverage for the same reason.

**2. GoM destroys spatial reasoning on exactly the questions it targets.** GQA left/right
(n=88), raw → best marked: Gemma **61.4 → 30.7**, Qwen 92.0 → 79.5, LlamaV 63.6 → 43.2. By what
the graph contains: both nouns marked but no relation between them (n=14) **−42.9** Gemma;
arrows only between the wrong pair (n=32) −18.8; correct relation present and drawn (n=39)
−38.5. Even a correct arrow hurt — that is the legibility failure above.

Root cause of the missing pair edges: **Algorithm 3 compared a detector label's WordNet aliases
to the raw question string.** `canonical_object_label` maps man/woman/child → `person`, and
`label_aliases("person")` is `{person, persons, persones}`, so a question saying "man" never
matched a human mark, Algorithm 3 fell into its zero-match branch, and the arrow it drew joined
whichever two objects were nearest. It now matches the question's parsed object terms, ranks
edges pair-first, emits both axes when both clear the margin, and treats a left/right question
as relevant to its whole axis. On the smoke set that moves arrows-on-the-queried-pair from
58.5% to 62.6% and left/right pair coverage from 48.6% to 51.4% — real but small; the binding
constraint is detector recall (only 70% of left/right questions have both objects detected at
all), which is the documented ceiling.

**Correction, measured at full GQA scale (996 images, gom_v3 vs gom_v4 on identical rows):**
restricting arrow *sources* to question-matched objects is essentially a **no-op** — mean
arrows/image is 2.07 in both runs. The projection that it would cut arrows by 38% came from a
proxy that matched detector labels against question tokens; the real Algorithm 3 matched set
(WordNet aliases + FastText + now the parsed object terms) is far broader, so
`matched ∩ kept ≈ kept`. What actually declutters is the **per-image cap**: max arrows 11 → 5,
and images with more than 5 arrows **31 → 0**, with the median render untouched. That is the
right shape — it removes the pathological renders the gallery complained about without
thinning the typical one — but the credit belongs to `max_relations_total`, not to the
head restriction.

**3. ID leakage on "who" questions was a labelling failure, not a prompting failure.** All 79
GQA `who` golds are person subtypes (man 22, woman 13, boy 10, girl 8, …) and the graph carried
126 marks labelled `person`. `normalize("man_1") == "man 1"`, which `gqa_hit` phrase-matches
against gold `man` for **1.0**; `person_1` scores 0. OWLv2 *does* detect the subtype
(man@0.47, boy@0.37, woman@0.32 on the gallery images) — dedup was dropping it against
`person@0.54` at mask IoU 1.000. The survivor now inherits the specific name (15 renames on 95
images). This is a partial fix: it needs the subtype detection to reach the dedup stage, which
it does on ~3 of 4 gallery cases.

### Scoring

`gqa_hit("strawberry", "strawberries")` was `0.0` — a scorer artifact the gom_v2 audit
recommended fixing and nobody did. The lenient metrics are now plural-tolerant
(`vqa_metrics.singularize`, applied symmetrically); **the official/primary VQA metric is
untouched**. Worth +0.4 to +1.3 points on every cell including raw, so Δ moves ≤0.3.
`score_table2.py --question-filter spatial` reports the spatial/relational slice
(GQA 653/996, VQAv1 102/988, VQAv2 132/991) alongside the full set.

### Smoke-run results (95 stratified rows, both prompts, 3 models)

**Read the scope first.** `make_audit_set.py` builds this set by *over-sampling the failure
modes* (who / existence-yes / existence-no / left-right / open-vocab / small) plus all 20 cases
from the gom_v3 gallery. It is a defect gate, not an accuracy benchmark: GQA carries n=67
(n=48 spatial), VQAv1/VQAv2 only n=14 (n=3 spatial), where one row is 7 to 33 points. The
deltas below are far worse than the full-run deltas by construction. **They are reported, not
gated** — the gates are the render and leakage tables.

Render gate, all 95 images x 6 variants: label overlaps **0**, edge-digest mismatch **0**,
arrowheads occluded **0**, relation labels off their arc **0** (max drift 55 px), relation
labels dropped **0**, mark/arrow budget **0** over, query coverage **90.3%**.

Leakage gate (worst marked condition), both prompts **PASS**:

| | gom_v3 (`data_v6`, 1000 rows) | gom_v4 (95 rows) |
|---|---:|---:|
| Qwen ID-shaped answers, text conditions | 87-90 | 12-14 |
| **generic `person_N` answers (gated, <= 5)** | — | **5** |
| LlamaV generic `person_N` answers | 152-172 | **0** |
| relation-word answers | 1 -> 31 under prompt v3 | **0** (v2 prompt), 2 (v4 prompt) |
| plan-mode ("I will analyze…") | 0 (v2) / 9-14 (v3) | **0** |
| numeric-ID conditions, any leak | 4-5 | **0** |

Δ = best marked − raw, lenient:

| model | dataset | n | raw | Δ `gom_v2_concise` | Δ `gom_v4_concise` |
|---|---|---:|---:|---:|---:|
| gemma3_4b | gqa | 67 | 56.72 | −1.49 | −1.49 |
| gemma3_4b | gqa *(spatial)* | 48 | 64.58 | **+2.08** | −2.08 |
| qwen25_vl_7b | gqa | 67 | 88.06 | −19.40 | −20.90 |
| qwen25_vl_7b | gqa *(spatial)* | 48 | 93.75 | −14.58 | −14.58 |
| llamav_o1_11b | gqa | 67 | 70.15 | −8.96 | −7.46 |
| llamav_o1_11b | gqa *(spatial)* | 48 | 79.17 | −10.42 | −10.42 |

(VQAv1/VQAv2 cells are n=14 and n=3 — omitted as noise; the full table is in the run's
`score.*.json`.)

**The prompt fix does not pay, and this is the third time.** `gom_v4_concise` adds two
*positive-framed* sentences — marks are partial and fallible, and answer about the two things
the question names — precisely because prompt v3's prohibitions backfired by negation priming.
It is neutral or worse on almost every cell (Gemma's GQA spatial slice goes +2.08 → −2.08), and
it reintroduces 2 relation-word answers that `gom_v2_concise` has none of. **Keep
`gom_v2_concise`.** The profile stays in the code with this measurement recorded, as v3 did.

**What the render fixes did and did not buy.** They removed every measurable render defect and
they eliminated LlamaV's generic-tag leakage outright (152-172 → 0 per condition). They did not
turn the VQA sign positive on this set — but this set is stratified to be hard, so that is not
evidence either way. The honest next step is a full curated re-run under `gom_v4` +
`gom_v2_concise`; nothing here supersedes §gom_v3's Table 2.

**Residual, characterized rather than hidden.** The 5-10 remaining "who leaks" are no longer
`person_1`: 6 of 11 are the model answering a correct person subtype (`man_2`, `guy_3`,
`lady_1`) where the gold wants a *role* noun — `chef`, `umpire`, `catcher`, `doctor`, `skier`,
`player` — that no detector emits. Query coverage misses (`ladle`, `jet`, `shadow`, `house`,
`television`) are detector recall. Both are ceilings, not render defects, which is why the gate
counts only generic answer-less tags and why coverage is gated at 90% rather than 100%.

### Full-scale preprocessing verification (`data_v7`, 3,975 curated images)

Run before inference, because a defect found after 3 h of GPU is a defect found too late.
`data_v7/prepared/` is a byte copy of `data_v6/prepared/`, so gom_v3 and gom_v4 differ only in
the pipeline. Three things the 95-row stratified smoke set got wrong, all caught here:

**1. Two gate thresholds were calibrated on 95 rows and were simply wrong at 3,975.**
`--max-marks 10` gated *below the profile's own contract* (`max_detections_total: 15`) and
flagged 30 images that were never out of budget; it is now 15. `--min-coverage 90` was
unreachable — measured over all four datasets, **gom_v3 sits at 84.1% and gom_v4 at 83.7%**,
so the floor is now 80, which still catches a regression of the kind gom_v3 fixed (a closed
vocabulary would land far below) without chasing a number detector recall cannot reach.

**2. gom_v4's 1-point coverage dip was my metric, not the pipeline.** `_inherit_specific_label`
renames a `person` mark to `man`, and `check_render_quality.label_terms` did not canonicalize
mark labels the way it canonicalizes question terms — so a question about a "person" stopped
matching its own (more specific) mark. Canonicalizing recovers 83.1% → 83.7%; the residual
0.4pp against gom_v3 is real and small.

**3. A pre-existing render defect, newly measured: 10 of 3,975 renders (0.25%) come out
≥25 percentage points whiter than their own source photo** — in the worst case a 640x154 image
whose content occupies 162x48 px inside the frame, 94% white. It is **not** caused by gom_v4:
gom_v3 shows 88.6% white on the same image and 93.8% on the label-free `segmented` variant, so
it is present in every recorded run, including the paper-profile ones. It is also **not** an
aspect-ratio problem (most affected images are 1.50 / 1.33 / 0.66) and it is not the label
placer — `segmented` has no labels. Two attempted reproductions (axes autoscale; an
out-of-bounds artist) both failed to trigger it, so the mechanism is *unknown* and no
speculative fix was shipped. Reproduce with the render-vs-source whiteness scan in this
section's commit. Affected rows are guaranteed raw-wins in every marked condition.

**Relation-label drift: the 72 px threshold is calibrated, and here is the calibration.**
Run-wide worst drift is 66 px on 4 images (0.10%), every one carrying a leader line to its arc.
A finer 24-angle spiral was implemented, measured, and **reverted**: it fixed 1 of the 4 while
changing renders on unrelated images (2 of 120 controls), which is the worst of both. After the
revert all 120 control renders hash-match their originals, so the run has one rendering
behaviour throughout. The threshold guards against drift back toward gom_v3's 200 px-with-no-leader,
not against 66 px-with-a-leader.

Final gate on all 3,975 images x 6 variants: label overlaps **0**, edge-digest mismatch **0**,
arrowheads occluded **0**, relation labels off arc **0**, dropped **0**, over budget **0**,
query coverage **83.7%**. `audit_relations.py`: **0 hard consistency errors** on all four
datasets. Arrows max **11 → 5**; marks mean 3.77 → 3.75 (unchanged). GQA who-questions:
generic `person` marks **126 → 70**, specific subtype marks **0 → 123**.

### Full gom_v4 run (`data_v7`, 3,975 curated images, 3 models, 4 datasets)

`data_v7/prepared/` is a byte copy of `data_v6/prepared/`, so gom_v3 and gom_v4 differ **only**
in the pipeline. The gom_v3 column below is that run's predictions **re-scored with the current
plural-tolerant metric**, because the scorer changed too: leaving it un-rescored would credit
the scorer fix to the pipeline (the metric alone moves deltas by −0.9 to +0.2).

Δ = best marked − raw, lenient:

| model | dataset | raw | Δ gom_v3* | Δ gom_v4 | Δ gom_v4 (spatial) |
|---|---|---:|---:|---:|---:|
| gemma3_4b | gqa | 54.52 | −0.40 | −1.71 | −1.23 |
| gemma3_4b | vqav1 | 69.62 | −1.62 | −2.61 | +12.65 |
| gemma3_4b | vqav2 | 67.31 | −1.47 | −0.42 | +5.15 |
| gemma3_4b | refcocog | — | 51.84 | 51.58 | (absolute) |
| qwen25_vl_7b | gqa | 73.49 | −7.13 | −6.22 | −6.58 |
| qwen25_vl_7b | vqav1 | 89.17 | −7.96 | −7.65 | −6.57 |
| qwen25_vl_7b | vqav2 | 87.33 | −8.17 | −7.85 | −9.77 |
| qwen25_vl_7b | refcocog | — | 35.15 | 36.04 | (absolute) |
| llamav_o1_11b | gqa | 59.94 | −4.42 | −4.62 | −5.97 |
| llamav_o1_11b | vqav1 | 78.53 | −7.67 | −8.35 | −6.27 |
| llamav_o1_11b | vqav2 | 77.86 | −9.99 | −10.31 | −9.85 |
| llamav_o1_11b | refcocog | — | 28.67 | 29.21 | (absolute) |

\* re-scored with the current metric so only the pipeline differs.

**The VQA verdict is unchanged for the fourth consecutive overhaul.** Every gom_v3 → gom_v4
movement is within ±1.0 on n≈1000, where the standard error is ~1.5 — i.e. inside noise. Qwen
is consistently in the positive direction on all four datasets (+0.3 to +0.9) and LlamaV
consistently slightly negative; neither is significant. Gemma's spatial-slice numbers on
VQAv1/VQAv2 (+12.65 / +5.15) sit on n=100 and n=133 respectively — roughly 12 rows — and must
not be quoted as a result; the GQA spatial slice, with n=630, is −1.23.

**What the fixes did buy, measured.** The specific-person-label work is the one change with a
clear, attributable effect, and it is visible only when GQA is split by question type
(`gom_text_labeled`, identical rows):

| model | who-questions (n=79) | everything else (n=917) |
|---|---:|---:|
| gemma3_4b | 39.2 → 31.6 (**−7.6**) | 54.5 → 53.3 (−1.2) |
| qwen25_vl_7b | 15.2 → **21.5 (+6.3)** | 69.0 → 70.3 (+1.3) |

Qwen gains 6–7.6 points on exactly the rows the fix targeted, with non-who rows flat — the
`person_1` → `man_1` inheritance converting tag leaks into correct answers. **Gemma loses the
same rows**, and Gemma never leaked tags at all (0 in every condition): for it the specific
labels are not fixing a leak, they are a new failure surface, because the render now asserts
`man_1`/`girl_1` and a wrong subtype guess is something to copy. Verified mechanically: of
Gemma's 10 who-question flips, 6 answered a subtype that is on the image and wrong (0 answered a
correct one); all 5 of Qwen's did the same. **Specific person labels are therefore a per-model
trade, not a universal win**, and the two effects cancel in the aggregate.

Generic-tag leakage (the scoring-zero `person_N` answers), worst condition, same 2,975 rows:

| | gom_v3 | gom_v4 |
|---|---:|---:|
| qwen, text tags | 67 (2.25%) | **24 (0.81%)** |
| llamav, text tags | 14 | **3** |
| gemma, text tags | 1 | **0** |
| numeric-ID conditions | 0 | 0 |

Total ID-shaped answers rose (383 → 422) while *generic* ones fell ~65%: exactly the intended
conversion of `person_1` into `man_1`. Relation-word answers (117 → 112) and plan-mode
(16 → 20) are unchanged. `check_leakage.py` now gates on the **rate**, not a raw count — the
old `<= 5` was calibrated on a 95-row smoke set and is meaningless against 2,975 rows. The 1.5%
threshold is calibrated *between* the two measured runs and is a regression gate, not a quality
bar: it passes gom_v4 and fails gom_v3.

**Two self-inflicted defects found while attributing the above, both measured, neither fixed
in this run** (fixing them requires a re-preprocess + re-inference):

1. **`guy`/`lady` should never have been added to `_PERSON_SUBTYPES`.** They were added to lift
   who-coverage 56/79 → 62/79. But OWLv2 stamps `guy_1` on **26** of 79 who-images while only
   **4** golds are `guy` and **0** are `lady`; the other 22 golds are `man` (7), `girl` (4),
   `player` (3), `woman` (3), `boy` (2), `people` (2), `child` (1). Both models copy the tag.
   Cost: **Qwen −1.51 GQA**, Gemma −0.40 — larger than the entire gom_v3 → gom_v4 movement.
   Fix: drop them from the query list; the `_ALIASES` entries can stay.
2. **`boy`/`girl` inheritance is silently dead.** `_inherit_specific_label` decides via
   `canonical_object_label`, and `boy`/`girl` are absent from `_ALIASES`, so they canonicalize
   to themselves and never rename a `person` mark. 6 who-rows are stuck on `person_N`
   (6/6 wrong for Qwen). This is why `2408238` still rendered `person_1` despite OWLv2
   detecting `boy@0.37`.
   **The obvious fix is wrong**: adding them to `_ALIASES` would make every *direct* question
   about a boy or girl query the generic `person` (`('boy','bat')` → `('person','bat')`),
   reintroducing generic marks exactly where specific ones work today. The repair belongs in a
   dedicated subtype→generic map used only by inheritance, leaving global canonicalization alone.

## gom_v5 run (`data_v8`, 2026-08-17) — the render defects are gone and the VQA verdict holds

Driven by the user's review of the gom_v4 gallery: cases 10/14 showed bad segmentation, and
*"many images where the arrow is not clear — if it is too short only the arrowhead is visible"*.
Both were real and both were larger than the two cases. `data_v8/prepared/` is a byte copy of
`data_v7/prepared/`, so gom_v4 and gom_v5 differ only in the pipeline.

### The arrow defect was self-inflicted, and the obvious fix was the wrong one

gom_v4 clipped arrow endpoints to the box boundary (`patchA`/`patchB`). That is what made
arrowheads visible — and it deleted the shaft on **52.9% of arrows** (4,525/8,552), across 71.7%
of images with arrows, because for the median short pair the centroid distance is only 0.89x the
summed box half-extents: the chord lies inside both boxes and clipping leaves nothing.

Curving the arc — the suggested fix — is right in principle but insufficient alone: to clear
both boxes the required `arc3` rad has median **1.83** and p90 **3.52**, and a visually sane cap
of 1.0-1.2 fixes only 0-6%. The actual fix is that **gom_v4 applied two independent fixes for
buried arrowheads and needed only one.** Head visibility comes from reserving the head bbox in
the label registry, not from clipping. Dropping the clip:

| | gom_v4 | gom_v5 |
|---|---:|---:|
| arrow shaft, median (real graphs) | 16 px | **102 px** |
| min arrow shaft per render, median (run-wide) | — | **190 px** |
| renders with an arrow under 25 px | ~53% of arrows | **24 / 15,900 = 0.15%** |
| arrowheads hidden under a label | 0 | **0** |

Adaptive curvature (capped at 1.4, calibrated on the densest real scene) handles the residual
overlapping/nested pairs, and 94 relations between near-coincident centroids — undrawable at any
curvature — are now filtered at selection so graph, triples and render keep one edge multiset.

### The scribbled masks were a regression from the legacy monolith

`_draw_segmentation` used `cv2.RETR_CCOMP`, which returns interior **hole** boundaries as well
as outer ones, and stroked every contour with no area floor. `all_in_one_gom.py:1983` did it
correctly (`RETR_EXTERNAL` + largest contour); the refactor into `gom/viz/` lost it, and since
the gom_v* profiles render outline-only the scribble was the entire visual. Now `RETR_EXTERNAL`
plus an area floor — an area floor rather than largest-only, so a genuinely two-part object
keeps both parts.

**The contour-count gate was dropped to informational, and that decision is not a concession
to the numbers.** Measured over 3,975 images: 77% of renders stroke one contour, 97.6% three or
fewer, thin tail to 13 — the shape of legitimately fragmented objects (a bicycle is 6). The
count measures fragmentation, not the defect; hole boundaries are now structurally impossible,
and `test_mask_outline_ignores_holes_and_specks` asserts that directly, which is a sharper guard
than any run-wide threshold.

### Result: Δ = best marked − raw, gom_v4 vs gom_v5 on identical rows, same metric

| model | dataset | raw | Δ gom_v4 | Δ gom_v5 | change | Δ gom_v5 spatial |
|---|---|---:|---:|---:|---:|---:|
| gemma3_4b | gqa | 54.52 | −1.71 | −1.91 | −0.20 | −2.14 |
| gemma3_4b | vqav1 | 69.62 | −2.61 | −2.18 | +0.44 | +5.49 |
| gemma3_4b | vqav2 | 67.31 | −0.42 | −0.89 | −0.46 | +2.35 |
| gemma3_4b | refcocog | — | 51.58 | 50.93 | −0.65 | (absolute) |
| qwen25_vl_7b | gqa | 73.49 | −6.22 | −7.13 | −0.90 | −7.66 |
| qwen25_vl_7b | vqav1 | 89.17 | −7.65 | −8.71 | −1.06 | −8.04 |
| qwen25_vl_7b | vqav2 | 87.33 | −7.85 | −7.83 | +0.02 | −9.09 |
| qwen25_vl_7b | refcocog | — | 36.04 | 37.07 | +1.03 | (absolute) |
| llamav_o1_11b | gqa | 59.94 | −4.62 | −4.52 | +0.10 | −4.59 |
| llamav_o1_11b | vqav1 | 78.53 | −8.35 | −7.96 | +0.39 | −8.63 |
| llamav_o1_11b | vqav2 | 77.86 | −10.31 | −10.31 | −0.00 | −10.15 |
| llamav_o1_11b | refcocog | — | 29.21 | 27.67 | −1.54 | (absolute) |

**Fifth consecutive overhaul to land neutral.** Every movement is within ±1.1 on n≈1000 (SE
~1.5). Making the arrows legible and the outlines clean did not change the VQA verdict.
Gemma's spatial VQAv1/VQAv2 slices (+5.49/+2.35) sit on n=100/133 and must not be quoted.

### The one large, precisely-located effect: subtype labelling

Splitting GQA by question type (identical rows, gom_v4 → gom_v5):

| model | condition | who-questions (n=79) | everything else (n=917) |
|---|---|---:|---:|
| qwen25_vl_7b | gom_text | 22.8 → **36.7 (+13.9)** | 69.2 → 68.9 (−0.3) |
| qwen25_vl_7b | gom_text_labeled | 21.5 → **34.2 (+12.7)** | 70.3 → 68.0 (−2.3) |
| gemma3_4b | gom_text_labeled | 31.6 → 34.2 (+2.5) | 53.3 → 54.2 (+0.9) |

Removing `guy`/`lady` from the queries (26 → 0 marks on who-images) and giving
`_inherit_specific_label` its own subtype map — rather than routing through
`canonical_object_label`, where `boy`/`girl` canonicalize to themselves — moved exactly the rows
they targeted. Generic `person_N` answers are now 0.64% of rows against gom_v3's 2.25%.

**The aggregate still worsened, and the arithmetic is the point:** +12.7 on 79 rows is ~10 rows
gained; −2.3 on 917 rows is ~21 lost. The non-who loss is concentrated on `gom_text_labeled`
(−2.3) versus `gom_text` (−0.3), and the only difference between those conditions is whether the
relation words are drawn — so the more legible relation labels appear to cost accuracy on
questions the graph is not about. At ~1.5 SE this is suggestive, not established.

### Cross-detector corroboration does not separate wrong labels either

The probe recorded, for 19,764 marks, whether the survivor beat a rival detection claiming a
different class. Joined to predictions by box IoU:

| bucket | n | beat a different-class rival |
|---|---:|---:|
| marks the model copied and got wrong | 83 | 19 (22.9%) |
| every other mark | 9,991 | 1,643 (16.4%) |

22.9% vs 16.4% is not a usable separator. Combined with the earlier null result on confidence
(0.537 vs 0.529) and on open-vocab origin (50/50), **the wrong-class-name problem has no
detector-side signal we have found** and should be treated as a ceiling.

## GEPA prompt optimization (2026-08-17..19) — three runs, no prompt beats the hand-written one

The last untested lever after five render overhauls was the marked-condition system prompt.
Three DSPy-GEPA runs (~20h GPU) optimized `SYSTEM_GOM_V2` against `data_v8` renders, all four
gom_* conditions at once, on Qwen2.5-VL-7B. **Every run is null out of sample.** The harness is
local-only (`prompt_opt/`, git-ignored); this section is the record.

Each run selects on a val split and is then judged on images the optimizer never saw:

| run | setup | candidates | best val | **best holdout** |
|---|---|---:|---:|---:|
| A | bf16 vLLM, GPT-5.6-luna judge, GEPA's default proposer | 121 | +1.95 | **−0.09** |
| B | Q4 GGUF on llama.cpp, local Qwen3.8-27B judge, generalisation-only proposer | 37 | +0.43 | **−0.12** |
| C | as B, warm-started from B's best prompt, 8h | 108 | +1.11 | **+0.04** |

Run C's full holdout table (500 held-out images, 2,000 pairs each), against the production seed:

| prompt | words | val | holdout | vs seed |
|---|---:|---:|---:|---:|
| cand 104 | 375 | 73.91 | 74.07 | +0.04 |
| **production seed** | **138** | — | **74.03** | — |
| warm start | 288 | 73.73 | 73.90 | −0.13 |
| cand 88 (**val winner**) | 369 | **74.53** | 73.55 | **−0.48** |

**The val winner is the worst prompt out of sample**, and across the scored candidates val rank is
anti-correlated with holdout rank (Spearman −0.26). In all three runs the seed scored *lowest* on
val — which is why GEPA never selected it — and *highest* on the holdout. The decisive statistic
is the same every time: for run C, 108 candidates with mean 72.89 and sd 0.68 give an expected
maximum from pure noise of **74.96**, against an observed best of **74.53**. The search did worse
than chance applied to its own candidate population.

**What GEPA actually learned, and why it did not transfer.** Every candidate trades yes/no gain
for open-ended loss, monotonically (cand 88: +0.74 / −1.42; answer length is flat, so this is not
verbosity). Across 491 proposals, 84.9% add yes/no or existence rules while only 45.8% retain
naming guidance — and naming is what open-ended questions need. The cause is a defect in the
harness's own sampler: it round-robined the strata, so **val was 58.6% yes/no against a 43.4% pool
rate** while the holdout matched the pool, and `who` questions were 11.7% of train against 0.4% of
the holdout. Trading open-ended accuracy into yes/no accuracy paid on val and was charged for on a
representative test set. The sampler now draws proportionally (all splits within 0.2 points of the
pool).

**How much this actually tested the lever is limited by four more harness defects**, all found by
auditing the traces afterwards and all now fixed: the shared server's reasoning budget decapitated
the *reflection* model (90% of calls cut mid-thought, producing 5.1% scratchpad proposals, 8
candidates descended from one); a dead template placeholder meant the proposer was never given a
length budget and 78% of proposals had their tail — where the new rule goes — silently trimmed;
the judge's evidence line re-introduced the gold answer into 46.8% of reflection feedback; and
`R7_overlay_colour` never fired once in 1,156 judge calls. Runs A–C therefore tested a search that
was misdirected in several ways at once, which is the honest caveat on reading them as evidence
about prompt optimization in general.

**What is unchanged**: the marked-vs-raw deficit. The seed sits at −7.8 against raw on val, and no
candidate moved it — the optimizer's cheapest move is to converge on the unannotated answers
(+3.7 points of raw-agreement on the "winning" candidate), which is the null hypothesis.
