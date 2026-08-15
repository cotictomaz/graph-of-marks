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
