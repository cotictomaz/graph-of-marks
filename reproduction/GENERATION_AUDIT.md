# Generation-level audit of the corrected run (`data_v2`, 2026-08-15)

Question-by-question analysis of the actual generations and rendered images, answering:
was anything truncated, were the decoding parameters wrong, are dataset instances bad, and
what actually causes the residual marked-condition deficits. Produced by
`reproduction/audit_generations.py` (counts reproducible by rerunning it) plus manual
inspection of renders cited by stem below. All numbers are `supplementary_concise` on
`data_v2` unless stated.

## 1. Truncation — ruled out

`max_tokens=512` is never approached. Word-length p90 per generation: Gemma/Qwen ≤ 2 words
in every condition; LlamaV marked p90 = 33 words, maximum 100 words, only 3/1000 at ≥ 80
words. Zero empty generations anywhere. Nothing is cut off mid-answer.

## 2. Decoding parameters — one real discrepancy, tested

The single setting used everywhere (seed 0, temperature 0.2, top_p 0.9) is **not in the
paper's published grid** (seeds {42,123,456} × temperatures {0.1,0.3,0.5} × top_p
{0.7,0.9,0.95}); it is an invented midpoint. `max_tokens` matches the spec (512).
Experiment (a) below reruns Qwen/VQAv2 at a true grid point (seed 42, temp 0.1, top_p 0.9)
— see §6 for the outcome.

## 3. LlamaV-o1: the deficit is a chat-protocol artifact, not vision

**37–54% of every marked condition's generations are answer-less "plan" statements** —
"I will analyze the image to determine…" and then the model ends its turn (~20 words, far
below the token cap). On raw the rate is 0.2–1.7%.

| dataset | plan-only in raw | plan-only in marked conditions |
|---|---:|---:|
| GQA | 2/1000 | 363–421/1000 |
| VQAv1 | 3/1000 | 395–538/1000 |
| VQAv2 | 17/1000 | 398–536/1000 |

LlamaV-o1 is trained for staged curriculum reasoning; marked images flip it into its
plan-first mode and it emits only the first stage. These rows score 0 automatically — no
scorer can rescue an answer that is not there: lenient phrase-containment recovers only
+3.2 to +8.1 points (e.g. VQAv2 segmented 35.61 → 41.65, still vs raw 76.37). **LlamaV's
−32..−40 is therefore mostly a harness/model-protocol incompatibility and should not be
read as a visual-marking result at all.** Fixing it would need stepwise prompting or a
continuation turn, which is outside the paper's declared protocol.

## 4. Qwen: filled masks destroy the very objects Algorithm 3 selects

Flip taxonomy (raw-right → segmented-wrong, questions that survive the appearance filter):

| dataset | flips | kept | color_shift | label_leak | count=marks | yes→no | other |
|---|---:|---:|---:|---:|---:|---:|---:|
| GQA | 150 | 71 | 1 | 6 | 0 | 19 | 45 |
| VQAv1 | 98 | 56 | 1 | 2 | 0 | 15 | 38 |
| VQAv2 | 93 | 58 | 1 | 2 | 0 | 10 | 45 |

The dominant identifiable category is **yes→no existence/state denial** (~17–27% of kept
flips). Visual inspection of the rendered images shows the mechanism, and it is not
"unmarked = absent" (count == number-of-marks never fires); it is **"marked =
unrecognizable"**:

- `COCO_train2014_000000415089` — "How many potted plants?" gold 3 → **"0"**. Both plants
  are correctly selected by Algorithm 3 and masked solid blue; they no longer read as
  plants.
- `COCO_train2014_000000000049` — "Is there a flower arrangement?" gold yes → **"no"**. The
  arrangement is masked green-yellow; the flower colors that identify it are gone.
- `2382699` (GQA) — "Which kind of animal is running?" gold zebra → **"dog"**. The zebra's
  stripes are under a blue fill; the green-masked dog stays recognizable.
- `COCO_train2014_000000436929` — "What is floating near the bird?" gold ice → **"bird"**.
  All three birds masked; attention is pulled to the marked objects.

This is the self-defeating core of the method for VQA: **Algorithm 3 marks precisely the
query-relevant objects, and the fill then corrupts precisely the evidence the question
needs.** Same-class objects also share one palette color (`person_1`/`person_2` →
same color), so the queried class becomes a set of uniform blobs. The remaining "other"
flips are of the same character (2→1 tusks, knife→spatula, oranges→apples) plus a small
tail of metric brittleness ("ski poles" vs "ski pole").

## 5. Gemma's "gains" are mostly answer-format calibration

Gemma's raw-wrong → marked-right wins are dominated by style changes, not perception:
"Light brown"→"Brown", "Likely"→"Yes", "Cannot tell"→"No", "Possibly"→"Yes" — the raw
answer was semantically right but failed exact/consensus matching; the scene-graph prompt
pushes Gemma into canonical short answers. Rescoring kept questions with lenient
phrase-containment:

| dataset | Δ(seg−raw) strict | Δ(seg−raw) lenient |
|---|---:|---:|
| GQA | +1.99 | **−0.92** |
| VQAv1 | +6.66 | +5.11 |
| VQAv2 | +4.92 | +1.51 |

The GQA gain vanishes entirely under lenient scoring; ~70% of the VQAv2 gain is format.
Only VQAv1 retains a substantial genuine effect. **Claims of Gemma gains should be
qualified accordingly.**

## 6. Dataset instances and the appearance filter

The filter is tight: among kept questions, only 0/654 (GQA), 4/832 (VQAv1), 4/789 (VQAv2)
have a color-word gold answer; kept-side color_shift flips are 1–2 per dataset. The
remaining deficit is **not** residual appearance questions — it is the §4 mechanism, which
no instance filter can remove because it affects ordinary object/existence/count questions
whenever the queried object is masked.

## 7. GPU experiments (Qwen, VQAv2, corrected renders)

**(b) "Marks are optional aids" prompt (`visual_aid_concise`) — no effect.** Adding "The
colored outlines, object labels, and arrows are optional visual aids. If a mark conflicts
with visible image content, rely on the image." to every marked condition changes official
accuracy by −0.63 to +0.49 points (kept-only: −0.58 to +0.49) — noise. The model cannot
"rely on the image" because the fill has removed the image evidence; the deficit is not a
prompt-trust problem.

| condition | suppl. | visual_aid | Δ |
|---|---:|---:|---:|
| segmented | 75.23 | 75.29 | +0.06 |
| som_numeric | 71.53 | 72.02 | +0.49 |
| gom_text | 73.03 | 72.40 | −0.63 |
| gom_numeric | 71.11 | 70.63 | −0.48 |
| gom_text_labeled | 70.76 | 70.73 | −0.03 |
| gom_numeric_labeled | 68.75 | 68.81 | +0.06 |

**(a) Paper-grid decode setting (seed 42, temp 0.1, top_p 0.9) — no effect.** Every
condition moves by ≤ 0.70 points (raw 86.04 → 86.24, segmented 75.23 → 75.36, worst case
som_numeric +0.70). The off-grid single setting used in the main runs is not a confound.

| condition | seed0/t0.2/p0.9 | seed42/t0.1/p0.9 | Δ |
|---|---:|---:|---:|
| raw | 86.04 | 86.24 | +0.20 |
| segmented | 75.23 | 75.36 | +0.13 |
| som_numeric | 71.53 | 72.23 | +0.70 |
| gom_text | 73.03 | 72.99 | −0.04 |
| gom_numeric | 71.11 | 70.85 | −0.26 |
| gom_text_labeled | 70.76 | 71.23 | +0.47 |
| gom_numeric_labeled | 68.75 | 69.18 | +0.43 |

## 8. Verdict after the corrected run — ranked causes of the residual deficits

1. **LlamaV −32..−40: protocol artifact** (37–54% answer-less plan statements). Not a
   visual result.
2. **Qwen −7..−9 (filtered): evidence destruction of the queried objects** — Algorithm 3 +
   filled masks corrupt exactly what the question needs; existence denial and identity
   swaps follow. Not truncation, not decoding, not dataset instances.
3. **Gemma +1..+6.7: majority answer-format calibration**, genuine only on VQAv1 under
   lenient scoring.

The implication for the method: with `fill_segmentation` enabled (locked in the paper
profile), marks and VQA evidence are in direct conflict. The repo's own non-paper default
(`preprocessor.py`: "Outline-only preserves image evidence for VQA") is the built-in fix;
a paper-faithful reproduction cannot use it.

## 9. Best-config run (`data_v3`, 2026-08-15): every fix applied

Configuration chosen by evidence and a 120-image pilot: `paper_aaai26_outline` render
profile (outline beat a true α=0.10 fill in all six marked conditions) + `direct_concise`
prompt (final-answer-only on every condition) + filter v2 (appearance ∪ subjective).
Outcomes:

- **LlamaV's protocol artifact is cured**: plan-only generations fell from 37–54% to ≤0.1%,
  and its marked deficit collapsed from −32..−40 to −4.4..−6.2. It was never a visual
  result.
- **Gemma's gains were prompt-format effects**: with raw receiving the same direct-answer
  instruction, its raw scores jumped (VQAv1 60.5 → 68.0) and the marked advantage fell to
  −1.6..+1.2 ≈ parity. Bidirectionally, GoM rescues ≈ as many instances as it breaks for
  Gemma (e.g. VQAv1 70 vs 63).
- **Qwen/LlamaV keep a 2–4× hurt/help instance ratio** (Qwen VQAv1: 61 broken vs 14
  rescued kept-questions) even with outline marks.

Three-run convergence (Δ best marked − raw, appearance∪subjective filter applied uniformly):

| model | recorded | corrected | best-config |
|---|---|---|---|
| gemma3_4b (GQA/V1/V2) | +2.9/+3.9/+2.8 | +3.2/+6.2/+4.7 | −0.2/+1.2/+0.8 |
| qwen25_vl_7b | −8.6/−12.5/−12.6 | −7.1/−8.3/−9.1 | −2.6/−4.9/−4.5 |
| llamav_o1_11b | −35.2/−35.8/−36.0 | −33.6/−33.4/−33.6 | −3.4/−6.0/−7.1 |

**Residual mechanisms, from the outline-render flip images**: (a) contour clutter from
low-quality background masks — in `COCO_train2014_000000000081` the *sky* mask's sprawling
green outline makes the model see extra colors; (b) contours crossing the fine details the
question asks about — in `COCO_train2014_000000262204` the elephant's outline runs through
the tusk region and the count drops 2→1. (a) suggests one honest remaining knob
(drop stuff-classes / huge-area masks); (b) is intrinsic to drawing on pixels.

## 10. The zero-occlusion test and the closing assessment

`text_graph` condition (new `--extra-conditions text_graph`): the untouched original image
plus the textual triples in the prompt — the graph's information with zero pixels changed.

| model | GQA | VQAv1 | VQAv2 | (lenient) |
|---|---:|---:|---:|---|
| gemma3_4b | −1.00 | +0.53 | −0.23 | ≈0 everywhere |
| qwen25_vl_7b | −5.80 | −4.36 | −3.72 | −3.5..−5.3 |
| llamav_o1_11b | −7.50 | −13.21 | −13.61 | −4.1..−5.5 |

**Even with no occlusion at all, the graph's content does not help** — for the stronger
models it actively hurts: the detector-derived triples are coarser and noisier than what
the models extract from clean pixels, and the "use the graph" instruction makes them defer
to the worse source.

Evidence stack against any recoverable across-model gain, each item independent:
1. Marked-vs-raw negative under every render/prompt configuration tested (five runs).
2. Zero-occlusion textual graph ≈ neutral (Gemma) or harmful (Qwen/LlamaV).
3. No principled subset wins: spatial/relational questions — the paper's core claim — are
   the WORST category for every model (Qwen −4.8..−12.6); graph-covered instances are
   hurt more, not less.
4. Oracle ceilings (outcome-based deletion of every failing instance — diagnostic only,
   not a valid protocol): Qwen still lands at −0.9..+3.8; only Gemma's ceiling (+7.5..+9.5)
   can reach paper-sized gains, and only by construction.
5. The paper's own released VQAv2 artifacts score raw 63.25 vs best-GoM 53.77 under
   standard VQA scoring.

Where marks genuinely, reproducibly help: **RefCOCOg-style grounding** (raw 0 → 34–45;
best-config improved all three models further), and Gemma-sized models reach parity on
VQA. That is the defensible claim this pipeline supports.

Visual case files: `FLIP_EXAMPLES.md` (segmented condition) and
`FLIP_EXAMPLES_FULL_GOM.md` (full GoM — IDs + arrows + relation labels — on
spatial/relational questions, with the condition ladder and break/rescue counts).

## 12. Clutter: quantified, causally tested, and bounded (2026-08-15)

Visual inspection identified overlapping tags/arrows/labels as the failure mode of the
full-GoM conditions. Both halves tested:

**Dose-response (existing data_v3 predictions, `find_winning_config.py --clutter`).**
Binning kept questions by the recorded `rendered_edge_count`, Qwen's Δ(gom_text_labeled −
raw) degrades from −2.5..−4.2 at zero rendered relations to **−12..−15 at ≥10** (Gemma
VQAv2: −3.0 → −9.4..−10.1). Clutter is the dominant driver of the *labeled* conditions'
extra damage. At high edge counts the loss is largely shared with the IDs-only condition —
total annotation mass, not arrows specifically.

**Intervention (`paper_aaai26_declutter`: outline + mask-quality/stuff filter +
question-only relation rendering + per-head relation cap).** The worst case (`4928`,
~15 stacked labels) reduces to 3 objects / 3 arrows / 3 labels, fully readable
(render pair in `flip_examples_full_gom/`). 120-image GQA pilot: labeled conditions
+2.5/+3.3. Full 1,000-image GQA run, all three models: labeled conditions improve
consistently but modestly (+0.4..+1.8; pilot optimism was sample noise), and
Δ(best marked − raw) barely moves — Gemma −1.6 → −0.5, LlamaV −4.4 → −4.0,
**Qwen −4.9 → −4.9** — because every model's best marked condition is `segmented`,
which carries almost no annotations and was never clutter-limited. Note: under the paper
relation policy `max_relations_per_object` caps per relation *head*, so several edges per
object survive; deeper decluttering would need render-path changes.

**Bound.** Clutter explains why full GoM is the worst-performing mark family and its
removal recovers ~1 point at scale; it does not explain the residual −4..−5 gap between
the most minimal mark and raw, which remains the §8/§10 evidence-destruction and
information-content story.

## 11. Exhaustion: subsample-rule search and final config candidates (2026-08-15)

Final instruction was to find the subsample and configuration under which the whole GoM
pipeline improves as in the paper, and not stop until verified. Verification was defined
before running: an a-priori selection rule (question/image/graph features only) derived on
VQAv1 must transfer with Δ>0 for all three models to held-out GQA and VQAv2.

**Subsample axis — exhausted negative.** `find_winning_config.py --rule-search` enumerated
230 rules (all single-feature buckets and all cross-feature 2-way conjunctions over:
question type, relational form, question length, graph nodes/edges, queried-class
multiplicity, coverage, tiny-objects). **Zero rules met the criterion even on the training
dataset**, so held-out verification never engaged. The outcome-selected "GoM wins" sets
(exported to `data_v3/showcase_rescues.*.json`, diagnostic only) cover 34/2242 kept
instances for Qwen (1.5%), 68 for LlamaV, 164 for Gemma — too small and feature-incoherent
to support any selection rule.

**Config axis — exhausted at parity.** Two final evidence-backed variants piloted (120
VQAv2 images, Qwen, direct_concise, vs the plain-outline baseline):

| pilot config | best marked | Δ vs raw (84.58) |
|---|---:|---:|
| outline (baseline) | 82.58 | −2.00 |
| **outline_clean** (mask-quality filter on → stuff masks dropped; sky-contour case fixed, verified visually) | **83.67** | **−0.92** |
| outline_thin (contours 1.8 → 0.8) | 82.50 | −2.08 |

`outline_clean` is the best marked-condition result recorded anywhere in this
investigation — a single clean outline on query-relevant foreground objects — and it is
still below raw (inside noise of parity at n=120; promotion gate of +1.5 over baseline not
met; no full run spent).

**Closing statement.** Across three full runs, six render configurations, four prompt
profiles, decode variations, textual-only delivery, 230 subsample rules, and
outcome-selected ceilings, the verified finding is: **the GoM pipeline's across-model VQA
improvements reported in the paper do not exist in this data under any configuration or
defensible instance selection.** The monotone trend of the entire investigation — every
fix moved marked conditions toward raw, never past it — is itself the strongest evidence:
the best a mark can do on VQA is not be noticed. GoM's real, verified contributions are
referring-expression grounding (raw 0 → 34–45) and VQA parity for small models.
