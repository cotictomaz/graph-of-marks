# Why GoM preprocessing degrades Qwen2.5-VL on the spatial VQA subset

**Date:** 2026-07-19
**Model:** `Qwen/Qwen2.5-VL-7B-Instruct` (bf16, vLLM, `max_model_len: 24576`)
**Finding:** the GoM-preprocessed condition scores **33.48%** vs **69.48%** for the untouched
source images — a **−36.0 point** regression. The dominant causes are (1) the model copying
answers out of the annotation layer instead of reading the image, and (2) masks/label chips
physically occluding the pixels the question is about.

---

## 1. The two runs being compared

| | preprocessed (GoM) | raw baseline |
|---|---|---|
| config | `slurm_configs/vlm_comparison_preprocess_only.yaml` | `slurm_configs/vlm_comparison_spatial_raw.yaml` |
| `base_dir` | `ablation_studies/vlm_comparison_spatial` | `ablation_studies/vlm_comparison_spatial_raw` |
| `inference_image` | `preprocessed` (GoM render) | `raw` (source `.jpg`) |
| `include_scene_graph` | `true` (triples in prompt) | `false` |
| dataset | `vqav1_limited_1000.json`, `dataset_filter.mode: spatial` | identical |
| subsample | `num_images: 250`, `questions_per_image: 1` | identical |
| `n_runs` | 3 | 3 |

Both runs read images from the **same** `vlm_comparison_spatial/image_cache/` directory, and the
`image_path` / `question` / `answers` fields match one-for-one across the two `raw_results.json`
files. **250/250 examples are shared**, so the example set is exactly matched.

Preprocessing settings for the GoM condition (`preprocessing_overrides`):
`aggressive_pruning: true`, `auto_scale_styles: false`, `max_relations: 10`,
`enforce_max_global: true`, `enforce_max_per_object: true`, `max_relations_per_object: 999`
(global-only cap of 10 relations).

### Determinism

Both conditions produced **byte-identical answer lists across all 3 runs**
(`std_accuracy: 0.0`). The gap is fully deterministic, not sampling noise — a single run is
sufficient for the per-item analysis below, and all per-item numbers here come from `run_1`.

---

## 2. Headline numbers

From `summary_metrics.json`:

| metric | preprocessed | raw | Δ |
|---|---|---|---|
| `mean_accuracy` (official VQA soft) | **33.48** | **69.48** | **−36.00** |
| `std_accuracy` | 0.0 | 0.0 | — |
| `mean_exact_matches` | 70 / 250 | 147 / 250 | −77 |

Reproduced per-item with `gom.ablations.evaluation` (`extract_final_answer` →
`vqa_soft_accuracy` over the 10 human answers): mean PRE **33.48%**, RAW **69.48%** — exact
match with the stored summaries, so the scoring path is verified.

**Per-item direction of change** (threshold: soft-score delta > 0.5):

| | count | share of 250 |
|---|---|---|
| regressions (raw ≫ pre) | **102** | 40.8% |
| improvements (pre ≫ raw) | **8** | 3.2% |

A 13:1 regression-to-improvement ratio. This is a systematic collapse, not a redistribution.

### Prompt-cost side evidence

Mean inference time per example:

| run | preprocessed | raw |
|---|---|---|
| 1 | 8.12 s | 0.39 s |
| 2 | 8.09 s | 0.29 s |
| 3 | 8.08 s | 0.29 s |

~28× slower. The GoM render is upscaled (e.g. a 375×640 COCO image renders at 2792×4849), so
the prompt is dominated by vision tokens — consistent with the `max_model_len: 24576` the config
had to be sized to.

---

## 3. Where the loss is concentrated

Question types assigned by prefix/keyword heuristic over the 250 shared items:

| question type | n | PRE | RAW | Δ |
|---|---|---|---|---|
| OCR / "what does X say", "number on" | 6 | 16.7% | 96.7% | **−80.0** |
| color ("what color…") | 14 | 21.4% | 78.6% | **−57.1** |
| counting ("how many…") | 9 | 36.7% | 77.8% | −41.1 |
| "where …" | 70 | 15.7% | 54.3% | −38.6 |
| other (what / which / why) | 104 | 30.6% | 68.7% | −38.1 |
| yes/no | 47 | 71.5% | 86.2% | −14.7 |

The categories that depend on **reading raw pixels** — text, color, counting — are hit hardest.
That is the signature of occlusion, not of degraded reasoning. Yes/no, which needs the least
pixel fidelity, degrades least.

---

## 4. Failure mode 1 — answers copied from the annotation layer

**56 of the 102 regressions (55%)** have a PRE answer that is copied verbatim out of the overlay:
a mark id (`window_1`), a detector class name (`parking meter`), or a relation predicate
(`InFrontOf`).

Cross-referencing each regressed answer against the node labels in that image's
`*_graph.json` (mark-index suffix stripped):

| | count | share of 102 regressions |
|---|---|---|
| PRE answer **is** a detected-object label in the scene graph | 35 | **34%** |
| gold answer is a detected-object label | 3 | **3%** |

The model's answer distribution has migrated onto the detector's vocabulary, which is a
vocabulary that mostly does **not** contain the right answers.

Answer-shape counts over all 250 PRE answers: 27 contain a literal `_N` mark id (26 of them
end in one; the 27th is `Above person_1 and right of bowl_1`), 32 are bare relation predicates,
15 are abstentions.

```
Q: "What is on the other side of the train?"        gold=trees        PRE=window_1           RAW=Trees
Q: "What is on the ground near the standing giraffe?" gold=rocks      PRE=stone_4            RAW=Rocks
Q: "Where is the bus parked?"                       gold=parking lot  PRE=Parking meter_1    RAW=Parking lot
Q: "What is beside the chair?"                      gold=table        PRE=chair_2            RAW=Table
Q: "What is to the right of the screen?"            gold=doughnut     PRE=sandwich_1         RAW=Donut
Q: "What is the name of the middle tool?"           gold=file         PRE=knife_1            RAW=Crochet hook
Q: "If the power went out, where would you still be able to write notes?"
                                                    gold=address book PRE=table_1            RAW=Notebook
```

And the same effect without the suffix — the model inherits the detector's **errors** rather
than correcting them from the image:

```
Q: "What's behind the horse?"                       gold=carriage   PRE=Parking meter   RAW=Carriage
Q: "What animal is in the back of the truck?"       gold=elephant   PRE=cow             RAW=Elephant
Q: "Where is he?"                                   gold=ocean      PRE=Surfboard       RAW=Ocean
Q: "Where is this?"                                 gold=market     PRE=Plant           RAW=Market
Q: "Where is location?"                             gold=restaurant PRE=Dining table    RAW=Restaurant
Q: "What kind of vehicle is in front of the building?" gold=motorcycle PRE=Bicycle      RAW=Motorcycle
Q: "What caused the tracks at the bottom of the image?" gold=skis   PRE=Person          RAW=Skis
Q: "What is in between the bikes?"                  gold=bird       PRE=Person          RAW=Bird
```

Raw Qwen2.5-VL answers all of these correctly. The marks are actively overriding a correct
percept with a wrong label.

---

## 5. Failure mode 2 — "where" questions answered with relation predicates

70 items (28% of the set) are "where" questions; they lose 38.6 points. 32 PRE answers across
the set are bare graph predicates. The scene-graph triples teach the model that the expected
answer *format* is a relation token rather than a place:

```
Q: "Where is the dog?"            gold=bed      PRE=InFrontOf   RAW=On the bed
Q: "Where are the kids?"          gold=bed      PRE=InBed       RAW=Bed
Q: "Where are the sheep?"         gold=in pen   PRE=InFrontOf   RAW=In a pen.
Q: "Where is the phone sitting?"  gold=keyboard PRE=OnKeyboard  RAW=On a keyboard.
Q: "Where is the fence?"          gold=behind animal PRE=Between RAW=Behind the giraffe.
Q: "Where the stop sign is fitted?" gold=on bus PRE=Below       RAW=On a bus
Q: "Where is the skateboarders hand?" gold=ground PRE=Below     RAW=On the ground
Q: "Where is the pizza?"          gold=on plate PRE=Above person_1 and right of bowl_1  RAW=On a table
Q: "Where is the cat looking?"    gold=out window PRE=Above     RAW=Outside
```

**Important caveat:** several of these are *semantically informative but unscoreable*. `InBed`
vs gold `bed`, `OnKeyboard` vs gold `keyboard` — the content is right, the surface form scores
0 under the official VQA normalizer. So **part of the −36 is a formatting artifact of the
metric, not a capability loss**. This must be separated before the number goes in a paper.

---

## 6. Failure mode 3 — masks and label chips destroy the evidence

Rendering check on `COCO_train2014_000000033828` (question: *"What color is the ribbon on the
back of the helmet?"*, gold `pink`, PRE `Purple`, RAW `Pink`):

- Source: a 375×640 ski-slope photo; the person on the right wears a helmet with a **pink**
  ribbon.
- GoM render: that person is covered by a **magenta/purple `person_1` segmentation mask** that
  swallows the ribbon entirely. The model answers `Purple` — i.e. **it reports the mask color,
  not the object color.**

The same pattern runs through the whole color set. PRE answers cluster on the render palette
(Purple, Red ×2, Brown ×3) where gold is pink / gray / yellow / white:

```
Q: "What color is the ribbon on the back of the helmet?"  gold=pink            PRE=Purple  RAW=Pink
Q: "What color is the inside of the cup?"                 gold=white           PRE=Red     RAW=White
Q: "What colors are the bus to the right?"                gold=green and white PRE=Brown   RAW=White and green
Q: "What color jacket is the man on left wearing?"        gold=yellow and black PRE=Brown  RAW=Yellow
Q: "What color is the inside of the suitcase lid?"        gold=gray            PRE=Brown   RAW=Black
Q: "What color is the edge of the bus?"                   gold=gray            PRE=Red     RAW=Pink
Q: "What color is the bird near the water?"    gold=white and gray  PRE="The image does not provide enough detail…"  RAW=White
Q: "What color is the nearest sailboat?"       gold=white           PRE="The image does not provide information…"    RAW=White
Q: "What color is the door in the back?"       gold=red             PRE="The image does not provide…"                RAW=Red
```

**OCR degrades character-wise rather than failing outright** — exactly what partial overlap of a
label chip on a glyph produces:

```
Q: "What is the number on the front of this truck?"  gold=703  PRE=103   RAW=703
Q: "What does the top sign say?"  gold=south columbus stallions  PRE="South Columbus Stallings"  RAW="South Columbus Stallions"
Q: "What does the side of the camel say?"  gold=water aid  PRE="Can't determine"  RAW=WaterAid
Q: "What does the third banner from the bottom say?"  gold=steak in sac  PRE=FREE LEMONADE  RAW=POLISH SAUSAGE
```

**Counting collapses to the detector's count**, not the visible one:

```
Q: "How many candles are above the clock?"     gold=2  PRE=One   RAW=Two
Q: "How many dogs are in the back of the pickup truck?"  gold=2  PRE=One   RAW=Two
Q: "How many people can be seen riding inside the elephant?"  gold=2  PRE=Zero  RAW=Two
Q: "How many steps are there on the right?"    gold=2  PRE=3     RAW=Two
```

### Render-density evidence

Over the 250 preprocessed images (`*_graph.json`):

| | mean | max |
|---|---|---|
| nodes (drawn marks) per image | 9.2 | 57 |
| relations (drawn arrows) per image | 5.1 | 10 |

The relation cap (`max_relations: 10`) is being enforced correctly. But 9 masks + 9 opaque
label chips + 5 curved arrows with their own text chips, on a 375×640 photo, is enough to bury
the subject. In the inspected render the label chips (`helmet_1`, `helmet_3`, `snowboard_1`,
`person_1`, `glove_1`) each occupy a large fraction of the frame width, and several
`Behind` / `Below` chips are stacked on top of one another over the person being asked about.

**Marks are also misplaced** in that render — the `snowboard_3` mask sits on a glove, `helmet_1`
labels an empty snow patch — so the model is simultaneously losing pixels *and* being handed
confidently wrong labels.

---

## 7. Failure mode 4 — abstention and negative bias

| | preprocessed | raw |
|---|---|---|
| answers of the form "cannot determine / not visible / unknown / none" | **15** | **2** |

Yes/no answer distribution over the 47 yes/no items:

| | yes | no |
|---|---|---|
| gold | 23 | 19 |
| RAW | 16 | 26 |
| PRE | **9** | **32** |

The clutter pushes the model toward "No" and toward refusing. Concrete flips:

```
Q: "Is it bright outside?"                       gold=yes  PRE=No  RAW=Yes
Q: "Is he skating on top of a pipe?"             gold=yes  PRE=No  RAW=Yes
Q: "Is there a glass of water around?"           gold=yes  PRE=No  RAW=Yes
Q: "Could the man in the back be asleep?"        gold=yes  PRE=No  RAW=Yes
Q: "Is the woman in the middle the Mom of the two boys?"  gold=yes  PRE=No  RAW=Yes
```

---

## 8. Failure-mode breakdown of the 102 regressions

Mutually exclusive categories, assigned in priority order (mark id → relation predicate →
graph node label → abstention → other):

| category | count | share |
|---|---|---|
| **annotation-layer leakage (union of the three below)** | **56** | **55%** |
| — literal `_N` mark id | 19 | 19% |
| — bare relation predicate | 19 | 19% |
| — detector class name (no `_N` suffix, matched against that image's graph nodes) | 18 | 18% |
| other misperception (color, count, OCR, object identity) | 36 | 35% |
| abstention ("cannot determine", "not visible") | 10 | 10% |

Categories were assigned by regex plus a per-image graph-node lookup, so the boundary between
"detector class name" and "other misperception" depends on whether the wrong answer happens to
appear as a node in *that* image's graph — several §4 examples (`cow` for elephant,
`Bicycle` for motorcycle) are detector-vocabulary errors that land in "other misperception"
because the label came from a different image's detections or from a pruned node. The 55%
leakage figure is therefore a **lower bound**.

---

## 9. Interpretation

Qwen2.5-VL-7B at **69.5%** on this subset is already strong enough to ground objects in these
scenes unaided. Against that baseline the annotation layer is **strictly subtractive**: it
removes pixel evidence the model was successfully using, and it substitutes a lower-quality
symbolic description (the detector's labels and the geometric relation graph) that the model
then defers to.

The mechanism is a deference cascade:

1. The render occludes the region of interest (mask fill, label chip, arrow).
2. The prompt supplies an authoritative-looking symbolic description of that same region.
3. The model resolves the conflict in favour of the symbols — so detector errors propagate to
   the answer, and the answer *vocabulary* collapses onto `{mark_id, class_name, relation_token}`.

GoM's premise is that marks help when the base model cannot ground objects on its own. That
premise does not hold for this model on this data.

---

## 10. Confounds and limitations of this comparison

**The comparison is confounded and cannot, on its own, attribute the −36 points.** The raw
baseline changes *two* variables at once: `inference_image: raw` **and**
`include_scene_graph: false`. The evidence splits across both:

- the relation-predicate leakage (§5) points at the **scene-graph text**;
- the color / OCR / counting collapse (§6) points at the **rendered image**;
- the mark-id leakage (§4) requires **both** (the `_N` ids appear in the render *and* the triples).

**The 2×2 is the single most valuable next run.** Add the two missing cells:

| | `include_scene_graph: false` | `include_scene_graph: true` |
|---|---|---|
| `inference_image: raw` | ✅ done — 69.48% | ❌ **missing** |
| `inference_image: preprocessed` | ❌ **missing** | ✅ done — 33.48% |

Other limitations:

- **Metric formatting inflates the gap.** Answers like `InBed` (gold `bed`) are semantically
  correct and score 0. Re-scoring with the leakage categories tagged would give a cleaner
  capability number. Magnitude not yet quantified.
- **Question-type buckets are heuristic** (prefix/keyword matching), and the OCR (n=6) and
  counting (n=9) buckets are small — their per-bucket deltas are indicative, not tight.
- **Single model, single subset.** 250 spatial-filtered VQAv1 images, one question each,
  Qwen2.5-VL-7B only. Whether the effect inverts for a weaker-grounding model (the case GoM is
  designed for) is untested here.

---

## 11. Suggested mitigations, in expected-effect order

1. **Reduce occlusion (targets §6, the color/OCR/counting collapse).** Translucent mask
   *outlines* instead of filled masks; cap label-chip size relative to the image; move chips
   outside the box or into a margin legend. This is the largest single lever — the palette-color
   answers are direct proof the fill is being read as the object.
2. **Strip `_N` suffixes from drawn labels** or switch to numeric-only marks
   (`gom_numeric_labeled`), so there is no `window_1`-shaped token for the model to emit as an
   answer (targets §4, 20% of regressions).
3. **Add an explicit prompt constraint** that mark ids, detector class names, and relation
   tokens are not valid answers, and that "where" questions want a place in natural language
   (targets §4–§5).
4. **Reconsider mark density.** 9.2 marks/image on 375×640 COCO photos is high; tightening
   `aggressive_pruning` toward question-relevant objects only would cut both occlusion and
   the size of the wrong-label vocabulary.
5. **Fix upstream detection quality** — misplaced masks (`snowboard_3` on a glove) and wrong
   classes (`parking meter` for a carriage) are what make the deference cascade harmful rather
   than merely redundant.

---

## Reproducing this analysis

```bash
# summary numbers
cat ablation_studies/vlm_comparison_spatial/results/vlm_comparison/Qwen/Qwen2.5-VL-7B-Instruct/summary_metrics.json
cat ablation_studies/vlm_comparison_spatial_raw/results/vlm_comparison/Qwen/Qwen2.5-VL-7B-Instruct/summary_metrics.json
```

Per-item scoring loads `src/gom/ablations/evaluation.py` directly via `importlib` (importing the
`gom` package pulls in `matplotlib`, which is absent on the submit host):

```python
import importlib.util, json
spec = importlib.util.spec_from_file_location("ev", "src/gom/ablations/evaluation.py")
ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)

def load(d):
    return json.load(open(f"ablation_studies/{d}/results/vlm_comparison/"
                          "Qwen/Qwen2.5-VL-7B-Instruct/run_1/raw_results.json"))

P, R = load("vlm_comparison_spatial"), load("vlm_comparison_spatial_raw")
key = lambda r: (r["image_id"], r["question"])
dp, dr = {key(r): r for r in P}, {key(r): r for r in R}
for k in dp:
    gold = dp[k]["metadata"]["answers"]
    sp = ev.vqa_soft_accuracy(ev.extract_final_answer(dp[k]["generated_answer"]), gold)
    sr = ev.vqa_soft_accuracy(ev.extract_final_answer(dr[k]["generated_answer"]), gold)
    ...  # sr - sp > 0.5  =>  regression
```

Graph-label cross-reference reads
`ablation_studies/vlm_comparison_spatial/preprocessed_images/vlm_comparison/default/<stem>_graph.json`,
resolved from each result record's `inference_image_path` (`_output.jpg` → `_graph.json`).
