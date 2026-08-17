# GoM failures after the gom_v3 fixes (20 previously unshown cases)

Condition `gom_text_labeled` from the `data_v6` run - outline renders, open-vocabulary
question-driven marks, deterministic zero-overlap label placement, `gom_v2_concise`
prompt, curated eval. Model: Qwen2.5-VL-7B. Every case is correct on the clean image
and wrong on the GoM image (lenient scoring). These 20 are drawn from the 101 flips
that no earlier gallery has shown; each mechanism below was read off the render, not
inferred from the answer string.

The defects the earlier audits found are gone: no label overlaps anywhere in the run
(3,975 images x 6 variants), 3.6 marks per image, no fragment duplicates, and question
nouns outside the closed ontology are now detected. What these 20 show instead:

| mechanism | cases | fixable |
|---|---|---|
| ID leak - answers a reference tag on a clean render | 1-4 | no (prompting fails; numeric IDs avoid it) |
| queried object never outlined -> existence denied | 5-8, 11 | detector recall ceiling |
| marks point at other objects / assert a false premise | 9, 12-14 | partly: relation choice |
| **question words became marks** | **15, 20** | **yes - new defect, see below** |
| generic or wrong class on the queried object | 16, 17 | yes - label preference |
| left/right inversion with the arrow drawn | 10 | no |
| singular/plural, model error | 18, 19 | scorer / none |

**The one regression this gallery introduces is mine.** Opening the detector vocabulary
to any content noun also let adjectives and verbs through. In case 15 the question's own
answer options are drawn on the image as `peeled_1` and `unpeeled_2`; in case 20 a frisbee
is labelled `throwing_1`. Marks that spell out candidate answers are worse than no marks.
The `_META_WORDS` stoplist needs the question's answer-bearing adjectives and -ing verbs,
and a re-run is required before this gallery is superseded.

## 1. `2347957` (GQA) - ID leak

**Question:** Who is wearing a jacket?
**Gold:** woman, The woman is wearing a jacket.

| condition | model output |
|---|---|
| raw (clean image) | **woman** correct |
| GoM (outline + IDs + arrows) | **person_1** wrong |

*Mechanism:* clean 3-mark render; the model answers the tag `person_1` instead of naming the woman

| original | GoM render (what the model saw) |
|---|---|
| ![o](flip_examples_paper_gom/v4_01_2347957_original.jpg) | ![g](flip_examples_paper_gom/v4_01_2347957_gom.jpg) |

## 2. `2348675` (GQA) - ID leak

**Question:** Who is holding the racket in the center?
**Gold:** girl, The girl is holding the racket.

| condition | model output |
|---|---|
| raw (clean image) | **girl** correct |
| GoM (outline + IDs + arrows) | **person_1** wrong |

*Mechanism:* clean render (girl + two racket marks); answers `person_1` instead of `girl`

| original | GoM render (what the model saw) |
|---|---|
| ![o](flip_examples_paper_gom/v4_02_2348675_original.jpg) | ![g](flip_examples_paper_gom/v4_02_2348675_gom.jpg) |

## 3. `2380524` (GQA) - ID leak

**Question:** Who is wearing a tee shirt?
**Gold:** man, The man is wearing a tee shirt.

| condition | model output |
|---|---|
| raw (clean image) | **Man** correct |
| GoM (outline + IDs + arrows) | **person_2** wrong |

*Mechanism:* answers `person_2` instead of `man`

| original | GoM render (what the model saw) |
|---|---|
| ![o](flip_examples_paper_gom/v4_03_2380524_original.jpg) | ![g](flip_examples_paper_gom/v4_03_2380524_gom.jpg) |

## 4. `2408238` (GQA) - ID leak

**Question:** Who is wearing a watch?
**Gold:** boy, The boy is wearing a watch.

| condition | model output |
|---|---|
| raw (clean image) | **Boy** correct |
| GoM (outline + IDs + arrows) | **person_1** wrong |

*Mechanism:* answers `person_1` instead of `boy`

| original | GoM render (what the model saw) |
|---|---|
| ![o](flip_examples_paper_gom/v4_04_2408238_original.jpg) | ![g](flip_examples_paper_gom/v4_04_2408238_gom.jpg) |

## 5. `2341832` (GQA) - queried object unmarked

**Question:** Do you see benches to the right of the bottle that is not open?
**Gold:** yes, Yes, there is a bench to the right of the bottle.

| condition | model output |
|---|---|
| raw (clean image) | **yes** correct |
| GoM (outline + IDs + arrows) | **no** wrong |

*Mechanism:* no bench is outlined - only the cat and the bottle are marked - and the model reads the absence of a mark as absence of the object

| original | GoM render (what the model saw) |
|---|---|
| ![o](flip_examples_paper_gom/v4_05_2341832_original.jpg) | ![g](flip_examples_paper_gom/v4_05_2341832_gom.jpg) |

## 6. `2315568` (GQA) - queried object unmarked

**Question:** Do you see a ladle next to the computer that is sitting on the floor?
**Gold:** yes, Yes, there is a ladle next to the computer.

| condition | model output |
|---|---|
| raw (clean image) | **Yes** correct |
| GoM (outline + IDs + arrows) | **No** wrong |

*Mechanism:* no ladle is outlined; the model answers no

| original | GoM render (what the model saw) |
|---|---|
| ![o](flip_examples_paper_gom/v4_06_2315568_original.jpg) | ![g](flip_examples_paper_gom/v4_06_2315568_gom.jpg) |

## 7. `2389557` (GQA) - queried object unmarked

**Question:** Is the smiling person above a bench?
**Gold:** yes, Yes, the girl is above a bench.

| condition | model output |
|---|---|
| raw (clean image) | **Yes** correct |
| GoM (outline + IDs + arrows) | **No** wrong |

*Mechanism:* the bench is not outlined; the marked person alone does not support the relation

| original | GoM render (what the model saw) |
|---|---|
| ![o](flip_examples_paper_gom/v4_07_2389557_original.jpg) | ![g](flip_examples_paper_gom/v4_07_2389557_gom.jpg) |

## 8. `2370503` (GQA) - relation names the wrong pair

**Question:** Is the bicycle behind the tree in the field?
**Gold:** yes, Yes, the bicycle is behind the tree.

| condition | model output |
|---|---|
| raw (clean image) | **Yes** correct |
| GoM (outline + IDs + arrows) | **No** wrong |

*Mechanism:* the tree is unmarked, so the only `Behind` arrow runs bicycle -> field_1; the graph asserts a relation the question did not ask about

| original | GoM render (what the model saw) |
|---|---|
| ![o](flip_examples_paper_gom/v4_08_2370503_original.jpg) | ![g](flip_examples_paper_gom/v4_08_2370503_gom.jpg) |

## 9. `2403371` (GQA) - marks point at other objects

**Question:** Is the truck to the left or to the right of the bench on the right?
**Gold:** left, The truck is to the left of the bench.

| condition | model output |
|---|---|
| raw (clean image) | **left** correct |
| GoM (outline + IDs + arrows) | **right** wrong |

*Mechanism:* neither the truck nor the bench is outlined - the two marks are distant cars - so the arrow describes an unrelated pair

| original | GoM render (what the model saw) |
|---|---|
| ![o](flip_examples_paper_gom/v4_09_2403371_original.jpg) | ![g](flip_examples_paper_gom/v4_09_2403371_gom.jpg) |

## 10. `2361897` (GQA) - left/right inversion

**Question:** Is the ball to the left or to the right of the man that is wearing socks?
**Gold:** right, The ball is to the right of the man.

| condition | model output |
|---|---|
| raw (clean image) | **right** correct |
| GoM (outline + IDs + arrows) | **left** wrong |

*Mechanism:* both objects are marked and the arrow is drawn; the model still inverts the direction

| original | GoM render (what the model saw) |
|---|---|
| ![o](flip_examples_paper_gom/v4_10_2361897_original.jpg) | ![g](flip_examples_paper_gom/v4_10_2361897_gom.jpg) |

## 11. `2367686` (GQA) - queried object unmarked

**Question:** Is the speaker to the right or to the left of the man?
**Gold:** left, The speaker is to the left of the man.

| condition | model output |
|---|---|
| raw (clean image) | **left** correct |
| GoM (outline + IDs + arrows) | **right** wrong |

*Mechanism:* the speaker is not outlined; only two people and a tie are marked

| original | GoM render (what the model saw) |
|---|---|
| ![o](flip_examples_paper_gom/v4_11_2367686_original.jpg) | ![g](flip_examples_paper_gom/v4_11_2367686_gom.jpg) |

## 12. `2315716` (GQA) - marks supply a false premise

**Question:** Do you see people to the left of the tall palm trees?
**Gold:** no, No, the people are to the right of the palms.

| condition | model output |
|---|---|
| raw (clean image) | **no** correct |
| GoM (outline + IDs + arrows) | **Yes** wrong |

*Mechanism:* four person marks sit left of the palm-tree marks, so the layout of the labels suggests the very arrangement the gold answer denies

| original | GoM render (what the model saw) |
|---|---|
| ![o](flip_examples_paper_gom/v4_12_2315716_original.jpg) | ![g](flip_examples_paper_gom/v4_12_2315716_gom.jpg) |

## 13. `2376059` (GQA) - marks supply a false premise

**Question:** Are there bottles to the right of the doll?
**Gold:** no, No, there is a cup to the right of the doll.

| condition | model output |
|---|---|
| raw (clean image) | **no** correct |
| GoM (outline + IDs + arrows) | **Yes** wrong |

*Mechanism:* the marked bottle/doll pair reads as the asked-about arrangement

| original | GoM render (what the model saw) |
|---|---|
| ![o](flip_examples_paper_gom/v4_13_2376059_original.jpg) | ![g](flip_examples_paper_gom/v4_13_2376059_gom.jpg) |

## 14. `2378259` (GQA) - marks supply a false premise

**Question:** Is the water bottle to the right of a refrigerator?
**Gold:** no, No, the water bottle is to the right of the dish washer.

| condition | model output |
|---|---|
| raw (clean image) | **no** correct |
| GoM (outline + IDs + arrows) | **Yes** wrong |

*Mechanism:* dense scene; the marked pair reads as the asked-about arrangement

| original | GoM render (what the model saw) |
|---|---|
| ![o](flip_examples_paper_gom/v4_14_2378259_original.jpg) | ![g](flip_examples_paper_gom/v4_14_2378259_gom.jpg) |

## 15. `2346557` (GQA) - question words became marks

**Question:** Are the oranges to the right of the other oranges unpeeled or peeled?
**Gold:** peeled, The oranges are peeled.

| condition | model output |
|---|---|
| raw (clean image) | **peeled** correct |
| GoM (outline + IDs + arrows) | **unpeeled** wrong |

*Mechanism:* the adjectives from the question are open-vocabulary queries, so the render literally contains `peeled_1`, `peeled_2`, `unpeeled_1` and `unpeeled_2` - the two answer options drawn as labels on arbitrary fruit - and the model picks the wrong one

| original | GoM render (what the model saw) |
|---|---|
| ![o](flip_examples_paper_gom/v4_15_2346557_original.jpg) | ![g](flip_examples_paper_gom/v4_15_2346557_gom.jpg) |

## 16. `2372647` (GQA) - generic label on the queried object

**Question:** What is the vegetable that is to the left of the sponge?
**Gold:** lettuce, The vegetable is lettuce.

| condition | model output |
|---|---|
| raw (clean image) | **lettuce** correct |
| GoM (outline + IDs + arrows) | **carrot** wrong |

*Mechanism:* the lettuce carries the generic open-vocabulary label `vegetable_1` while the neighbouring carrots carry specific ones, so the model answers `carrot`

| original | GoM render (what the model saw) |
|---|---|
| ![o](flip_examples_paper_gom/v4_16_2372647_original.jpg) | ![g](flip_examples_paper_gom/v4_16_2372647_gom.jpg) |

## 17. `713865` (GQA) - mark's class is wrong

**Question:** What vehicle is to the left of the vehicle on the sidewalk?
**Gold:** van, The vehicle is a van.

| condition | model output |
|---|---|
| raw (clean image) | **van** correct |
| GoM (outline + IDs + arrows) | **bus** wrong |

*Mechanism:* the van in the garage is outlined and labelled `bus_2`; the model reports the label, not the vehicle

| original | GoM render (what the model saw) |
|---|---|
| ![o](flip_examples_paper_gom/v4_17_713865_original.jpg) | ![g](flip_examples_paper_gom/v4_17_713865_gom.jpg) |

## 18. `2319253` (GQA) - singular/plural

**Question:** What type of fruit is to the right of the food on the left side?
**Gold:** strawberries, The fruits are strawberries.

| condition | model output |
|---|---|
| raw (clean image) | **Strawberries** correct |
| GoM (outline + IDs + arrows) | **strawberry** wrong |

*Mechanism:* answers `strawberry` where the gold list has only `strawberries`; the mark labels are singular

| original | GoM render (what the model saw) |
|---|---|
| ![o](flip_examples_paper_gom/v4_18_2319253_original.jpg) | ![g](flip_examples_paper_gom/v4_18_2319253_gom.jpg) |

## 19. `2378694` (GQA) - model error on a clean render

**Question:** What is the man wearing?
**Gold:** shorts, The man is wearing shorts.

| condition | model output |
|---|---|
| raw (clean image) | **shorts** correct |
| GoM (outline + IDs + arrows) | **wetsuit** wrong |

*Mechanism:* person and surfboard are correctly outlined and nothing is mislabelled; the model simply describes the wetsuit instead of the shorts

| original | GoM render (what the model saw) |
|---|---|
| ![o](flip_examples_paper_gom/v4_19_2378694_original.jpg) | ![g](flip_examples_paper_gom/v4_19_2378694_gom.jpg) |

## 20. `2380400` (GQA) - question words became marks

**Question:** What is the person in front of the trees throwing?
**Gold:** frisbee, The man is throwing the frisbee.

| condition | model output |
|---|---|
| raw (clean image) | **frisbee** correct |
| GoM (outline + IDs + arrows) | **ball** wrong |

*Mechanism:* the verb `throwing` became an open-vocabulary query, so the frisbee is outlined as `throwing_1`; with no usable class name the model answers `ball`

| original | GoM render (what the model saw) |
|---|---|
| ![o](flip_examples_paper_gom/v4_20_2380400_original.jpg) | ![g](flip_examples_paper_gom/v4_20_2380400_gom.jpg) |

---

**Scale.** 125 flips in 996 GQA rows, of which these 20 and the 24 shown in earlier
galleries are a sample; the same renders also produce 100+ rescues, so the net per-model
effect is what `RESULTS.md` section gom_v3 reports, not this gallery. ID leakage remains
the largest single mechanism on text-tag renders (25 of 125) and is 20x rarer on the
numeric-ID conditions (4-5 vs 87-90 across the full run); both are evaluated.
