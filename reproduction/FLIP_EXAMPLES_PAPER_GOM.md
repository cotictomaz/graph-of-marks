# GoM failures after the gom_v5 fixes (`data_v8`, full curated run)

Regenerated from the **full gom_v5 run** — 3,975 curated images, all render gates pass,
`audit_relations.py` 0 hard errors. Qwen2.5-VL-7B / `gom_text_labeled` / `gom_v2_concise`,
the same axis as the previous two galleries. Produced by `reproduction/make_flip_gallery.py`.

## What is fixed, so do not go looking for it

| defect | status |
|---|---|
| arrows too short to show direction (52.9% of arrows in gom_v4) | **0.15% of renders**, shaft median 16px → 190px |
| mask outlines drawn as scribbles | fixed at source: `RETR_EXTERNAL` + area floor, so hole boundaries cannot be stroked |
| arrowheads hidden under label boxes | 0, run-wide |
| relation labels floating off their arc | 0.01% |
| relations between coincident centroids (undrawable) | 94 → 0 |
| `guy`/`lady` marks on who-images | 26 → 0 |
| generic `person_N` answers | 0.64% (gom_v3 was 2.25%) |

## What to look for instead

`other` is 52% of flips — that is where the heuristic classifier gives up and your judgement
is worth more than another regex. `absence_denied` (65) and `false_premise_asserted` (41) are
the existence-oracle mechanisms: the model treats an absent mark as "not there" and a present
mark as proof. `mark_label_copied` (37) is a mark whose class is simply wrong.

**The mechanism label on each case is a heuristic guess, not a verdict.** Several of the gom_v3
gallery's stated mechanisms turned out wrong when checked against the artifacts.

**Mechanism census over all 349 flips** (heuristic labels, to be confirmed against each render):

| mechanism | count |
|---|---:|
| `other` | 181 |
| `absence_denied` | 65 |
| `false_premise_asserted` | 41 |
| `mark_label_copied` | 37 |
| `wrong_subtype_copied` | 16 |
| `generic_tag_answer` | 5 |
| `relation_word_answer` | 4 |

## 1. `4928` (gqa) — `other`

**Question:** Is the pot to the right or to the left of the kettle?
**Gold:** left

| condition | model output |
|---|---|
| raw (clean image) | **left** correct |
| GoM (`gom_text_labeled`) | **Right** wrong |

*Marks (5):* pot_1, kettle_1, pot_2, kettle_2, pot_3

```
Triples:
pot_1 -(left_of)-> kettle_2
kettle_1 -(touching_right_of)-> pot_3
pot_2 -(close_above)-> kettle_1
kettle_2 -(left_of)-> kettle_1
```

| original | GoM render (what the model saw) |
|---|---|
| ![o](flip_examples_paper_gom_images/01_4928_original.jpg) | ![g](flip_examples_paper_gom_images/01_4928_gom.jpg) |

## 2. `2326540` (gqa) — `absence_denied`

**Question:** Do you see vans to the left of the bus on the left?
**Gold:** yes

| condition | model output |
|---|---|
| raw (clean image) | **yes** correct |
| GoM (`gom_text_labeled`) | **No** wrong |

*Marks (4):* bus_1, bus_2, bus_3, truck_1

```
Triples:
bus_1 -(left_of)-> bus_3
bus_2 -(right_of)-> bus_3
truck_1 -(above)-> bus_1
```

| original | GoM render (what the model saw) |
|---|---|
| ![o](flip_examples_paper_gom_images/02_2326540_original.jpg) | ![g](flip_examples_paper_gom_images/02_2326540_gom.jpg) |

## 3. `2353443` (gqa) — `false_premise_asserted`

**Question:** Is the computer to the left of books?
**Gold:** no

| condition | model output |
|---|---|
| raw (clean image) | **no** correct |
| GoM (`gom_text_labeled`) | **Yes** wrong |

*Marks (3):* mouse_1, computer_1, book_1

```
Triples:
mouse_1 -(above)-> computer_1
book_1 -(right_of)-> mouse_1
```

| original | GoM render (what the model saw) |
|---|---|
| ![o](flip_examples_paper_gom_images/03_2353443_original.jpg) | ![g](flip_examples_paper_gom_images/03_2353443_gom.jpg) |

## 4. `2390572` (gqa) — `mark_label_copied`

**Question:** Are there bartenders or students in the photo?
**Gold:** no

| condition | model output |
|---|---|
| raw (clean image) | **No** correct |
| GoM (`gom_text_labeled`) | **Bartenders** wrong |

*Marks (5):* person_1, motorcycle_1, motorcycle_2, bartenders_1, person_2

```
Triples:
bartenders_1 -(touching_below)-> person_1
```

| original | GoM render (what the model saw) |
|---|---|
| ![o](flip_examples_paper_gom_images/04_2390572_original.jpg) | ![g](flip_examples_paper_gom_images/04_2390572_gom.jpg) |

## 5. `2333988` (gqa) — `wrong_subtype_copied`

**Question:** Who is wearing a jacket?
**Gold:** woman

| condition | model output |
|---|---|
| raw (clean image) | **woman** correct |
| GoM (`gom_text_labeled`) | **man_3** wrong |

*Marks (7):* person_1, person_2, person_3, man_1, woman_1, man_2, man_3

```
Triples:
person_1 -(right_of)-> person_2
man_1 -(left_of)-> man_2
man_3 -(touching_above)-> person_1
```

| original | GoM render (what the model saw) |
|---|---|
| ![o](flip_examples_paper_gom_images/05_2333988_original.jpg) | ![g](flip_examples_paper_gom_images/05_2333988_gom.jpg) |

## 6. `2397817` (gqa) — `generic_tag_answer`

**Question:** Who is wearing the hat?
**Gold:** umpire

| condition | model output |
|---|---|
| raw (clean image) | **umpire** correct |
| GoM (`gom_text_labeled`) | **person_3** wrong |

*Marks (3):* person_1, person_2, person_3

```
Triples:
person_1 -(left_of)-> person_2
person_3 -(above)-> person_2
```

| original | GoM render (what the model saw) |
|---|---|
| ![o](flip_examples_paper_gom_images/06_2397817_original.jpg) | ![g](flip_examples_paper_gom_images/06_2397817_gom.jpg) |

## 7. `COCO_train2014_000000023357` (vqav1) — `relation_word_answer`

**Question:** Where are the spectators?
**Gold:** bleachers

| condition | model output |
|---|---|
| raw (clean image) | **bleachers** correct |
| GoM (`gom_text_labeled`) | **above** wrong |

*Marks (7):* person_1, person_2, person_3, person_4, elephant_1, spectators_1, bench_1

```
Triples:
spectators_1 -(above)-> person_1
```

| original | GoM render (what the model saw) |
|---|---|
| ![o](flip_examples_paper_gom_images/07_COCO_train2014_000000023357_original.jpg) | ![g](flip_examples_paper_gom_images/07_COCO_train2014_000000023357_gom.jpg) |

## 8. `2385364` (gqa) — `other`

**Question:** Is the open can to the left or to the right of the shelf the towels are on?
**Gold:** right

| condition | model output |
|---|---|
| raw (clean image) | **right** correct |
| GoM (`gom_text_labeled`) | **left** wrong |

*Marks (5):* toilet_1, sink_1, shelf_1, toilet_2, shelf_2

```
Triples:
toilet_1 -(touching_left_of)-> sink_1
shelf_1 -(left_of)-> toilet_2
shelf_2 -(above)-> shelf_1
```

| original | GoM render (what the model saw) |
|---|---|
| ![o](flip_examples_paper_gom_images/08_2385364_original.jpg) | ![g](flip_examples_paper_gom_images/08_2385364_gom.jpg) |

## 9. `2365147` (gqa) — `absence_denied`

**Question:** Is the teddy bear that is to the left of the cheeseburger sitting in a toy car?
**Gold:** yes

| condition | model output |
|---|---|
| raw (clean image) | **yes** correct |
| GoM (`gom_text_labeled`) | **No** wrong |

*Marks (9):* teddy bear_1, teddy bear_2, teddy bear_3, teddy bear_4, bear_1, bear_2, bear_3, bear_4, car_1

```
Triples:
teddy bear_1 -(right_of)-> teddy bear_3
teddy bear_4 -(left_of)-> teddy bear_3
bear_1 -(right_of)-> bear_4
bear_2 -(left_of)-> bear_3
car_1 -(touching_below)-> teddy bear_4
```

| original | GoM render (what the model saw) |
|---|---|
| ![o](flip_examples_paper_gom_images/09_2365147_original.jpg) | ![g](flip_examples_paper_gom_images/09_2365147_gom.jpg) |

## 10. `2390673` (gqa) — `false_premise_asserted`

**Question:** Are there any pizzas to the left of the utensil that is on the right?
**Gold:** no

| condition | model output |
|---|---|
| raw (clean image) | **no** correct |
| GoM (`gom_text_labeled`) | **yes** wrong |

*Marks (2):* sandwich_1, pizza_1

```
Triples:
sandwich_1 -(right_of)-> pizza_1
```

| original | GoM render (what the model saw) |
|---|---|
| ![o](flip_examples_paper_gom_images/10_2390673_original.jpg) | ![g](flip_examples_paper_gom_images/10_2390673_gom.jpg) |

## 11. `2374990` (gqa) — `mark_label_copied`

**Question:** What is the vegetable that is on the food the napkin is to the left of?
**Gold:** carrot

| condition | model output |
|---|---|
| raw (clean image) | **carrot** correct |
| GoM (`gom_text_labeled`) | **lettuce** wrong |

*Marks (12):* carrot_1, carrot_2, carrot_3, lettuce_1, vegetable_1, lettuce_2, lettuce_3, potato_1, potato_2, vegetable_2, sandwich_1, tomato_1

```
Triples:
carrot_1 -(right_of)-> carrot_2
lettuce_1 -(touching_below)-> vegetable_1
lettuce_2 -(touching_below)-> potato_1
potato_2 -(touching_right_of)-> sandwich_1
vegetable_2 -(behind)-> carrot_2
```

| original | GoM render (what the model saw) |
|---|---|
| ![o](flip_examples_paper_gom_images/11_2374990_original.jpg) | ![g](flip_examples_paper_gom_images/11_2374990_gom.jpg) |

## 12. `2368367` (gqa) — `wrong_subtype_copied`

**Question:** Who is holding the open laptop?
**Gold:** boy

| condition | model output |
|---|---|
| raw (clean image) | **Boy** correct |
| GoM (`gom_text_labeled`) | **man_2** wrong |

*Marks (5):* tie_1, glasses_1, man_1, man_2, girl_1

```
Triples:
tie_1 -(above)-> man_2
glasses_1 -(touching_above)-> girl_1
man_1 -(touching_below)-> girl_1
```

| original | GoM render (what the model saw) |
|---|---|
| ![o](flip_examples_paper_gom_images/12_2368367_original.jpg) | ![g](flip_examples_paper_gom_images/12_2368367_gom.jpg) |

## 13. `2326720` (gqa) — `generic_tag_answer`

**Question:** Who is wearing a cap?
**Gold:** player

| condition | model output |
|---|---|
| raw (clean image) | **Player** correct |
| GoM (`gom_text_labeled`) | **person_2** wrong |

*Marks (3):* person_1, person_2, cap_1

```
Triples:
person_1 -(right_of)-> person_2
person_2 -(touching_below)-> cap_1
```

| original | GoM render (what the model saw) |
|---|---|
| ![o](flip_examples_paper_gom_images/13_2326720_original.jpg) | ![g](flip_examples_paper_gom_images/13_2326720_gom.jpg) |

## 14. `COCO_train2014_000000496302` (vqav1) — `relation_word_answer`

**Question:** Where is the banner?
**Gold:** on pole

| condition | model output |
|---|---|
| raw (clean image) | **on pole** correct |
| GoM (`gom_text_labeled`) | **above** wrong |

*Marks (9):* person_1, traffic light_1, car_1, banner_1, person_2, traffic light_2, person_3, person_4, traffic light_3

```
Triples:
banner_1 -(above)-> person_4
```

| original | GoM render (what the model saw) |
|---|---|
| ![o](flip_examples_paper_gom_images/14_COCO_train2014_000000496302_original.jpg) | ![g](flip_examples_paper_gom_images/14_COCO_train2014_000000496302_gom.jpg) |

## 15. `2413895` (gqa) — `other`

**Question:** Does the dark cat appear to be standing or lying?
**Gold:** standing

| condition | model output |
|---|---|
| raw (clean image) | **standing** correct |
| GoM (`gom_text_labeled`) | **lying** wrong |

*Marks (3):* cat_1, cat_2, cat_3

```
Triples:
cat_1 -(right_of)-> cat_3
cat_2 -(below)-> cat_3
```

| original | GoM render (what the model saw) |
|---|---|
| ![o](flip_examples_paper_gom_images/15_2413895_original.jpg) | ![g](flip_examples_paper_gom_images/15_2413895_gom.jpg) |

## 16. `2320463` (gqa) — `absence_denied`

**Question:** Do you see trays to the right of the pizza cutter on the shelf?
**Gold:** yes

| condition | model output |
|---|---|
| raw (clean image) | **yes** correct |
| GoM (`gom_text_labeled`) | **No** wrong |

*Marks (7):* pizza_1, pizza_2, pizza_3, pizza_4, person_1, cutter_1, trays_1

```
Triples:
pizza_1 -(right_of)-> pizza_4
pizza_2 -(touching_right_of)-> cutter_1
pizza_3 -(above)-> person_1
pizza_4 -(touching_below)-> person_1
trays_1 -(left_of)-> cutter_1
```

| original | GoM render (what the model saw) |
|---|---|
| ![o](flip_examples_paper_gom_images/16_2320463_original.jpg) | ![g](flip_examples_paper_gom_images/16_2320463_gom.jpg) |

## 17. `2368619` (gqa) — `false_premise_asserted`

**Question:** Is the girl to the left of a woman?
**Gold:** no

| condition | model output |
|---|---|
| raw (clean image) | **no** correct |
| GoM (`gom_text_labeled`) | **Yes** wrong |

*Marks (5):* person_1, person_2, person_3, person_4, girl_1

```
Triples:
person_1 -(touching_left_of)-> girl_1
person_2 -(left_of)-> person_4
person_3 -(right_of)-> person_1
```

| original | GoM render (what the model saw) |
|---|---|
| ![o](flip_examples_paper_gom_images/17_2368619_original.jpg) | ![g](flip_examples_paper_gom_images/17_2368619_gom.jpg) |

## 18. `2369075` (gqa) — `mark_label_copied`

**Question:** What is the dessert that is to the right of the table?
**Gold:** cupcakes

| condition | model output |
|---|---|
| raw (clean image) | **cupcakes** correct |
| GoM (`gom_text_labeled`) | **cake** wrong |

*Marks (6):* cake_1, cake_2, cake_3, cake_4, dining table_1, table_1

```
Triples:
cake_1 -(left_of)-> cake_4
cake_2 -(left_of)-> cake_3
cake_3 -(left_of)-> cake_1
dining table_1 -(touching_above)-> cake_2
table_1 -(above)-> dining table_1
```

| original | GoM render (what the model saw) |
|---|---|
| ![o](flip_examples_paper_gom_images/18_2369075_original.jpg) | ![g](flip_examples_paper_gom_images/18_2369075_gom.jpg) |

## 19. `2380743` (gqa) — `wrong_subtype_copied`

**Question:** Who is wearing shorts?
**Gold:** man

| condition | model output |
|---|---|
| raw (clean image) | **Man** correct |
| GoM (`gom_text_labeled`) | **boy** wrong |

*Marks (3):* shorts_1, shorts_2, boy_1

```
Triples:
shorts_1 -(touching_below)-> boy_1
```

| original | GoM render (what the model saw) |
|---|---|
| ![o](flip_examples_paper_gom_images/19_2380743_original.jpg) | ![g](flip_examples_paper_gom_images/19_2380743_gom.jpg) |

## 20. `2316018` (gqa) — `generic_tag_answer`

**Question:** Who is wearing a dress?
**Gold:** woman

| condition | model output |
|---|---|
| raw (clean image) | **Woman** correct |
| GoM (`gom_text_labeled`) | **person 1** wrong |

*Marks (3):* person_1, bed_1, dress_1

```
Triples:
person_1 -(touching_above)-> dress_1
bed_1 -(touching_right_of)-> dress_1
```

| original | GoM render (what the model saw) |
|---|---|
| ![o](flip_examples_paper_gom_images/20_2316018_original.jpg) | ![g](flip_examples_paper_gom_images/20_2316018_gom.jpg) |
