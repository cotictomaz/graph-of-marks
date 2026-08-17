# GoM failures under the final configuration (gom_v3 renders)

Every render below is the final GoM form - condition `gom_text_labeled` from the
`data_v6` run: outline renders, open-vocabulary question-driven marks, deterministic
zero-overlap label placement, `gom_v2_concise` prompt, curated eval. Model:
Qwen2.5-VL-7B. All cases: correct on the clean image, wrong on the GoM image
(lenient scoring).

The renders no longer show the defects the previous audit found: label boxes never
overlap (verified run-wide, 3,975 images x 6 variants), marks average 3.6 per image,
part-of-object fragment marks are removed, and question nouns outside the closed
ontology are detected. What remains are the failures that survive a clean render.

Flip pool: 125 raw-right -> GoM-wrong instances out of 996 GQA rows
(identity/other 37, ID leak 25, yes->no 22, label echo 15, no->yes 14, left/right 12).
Run record: `RESULTS.md` section gom_v3.

## 1. `2387333` (GQA)

**Question:** Who is in front of the house?
**Gold:** boy, The boy is in front of the house.

| condition | model output |
|---|---|
| raw (clean image) | **Boy** OK |
| GoM (outline + IDs + arrows) | **person_1** WRONG |

*Mechanism:* ID leakage: the model answers with a mark's reference tag instead of naming the person

| original | GoM render (what the model saw) |
|---|---|
| ![o](flip_examples_paper_gom/v3_01_2387333_original.jpg) | ![g](flip_examples_paper_gom/v3_01_2387333_gom.jpg) |

## 2. `2333988` (GQA)

**Question:** Who is wearing a jacket?
**Gold:** woman, The woman is wearing a jacket.

| condition | model output |
|---|---|
| raw (clean image) | **woman** OK |
| GoM (outline + IDs + arrows) | **person_1** WRONG |

*Mechanism:* ID leakage: the model answers with a mark's reference tag instead of naming the person

| original | GoM render (what the model saw) |
|---|---|
| ![o](flip_examples_paper_gom/v3_02_2333988_original.jpg) | ![g](flip_examples_paper_gom/v3_02_2333988_gom.jpg) |

## 3. `2397817` (GQA)

**Question:** Who is wearing the hat?
**Gold:** umpire, The umpire is wearing a hat.

| condition | model output |
|---|---|
| raw (clean image) | **umpire** OK |
| GoM (outline + IDs + arrows) | **person_2** WRONG |

*Mechanism:* ID leakage: the model answers with a mark's reference tag instead of naming the person

| original | GoM render (what the model saw) |
|---|---|
| ![o](flip_examples_paper_gom/v3_03_2397817_original.jpg) | ![g](flip_examples_paper_gom/v3_03_2397817_gom.jpg) |

## 4. `2368367` (GQA)

**Question:** Who is holding the open laptop?
**Gold:** boy, The boy is holding the laptop.

| condition | model output |
|---|---|
| raw (clean image) | **Boy** OK |
| GoM (outline + IDs + arrows) | **person_2** WRONG |

*Mechanism:* ID leakage: the model answers with a mark's reference tag instead of naming the person

| original | GoM render (what the model saw) |
|---|---|
| ![o](flip_examples_paper_gom/v3_04_2368367_original.jpg) | ![g](flip_examples_paper_gom/v3_04_2368367_gom.jpg) |

## 5. `2326540` (GQA)

**Question:** Do you see vans to the left of the bus on the left?
**Gold:** yes, Yes, there is a van to the left of the bus.

| condition | model output |
|---|---|
| raw (clean image) | **yes** OK |
| GoM (outline + IDs + arrows) | **no** WRONG |

*Mechanism:* existence denial: the queried object is outlined, the model still answers no

| original | GoM render (what the model saw) |
|---|---|
| ![o](flip_examples_paper_gom/v3_05_2326540_original.jpg) | ![g](flip_examples_paper_gom/v3_05_2326540_gom.jpg) |

## 6. `392` (GQA)

**Question:** Is the keyboard to the left of the computer?
**Gold:** yes, Yes, the keyboard is to the left of the computer.

| condition | model output |
|---|---|
| raw (clean image) | **Yes** OK |
| GoM (outline + IDs + arrows) | **No** WRONG |

*Mechanism:* existence denial: the queried object is outlined, the model still answers no

| original | GoM render (what the model saw) |
|---|---|
| ![o](flip_examples_paper_gom/v3_06_392_original.jpg) | ![g](flip_examples_paper_gom/v3_06_392_gom.jpg) |

## 7. `2411265` (GQA)

**Question:** Is there any elephant in the zoo?
**Gold:** yes, Yes, there are elephants in the zoo.

| condition | model output |
|---|---|
| raw (clean image) | **Yes** OK |
| GoM (outline + IDs + arrows) | **No** WRONG |

*Mechanism:* existence denial: the queried object is outlined, the model still answers no

| original | GoM render (what the model saw) |
|---|---|
| ![o](flip_examples_paper_gom/v3_07_2411265_original.jpg) | ![g](flip_examples_paper_gom/v3_07_2411265_gom.jpg) |

## 8. `2320463` (GQA)

**Question:** Do you see trays to the right of the pizza cutter on the shelf?
**Gold:** yes, Yes, there is a tray to the right of the pizza cutter.

| condition | model output |
|---|---|
| raw (clean image) | **yes** OK |
| GoM (outline + IDs + arrows) | **No** WRONG |

*Mechanism:* existence denial: the queried object is outlined, the model still answers no

| original | GoM render (what the model saw) |
|---|---|
| ![o](flip_examples_paper_gom/v3_08_2320463_original.jpg) | ![g](flip_examples_paper_gom/v3_08_2320463_gom.jpg) |

## 9. `2385364` (GQA)

**Question:** Is the open can to the left or to the right of the shelf the towels are on?
**Gold:** right, The can is to the right of the shelf.

| condition | model output |
|---|---|
| raw (clean image) | **right** OK |
| GoM (outline + IDs + arrows) | **left** WRONG |

*Mechanism:* left/right inversion on a direct spatial question, arrows present

| original | GoM render (what the model saw) |
|---|---|
| ![o](flip_examples_paper_gom/v3_09_2385364_original.jpg) | ![g](flip_examples_paper_gom/v3_09_2385364_gom.jpg) |

## 10. `2376638` (GQA)

**Question:** Are the books of the coffee table to the left or to the right of the couch?
**Gold:** right, The books are to the right of the couch.

| condition | model output |
|---|---|
| raw (clean image) | **right** OK |
| GoM (outline + IDs + arrows) | **Left** WRONG |

*Mechanism:* left/right inversion on a direct spatial question, arrows present

| original | GoM render (what the model saw) |
|---|---|
| ![o](flip_examples_paper_gom/v3_10_2376638_original.jpg) | ![g](flip_examples_paper_gom/v3_10_2376638_gom.jpg) |

## 11. `2322351` (GQA)

**Question:** Is the towel to the left or to the right of the cabinet that is on the right side of the photo?
**Gold:** left, The towel is to the left of the cabinet.

| condition | model output |
|---|---|
| raw (clean image) | **left** OK |
| GoM (outline + IDs + arrows) | **right** WRONG |

*Mechanism:* left/right inversion on a direct spatial question, arrows present

| original | GoM render (what the model saw) |
|---|---|
| ![o](flip_examples_paper_gom/v3_11_2322351_original.jpg) | ![g](flip_examples_paper_gom/v3_11_2322351_gom.jpg) |

## 12. `2353443` (GQA)

**Question:** Is the computer to the left of books?
**Gold:** no, No, the computer is to the left of the papers.

| condition | model output |
|---|---|
| raw (clean image) | **no** OK |
| GoM (outline + IDs + arrows) | **Yes** WRONG |

*Mechanism:* asserts a relation the photograph contradicts

| original | GoM render (what the model saw) |
|---|---|
| ![o](flip_examples_paper_gom/v3_12_2353443_original.jpg) | ![g](flip_examples_paper_gom/v3_12_2353443_gom.jpg) |

## 13. `2390673` (GQA)

**Question:** Are there any pizzas to the left of the utensil that is on the right?
**Gold:** no, No, there is a sandwich to the left of the utensil.

| condition | model output |
|---|---|
| raw (clean image) | **no** OK |
| GoM (outline + IDs + arrows) | **yes** WRONG |

*Mechanism:* asserts a relation the photograph contradicts

| original | GoM render (what the model saw) |
|---|---|
| ![o](flip_examples_paper_gom/v3_13_2390673_original.jpg) | ![g](flip_examples_paper_gom/v3_13_2390673_gom.jpg) |

## 14. `2390572` (GQA)

**Question:** Are there bartenders or students in the photo?
**Gold:** no, No, there are no bartenders or students.

| condition | model output |
|---|---|
| raw (clean image) | **No** OK |
| GoM (outline + IDs + arrows) | **Bartenders** WRONG |

*Mechanism:* mark-label echo: the model repeats a mark's class word instead of the finer-grained truth

| original | GoM render (what the model saw) |
|---|---|
| ![o](flip_examples_paper_gom/v3_14_2390572_original.jpg) | ![g](flip_examples_paper_gom/v3_14_2390572_gom.jpg) |

## 15. `2383493` (GQA)

**Question:** What kind of animal is the water behind of?
**Gold:** elephants, The water is behind the elephants.

| condition | model output |
|---|---|
| raw (clean image) | **elephants** OK |
| GoM (outline + IDs + arrows) | **elephant** WRONG |

*Mechanism:* mark-label echo: the model repeats a mark's class word instead of the finer-grained truth

| original | GoM render (what the model saw) |
|---|---|
| ![o](flip_examples_paper_gom/v3_15_2383493_original.jpg) | ![g](flip_examples_paper_gom/v3_15_2383493_gom.jpg) |

## 16. `2369075` (GQA)

**Question:** What is the dessert that is to the right of the table?
**Gold:** cupcakes, The dessert is cupcakes.

| condition | model output |
|---|---|
| raw (clean image) | **cupcakes** OK |
| GoM (outline + IDs + arrows) | **cake** WRONG |

*Mechanism:* mark-label echo: the model repeats a mark's class word instead of the finer-grained truth

| original | GoM render (what the model saw) |
|---|---|
| ![o](flip_examples_paper_gom/v3_16_2369075_original.jpg) | ![g](flip_examples_paper_gom/v3_16_2369075_gom.jpg) |

## 17. `2413895` (GQA)

**Question:** Does the dark cat appear to be standing or lying?
**Gold:** standing, The cat is standing.

| condition | model output |
|---|---|
| raw (clean image) | **standing** OK |
| GoM (outline + IDs + arrows) | **lying** WRONG |

*Mechanism:* identity/state shift on a marked object

| original | GoM render (what the model saw) |
|---|---|
| ![o](flip_examples_paper_gom/v3_17_2413895_original.jpg) | ![g](flip_examples_paper_gom/v3_17_2413895_gom.jpg) |

## 18. `2404832` (GQA)

**Question:** What kind of furniture is holding the books in the top of the photo?
**Gold:** bookshelf, The piece of furniture is a bookshelf.

| condition | model output |
|---|---|
| raw (clean image) | **bookshelf** OK |
| GoM (outline + IDs + arrows) | **shelf** WRONG |

*Mechanism:* identity/state shift on a marked object

| original | GoM render (what the model saw) |
|---|---|
| ![o](flip_examples_paper_gom/v3_18_2404832_original.jpg) | ![g](flip_examples_paper_gom/v3_18_2404832_gom.jpg) |

## 19. `2375522` (GQA)

**Question:** Who stands next to the person the window behind of?
**Gold:** man, The man stands next to the lady.

| condition | model output |
|---|---|
| raw (clean image) | **man** OK |
| GoM (outline + IDs + arrows) | **woman** WRONG |

*Mechanism:* identity/state shift on a marked object

| original | GoM render (what the model saw) |
|---|---|
| ![o](flip_examples_paper_gom/v3_19_2375522_original.jpg) | ![g](flip_examples_paper_gom/v3_19_2375522_gom.jpg) |

## 20. `2346478` (GQA)

**Question:** What type of food is to the right of the wine on the left?
**Gold:** nuts, The food is nuts.

| condition | model output |
|---|---|
| raw (clean image) | **Nuts** OK |
| GoM (outline + IDs + arrows) | **none** WRONG |

*Mechanism:* identity/state shift on a marked object

| original | GoM render (what the model saw) |
|---|---|
| ![o](flip_examples_paper_gom/v3_20_2346478_original.jpg) | ![g](flip_examples_paper_gom/v3_20_2346478_gom.jpg) |

---

**Scale.** These 20 are a stratified sample of 125 flips in 996 GQA rows; the same
renders also produce 100+ rescues, so the net effect per model is what `RESULTS.md`
reports, not this gallery. The dominant remaining mechanism is ID leakage on text-tag
renders (25 of 125), which the numeric-ID conditions largely avoid (4-5 leaks vs 87-90
across the full run) - both conditions are evaluated.
