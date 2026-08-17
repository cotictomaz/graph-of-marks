# Defect audit of the 20 gom_v2 flip cases

Visual classification of every case in `FLIP_EXAMPLES_PAPER_GOM.md` (data_v5, Qwen2.5-VL-7B,
`gom_text_labeled`). Each case is assigned the **primary** defect that caused the flip, plus any
secondary defects visible in the render. This drives the gom_v3 fix list at the bottom.

## Taxonomy

| code | defect | fixable by |
|---|---|---|
| **P** | prompt — model answers with mark vocabulary (ID tag / relation word) | prompt v3 |
| **D-miss** | queried object not detected/marked at all | open-vocab queries |
| **D-class** | queried object marked with the wrong class | open-vocab queries / detector ceiling |
| **D-generic** | mark carries a category label ("animal_1") instead of the specific class | targeted-label preference |
| **D-dup** | same object carries 2+ marks (fragment or duplicate) | fragment dedup |
| **D-fp** | mark on something that is not a real scene object (picture-in-image, corner artifact) | picture/area filter |
| **L-overlap** | label boxes overlap each other or hide a label | deterministic placement |
| **E-echo** | model copies a mark's class word instead of the finer-grained truth | prompt + specific labels |
| **E-plural** | answer correct but singular/plural mismatch vs gold | scorer |
| **M** | model reasoning error; render is clean and correct | not a pipeline defect |

## Per-case classification

| # | image | question | GoM answer | primary | secondary | what I see in the render |
|---|---|---|---|---|---|---|
| 1 | `2375522` | Who stands next to the person …? | person_2 | **P** | — | Clean 3-mark render, correct outlines. Pure ID leak. |
| 2 | `2387333` | Who is in front of the house? | person_1 | **P** | — | Clean 3-mark render (person/bat/ball). Pure ID leak. |
| 3 | `2333988` | Who is wearing a jacket? | person_1 | **P** | — | Clean. Pure ID leak. |
| 4 | `2397817` | Who is wearing the hat? | person_3 | **P** | L-overlap | ID leak; also `person_1` label sits far left of its person, `Left Of`/`Above` float mid-field. |
| 5 | `2326540` | Do you see vans to the left of the bus …? | no | **D-miss** | D-class, D-fp | The prominent left bus is unmarked; the van is marked `truck_1`; `bus_1` is a corner false-positive on a boat. "van" is not in the closed vocabulary → never queried. |
| 6 | `2411991` | Do you see skiers to the left of the bag …? | No | **L-overlap** | D-miss | `person_2`, `Touching Below`, `Left Of`, `Close Right Of` all stack into one unreadable pile over the skier. "skier" never queried (not in vocab). |
| 7 | `2365147` | Is the teddy bear … sitting in a toy car? | No | **D-miss** | D-fp | Cheeseburger never marked ("cheeseburger" absent from vocab); 3 of 8 marks are tiny background bear figurines (`bear_1..3`). |
| 8 | `2411265` | Is there any elephant in the zoo? | No | **D-dup** | — | Two elephants carry three marks: `elephant_3` is a leg fragment of `elephant_1`. Existence denial despite correct outlines. |
| 9 | `2385364` | Is the open can left or right of the shelf …? | left | **D-class** | — | Verified NOT a binding bug: every label matches its own box. The trash bin is misclassified `toilet_2`, a toilet part is `sink_1`; the queried can is unmarked. |
| 10 | `2376638` | Are the books … left or right of the couch? | Left | **D-dup** | L-overlap | Books marked in two places (`book_1/2` on the sofa, `book_3` on the table) — the question's "books of the coffee table" is ambiguous under duplicate same-class marks; `book_1`/`Left Of`/`Touching Above` crowd the bottom-left. |
| 11 | `2322351` | Is the towel left or right of the cabinet …? | right | **D-miss** | L-overlap | Neither towel nor cabinet is marked; `Touching Above` covers `oven_1` so it reads "en_1". |
| 12 | `2353443` | Is the computer to the left of books? | Yes | **D-class** | — | The papers are marked `book_1`; gold is "no, the computer is left of the *papers*". The mark asserts the false premise. |
| 13 | `2368619` | Is the girl to the left of a woman? | yes | **D-generic** | L-overlap, M | Four people all labeled `person_N` — girl/woman distinction erased; `person_2`/`Left Of` touch. |
| 14 | `2383493` | What kind of animal is the water behind of? | elephant | **E-plural** | L-overlap | Answer is semantically right; gold is "elephants". `Left Of`/`Touching Left Of`/`Right Of` pile up near `elephant_2`. |
| 15 | `2369075` | What is the dessert right of the table? | cake | **E-echo** | L-overlap | Cupcakes are marked `cake_1..4`; `cake_3` is half-hidden under `cake_4`. Model echoes the mark's class word. |
| 16 | `2340160` | What furniture does the pillow lie on top of? | chair | **D-miss** | E-echo | Pillow unmarked; the couch is only clipped at the frame edge as `sofa_1` while `chair_1` is prominent → echo. |
| 17 | `2413895` | Does the dark cat appear standing or lying? | lying | **M** | D-fp, L-overlap | `cat_3` is a cat *in a wall poster*; `cat_3`/`Below`/`Right Of` crowd. The dark cat itself is correctly outlined. |
| 18 | `2346478` | What type of food is right of the wine …? | chips | **D-miss** | — | The nuts are not marked (only the three wine glasses are); "food"/"nuts" never queried. |
| 19 | `2401706` | What animal is left of the zebra …? | goat | **D-generic** | D-dup | The cow carries two marks, both labeled `animal_N` (the category query became the label) — no species evidence for the model. |
| 20 | `2369026` | What instrument is right of the person …? | knife | **D-miss** | L-overlap | The guitar is unmarked ("instrument"/"guitar" never queried); `Right Of`/`Touching Below`/`person_2` overlap top-left. |

## Rates

Primary: **D-miss 6**, **P 4**, **D-class 3**, **D-dup 2**, **D-generic 2**, **L-overlap 1**,
**E-echo 1**, **E-plural 1**, **M 1**. As a *secondary* defect, **L-overlap appears in 9 of 20
renders** — it is the most pervasive visual problem even when it is not the proximate cause.

## Fixes these defects imply (implemented in gom_v3)

1. **Open-vocabulary question queries** (D-miss 6, D-class 3): drop the closed `_VISUAL_OBJECTS`
   gate on detector queries. "van", "cheeseburger", "towel", "cabinet", "nuts", "guitar",
   "skier", "pillow" all become OWLv2 queries.
2. **Specific-label preference** (D-generic 2): when a targeted detection's query is a *category*
   word ("animal", "food", "instrument", "furniture"), prefer an overlapping ensemble detection's
   specific class ("cow") for the mark label instead of the query string.
3. **Deterministic no-overlap label placement** (L-overlap 1 primary + 9 secondary): a single
   placement pass with a hard zero-overlap constraint over object labels, relation labels, and
   precomputed arrow paths, plus a per-render `label_overlap_count` recorded in the artifact
   metadata so a whole run can be asserted overlap-free.
4. **Fragment/duplicate dedup** (D-dup 2): drop a same-class detection whose box is largely
   contained in a bigger same-class box while their masks barely intersect.
5. **Prompt v3** (P 4, E-echo 1): presence assertion + few-shot exemplars showing a `who`
   question answered "woman" (never `person_2`), an existence question answered "yes", and a
   `where` question answered with a place (never an arrow word).
6. **Plural-tolerant lenient scoring** (E-plural 1): treat "elephant"/"elephants" as a match in
   `gqa_hit`, the same class of eval artifact as the earlier "Right of" vs "right".
7. **Picture-in-image false positives** (D-fp 3, secondary): the existing
   `max_picture_area_ratio` path only covers picture/painting/frame classes; the tiny-background
   figurine and wall-poster cases are left to the detection caps and are documented, not
   special-cased.

Not fixable in the pipeline: **M 1** (model reasoning on a correct render) and the residual
detector ceiling behind D-class (a trash bin that OWLv2/Detectron2 call a toilet).
