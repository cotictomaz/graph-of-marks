# Paper-form GoM failures: filled colored segments + ID labels + relation arrows

Every render below is the paper's declared GoM form — condition `gom_text_labeled` from
the paper-faithful run (`data_v2`: fill α=0.25, text IDs, relation arrows and labels,
`supplementary_concise` prompt). Model: Qwen2.5-VL-7B. All cases: correct on the clean
image, wrong on the GoM image; appearance/subjective questions excluded.

## 1. `2354833` (GQA)

**Question:** On which side of the picture are the shelves?
**Gold:** The shelves are on the left of the image., left

| condition | model output |
|---|---|
| raw (clean image) | **left** ✅ |
| GoM (fill + IDs + arrows) | **Right** ❌ |

*Mechanism:* the queried shelves are unmarked while 17 'bottle_N' labels blanket the scene; answer flips left→Right

| original | GoM render (what the model saw) |
|---|---|
| ![o](flip_examples_paper_gom/01_2354833_original.jpg) | ![g](flip_examples_paper_gom/01_2354833_gom.jpg) |

## 2. `2335852` (GQA)

**Question:** Is the man to the right or to the left of the elephant?
**Gold:** The man is to the left of the elephant., left

| condition | model output |
|---|---|
| raw (clean image) | **left** ✅ |
| GoM (fill + IDs + arrows) | **Right** ❌ |

*Mechanism:* left/right inversion on a direct spatial question, arrows present

| original | GoM render (what the model saw) |
|---|---|
| ![o](flip_examples_paper_gom/02_2335852_original.jpg) | ![g](flip_examples_paper_gom/02_2335852_gom.jpg) |

## 3. `2333988` (GQA)

**Question:** Who is wearing a jacket?
**Gold:** The woman is wearing a jacket., woman

| condition | model output |
|---|---|
| raw (clean image) | **woman** ✅ |
| GoM (fill + IDs + arrows) | **person 1** ❌ |

*Mechanism:* ID leakage: answers 'person 1' instead of 'woman'

| original | GoM render (what the model saw) |
|---|---|
| ![o](flip_examples_paper_gom/03_2333988_original.jpg) | ![g](flip_examples_paper_gom/03_2333988_gom.jpg) |

## 4. `2392912` (GQA)

**Question:** Who is wearing the shirt?
**Gold:** The man is wearing a shirt., man

| condition | model output |
|---|---|
| raw (clean image) | **man** ✅ |
| GoM (fill + IDs + arrows) | **person_1** ❌ |

*Mechanism:* ID leakage: answers 'person_1' instead of 'man'

| original | GoM render (what the model saw) |
|---|---|
| ![o](flip_examples_paper_gom/04_2392912_original.jpg) | ![g](flip_examples_paper_gom/04_2392912_gom.jpg) |

## 5. `2359506` (GQA)

**Question:** That snow is where?
**Gold:** The snow is on the ground., ground

| condition | model output |
|---|---|
| raw (clean image) | **ground** ✅ |
| GoM (fill + IDs + arrows) | **Above** ❌ |

*Mechanism:* relation-vocabulary leakage: answers 'Above' (arrow label) instead of 'ground'

| original | GoM render (what the model saw) |
|---|---|
| ![o](flip_examples_paper_gom/05_2359506_original.jpg) | ![g](flip_examples_paper_gom/05_2359506_gom.jpg) |

## 6. `2321902` (GQA)

**Question:** Is the child to the right or to the left of the woman that is to the right of the man?
**Gold:** The child is to the right of the woman., right

| condition | model output |
|---|---|
| raw (clean image) | **right** ✅ |
| GoM (fill + IDs + arrows) | **Right of** ❌ |

*Mechanism:* relation-vocabulary leakage: answers 'Right of' instead of 'right'

| original | GoM render (what the model saw) |
|---|---|
| ![o](flip_examples_paper_gom/06_2321902_original.jpg) | ![g](flip_examples_paper_gom/06_2321902_gom.jpg) |

## 7. `2382290` (GQA)

**Question:** What is the man wearing?
**Gold:** The man is wearing a hat., hat

| condition | model output |
|---|---|
| raw (clean image) | **hat** ✅ |
| GoM (fill + IDs + arrows) | **green** ❌ |

*Mechanism:* palette leakage: answers 'green' (a mask color) to 'What is the man wearing?'

| original | GoM render (what the model saw) |
|---|---|
| ![o](flip_examples_paper_gom/07_2382290_original.jpg) | ![g](flip_examples_paper_gom/07_2382290_gom.jpg) |

## 8. `2401706` (GQA)

**Question:** What kind of animal is to the left of the zebra that is eating grass?
**Gold:** The animal is a cow., cow

| condition | model output |
|---|---|
| raw (clean image) | **cow** ✅ |
| GoM (fill + IDs + arrows) | **horse** ❌ |

*Mechanism:* identity swap on a relational question: cow → horse

| original | GoM render (what the model saw) |
|---|---|
| ![o](flip_examples_paper_gom/08_2401706_original.jpg) | ![g](flip_examples_paper_gom/08_2401706_gom.jpg) |

## 9. `2316593` (GQA)

**Question:** What is the small item of furniture called?
**Gold:** The piece of furniture is a bed., bed

| condition | model output |
|---|---|
| raw (clean image) | **bed** ✅ |
| GoM (fill + IDs + arrows) | **chair** ❌ |

*Mechanism:* filled mask destroys object identity: bed → chair

| original | GoM render (what the model saw) |
|---|---|
| ![o](flip_examples_paper_gom/09_2316593_original.jpg) | ![g](flip_examples_paper_gom/09_2316593_gom.jpg) |

## 10. `2412283` (GQA)

**Question:** Is the bookcase to the left of the chair that is not antique?
**Gold:** Yes, the bookcase is to the left of the chair., yes

| condition | model output |
|---|---|
| raw (clean image) | **yes** ✅ |
| GoM (fill + IDs + arrows) | **No** ❌ |

*Mechanism:* yes→No denial on a spatial-relation question with the arrow drawn

| original | GoM render (what the model saw) |
|---|---|
| ![o](flip_examples_paper_gom/10_2412283_original.jpg) | ![g](flip_examples_paper_gom/10_2412283_gom.jpg) |

## 11. `2387622` (GQA)

**Question:** Are there any benches near the sidewalk?
**Gold:** Yes, there is a bench near the sidewalk., yes

| condition | model output |
|---|---|
| raw (clean image) | **yes** ✅ |
| GoM (fill + IDs + arrows) | **no** ❌ |

*Mechanism:* existence denial: marked scene, benches 'not there'

| original | GoM render (what the model saw) |
|---|---|
| ![o](flip_examples_paper_gom/11_2387622_original.jpg) | ![g](flip_examples_paper_gom/11_2387622_gom.jpg) |

## 12. `2396350` (GQA)

**Question:** Where is the dog?
**Gold:** The dog is in the car., car

| condition | model output |
|---|---|
| raw (clean image) | **car** ✅ |
| GoM (fill + IDs + arrows) | **In the mirror** ❌ |

*Mechanism:* attention artifact: 'Where is the dog?' → 'In the mirror'

| original | GoM render (what the model saw) |
|---|---|
| ![o](flip_examples_paper_gom/12_2396350_original.jpg) | ![g](flip_examples_paper_gom/12_2396350_gom.jpg) |
