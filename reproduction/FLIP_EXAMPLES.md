# Visual inspection: 10 instances where Qwen2.5-VL-7B was right on the raw image and wrong after GoM

Run: `data_v3` (best config — outline-only marks, `direct_concise` prompt), condition `segmented`.
Each case shows the original photo, the preprocessed image the model actually saw, and the
verbatim generations. Images are copied under `reproduction/flip_examples/`.

## 1. `COCO_train2014_000000458752` (VQAV2)

**Question:** What is this photo taken looking through?
**Gold answer(s):** mesh, net, netting

| condition | model output |
|---|---|
| raw (clean image) | **net** ✅ |
| segmented (marked image) | **Fence** ❌ |
| gom_text (marked + IDs) | net |

*Mechanism:* texture/identity through contour (net -> fence)

| original | preprocessed (what the model saw) |
|---|---|
| ![original](flip_examples/01_COCO_train2014_000000458752_original.jpg) | ![marked](flip_examples/01_COCO_train2014_000000458752_marked.jpg) |

## 2. `COCO_train2014_000000262204` (VQAV2)

**Question:** How many tusks are visible?
**Gold answer(s):** 2

| condition | model output |
|---|---|
| raw (clean image) | **2** ✅ |
| segmented (marked image) | **1** ❌ |
| gom_text (marked + IDs) | 1 |

*Mechanism:* contour crosses the queried detail (tusks)

| original | preprocessed (what the model saw) |
|---|---|
| ![original](flip_examples/02_COCO_train2014_000000262204_original.jpg) | ![marked](flip_examples/02_COCO_train2014_000000262204_marked.jpg) |

## 3. `COCO_train2014_000000000081` (VQAV2)

**Question:** Does this planes tail have 4 colors?
**Gold answer(s):** 3, no, yes

| condition | model output |
|---|---|
| raw (clean image) | **No** ✅ |
| segmented (marked image) | **Yes** ❌ |
| gom_text (marked + IDs) | Yes |

*Mechanism:* contour colors contaminate a color-count question

| original | preprocessed (what the model saw) |
|---|---|
| ![original](flip_examples/03_COCO_train2014_000000000081_original.jpg) | ![marked](flip_examples/03_COCO_train2014_000000000081_marked.jpg) |

## 4. `COCO_train2014_000000436929` (VQAV2)

**Question:** What is floating near the bird?
**Gold answer(s):** fish, foam, fog, ice, nothing, water

| condition | model output |
|---|---|
| raw (clean image) | **ice** ✅ |
| segmented (marked image) | **shoes** ❌ |
| gom_text (marked + IDs) | ice |

*Mechanism:* attention pulled to marked objects (birds) away from answer (ice)

| original | preprocessed (what the model saw) |
|---|---|
| ![original](flip_examples/04_COCO_train2014_000000436929_original.jpg) | ![marked](flip_examples/04_COCO_train2014_000000436929_marked.jpg) |

## 5. `COCO_train2014_000000393242` (VQAV2)

**Question:** How many arrows can be seen?
**Gold answer(s):** 1, 2, 3

| condition | model output |
|---|---|
| raw (clean image) | **1** ✅ |
| segmented (marked image) | **2** ❌ |
| gom_text (marked + IDs) | 2 |

*Mechanism:* counting corrupted by added shapes (arrows)

| original | preprocessed (what the model saw) |
|---|---|
| ![original](flip_examples/05_COCO_train2014_000000393242_original.jpg) | ![marked](flip_examples/05_COCO_train2014_000000393242_marked.jpg) |

## 6. `392` (GQA)

**Question:** Is the keyboard to the left of the computer?
**Gold answer(s):** Yes, the keyboard is to the left of the computer., yes

| condition | model output |
|---|---|
| raw (clean image) | **Yes** ✅ |
| segmented (marked image) | **No** ❌ |
| gom_text (marked + IDs) | No |

| original | preprocessed (what the model saw) |
|---|---|
| ![original](flip_examples/06_392_original.jpg) | ![marked](flip_examples/06_392_marked.jpg) |

## 7. `2369075` (GQA)

**Question:** What is the dessert that is to the right of the table?
**Gold answer(s):** The dessert is cupcakes., cupcakes

| condition | model output |
|---|---|
| raw (clean image) | **cupcakes** ✅ |
| segmented (marked image) | **cake** ❌ |
| gom_text (marked + IDs) | cake |

| original | preprocessed (what the model saw) |
|---|---|
| ![original](flip_examples/07_2369075_original.jpg) | ![marked](flip_examples/07_2369075_marked.jpg) |

## 8. `2333988` (GQA)

**Question:** Who is wearing a jacket?
**Gold answer(s):** The woman is wearing a jacket., woman

| condition | model output |
|---|---|
| raw (clean image) | **woman** ✅ |
| segmented (marked image) | **Everyone** ❌ |
| gom_text (marked + IDs) | person_2 |

| original | preprocessed (what the model saw) |
|---|---|
| ![original](flip_examples/08_2333988_original.jpg) | ![marked](flip_examples/08_2333988_marked.jpg) |

## 9. `2316593` (GQA)

**Question:** What is the small item of furniture called?
**Gold answer(s):** The piece of furniture is a bed., bed

| condition | model output |
|---|---|
| raw (clean image) | **bed** ✅ |
| segmented (marked image) | **chair** ❌ |
| gom_text (marked + IDs) | chair |

| original | preprocessed (what the model saw) |
|---|---|
| ![original](flip_examples/09_2316593_original.jpg) | ![marked](flip_examples/09_2316593_marked.jpg) |

## 10. `2318490` (GQA)

**Question:** Which kind of clothing is not knit?
**Gold answer(s):** The clothing is a jacket., jacket

| condition | model output |
|---|---|
| raw (clean image) | **jacket** ✅ |
| segmented (marked image) | **watch** ❌ |
| gom_text (marked + IDs) | tie |

| original | preprocessed (what the model saw) |
|---|---|
| ![original](flip_examples/10_2318490_original.jpg) | ![marked](flip_examples/10_2318490_marked.jpg) |
