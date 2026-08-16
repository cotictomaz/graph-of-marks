# Visual inspection: 10 instances where Qwen2.5-VL-7B was right on the raw image and wrong after GoM

**Scope note:** this file shows the `segmented` condition — masks only, no IDs/arrows —
because it is GoM's *best-scoring* marked condition. For the full GoM form (IDs + relation
arrows + relation labels) on spatial/relational questions, see
**`FLIP_EXAMPLES_FULL_GOM.md`**, including the per-condition score ladder showing all seven
conditions were measured.

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

---

## Further examples (11–20): other models and VQAv1 included

Same run and condition as above; cases 11–14 are Qwen2.5-VL-7B, 15–17 Gemma-3-4B,
18–20 LlamaV-o1-11B. Chosen to exclude scoring artifacts (synonym/format mismatches) —
each is a genuine perception flip.

## 11. `2403371` (GQA, Qwen2.5-VL-7B)

**Question:** Which side of the image is the wet bench on?
**Gold answer(s):** The bench is on the right of the image., right

| condition | model output |
|---|---|
| raw (clean image) | **right** ✅ |
| segmented (marked image) | **left** ❌ |
| gom_text (marked + IDs) | right |

*Mechanism:* spatial side flip: the marked bench reads as being on the other side

| original | preprocessed (what the model saw) |
|---|---|
| ![original](flip_examples/11_2403371_original.jpg) | ![marked](flip_examples/11_2403371_marked.jpg) |

## 12. `COCO_train2014_000000504259` (VQAV1, Qwen2.5-VL-7B)

**Question:** Are all of the sheep facing the same direction?
**Gold answer(s):** yes

| condition | model output |
|---|---|
| raw (clean image) | **yes** ✅ |
| segmented (marked image) | **No** ❌ |
| gom_text (marked + IDs) | No |

*Mechanism:* group-attribute denial: same-color masks on all sheep hide their orientation

| original | preprocessed (what the model saw) |
|---|---|
| ![original](flip_examples/12_COCO_train2014_000000504259_original.jpg) | ![marked](flip_examples/12_COCO_train2014_000000504259_marked.jpg) |

## 13. `COCO_train2014_000000520112` (VQAV1, Qwen2.5-VL-7B)

**Question:** What are the men doing?
**Gold answer(s):** drinking, drinking wine, talking, tasting wine, watching something, wine tasting

| condition | model output |
|---|---|
| raw (clean image) | **drinking** ✅ |
| segmented (marked image) | **standing** ❌ |
| gom_text (marked + IDs) | standing |

*Mechanism:* activity misread: contours freeze the action into 'standing'

| original | preprocessed (what the model saw) |
|---|---|
| ![original](flip_examples/13_COCO_train2014_000000520112_original.jpg) | ![marked](flip_examples/13_COCO_train2014_000000520112_marked.jpg) |

## 14. `COCO_train2014_000000305527` (VQAV2, Qwen2.5-VL-7B)

**Question:** Is the Frisbee upside down?
**Gold answer(s):** yes

| condition | model output |
|---|---|
| raw (clean image) | **Yes** ✅ |
| segmented (marked image) | **No** ❌ |
| gom_text (marked + IDs) | No |

*Mechanism:* object-state flip (upside down) with contour over the frisbee

| original | preprocessed (what the model saw) |
|---|---|
| ![original](flip_examples/14_COCO_train2014_000000305527_original.jpg) | ![marked](flip_examples/14_COCO_train2014_000000305527_marked.jpg) |

## 15. `2317129` (GQA, Gemma-3-4B)

**Question:** Are the olives to the left or to the right of the cheese?
**Gold answer(s):** The olives are to the left of the cheese., left

| condition | model output |
|---|---|
| raw (clean image) | **Left** ✅ |
| segmented (marked image) | **Right** ❌ |
| gom_text (marked + IDs) | Right |

*Mechanism:* left/right inversion on the exact spatial question type GoM targets

| original | preprocessed (what the model saw) |
|---|---|
| ![original](flip_examples/15_2317129_original.jpg) | ![marked](flip_examples/15_2317129_marked.jpg) |

## 16. `COCO_train2014_000000131172` (VQAV2, Gemma-3-4B)

**Question:** How many men in the picture are wearing sunglasses?
**Gold answer(s):** 1, 2

| condition | model output |
|---|---|
| raw (clean image) | **One** ✅ |
| segmented (marked image) | **Two** ❌ |
| gom_text (marked + IDs) | Two |

*Mechanism:* count inflated: contour edges read as an extra person with sunglasses

| original | preprocessed (what the model saw) |
|---|---|
| ![original](flip_examples/16_COCO_train2014_000000131172_original.jpg) | ![marked](flip_examples/16_COCO_train2014_000000131172_marked.jpg) |

## 17. `COCO_train2014_000000291855` (VQAV1, Gemma-3-4B)

**Question:** Is this a private bus?
**Gold answer(s):** no, yes

| condition | model output |
|---|---|
| raw (clean image) | **No** ✅ |
| segmented (marked image) | **Yes** ❌ |
| gom_text (marked + IDs) | Yes |

*Mechanism:* attribute hallucination under marks (private bus)

| original | preprocessed (what the model saw) |
|---|---|
| ![original](flip_examples/17_COCO_train2014_000000291855_original.jpg) | ![marked](flip_examples/17_COCO_train2014_000000291855_marked.jpg) |

## 18. `2409037` (GQA, LlamaV-o1-11B)

**Question:** Is the clock on the right side?
**Gold answer(s):** No, the clock is on the left of the image., no

| condition | model output |
|---|---|
| raw (clean image) | **No** ✅ |
| segmented (marked image) | **Yes** ❌ |
| gom_text (marked + IDs) | Yes |

*Mechanism:* spatial side flip for the weakest model

| original | preprocessed (what the model saw) |
|---|---|
| ![original](flip_examples/18_2409037_original.jpg) | ![marked](flip_examples/18_2409037_marked.jpg) |

## 19. `COCO_train2014_000000262171` (VQAV2, LlamaV-o1-11B)

**Question:** How many people are on the boat?
**Gold answer(s):** 5, 6, 7, 9

| condition | model output |
|---|---|
| raw (clean image) | **5** ✅ |
| segmented (marked image) | **4** ❌ |
| gom_text (marked + IDs) | 5 |

*Mechanism:* count dropped: one masked person not counted

| original | preprocessed (what the model saw) |
|---|---|
| ![original](flip_examples/19_COCO_train2014_000000262171_original.jpg) | ![marked](flip_examples/19_COCO_train2014_000000262171_marked.jpg) |

## 20. `COCO_train2014_000000025241` (VQAV1, LlamaV-o1-11B)

**Question:** What part of the body are these worn around?
**Gold answer(s):** around neck, man, neck

| condition | model output |
|---|---|
| raw (clean image) | **Neck** ✅ |
| segmented (marked image) | **Belt** ❌ |
| gom_text (marked + IDs) | waist |

*Mechanism:* worn-around-neck -> 'Belt': identity displaced by marks

| original | preprocessed (what the model saw) |
|---|---|
| ![original](flip_examples/20_COCO_train2014_000000025241_original.jpg) | ![marked](flip_examples/20_COCO_train2014_000000025241_marked.jpg) |
