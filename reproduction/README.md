# AAAI 2026 Table 2 Reproduction

This directory is the canonical, fail-closed reconstruction path. It implements the
published Algorithms 1-4, produces all six marked-image variants from one graph, runs the
three paper models over the Cartesian 27-setting decode grid, and scores each dataset with
its declared protocol. Exact image membership is enforced. Query provenance and known
paper/data conflicts are reported rather than silently treated as exact.

## Fresh Machine Setup

Host prerequisites: Docker with the NVIDIA container runtime (`--gpus all`),
`nvidia-smi`, python3 on the host (the harness itself is stdlib-only), and about
120 GB free — ~40 GB of Docker images, ~44 GB of model weights, ~7 GB of FastText
vectors, and the data root.

Exactly one artifact cannot be fetched from the internet and must be copied from a
machine that already has it:

```
data_paper/gom_datasets.zip     605 MB
sha256 a9c0f446ed4d99bcb7e00cbc3cd686d9fe19149ad3a1015a379e05569992f404
```

It holds the four exact 1,000-image author splits. `GOM_DATASET_ARCHIVE` overrides
the path. Everything else — Docker images, detector/segmenter/depth weights, the
three VLMs, the FastText vectors, the annotations — is rebuilt or downloaded.

```bash
cp /path/to/gom_datasets.zip data_paper/     # the one manual step
cp .env.example .env                         # then set HF_TOKEN (Gemma-3 is gated)
reproduction/run_afk.sh                      # or nohup setsid ... & to go AFK
```

`run_afk.sh` preflights the host, builds both images, downloads every model into
`$MODEL_CACHE`, SHA-256-verifies the four pinned weights, installs the image
splits, converts FastText, then preprocesses, audits, infers and scores. Every
stage is idempotent, so re-running after a failure resumes. Logs land in
`reproduction/afk_runs/<timestamp>/`.

To drive the stages yourself instead, see **One Command** below; `run_afk.sh` is a
wrapper over the same `reproduce.py` entry points.

### Knobs

| variable | default | why |
|---|---|---|
| `GOM_MODEL_CACHE` | `~/.cache/gom-paper` | mounted at `/model-cache`; holds all HF/torch.hub downloads |
| `GOM_DATASET_ARCHIVE` | `data_paper/gom_datasets.zip` | the hand-carried archive |
| `GOM_BLACKWELL` | `0` | set to `1` on sm_120 (RTX 5090 etc.) — pins the Qwen2.5-VL vision tower to `TORCH_SDPA`, working around an xformers FA3 crash. Unnecessary on Ampere/Ada. |
| `GOM_VRAM_GEMMA3_4B` | `16000` | free-VRAM floor in MiB before the model is launched |
| `GOM_VRAM_QWEN25_VL_7B` | `24000` | |
| `GOM_VRAM_LLAMAV_O1_11B` | `28000` | |
| `GOM_PROFILES` | `paper_declared supplementary_concise` | prompt profiles to run |

The floors are the levels at which each model actually loaded on the original box
(weights + vLLM's multimodal profile run + KV cache, not weights alone). Lower them
on a smaller GPU and expect to also lower `--inference-batch-size`.

The cu128 images run unchanged on Ampere and Ada; `GOM_BLACKWELL` and the VRAM
floors are the only GPU-specific settings.

### If a weight hash mismatches

`weights.yaml` pins SHA-256s for `yolov8x.pt`, `sam_hq_vit_h.pth`,
`model_final_a3ec72.pkl` and `dpt_large_384.pt`. `ultralytics` and the detectron2
and MiDaS revisions are version-pinned, so those downloads are deterministic, but
SAM-HQ is fetched from the mutable `lkeab/hq-sam` HF main branch. If `verify_weights`
reports a mismatch, that is the check working — copy `checkpoints/sam_hq_vit_h.pth`
from the source machine rather than re-registering the new hash.

## Data Provenance

All four exact 1,000-image author subsets are stored in `data_paper/gom_datasets.zip`
(SHA-256 `a9c0f446ed4d99bcb7e00cbc3cd686d9fe19149ad3a1015a379e05569992f404`).
The image splits and query manifests have separate provenance in
`reproduction/manifests.yaml`. VQAv1's original 3,000-row manifest was recovered from the
pre-refactor author archive, and VQAv2 has the released author manifest. GQA is the exact
ZIP image set joined to every associated row in the released 943,000-row balanced pool.
RefCOCOg retains one prompt per image and all associated target instances reconstructed
from official annotations.

The recovered data exposes paper inconsistencies. VQAv1 has exactly 3 queries per image,
although the paper reports 4; VQAv2 has 5.18 rather than 4. GQA has 15,334 canonical
queries (15.334 per image) when all questions associated with the author images are
retained, although the paper reports an average of 3. RefCOCOg's author query selection was not released. These facts
are carried into every generated provenance and score report; exact image membership is
never presented as proof of an exact undocumented query protocol.

Install only these subsets, without retaining full upstream datasets, with:

```bash
./reproduce.sh datasets --data-root /path/to/gom-paper-data
```

The installer rejects any archive, manifest, count, or basename mismatch and records a
content hash for each installed image collection.

## One Command

```bash
./reproduce.sh table2 \
  --data-root /path/to/gom-paper-data \
  --fasttext /path/to/cc.en.300.kv \
  --model-cache /path/to/persistent/model-cache \
  --preprocess-workers 2 \
  --resume
```

`table2` installs and verifies the paper image split before preprocessing. To run only
VQAv2, replace `table2` with `vqav2`.
Use `--resume` for an interrupted run and `--no-build` after the two Docker images exist.
On a GPU with at least 24 GB VRAM, `--preprocess-workers 2` runs disjoint image chunks
concurrently; one worker is the conservative default for smaller GPUs.
Inference submits 256 multimodal requests per vLLM call by default, matching the
previous completed VQAv2 grid; override `--inference-batch-size` if needed.
`./reproduce.sh plan` prints the matrix, provenance levels, and exact generation count
without loading models.
The model cache persists Hugging Face, MiDaS, and detector downloads across container runs;
it defaults to `~/.cache/gom-paper` and can also be set with `GOM_MODEL_CACHE`.
`GOM_PAPER_DATA` and `GOM_FASTTEXT` provide environment defaults for `--data-root`
and `--fasttext`, allowing subsequent runs to use simply `./reproduce.sh vqav2 --resume`.

The command performs, in order:

1. Exact-manifest hash/count validation and image validation.
2. Canonical duplicate removal with stable question IDs and provenance output.
3. Image-level `paper_aaai26` preprocessing with SAM-HQ, MiDaS DPT-Large, Algorithms 2/3, and all six renders. Repeated questions reuse the graph for their image, matching the paper's one-render-per-image contract; use `--artifact-granularity question` only for a question-conditioned ablation.
4. SHA-256 verification of the four detector/segmenter/depth weight files.
5. Graph/triple/render edge digest audit.
6. Model-revision-locked inference for 27 seed/temperature/top-p combinations.
7. Official VQA consensus, normalized GQA exact match, or RefCOCOg region IoU scoring.
   VQA reports also include the released code's lowercase exact-match score as a clearly
   labeled compatibility metric.

Results are written below `<data-root>/table2_report.{json,md}`. Every prediction has a
sidecar containing the dataset hash, model revision, prompt profile, decode parameters,
code commit, and render/edge provenance.

See `RESULTS.md` for the exact-split VQAv2 results already verified in this workspace and
for the incompatible released-artifact scores. They are labeled separately from the fresh
`paper_declared` Table 2 run.

## FastText

Algorithm 3 requires the paper's `cc.en.300.vec` vectors. Convert the decompressed official
file once:

```bash
docker run --rm -v "$PWD:$PWD" -w "$PWD" gom-paper-preprocess:1 \
  python3 reproduction/prepare_fasttext.py \
  /path/to/cc.en.300.vec /path/to/cc.en.300.kv
```

Add `--download` to fetch `cc.en.300.vec.gz` from
`dl.fbaipublicfiles.com/fasttext/vectors-crawl/` and decompress it first when the
source file is absent (~1.3 GB compressed, 4.3 GB expanded). `run_afk.sh` does this
automatically when the `.kv` pair is missing.

The `.kv` output has a companion NumPy file created by gensim; keep both together.
The reproduction command requires both files and records their SHA-256 hashes in
`<data-root>/artifacts/fasttext.json`.

## Published Spec vs Released Artifact

The primary profile is `paper_declared`, which uses the exact supplementary visual-SG
prompt without appending a short-answer instruction. The conflicting released VQAv2 path
is retained as `released_artifact_bare`. Run it separately with:

```bash
./reproduce.sh inference --datasets vqav2 \
  --prompt-profile released_artifact_bare --data-root /path/to/gom-paper-data
```

Do not average or compare those generations as though they came from the same protocol.
The released evaluator also conflicts with the paper's dataset-creator-metric statement:
it lowercases and exactly compares against one stored answer instead of computing VQA
annotator consensus. The reconstruction reports both metrics without substituting one for
the other.

The locked profile includes the released implementation's per-class NMS at IoU 0.5 after
WBF. That step is absent from Algorithm 1's pseudocode but is necessary to prevent the three
detectors from turning one physical object into several query matches. This implementation
detail is recorded separately in `paper_spec.yaml` rather than presented as published text.

The profile also records one deliberate correction: after SAM-HQ it removes same-class
detections when the boxes strongly overlap or the masks identify the same physical instance.
Cross-class suppression is disabled. The released box-only NMS leaves nested duplicate boxes
in the giraffe/dog example, causing Algorithm 3 to retain two dog marks and discard the
giraffe. This correction is reported separately from the published pseudocode in every run.

Relation-label placement is collision-checked with a bounded 50-pixel search budget. The
graph JSON, triples text, and every render variant carry the same edge digest; the audit
compares their edge multisets so serializer traversal order cannot create a false failure.

## Manifest Schema

VQAv1/VQAv2 rows require `image_path`, `question`, and exactly ten `answers`. GQA rows
require `image_path`, `question`, and `answer`. RefCOCOg rows contain one image and a
`targets` list; every target requires `description` and `bbox_xywh`. Image paths in manifests
are reduced to basenames and resolved under the dataset image directory, preventing hidden
local paths from changing the sample.
