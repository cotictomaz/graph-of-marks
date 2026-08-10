#!/usr/bin/env bash
# Unattended end-to-end driver for the AAAI-26 Table 2 reproduction.
#
# Rebuilds both Docker images, refetches every model into $MODEL_CACHE, verifies
# the pinned detector/segmenter/depth weights, finishes preprocessing, audits the
# graphs, runs inference for each model, and scores the result.
#
# Every stage is idempotent (--resume / cache hits), so re-running this script
# after a failure or a reboot picks up where it stopped.
#
#   nohup setsid reproduction/run_afk.sh > /dev/null 2>&1 &
#
set -euo pipefail

ROOT=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
DATA_ROOT=$ROOT/reproduction/data
FASTTEXT=$DATA_ROOT/cc.en.300.kv
MODEL_CACHE=${GOM_MODEL_CACHE:-/llms}
MODELS="gemma3_4b qwen25_vl_7b llamav_o1_11b"
PROFILES="paper_declared supplementary_concise"
DATASETS=gqa,vqav1,vqav2,refcocog

RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)
LOGS=$ROOT/reproduction/afk_runs/$RUN_ID
mkdir -p "$LOGS"
ln -sfn "$LOGS" "$ROOT/reproduction/afk_runs/latest"

STAGE=startup
trap 'code=$?; [ $code -eq 0 ] || say "FAILED in stage \"$STAGE\" (exit $code) — see $LOGS/$STAGE.log"' EXIT

say() { printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$LOGS/driver.log"; }

# Run one stage, streaming to its own log.  Fails the script on a non-zero exit.
stage() {
    STAGE=$1
    shift
    say "=== $STAGE: $* "
    "$@" > "$LOGS/$STAGE.log" 2>&1
    say "=== $STAGE: done"
}

# Like stage(), but a failure is reported and tolerated.  One model that cannot
# load must not cost us the models that can, nor the scoring of what completed.
stage_soft() {
    STAGE=$1
    shift
    say "=== $STAGE: $* "
    if "$@" > "$LOGS/$STAGE.log" 2>&1; then
        say "=== $STAGE: done"
        return 0
    fi
    say "=== $STAGE: FAILED (continuing) — see $LOGS/$STAGE.log"
    return 1
}

vram_free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
vram_total() { nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1; }

# Block until the GPU has room.  The blocker is usually another user's job, so the
# wait is deliberately unbounded: we are AFK and would rather wait than OOM.
wait_for_vram() {
    local need=$1 label=$2 free
    while :; do
        free=$(vram_free)
        if [ "$free" -ge "$need" ]; then
            say "$label: ${free} MiB free >= ${need} MiB needed — starting"
            VRAM_FREE=$free
            return 0
        fi
        say "$label: waiting for VRAM (${free} MiB free < ${need} MiB needed)"
        sleep 300
    done
}

docker_prefetch() {
    local image=$1
    shift
    docker run --rm \
        --env-file "$ROOT/.env" \
        -v "$ROOT:$ROOT" -v "$MODEL_CACHE:/model-cache" \
        -w "$ROOT" \
        -e PYTHONPATH="$ROOT/src" -e PYTHONUNBUFFERED=1 \
        -e HF_HOME=/model-cache \
        -e TORCH_HOME=/model-cache/torch_cache \
        -e FVCORE_CACHE=/model-cache/torch_cache/iopath_cache \
        "$image" "$@"
}

reproduce() {
    local command=$1
    shift
    python3 "$ROOT/reproduction/reproduce.py" "$command" \
        --data-root "$DATA_ROOT" \
        --datasets "$DATASETS" \
        --fasttext "$FASTTEXT" \
        --model-cache "$MODEL_CACHE" \
        --artifact-granularity image \
        --no-build --resume "$@"
}

say "run $RUN_ID · model cache $MODEL_CACHE · logs $LOGS"

# ---------------------------------------------------------------- environment
stage build_preprocess docker build -f "$ROOT/reproduction/docker/preprocess.Dockerfile" \
    -t gom-paper-preprocess:1 "$ROOT"
stage build_inference docker build -f "$ROOT/reproduction/docker/inference.Dockerfile" \
    -t gom-paper-inference:1 "$ROOT"

stage fetch_preprocess_models docker_prefetch gom-paper-preprocess:1 \
    python3 reproduction/prefetch_models.py --stage preprocess
stage fetch_inference_models docker_prefetch gom-paper-inference:1 \
    python3 reproduction/prefetch_models.py --stage inference

stage verify_weights docker_prefetch gom-paper-preprocess:1 \
    python3 reproduction/verify_weights.py --cache /model-cache \
    --output "$DATA_ROOT/artifacts/preprocessing_weights.json"

# ------------------------------------------------------------- preprocessing
# Two workers each load a full model stack; only spawn the second one when the
# GPU has obvious headroom.
# A restart should not reload the detector stack onto a GPU we are rationing just
# to re-confirm 4,000 existing artifacts.
preprocessing_complete() {
    local dataset expected count
    for dataset in $(echo "$DATASETS" | tr ',' ' '); do
        expected=$(python3 -c "import json,sys;print(len(json.load(open(sys.argv[1]))))" \
            "$DATA_ROOT/prepared/$dataset/preproc_input.json" 2>/dev/null) || return 1
        count=$(ls "$DATA_ROOT/artifacts/$dataset/preprocessing"/*_graph.json 2>/dev/null | wc -l)
        [ "$count" -ge "$expected" ] || return 1
    done
    return 0
}

if preprocessing_complete; then
    say "preprocessing already complete for every dataset — skipping"
else
    WORKERS=1
    if [ "$(vram_free)" -ge 24000 ]; then WORKERS=2; fi
    say "preprocessing with $WORKERS worker(s) ($(vram_free) MiB free)"
    stage preprocess reproduce preprocess --preprocess-workers "$WORKERS"
fi

stage audit reproduce audit

# ---------------------------------------------------------------- inference
# Thresholds are weights + vLLM's multimodal profile run + KV cache, not weights
# alone: gemma3_4b OOM'd at 12.2 GB free with 11.26 GB already resident, dying in
# the SigLIP profile pass.  These are the floors at which each model actually loads.
# Two prompt profiles, scored separately and never pooled:
#   paper_declared       - the supplementary visual-SG prompt verbatim.  Only the
#                          raw condition carries a short-answer instruction, so the
#                          marked conditions generate sentences and score ~0 under
#                          exact match; read these with the phrase-compatibility metric.
#   supplementary_concise - the same prompts with the short-answer instruction on
#                          every condition, making the official exact-match metric
#                          meaningful for raw and marked arms alike.
TOTAL=$(vram_total)
for profile in $PROFILES; do
    done_models=""
    for model in $MODELS; do
        # Per-model engine settings, each forced by a measured failure on this
        # Blackwell (sm_120) GPU -- see reproduction/compat/sitecustomize.py:
        #   qwen  - its ViT crashes on xformers' FA3 Hopper kernel, so the vision
        #           tower is pinned to TORCH_SDPA via the compat sitecustomize.
        #   qwen/llamav - their worst-case multimodal profile run OOMs at vLLM's
        #           default max_num_seqs regardless of gpu_memory_utilization.
        extra=""
        case $model in
            gemma3_4b)
                need=16000 ;;
            qwen25_vl_7b)
                need=24000
                extra="--max-num-seqs 8
                    --container-env GOM_VIT_ATTN_BACKEND=TORCH_SDPA
                    --container-env PYTHONPATH=$ROOT/reproduction/compat:$ROOT/src" ;;
            llamav_o1_11b)
                need=28000
                # Mllama's chat template rejects a system role next to an image.
                extra="--max-num-seqs 8 --fold-system-into-user" ;;
        esac
        wait_for_vram "$need" "$profile/$model"
        util=$(awk -v f="$VRAM_FREE" -v t="$TOTAL" \
            'BEGIN { u = (f - 1500) / t; if (u > 0.90) u = 0.90; printf "%.2f", u }')
        batch=256
        if [ "$VRAM_FREE" -lt 20000 ]; then batch=64; fi
        say "$profile/$model: gpu_memory_utilization=$util batch=$batch"
        # shellcheck disable=SC2086  # $extra is a deliberate word-split flag list
        if stage_soft "inference_${profile}_${model}" reproduce inference \
            --models "$model" --prompt-profile "$profile" \
            --one-per-image --single-setting \
            --gpu-memory-utilization "$util" --inference-batch-size "$batch" \
            $extra; then
            done_models="$done_models $model"
        fi
    done

    if [ -z "$done_models" ]; then
        say "$profile: no model completed inference — nothing to score"
        continue
    fi
    say "scoring $profile:$done_models"
    stage_soft "score_$profile" reproduce score --prompt-profile "$profile" \
        --models "$(echo $done_models | tr -s ' ' | sed 's/^ //;s/ /,/g')" \
        --one-per-image --single-setting || true
done

STAGE=complete
say "ALL DONE — reports at $DATA_ROOT/table2_report.<profile>.md"
