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
DATA_ROOT=${GOM_PAPER_DATA:-$ROOT/reproduction/data}
FASTTEXT=${GOM_FASTTEXT:-$DATA_ROOT/cc.en.300.kv}
# Same default as reproduce.py, so both entry points share one cache.  The box
# this ran on originally used GOM_MODEL_CACHE=/llms.
MODEL_CACHE=${GOM_MODEL_CACHE:-$HOME/.cache/gom-paper}
MODELS="gemma3_4b qwen25_vl_7b llamav_o1_11b"
PROFILES=${GOM_PROFILES:-"paper_declared supplementary_concise"}
RENDER_PROFILE=${GOM_RENDER_PROFILE:-paper_aaai26}
DATASETS=gqa,vqav1,vqav2,refcocog
DATASET_ARCHIVE=${GOM_DATASET_ARCHIVE:-$ROOT/data_paper/gom_datasets.zip}
# Free VRAM each model needs before it is launched; see the inference section.
VRAM_GEMMA3_4B=${GOM_VRAM_GEMMA3_4B:-16000}
VRAM_QWEN25_VL_7B=${GOM_VRAM_QWEN25_VL_7B:-24000}
VRAM_LLAMAV_O1_11B=${GOM_VRAM_LLAMAV_O1_11B:-28000}
# sm_120 (Blackwell) needs the Qwen vision tower forced off xformers' FA3 kernel.
# Harmless elsewhere, but it costs throughput, so keep it opt-in.
BLACKWELL=${GOM_BLACKWELL:-0}

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
    local env_file=()
    [ -f "$ROOT/.env" ] && env_file=(--env-file "$ROOT/.env")
    docker run --rm \
        "${env_file[@]}" \
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
        --render-profile "$RENDER_PROFILE" \
        --no-build --resume "$@"
}

say "run $RUN_ID · model cache $MODEL_CACHE · logs $LOGS"

# --------------------------------------------------------------- host preflight
# Everything below takes hours.  Fail in seconds instead, naming the fix, so a
# fresh machine does not discover a missing archive after a 20-minute image build.
host_preflight() {
    local problems=()
    command -v docker  >/dev/null || problems+=("docker not on PATH")
    command -v nvidia-smi >/dev/null || problems+=("nvidia-smi not on PATH (NVIDIA container runtime required)")
    [ -f "$ROOT/.env" ] || problems+=("missing $ROOT/.env — copy .env.example and set HF_TOKEN (Gemma-3 is gated)")
    [ -f "$DATASET_ARCHIVE" ] || problems+=("missing $DATASET_ARCHIVE — copy it from the source machine or set GOM_DATASET_ARCHIVE")
    mkdir -p "$MODEL_CACHE" 2>/dev/null || problems+=("cannot create model cache $MODEL_CACHE")
    [ -w "$MODEL_CACHE" ] || problems+=("model cache $MODEL_CACHE is not writable")
    local free_gb
    free_gb=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc '0-9')
    [ "${free_gb:-0}" -ge 120 ] || problems+=("only ${free_gb}G free at $ROOT; ~120G needed (images, weights, artifacts)")
    if [ ${#problems[@]} -gt 0 ]; then
        local problem
        say "preflight failed:"
        for problem in "${problems[@]}"; do say "  - $problem"; done
        return 1
    fi
    say "gpu: $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader)"
}
# Called directly rather than through stage(), so the fix-it messages reach the
# terminal instead of a log file nobody is watching yet.
STAGE=host_preflight
host_preflight

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

# Idempotent: prepare_datasets.py re-verifies the archive hash and every installed
# image basename, and skips files already in place.
stage datasets reproduce datasets --dataset-archive "$DATASET_ARCHIVE"

# Algorithm 3's semantic half is inert without these, and preflight hard-requires
# both the .kv and its .vectors.npy companion.  ~4.3 GB download, once per machine.
if [ -f "$FASTTEXT" ] && [ -f "$FASTTEXT.vectors.npy" ]; then
    say "fasttext already converted at $FASTTEXT — skipping"
else
    stage fasttext docker_prefetch gom-paper-preprocess:1 \
        python3 reproduction/prepare_fasttext.py --download \
        "$DATA_ROOT/cc.en.300.vec" "$FASTTEXT"
fi

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
    stage preprocess reproduce preprocess --preprocess-workers "$WORKERS" --one-per-image
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
        # Per-model engine settings, each forced by a measured failure:
        #   qwen  - on sm_120 its ViT crashes on xformers' FA3 Hopper kernel, so the
        #           vision tower is pinned to TORCH_SDPA via the compat sitecustomize
        #           (GOM_BLACKWELL=1; unnecessary on Ampere/Ada).
        #   qwen/llamav - their worst-case multimodal profile run OOMs at vLLM's
        #           default max_num_seqs regardless of gpu_memory_utilization.
        # VRAM floors are the levels at which each model actually loaded here;
        # override per model with GOM_VRAM_<MODEL> on a differently sized GPU.
        extra=""
        case $model in
            gemma3_4b)
                need=$VRAM_GEMMA3_4B ;;
            qwen25_vl_7b)
                need=$VRAM_QWEN25_VL_7B
                extra="--max-num-seqs 8"
                if [ "$BLACKWELL" = 1 ]; then
                    extra="$extra
                        --container-env GOM_VIT_ATTN_BACKEND=TORCH_SDPA
                        --container-env PYTHONPATH=$ROOT/reproduction/compat:$ROOT/src"
                fi ;;
            llamav_o1_11b)
                need=$VRAM_LLAMAV_O1_11B
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
