#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash run_variable_length_pipeline.sh --dataset DATASET [options]

Starts after item embeddings have been generated. It trains or reuses an
RQ-VAE tokenizer, creates a fixed index, converts it to a collision-free
variable-length index, then trains and evaluates selected recommenders.

Required:
  --dataset NAME

Options:
  --data-root PATH             Dataset directory parent (default: <repo>/data)
  --embedding-file PATH        Item embedding .npy path
  --cf-embedding PATH          Collaborative-filtering embedding .pt path
  --rqvae-checkpoint PATH      Reuse an existing RQ-VAE checkpoint (autodetected if omitted)
  --retrain-rqvae              Force training RQ-VAE even if a checkpoint exists
  --rqvae-epochs COUNT         RQ-VAE epochs (default: 10000)
  --rqvae-eval-step COUNT      RQ-VAE validation interval (default: 2000)
  --rqvae-device DEVICE        RQ-VAE device (default: cuda:0)
  --alpha VALUE                Collaborative-loss weight (default: 0.01)
  --beta VALUE                 Diversity-loss weight (default: 0.0001)
  --min-length COUNT           Minimum SID length (default: 1)
  --max-length COUNT           Maximum SID length (default: 4)
  --index-name NAME            Variable index filename (default: <dataset>.index.varlen.json)
  --overwrite-index            Replace an existing variable index
  --tokenizer-only             Stop after variable-length index generation
  --models LIST                Comma-separated: tiger,lcrec (default: tiger)
  --base-model PATH            Base model for LC-Rec (default: huggyllama/llama-7b)
  --tiger-gpus IDS             CUDA devices for TIGER (default: autodetect, up to 2)
  --lcrec-gpus IDS             CUDA devices for LC-Rec (default: autodetect, up to 4)
  --skip-evaluation            Train selected recommenders without evaluation
  --python PATH                Python executable (default: python3)
  -h, --help                   Show this help
EOF
}

format_duration() {
  local total_seconds="$1"
  printf '%02dh %02dm %02ds' \
    "$((total_seconds / 3600))" \
    "$(((total_seconds % 3600) / 60))" \
    "$((total_seconds % 60))"
}

PHASE_NAMES=()
PHASE_DURATIONS=()

record_phase() {
  PHASE_NAMES+=("$1")
  PHASE_DURATIONS+=("$2")
}

print_phase_durations() {
  local index
  printf '\nPhase durations:\n'
  for index in "${!PHASE_NAMES[@]}"; do
    printf '  - %s: %s\n' "${PHASE_NAMES[$index]}" \
      "$(format_duration "${PHASE_DURATIONS[$index]}")"
  done
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CALLER_DIR="$(pwd)"
PIPELINE_START="$SECONDS"
DATASET=""
DATA_ROOT="$REPO_ROOT/data"
EMBEDDING_FILE=""
CF_EMBEDDING=""
RQ_CHECKPOINT=""
RETRAIN_RQVAE=false
RQ_EPOCHS="10000"
RQ_EVAL_STEP="2000"
RQ_DEVICE="cuda:0"
ALPHA="0.01"
BETA="0.0001"
MIN_LENGTH="1"
MAX_LENGTH="4"
INDEX_NAME=""
MODELS="tiger"
BASE_MODEL="${BASE_MODEL:-huggyllama/llama-7b}"
TIGER_GPUS=""
LCREC_GPUS=""
SKIP_EVALUATION=false
OVERWRITE_INDEX=false
TOKENIZER_ONLY=false
PYTHON_BIN="${PYTHON_BIN:-python3}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --embedding-file) EMBEDDING_FILE="$2"; shift 2 ;;
    --cf-embedding) CF_EMBEDDING="$2"; shift 2 ;;
    --rqvae-checkpoint) RQ_CHECKPOINT="$2"; shift 2 ;;
    --retrain-rqvae) RETRAIN_RQVAE=true; shift ;;
    --rqvae-epochs) RQ_EPOCHS="$2"; shift 2 ;;
    --rqvae-eval-step) RQ_EVAL_STEP="$2"; shift 2 ;;
    --rqvae-device) RQ_DEVICE="$2"; shift 2 ;;
    --alpha) ALPHA="$2"; shift 2 ;;
    --beta) BETA="$2"; shift 2 ;;
    --min-length) MIN_LENGTH="$2"; shift 2 ;;
    --max-length) MAX_LENGTH="$2"; shift 2 ;;
    --index-name) INDEX_NAME="$2"; shift 2 ;;
    --models) MODELS="$2"; shift 2 ;;
    --base-model) BASE_MODEL="$2"; shift 2 ;;
    --tiger-gpus) TIGER_GPUS="$2"; shift 2 ;;
    --lcrec-gpus) LCREC_GPUS="$2"; shift 2 ;;
    --skip-evaluation) SKIP_EVALUATION=true; shift ;;
    --overwrite-index) OVERWRITE_INDEX=true; shift ;;
    --tokenizer-only) TOKENIZER_ONLY=true; shift ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$DATASET" ]]; then
  printf '%s\n' '--dataset is required.' >&2
  usage >&2
  exit 2
fi
if [[ "$DATA_ROOT" != /* ]]; then
  DATA_ROOT="$CALLER_DIR/$DATA_ROOT"
fi
if [[ -n "$EMBEDDING_FILE" && "$EMBEDDING_FILE" != /* ]]; then
  EMBEDDING_FILE="$CALLER_DIR/$EMBEDDING_FILE"
fi
if [[ -n "$CF_EMBEDDING" && "$CF_EMBEDDING" != /* ]]; then
  CF_EMBEDDING="$CALLER_DIR/$CF_EMBEDDING"
fi
if [[ -n "$RQ_CHECKPOINT" && "$RQ_CHECKPOINT" != /* ]]; then
  RQ_CHECKPOINT="$CALLER_DIR/$RQ_CHECKPOINT"
fi
if [[ -n "$BASE_MODEL" && -d "$CALLER_DIR/$BASE_MODEL" ]]; then
  BASE_MODEL="$CALLER_DIR/$BASE_MODEL"
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  printf 'Python executable not found: %s\n' "$PYTHON_BIN" >&2
  exit 1
fi

INDEX_NAME="${INDEX_NAME:-$DATASET.index.varlen.json}"
INDEX_FILE="$DATA_ROOT/$DATASET/$INDEX_NAME"
INDEX_SUFFIX="${INDEX_NAME#"$DATASET"}"
FIXED_INDEX_NAME="$DATASET.index.fixed-for-varlen.json"
FIXED_INDEX_FILE="$DATA_ROOT/$DATASET/$FIXED_INDEX_NAME"

if [[ "$INDEX_SUFFIX" == "$INDEX_NAME" || "$INDEX_SUFFIX" != *.json ]]; then
  printf 'Index name must start with %s and end with .json: %s\n' "$DATASET" "$INDEX_NAME" >&2
  exit 2
fi
if [[ "$TOKENIZER_ONLY" != true ]]; then
  contains_model() {
    [[ ",$MODELS," == *",$1,"* ]]
  }

  if ! contains_model tiger && ! contains_model lcrec; then
    printf 'Select at least one supported model: tiger,lcrec\n' >&2
    exit 2
  fi
  if contains_model lcrec && [[ -z "$BASE_MODEL" ]]; then
    printf '%s\n' '--base-model is required when selecting lcrec.' >&2
    exit 2
  fi
else
  contains_model() {
    false
  }
fi

gpu_count() {
  awk -F',' '{ print NF }' <<<"$1"
}

detect_available_gpus() {
  local max_count="$1"
  "$PYTHON_BIN" -c "
import torch
count = torch.cuda.device_count()
if count > 0:
    limit = min(count, $max_count)
    print(','.join(str(i) for i in range(limit)))
else:
    print('0')
" 2>/dev/null || echo "0"
}

if [[ -z "$TIGER_GPUS" ]]; then
  TIGER_GPUS="$(detect_available_gpus 2)"
fi
if [[ -z "$LCREC_GPUS" ]]; then
  LCREC_GPUS="$(detect_available_gpus 4)"
fi

RQ_CHECKPOINT_ROOT="$REPO_ROOT/checkpoint/$DATASET"

find_latest_checkpoint() {
  local root="$1"
  if [[ ! -d "$root" ]]; then
    return 0
  fi
  "$PYTHON_BIN" -c '
import sys, glob, os
root = sys.argv[1]
candidates = glob.glob(os.path.join(root, "**", "best_collision_model.pth"), recursive=True)
if not candidates:
    candidates = glob.glob(os.path.join(root, "**", "best_loss_model.pth"), recursive=True)
if candidates:
    print(max(candidates, key=os.path.getmtime))
' "$root" 2>/dev/null || true
}

if [[ -z "$RQ_CHECKPOINT" && "$RETRAIN_RQVAE" != true ]]; then
  DETECTED_CKPT="$(find_latest_checkpoint "$RQ_CHECKPOINT_ROOT")"
  if [[ -n "$DETECTED_CKPT" && -f "$DETECTED_CKPT" ]]; then
    RQ_CHECKPOINT="$DETECTED_CKPT"
    printf '\n[RQ-VAE] Autodetected existing checkpoint: %s\n' "$RQ_CHECKPOINT"
    printf '[RQ-VAE] Reusing existing checkpoint (pass --retrain-rqvae to force training).\n'
  fi
fi

if [[ -f "$INDEX_FILE" && "$OVERWRITE_INDEX" != true ]]; then
  printf '\n[Variable index] Found existing variable-length index: %s\n' "$INDEX_FILE"
  printf '[Variable index] Reusing existing index (pass --overwrite-index to regenerate).\n'
else
  fixed_args=(
    --dataset "$DATASET"
    --data-root "$DATA_ROOT"
    --rqvae-epochs "$RQ_EPOCHS"
    --rqvae-eval-step "$RQ_EVAL_STEP"
    --rqvae-device "$RQ_DEVICE"
    --alpha "$ALPHA"
    --beta "$BETA"
    --index-name "$FIXED_INDEX_NAME"
    --models tiger
    --tokenizer-only
    --python "$PYTHON_BIN"
  )
  if [[ "$OVERWRITE_INDEX" == true ]]; then
    fixed_args+=(--overwrite-index)
  fi
  if [[ "$RETRAIN_RQVAE" == true ]]; then
    fixed_args+=(--retrain-rqvae)
  fi
  if [[ -n "$EMBEDDING_FILE" ]]; then
    fixed_args+=(--embedding-file "$EMBEDDING_FILE")
  fi
  if [[ -n "$CF_EMBEDDING" ]]; then
    fixed_args+=(--cf-embedding "$CF_EMBEDDING")
  fi
  if [[ -n "$RQ_CHECKPOINT" ]]; then
    fixed_args+=(--rqvae-checkpoint "$RQ_CHECKPOINT")
  fi

  if [[ -f "$FIXED_INDEX_FILE" && "$OVERWRITE_INDEX" != true && "$RETRAIN_RQVAE" != true ]]; then
    printf '\n[Fixed index] Found existing intermediate index: %s\n' "$FIXED_INDEX_FILE"
    printf '[Fixed index] Reusing existing fixed index (pass --overwrite-index to regenerate).\n'
  else
    printf '\n[Fixed index] Training or reusing RQ-VAE and generating a fixed index...\n'
    STEP_START="$SECONDS"
    bash "$REPO_ROOT/run_fixed_length_pipeline.sh" "${fixed_args[@]}"
    printf 'Completed fixed-index preparation in %s.\n' "$(format_duration "$((SECONDS - STEP_START))")"
    record_phase "Fixed-index preparation" "$((SECONDS - STEP_START))"
    printf 'Stored intermediate fixed-length index: %s\n' "$FIXED_INDEX_FILE"
  fi

  printf '\n[Variable index] Creating the variable-length item index...\n'
  STEP_START="$SECONDS"
  "$PYTHON_BIN" "$REPO_ROOT/RQ-VAE/truncate_indices.py" \
    --input "$FIXED_INDEX_FILE" \
    --output "$INDEX_FILE" \
    --min-length "$MIN_LENGTH" \
    --max-length "$MAX_LENGTH"

  if [[ ! -f "$INDEX_FILE" ]]; then
    printf 'Variable-length index generation completed without creating: %s\n' "$INDEX_FILE" >&2
    exit 1
  fi
  printf 'Completed variable-length index generation in %s.\n' \
    "$(format_duration "$((SECONDS - STEP_START))")"
  record_phase "Variable-length index generation" "$((SECONDS - STEP_START))"
  printf 'Stored variable-length item index: %s\n' "$INDEX_FILE"
  printf 'Stored variable-length index summary: %s\n' "${INDEX_FILE%.json}.summary.json"
fi

if [[ "$TOKENIZER_ONLY" == true ]]; then
  print_phase_durations
  printf '\nTokenizer and variable-length index pipeline completed in %s.\n' \
    "$(format_duration "$((SECONDS - PIPELINE_START))")"
  exit 0
fi

if contains_model tiger; then
  printf '\n[TIGER] Training with variable-length IDs...\n'
  STEP_START="$SECONDS"
  mkdir -p "$REPO_ROOT/LETTER-TIGER/results/$DATASET"
  (
    cd "$REPO_ROOT/LETTER-TIGER"
    CUDA_VISIBLE_DEVICES="$TIGER_GPUS" torchrun \
      --nproc_per_node="$(gpu_count "$TIGER_GPUS")" \
      --master_port=2314 \
      finetune.py \
      --output_dir "./ckpt/$DATASET-varlen" \
      --dataset "$DATASET" \
      --data_path "$DATA_ROOT" \
      --per_device_batch_size 256 \
      --learning_rate 5e-4 \
      --epochs 200 \
      --index_file "$INDEX_SUFFIX" \
      --temperature 1.0
  )
  printf 'Completed variable-length LETTER-TIGER training in %s.\n' \
    "$(format_duration "$((SECONDS - STEP_START))")"
  record_phase "Variable-length LETTER-TIGER training" "$((SECONDS - STEP_START))"
  printf 'Stored LETTER-TIGER checkpoint: %s\n' \
    "$REPO_ROOT/LETTER-TIGER/ckpt/$DATASET-varlen"

  if [[ "$SKIP_EVALUATION" != true ]]; then
    printf '\n[TIGER] Evaluating variable-length IDs...\n'
    STEP_START="$SECONDS"
    (
      cd "$REPO_ROOT/LETTER-TIGER"
      "$PYTHON_BIN" test.py \
        --gpu_id 0 \
        --ckpt_path "./ckpt/$DATASET-varlen" \
        --dataset "$DATASET" \
        --data_path "$DATA_ROOT" \
        --results_file "./results/$DATASET/varlen.json" \
        --test_batch_size 32 \
        --num_beams 20 \
        --test_prompt_ids 0 \
        --index_file "$INDEX_SUFFIX"
    )
    printf 'Completed variable-length LETTER-TIGER evaluation in %s.\n' \
      "$(format_duration "$((SECONDS - STEP_START))")"
    record_phase "Variable-length LETTER-TIGER evaluation" "$((SECONDS - STEP_START))"
    printf 'Stored LETTER-TIGER metrics: %s\n' \
      "$REPO_ROOT/LETTER-TIGER/results/$DATASET/varlen.json"
  fi
fi

if contains_model lcrec; then
  printf '\n[LC-Rec] Training with variable-length IDs...\n'
  STEP_START="$SECONDS"
  mkdir -p "$REPO_ROOT/LETTER-LC-Rec/results/$DATASET"
  (
    cd "$REPO_ROOT/LETTER-LC-Rec"
    CUDA_VISIBLE_DEVICES="$LCREC_GPUS" torchrun \
      --nproc_per_node="$(gpu_count "$LCREC_GPUS")" \
      --master_port=3325 \
      lora_finetune.py \
      --base_model "$BASE_MODEL" \
      --output_dir "./ckpt/$DATASET-varlen" \
      --dataset "$DATASET" \
      --data_path "$DATA_ROOT" \
      --per_device_batch_size 16 \
      --learning_rate 1e-4 \
      --epochs 4 \
      --tasks seqrec \
      --train_prompt_sample_num 1 \
      --train_data_sample_num 0 \
      --index_file "$INDEX_SUFFIX" \
      --wandb_run_name "${DATASET}-varlen" \
      --temperature 1.0
  )
  printf 'Completed variable-length LETTER-LC-Rec training in %s.\n' \
    "$(format_duration "$((SECONDS - STEP_START))")"
  record_phase "Variable-length LETTER-LC-Rec training" "$((SECONDS - STEP_START))"
  printf 'Stored LETTER-LC-Rec checkpoint: %s\n' \
    "$REPO_ROOT/LETTER-LC-Rec/ckpt/$DATASET-varlen"

  if [[ "$SKIP_EVALUATION" != true ]]; then
    printf '\n[LC-Rec] Evaluating variable-length IDs...\n'
    STEP_START="$SECONDS"
    (
      cd "$REPO_ROOT/LETTER-LC-Rec"
      CUDA_VISIBLE_DEVICES="$LCREC_GPUS" torchrun \
        --nproc_per_node="$(gpu_count "$LCREC_GPUS")" \
        --master_port=4324 \
        test_ddp.py \
        --ckpt_path "./ckpt/$DATASET-varlen" \
        --base_model "$BASE_MODEL" \
        --dataset "$DATASET" \
        --data_path "$DATA_ROOT" \
        --results_file "./results/$DATASET/varlen.json" \
        --test_batch_size 1 \
        --num_beams 20 \
        --test_prompt_ids 0 \
        --index_file "$INDEX_SUFFIX"
    )
    printf 'Completed variable-length LETTER-LC-Rec evaluation in %s.\n' \
      "$(format_duration "$((SECONDS - STEP_START))")"
    record_phase "Variable-length LETTER-LC-Rec evaluation" "$((SECONDS - STEP_START))"
    printf 'Stored LETTER-LC-Rec metrics: %s\n' \
      "$REPO_ROOT/LETTER-LC-Rec/results/$DATASET/varlen.json"
  fi
fi

print_phase_durations
printf '\nVariable-length pipeline completed in %s.\n' \
  "$(format_duration "$((SECONDS - PIPELINE_START))")"
