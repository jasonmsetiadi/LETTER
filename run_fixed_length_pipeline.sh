#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash run_fixed_length_pipeline.sh --dataset DATASET [options]

Starts after item embeddings have been generated. It trains the fixed four-code
RQ-VAE tokenizer, writes a new index file, then trains and evaluates selected
downstream recommenders.

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
  --index-name NAME            Generated index filename (default: <dataset>.index.fixed.json)
  --overwrite-index            Replace an existing generated index
  --tokenizer-only             Stop after fixed-length index generation
  --models LIST                Comma-separated: tiger,lcrec (default: tiger)
  --base-model PATH            Required when selecting lcrec
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
INDEX_NAME=""
MODELS="tiger"
BASE_MODEL=""
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
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  printf 'Python executable not found: %s\n' "$PYTHON_BIN" >&2
  exit 1
fi

EMBEDDING_FILE="${EMBEDDING_FILE:-$DATA_ROOT/$DATASET/$DATASET.emb-flan-t5-xl-td.npy}"
CF_EMBEDDING="${CF_EMBEDDING:-$REPO_ROOT/RQ-VAE/ckpt/$DATASET-32d-sasrec.pt}"
INDEX_NAME="${INDEX_NAME:-$DATASET.index.fixed.json}"
INDEX_FILE="$DATA_ROOT/$DATASET/$INDEX_NAME"
INDEX_SUFFIX="${INDEX_NAME#"$DATASET"}"
RQ_CHECKPOINT_ROOT="$REPO_ROOT/checkpoint/$DATASET"

if [[ "$INDEX_SUFFIX" == "$INDEX_NAME" || "$INDEX_SUFFIX" != *.json ]]; then
  printf 'Index name must start with %s and end with .json: %s\n' "$DATASET" "$INDEX_NAME" >&2
  exit 2
fi
if [[ ! -f "$INDEX_FILE" || "$OVERWRITE_INDEX" == true || "$RETRAIN_RQVAE" == true ]]; then
  if [[ ! -f "$EMBEDDING_FILE" ]]; then
    printf 'Item embeddings not found: %s\nRun data_process/preprocess_item_embeddings.sh first.\n' "$EMBEDDING_FILE" >&2
    exit 1
  fi
  if [[ ! -f "$CF_EMBEDDING" ]]; then
    printf 'Collaborative-filtering embeddings not found: %s\n' "$CF_EMBEDDING" >&2
    exit 1
  fi
fi

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

if [[ -f "$INDEX_FILE" && "$OVERWRITE_INDEX" != true && "$RETRAIN_RQVAE" != true ]]; then
  printf '\n[Index] Found existing fixed-length index: %s\n' "$INDEX_FILE"
  printf '[Index] Reusing existing index (pass --overwrite-index to regenerate).\n'
else
  DETECTED_CKPT=""
  if [[ -z "$RQ_CHECKPOINT" && "$RETRAIN_RQVAE" != true ]]; then
    DETECTED_CKPT="$(find_latest_checkpoint "$RQ_CHECKPOINT_ROOT")"
    if [[ -n "$DETECTED_CKPT" && -f "$DETECTED_CKPT" ]]; then
      RQ_CHECKPOINT="$DETECTED_CKPT"
      printf '\n[RQ-VAE] Autodetected existing checkpoint: %s\n' "$RQ_CHECKPOINT"
      printf '[RQ-VAE] Reusing existing checkpoint (pass --retrain-rqvae to force training).\n'
    fi
  fi

  if [[ -z "$RQ_CHECKPOINT" ]]; then
    printf '\n[RQ-VAE] Training the tokenizer...\n'
    STEP_START="$SECONDS"
    mkdir -p "$RQ_CHECKPOINT_ROOT"
    "$PYTHON_BIN" "$REPO_ROOT/RQ-VAE/main.py" \
      --device "$RQ_DEVICE" \
      --data_path "$EMBEDDING_FILE" \
      --cf_emb "$CF_EMBEDDING" \
      --alpha "$ALPHA" \
      --beta "$BETA" \
      --epochs "$RQ_EPOCHS" \
      --eval_step "$RQ_EVAL_STEP" \
      --ckpt_dir "$RQ_CHECKPOINT_ROOT"

    RQ_CHECKPOINT="$(find_latest_checkpoint "$RQ_CHECKPOINT_ROOT")"
    if [[ -z "$RQ_CHECKPOINT" ]]; then
      printf 'RQ-VAE training completed without a best_collision_model.pth checkpoint.\n' >&2
      exit 1
    fi
    printf 'Completed RQ-VAE training in %s.\n' "$(format_duration "$((SECONDS - STEP_START))")"
    record_phase "RQ-VAE training" "$((SECONDS - STEP_START))"
    printf 'Stored RQ-VAE checkpoint: %s\n' "$RQ_CHECKPOINT"
  else
    if [[ "$DETECTED_CKPT" != "$RQ_CHECKPOINT" ]]; then
      printf '\n[RQ-VAE] Reusing checkpoint: %s\n' "$RQ_CHECKPOINT"
    fi
  fi
  if [[ ! -f "$RQ_CHECKPOINT" ]]; then
    printf 'RQ-VAE checkpoint not found: %s\n' "$RQ_CHECKPOINT" >&2
    exit 1
  fi

  printf '\n[Index] Generating the fixed-length item index...\n'
  STEP_START="$SECONDS"
  "$PYTHON_BIN" "$REPO_ROOT/RQ-VAE/generate_indices.py" \
    --dataset "$DATASET" \
    --checkpoint-path "$RQ_CHECKPOINT" \
    --output-file "$INDEX_FILE" \
    --device "$RQ_DEVICE"

  if [[ ! -f "$INDEX_FILE" ]]; then
    printf 'Index generation completed without creating: %s\n' "$INDEX_FILE" >&2
    exit 1
  fi
  printf 'Completed fixed-index generation in %s.\n' "$(format_duration "$((SECONDS - STEP_START))")"
  record_phase "Fixed-length index generation" "$((SECONDS - STEP_START))"
  printf 'Stored fixed-length item index: %s\n' "$INDEX_FILE"
fi

if [[ "$TOKENIZER_ONLY" == true ]]; then
  print_phase_durations
  printf '\nTokenizer and fixed-index pipeline completed in %s.\n' \
    "$(format_duration "$((SECONDS - PIPELINE_START))")"
  exit 0
fi

if contains_model tiger; then
  printf '\n[TIGER] Training...\n'
  STEP_START="$SECONDS"
  mkdir -p "$REPO_ROOT/LETTER-TIGER/results/$DATASET"
  (
    cd "$REPO_ROOT/LETTER-TIGER"
    CUDA_VISIBLE_DEVICES="$TIGER_GPUS" torchrun \
      --nproc_per_node="$(gpu_count "$TIGER_GPUS")" \
      --master_port=2314 \
      finetune.py \
      --output_dir "./ckpt/$DATASET" \
      --dataset "$DATASET" \
      --data_path "$DATA_ROOT" \
      --per_device_batch_size 256 \
      --learning_rate 5e-4 \
      --epochs 200 \
      --index_file "$INDEX_SUFFIX" \
      --temperature 1.0
  )
  printf 'Completed LETTER-TIGER training in %s.\n' "$(format_duration "$((SECONDS - STEP_START))")"
  record_phase "LETTER-TIGER training" "$((SECONDS - STEP_START))"
  printf 'Stored LETTER-TIGER checkpoint: %s\n' "$REPO_ROOT/LETTER-TIGER/ckpt/$DATASET"

  if [[ "$SKIP_EVALUATION" != true ]]; then
    printf '\n[TIGER] Evaluating...\n'
    STEP_START="$SECONDS"
    (
      cd "$REPO_ROOT/LETTER-TIGER"
      "$PYTHON_BIN" test.py \
        --gpu_id 0 \
        --ckpt_path "./ckpt/$DATASET" \
        --dataset "$DATASET" \
        --data_path "$DATA_ROOT" \
        --results_file "./results/$DATASET/fixed.json" \
        --test_batch_size 32 \
        --num_beams 20 \
        --test_prompt_ids 0 \
        --index_file "$INDEX_SUFFIX"
    )
    printf 'Completed LETTER-TIGER evaluation in %s.\n' "$(format_duration "$((SECONDS - STEP_START))")"
    record_phase "LETTER-TIGER evaluation" "$((SECONDS - STEP_START))"
    printf 'Stored LETTER-TIGER metrics: %s\n' "$REPO_ROOT/LETTER-TIGER/results/$DATASET/fixed.json"
  fi
fi

if contains_model lcrec; then
  printf '\n[LC-Rec] Training...\n'
  STEP_START="$SECONDS"
  mkdir -p "$REPO_ROOT/LETTER-LC-Rec/results/$DATASET"
  (
    cd "$REPO_ROOT/LETTER-LC-Rec"
    CUDA_VISIBLE_DEVICES="$LCREC_GPUS" torchrun \
      --nproc_per_node="$(gpu_count "$LCREC_GPUS")" \
      --master_port=3325 \
      lora_finetune.py \
      --base_model "$BASE_MODEL" \
      --output_dir "./ckpt/$DATASET" \
      --dataset "$DATASET" \
      --data_path "$DATA_ROOT" \
      --per_device_batch_size 16 \
      --learning_rate 1e-4 \
      --epochs 4 \
      --tasks seqrec \
      --train_prompt_sample_num 1 \
      --train_data_sample_num 0 \
      --index_file "$INDEX_SUFFIX" \
      --wandb_run_name "${DATASET}-fixed" \
      --temperature 1.0
  )
  printf 'Completed LETTER-LC-Rec training in %s.\n' "$(format_duration "$((SECONDS - STEP_START))")"
  record_phase "LETTER-LC-Rec training" "$((SECONDS - STEP_START))"
  printf 'Stored LETTER-LC-Rec checkpoint: %s\n' "$REPO_ROOT/LETTER-LC-Rec/ckpt/$DATASET"

  if [[ "$SKIP_EVALUATION" != true ]]; then
    printf '\n[LC-Rec] Evaluating...\n'
    STEP_START="$SECONDS"
    (
      cd "$REPO_ROOT/LETTER-LC-Rec"
      CUDA_VISIBLE_DEVICES="$LCREC_GPUS" torchrun \
        --nproc_per_node="$(gpu_count "$LCREC_GPUS")" \
        --master_port=4324 \
        test_ddp.py \
        --ckpt_path "./ckpt/$DATASET" \
        --base_model "$BASE_MODEL" \
        --dataset "$DATASET" \
        --data_path "$DATA_ROOT" \
        --results_file "./results/$DATASET/fixed.json" \
        --test_batch_size 1 \
        --num_beams 20 \
        --test_prompt_ids 0 \
        --index_file "$INDEX_SUFFIX"
    )
    printf 'Completed LETTER-LC-Rec evaluation in %s.\n' "$(format_duration "$((SECONDS - STEP_START))")"
    record_phase "LETTER-LC-Rec evaluation" "$((SECONDS - STEP_START))"
    printf 'Stored LETTER-LC-Rec metrics: %s\n' "$REPO_ROOT/LETTER-LC-Rec/results/$DATASET/fixed.json"
  fi
fi

print_phase_durations
printf '\nFixed-length pipeline completed in %s.\n' \
  "$(format_duration "$((SECONDS - PIPELINE_START))")"
