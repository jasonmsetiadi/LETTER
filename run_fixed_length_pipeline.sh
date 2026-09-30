#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash run_fixed_length_pipeline.sh --dataset DATASET [options]

Starts after item embeddings have been generated. It trains or reuses the
selected tokenizer (letter or vanilla rqvae), writes a new index file, then
trains and evaluates selected downstream recommenders.

Required:
  --dataset NAME

Options:
  --tokenizer LIST             Comma-separated tokenizer(s): letter, rqvae (default: letter)
  --data-root PATH             Dataset directory parent (default: <repo>/data)
  --embedding-file PATH        Item embedding .npy path
  --cf-embedding PATH          Collaborative-filtering embedding .pt path (required for letter)
  --rqvae-checkpoint PATH      Reuse an existing RQ-VAE/LETTER checkpoint (autodetected if omitted)
  --retrain-rqvae              Force training RQ-VAE even if a checkpoint exists
  --rqvae-epochs COUNT         RQ-VAE epochs (default: 10000)
  --rqvae-eval-step COUNT      RQ-VAE validation interval (default: 2000)
  --rqvae-device DEVICE        RQ-VAE device (default: cuda:0)
  --alpha VALUE                Collaborative-loss weight (default: 0.01 for letter, 0.0 for rqvae)
  --beta VALUE                 Diversity-loss weight (default: 0.0001 for letter, 0.0 for rqvae)
  --num-layers COUNT           Number of RQ-VAE codebook layers / SID length (default: 4)
  --num-emb-list LIST          Explicit codebook sizes (e.g. "256 256 256 256 256")
  --index-name NAME            Generated index filename (default: <data-root>/<dataset>/<tok>/<dataset>.index.fixed[.L<k>].json)
  --overwrite-index            Replace an existing generated index
  --tokenizer-only             Stop after fixed-length index generation
  --models LIST                Comma-separated: tiger,lcrec (default: tiger)
  --base-model PATH            Base model for LC-Rec (default: huggyllama/llama-7b)
  --tiger-gpus IDS             CUDA devices for TIGER (default: autodetect, up to 2)
  --lcrec-gpus IDS             CUDA devices for LC-Rec (default: autodetect, up to 4)
  --results-file PATH          Results JSON path (default: <model>/results/<dataset>/<tok>/fixed[_L<k>].json)
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
TOKENIZERS="letter"
USER_ALPHA=""
USER_BETA=""
NUM_LAYERS="4"
USER_NUM_EMB_LIST=""
RESULTS_FILE=""
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
    --tokenizer|--tokenizer-type|--tokenizers) TOKENIZERS="$2"; shift 2 ;;
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --embedding-file) EMBEDDING_FILE="$2"; shift 2 ;;
    --cf-embedding) CF_EMBEDDING="$2"; shift 2 ;;
    --rqvae-checkpoint) RQ_CHECKPOINT="$2"; shift 2 ;;
    --retrain-rqvae) RETRAIN_RQVAE=true; shift ;;
    --rqvae-epochs) RQ_EPOCHS="$2"; shift 2 ;;
    --rqvae-eval-step) RQ_EVAL_STEP="$2"; shift 2 ;;
    --rqvae-device) RQ_DEVICE="$2"; shift 2 ;;
    --alpha) USER_ALPHA="$2"; shift 2 ;;
    --beta) USER_BETA="$2"; shift 2 ;;
    --num-layers) NUM_LAYERS="$2"; shift 2 ;;
    --num-emb-list) USER_NUM_EMB_LIST="$2"; shift 2 ;;
    --index-name) INDEX_NAME="$2"; shift 2 ;;
    --models) MODELS="$2"; shift 2 ;;
    --base-model) BASE_MODEL="$2"; shift 2 ;;
    --tiger-gpus) TIGER_GPUS="$2"; shift 2 ;;
    --lcrec-gpus) LCREC_GPUS="$2"; shift 2 ;;
    --results-file) RESULTS_FILE="$2"; shift 2 ;;
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

IFS=',' read -r -a RAW_TOKENIZERS <<< "$TOKENIZERS"
NORMALIZED_TOKENIZERS=()
for t in "${RAW_TOKENIZERS[@]}"; do
  t="$(echo "$t" | tr -d ' ' | tr '[:upper:]' '[:lower:]')"
  case "$t" in
    letter)
      NORMALIZED_TOKENIZERS+=(letter)
      ;;
    rqvae|rq-vae|vanilla|vanilla_rqvae|vanilla-rqvae)
      NORMALIZED_TOKENIZERS+=(rqvae)
      ;;
    *)
      printf 'Unknown tokenizer: %s (supported: letter, rqvae)\n' "$t" >&2
      exit 2
      ;;
  esac
done

if [[ "${#NORMALIZED_TOKENIZERS[@]}" -eq 0 ]]; then
  printf 'Select at least one supported tokenizer: letter, rqvae\n' >&2
  exit 2
fi

if [[ -n "$USER_NUM_EMB_LIST" ]]; then
  IFS=', ' read -r -a NUM_EMB_LIST <<< "$USER_NUM_EMB_LIST"
  NUM_LAYERS="${#NUM_EMB_LIST[@]}"
else
  if ! [[ "$NUM_LAYERS" =~ ^[1-9][0-9]*$ ]]; then
    printf '--num-layers must be a positive integer: %s\n' "$NUM_LAYERS" >&2
    exit 2
  fi
  NUM_EMB_LIST=()
  for (( i=0; i<NUM_LAYERS; i++ )); do
    NUM_EMB_LIST+=(256)
  done
fi

EMBEDDING_FILE="${EMBEDDING_FILE:-$DATA_ROOT/$DATASET/$DATASET.emb-flan-t5-xl-td.npy}"
CF_EMBEDDING="${CF_EMBEDDING:-$REPO_ROOT/RQ-VAE/ckpt/$DATASET-32d-sasrec.pt}"

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

find_free_port() {
  local default_port="${1:-29500}"
  "$PYTHON_BIN" -c '
import socket, sys
try:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        print(s.getsockname()[1])
except Exception:
    print(sys.argv[1])
' "$default_port" 2>/dev/null || echo "$default_port"
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
  local desired_layers="${2:-4}"
  if [[ ! -d "$root" ]]; then
    return 0
  fi
  "$PYTHON_BIN" -c '
import sys, glob, os, torch
root = sys.argv[1]
desired_layers = int(sys.argv[2])
candidates = glob.glob(os.path.join(root, "**", "best_collision_model.pth"), recursive=True)
if not candidates:
    candidates = glob.glob(os.path.join(root, "**", "best_loss_model.pth"), recursive=True)
valid = []
for c in candidates:
    try:
        ckpt = torch.load(c, map_location="cpu", weights_only=False)
    except TypeError:
        try:
            ckpt = torch.load(c, map_location="cpu")
        except Exception:
            continue
    except Exception:
        continue
    args = ckpt.get("args")
    if args and hasattr(args, "num_emb_list") and len(args.num_emb_list) == desired_layers:
        valid.append(c)
if valid:
    print(max(valid, key=os.path.getmtime))
' "$root" "$desired_layers" 2>/dev/null || true
}

for TOK in "${NORMALIZED_TOKENIZERS[@]}"; do
  if [[ "$TOK" == "letter" ]]; then
    TOK_NAME="letter"
    TOK_LABEL="LETTER"
    TOK_ALPHA="${USER_ALPHA:-0.01}"
    TOK_BETA="${USER_BETA:-0.0001}"
    TOK_CKPT_ROOT="$REPO_ROOT/checkpoint/$DATASET/$TOK_NAME"
    TOK_SK_ARGS=()
    TOK_NEEDS_CF=true

    TOK_INDEX_DIR="$DATA_ROOT/$DATASET/$TOK_NAME"
    if [[ "$NUM_LAYERS" -eq 4 ]]; then
      DEFAULT_INDEX_NAME="$DATASET.index.fixed.json"
      DEFAULT_TIGER_CKPT="./ckpt/$DATASET/$TOK_NAME"
      DEFAULT_LCREC_CKPT="./ckpt/$DATASET/$TOK_NAME"
      DEFAULT_TIGER_RESULTS="./results/$DATASET/$TOK_NAME/fixed.json"
      DEFAULT_LCREC_RESULTS="./results/$DATASET/$TOK_NAME/fixed.json"
      DEFAULT_LCREC_WANDB="${DATASET}-${TOK_NAME}-fixed"
    else
      DEFAULT_INDEX_NAME="$DATASET.index.fixed.L${NUM_LAYERS}.json"
      DEFAULT_TIGER_CKPT="./ckpt/$DATASET/$TOK_NAME-L${NUM_LAYERS}"
      DEFAULT_LCREC_CKPT="./ckpt/$DATASET/$TOK_NAME-L${NUM_LAYERS}"
      DEFAULT_TIGER_RESULTS="./results/$DATASET/$TOK_NAME/fixed_L${NUM_LAYERS}.json"
      DEFAULT_LCREC_RESULTS="./results/$DATASET/$TOK_NAME/fixed_L${NUM_LAYERS}.json"
      DEFAULT_LCREC_WANDB="${DATASET}-${TOK_NAME}-fixed-L${NUM_LAYERS}"
    fi
  else
    TOK_NAME="rqvae"
    TOK_LABEL="Vanilla RQ-VAE"
    TOK_ALPHA="${USER_ALPHA:-0.0}"
    TOK_BETA="${USER_BETA:-0.0}"
    TOK_CKPT_ROOT="$REPO_ROOT/checkpoint/$DATASET/$TOK_NAME"
    TOK_SK_ARGS=(--sk_epsilons)
    for (( i=0; i<NUM_LAYERS; i++ )); do
      TOK_SK_ARGS+=(0.0)
    done
    TOK_NEEDS_CF=false

    TOK_INDEX_DIR="$DATA_ROOT/$DATASET/$TOK_NAME"
    if [[ "$NUM_LAYERS" -eq 4 ]]; then
      DEFAULT_INDEX_NAME="$DATASET.index.fixed.json"
      DEFAULT_TIGER_CKPT="./ckpt/$DATASET/$TOK_NAME"
      DEFAULT_LCREC_CKPT="./ckpt/$DATASET/$TOK_NAME"
      DEFAULT_TIGER_RESULTS="./results/$DATASET/$TOK_NAME/fixed.json"
      DEFAULT_LCREC_RESULTS="./results/$DATASET/$TOK_NAME/fixed.json"
      DEFAULT_LCREC_WANDB="${DATASET}-${TOK_NAME}-fixed"
    else
      DEFAULT_INDEX_NAME="$DATASET.index.fixed.L${NUM_LAYERS}.json"
      DEFAULT_TIGER_CKPT="./ckpt/$DATASET/$TOK_NAME-L${NUM_LAYERS}"
      DEFAULT_LCREC_CKPT="./ckpt/$DATASET/$TOK_NAME-L${NUM_LAYERS}"
      DEFAULT_TIGER_RESULTS="./results/$DATASET/$TOK_NAME/fixed_L${NUM_LAYERS}.json"
      DEFAULT_LCREC_RESULTS="./results/$DATASET/$TOK_NAME/fixed_L${NUM_LAYERS}.json"
      DEFAULT_LCREC_WANDB="${DATASET}-${TOK_NAME}-fixed-L${NUM_LAYERS}"
    fi
  fi

  if [[ -n "$INDEX_NAME" ]]; then
    if [[ "${#NORMALIZED_TOKENIZERS[@]}" -gt 1 ]]; then
      BASE_NO_EXT="${INDEX_NAME%.json}"
      CUR_INDEX_FILENAME="${BASE_NO_EXT}.${TOK_NAME}.json"
    else
      CUR_INDEX_FILENAME="$INDEX_NAME"
    fi
    if [[ "$CUR_INDEX_FILENAME" == /* ]]; then
      CUR_INDEX_FILE="$CUR_INDEX_FILENAME"
      CUR_INDEX_ARG="$CUR_INDEX_FILENAME"
    elif [[ "$CUR_INDEX_FILENAME" == */* ]]; then
      CUR_INDEX_FILE="$DATA_ROOT/$DATASET/$CUR_INDEX_FILENAME"
      CUR_INDEX_ARG="$CUR_INDEX_FILENAME"
    else
      CUR_INDEX_FILE="$TOK_INDEX_DIR/$CUR_INDEX_FILENAME"
      CUR_INDEX_ARG="$TOK_NAME/$CUR_INDEX_FILENAME"
    fi
  else
    # Fall back to legacy index path in data/$DATASET/ if it already exists for letter
    if [[ "$TOK" == "letter" && ! -f "$TOK_INDEX_DIR/$DEFAULT_INDEX_NAME" && -f "$DATA_ROOT/$DATASET/$DEFAULT_INDEX_NAME" && "$OVERWRITE_INDEX" != true ]]; then
      CUR_INDEX_FILE="$DATA_ROOT/$DATASET/$DEFAULT_INDEX_NAME"
      CUR_INDEX_ARG="$DEFAULT_INDEX_NAME"
    else
      CUR_INDEX_FILE="$TOK_INDEX_DIR/$DEFAULT_INDEX_NAME"
      CUR_INDEX_ARG="$TOK_NAME/$DEFAULT_INDEX_NAME"
    fi
  fi

  if [[ -n "$RESULTS_FILE" ]]; then
    res_dir="$(dirname "$RESULTS_FILE")"
    res_base="$(basename "$RESULTS_FILE")"
    if [[ "$res_dir" == *"/$TOK_NAME" ]]; then
      CUR_TIGER_RESULTS="$RESULTS_FILE"
      CUR_LCREC_RESULTS="$RESULTS_FILE"
    else
      CUR_TIGER_RESULTS="$res_dir/$TOK_NAME/$res_base"
      CUR_LCREC_RESULTS="$res_dir/$TOK_NAME/$res_base"
    fi
  else
    CUR_TIGER_RESULTS="$DEFAULT_TIGER_RESULTS"
    CUR_LCREC_RESULTS="$DEFAULT_LCREC_RESULTS"
  fi

  CUR_TIGER_CKPT="$DEFAULT_TIGER_CKPT"
  CUR_LCREC_CKPT="$DEFAULT_LCREC_CKPT"
  CUR_LCREC_WANDB="$DEFAULT_LCREC_WANDB"

  if [[ ! -f "$CUR_INDEX_FILE" || "$OVERWRITE_INDEX" == true || "$RETRAIN_RQVAE" == true ]]; then
    if [[ ! -f "$EMBEDDING_FILE" ]]; then
      printf 'Item embeddings not found: %s\nRun data_process/preprocess_item_embeddings.sh first.\n' "$EMBEDDING_FILE" >&2
      exit 1
    fi
    if [[ "$TOK_NEEDS_CF" == true && ! -f "$CF_EMBEDDING" ]]; then
      printf 'Collaborative-filtering embeddings not found: %s\n' "$CF_EMBEDDING" >&2
      exit 1
    fi
  fi

  CUR_RQ_CHECKPOINT="$RQ_CHECKPOINT"

  if [[ -f "$CUR_INDEX_FILE" && "$OVERWRITE_INDEX" != true && "$RETRAIN_RQVAE" != true ]]; then
    printf '\n[Index] [%s] Found existing fixed-length index: %s\n' "$TOK_LABEL" "$CUR_INDEX_FILE"
    printf '[Index] Reusing existing index (pass --overwrite-index to regenerate).\n'
  else
    if [[ -n "$CUR_RQ_CHECKPOINT" && -f "$CUR_RQ_CHECKPOINT" ]]; then
      CHECK_LAYERS=$("$PYTHON_BIN" -c '
import sys, torch
try:
    ckpt = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
except TypeError:
    ckpt = torch.load(sys.argv[1], map_location="cpu")
args = ckpt.get("args")
if args and hasattr(args, "num_emb_list"):
    print(len(args.num_emb_list))
' "$CUR_RQ_CHECKPOINT" 2>/dev/null || true)
      if [[ -n "$CHECK_LAYERS" && "$CHECK_LAYERS" -ne "$NUM_LAYERS" ]]; then
        printf "Specified RQ-VAE checkpoint has %s layers, but requested --num-layers is %s.\n" "$CHECK_LAYERS" "$NUM_LAYERS" >&2
        exit 1
      fi
    fi

    DETECTED_CKPT=""
    if [[ -z "$CUR_RQ_CHECKPOINT" && "$RETRAIN_RQVAE" != true ]]; then
      DETECTED_CKPT="$(find_latest_checkpoint "$TOK_CKPT_ROOT" "$NUM_LAYERS")"
      if [[ -z "$DETECTED_CKPT" && "$TOK" == "letter" ]]; then
        LEGACY_CKPT="$(find_latest_checkpoint "$REPO_ROOT/checkpoint/$DATASET" "$NUM_LAYERS")"
        if [[ -n "$LEGACY_CKPT" && "$LEGACY_CKPT" != *"/rqvae/"* ]]; then
          DETECTED_CKPT="$LEGACY_CKPT"
        fi
      fi
      if [[ -n "$DETECTED_CKPT" && -f "$DETECTED_CKPT" ]]; then
        CUR_RQ_CHECKPOINT="$DETECTED_CKPT"
        printf '\n[RQ-VAE] [%s] Autodetected existing %s-layer checkpoint: %s\n' "$TOK_LABEL" "$NUM_LAYERS" "$CUR_RQ_CHECKPOINT"
        printf '[RQ-VAE] Reusing existing checkpoint (pass --retrain-rqvae to force training).\n'
      fi
    fi

    if [[ -z "$CUR_RQ_CHECKPOINT" ]]; then
      printf '\n[RQ-VAE] [%s] Training tokenizer with %s layers...\n' "$TOK_LABEL" "$NUM_LAYERS"
      STEP_START="$SECONDS"
      mkdir -p "$TOK_CKPT_ROOT"
      rq_train_cmd=(
        "$PYTHON_BIN" "$REPO_ROOT/RQ-VAE/main.py"
        --device "$RQ_DEVICE"
        --data_path "$EMBEDDING_FILE"
        --alpha "$TOK_ALPHA"
        --beta "$TOK_BETA"
        --epochs "$RQ_EPOCHS"
        --eval_step "$RQ_EVAL_STEP"
        --ckpt_dir "$TOK_CKPT_ROOT"
        --num_emb_list "${NUM_EMB_LIST[@]}"
      )
      if [[ "$TOK_NEEDS_CF" == true && -n "$CF_EMBEDDING" ]]; then
        rq_train_cmd+=(--cf_emb "$CF_EMBEDDING")
      fi
      if [[ "${#TOK_SK_ARGS[@]}" -gt 0 ]]; then
        rq_train_cmd+=("${TOK_SK_ARGS[@]}")
      fi

      "${rq_train_cmd[@]}"

      CUR_RQ_CHECKPOINT="$(find_latest_checkpoint "$TOK_CKPT_ROOT" "$NUM_LAYERS")"
      if [[ -z "$CUR_RQ_CHECKPOINT" ]]; then
        printf 'RQ-VAE training completed without a best_collision_model.pth checkpoint.\n' >&2
        exit 1
      fi
      printf 'Completed %s training in %s.\n' "$TOK_LABEL" "$(format_duration "$((SECONDS - STEP_START))")"
      record_phase "[$TOK_LABEL] Tokenizer training" "$((SECONDS - STEP_START))"
      printf 'Stored %s checkpoint: %s\n' "$TOK_LABEL" "$CUR_RQ_CHECKPOINT"
    else
      if [[ "$DETECTED_CKPT" != "$CUR_RQ_CHECKPOINT" ]]; then
        printf '\n[RQ-VAE] [%s] Reusing checkpoint: %s\n' "$TOK_LABEL" "$CUR_RQ_CHECKPOINT"
      fi
    fi
    if [[ ! -f "$CUR_RQ_CHECKPOINT" ]]; then
      printf '%s checkpoint not found: %s\n' "$TOK_LABEL" "$CUR_RQ_CHECKPOINT" >&2
      exit 1
    fi

    printf '\n[Index] [%s] Generating fixed-length item index...\n' "$TOK_LABEL"
    STEP_START="$SECONDS"
    mkdir -p "$(dirname "$CUR_INDEX_FILE")"
    "$PYTHON_BIN" "$REPO_ROOT/RQ-VAE/generate_indices.py" \
      --dataset "$DATASET" \
      --checkpoint-path "$CUR_RQ_CHECKPOINT" \
      --output-file "$CUR_INDEX_FILE" \
      --device "$RQ_DEVICE"

    if [[ ! -f "$CUR_INDEX_FILE" ]]; then
      printf 'Index generation completed without creating: %s\n' "$CUR_INDEX_FILE" >&2
      exit 1
    fi
    printf 'Completed %s index generation in %s.\n' "$TOK_LABEL" "$(format_duration "$((SECONDS - STEP_START))")"
    record_phase "[$TOK_LABEL] Index generation" "$((SECONDS - STEP_START))"
    printf 'Stored fixed-length item index: %s\n' "$CUR_INDEX_FILE"
  fi

  if [[ "$TOKENIZER_ONLY" == true ]]; then
    continue
  fi

  if contains_model tiger; then
    printf '\n[TIGER] [%s] Training...\n' "$TOK_LABEL"
    STEP_START="$SECONDS"
    mkdir -p "$(dirname "$CUR_TIGER_RESULTS")"
    (
      cd "$REPO_ROOT/LETTER-TIGER"
      TIGER_COUNT="$(gpu_count "$TIGER_GPUS")"
      if [[ "$TIGER_COUNT" -le 1 ]]; then
        printf '[TIGER] Single GPU mode (%s) - running directly without DDP.\n' "$TIGER_GPUS"
        CUDA_VISIBLE_DEVICES="$TIGER_GPUS" "$PYTHON_BIN" finetune.py \
          --output_dir "$CUR_TIGER_CKPT" \
          --dataset "$DATASET" \
          --data_path "$DATA_ROOT" \
          --per_device_batch_size 256 \
          --learning_rate 5e-4 \
          --epochs 200 \
          --index_file "$CUR_INDEX_ARG" \
          --temperature 1.0
      else
        TIGER_PORT="$(find_free_port 2314)"
        CUDA_VISIBLE_DEVICES="$TIGER_GPUS" torchrun \
          --nproc_per_node="$TIGER_COUNT" \
          --master_port="$TIGER_PORT" \
          finetune.py \
          --output_dir "$CUR_TIGER_CKPT" \
          --dataset "$DATASET" \
          --data_path "$DATA_ROOT" \
          --per_device_batch_size 256 \
          --learning_rate 5e-4 \
          --epochs 200 \
          --index_file "$CUR_INDEX_ARG" \
          --temperature 1.0
      fi
    )
    printf 'Completed [%s] LETTER-TIGER training in %s.\n' "$TOK_LABEL" "$(format_duration "$((SECONDS - STEP_START))")"
    record_phase "[$TOK_LABEL] LETTER-TIGER training" "$((SECONDS - STEP_START))"
    printf 'Stored LETTER-TIGER checkpoint: %s\n' "$CUR_TIGER_CKPT"

    if [[ "$SKIP_EVALUATION" != true ]]; then
      printf '\n[TIGER] [%s] Evaluating...\n' "$TOK_LABEL"
      STEP_START="$SECONDS"
      (
        cd "$REPO_ROOT/LETTER-TIGER"
        "$PYTHON_BIN" test.py \
          --gpu_id 0 \
          --ckpt_path "$CUR_TIGER_CKPT" \
          --dataset "$DATASET" \
          --data_path "$DATA_ROOT" \
          --results_file "$CUR_TIGER_RESULTS" \
          --test_batch_size 32 \
          --num_beams 20 \
          --test_prompt_ids 0 \
          --index_file "$CUR_INDEX_ARG"
      )
      printf 'Completed [%s] LETTER-TIGER evaluation in %s.\n' "$TOK_LABEL" "$(format_duration "$((SECONDS - STEP_START))")"
      record_phase "[$TOK_LABEL] LETTER-TIGER evaluation" "$((SECONDS - STEP_START))"
      printf 'Stored LETTER-TIGER metrics: %s\n' "$CUR_TIGER_RESULTS"
    fi
  fi

  if contains_model lcrec; then
    printf '\n[LC-Rec] [%s] Training...\n' "$TOK_LABEL"
    STEP_START="$SECONDS"
    mkdir -p "$(dirname "$CUR_LCREC_RESULTS")"
    (
      cd "$REPO_ROOT/LETTER-LC-Rec"
      LCREC_COUNT="$(gpu_count "$LCREC_GPUS")"
      if [[ "$LCREC_COUNT" -le 1 ]]; then
        printf '[LC-Rec] Single GPU mode (%s) - skipping DDP for 8-bit quantized training.\n' "$LCREC_GPUS"
        CUDA_VISIBLE_DEVICES="$LCREC_GPUS" "$PYTHON_BIN" lora_finetune.py \
          --base_model "$BASE_MODEL" \
          --output_dir "$CUR_LCREC_CKPT" \
          --dataset "$DATASET" \
          --data_path "$DATA_ROOT" \
          --per_device_batch_size 16 \
          --learning_rate 1e-4 \
          --epochs 4 \
          --tasks seqrec \
          --train_prompt_sample_num 1 \
          --train_data_sample_num 0 \
          --index_file "$CUR_INDEX_ARG" \
          --wandb_run_name "$CUR_LCREC_WANDB" \
          --temperature 1.0
      else
        LCREC_PORT="$(find_free_port 3325)"
        CUDA_VISIBLE_DEVICES="$LCREC_GPUS" torchrun \
          --nproc_per_node="$LCREC_COUNT" \
          --master_port="$LCREC_PORT" \
          lora_finetune.py \
          --base_model "$BASE_MODEL" \
          --output_dir "$CUR_LCREC_CKPT" \
          --dataset "$DATASET" \
          --data_path "$DATA_ROOT" \
          --per_device_batch_size 16 \
          --learning_rate 1e-4 \
          --epochs 4 \
          --tasks seqrec \
          --train_prompt_sample_num 1 \
          --train_data_sample_num 0 \
          --index_file "$CUR_INDEX_ARG" \
          --wandb_run_name "$CUR_LCREC_WANDB" \
          --temperature 1.0
      fi
    )
    printf 'Completed [%s] LETTER-LC-Rec training in %s.\n' "$TOK_LABEL" "$(format_duration "$((SECONDS - STEP_START))")"
    record_phase "[$TOK_LABEL] LETTER-LC-Rec training" "$((SECONDS - STEP_START))"
    printf 'Stored LETTER-LC-Rec checkpoint: %s\n' "$CUR_LCREC_CKPT"

    if [[ "$SKIP_EVALUATION" != true ]]; then
      printf '\n[LC-Rec] [%s] Evaluating...\n' "$TOK_LABEL"
      STEP_START="$SECONDS"
      (
        cd "$REPO_ROOT/LETTER-LC-Rec"
        TEST_PORT="$(find_free_port 4324)"
        CUDA_VISIBLE_DEVICES="$LCREC_GPUS" torchrun \
          --nproc_per_node="$(gpu_count "$LCREC_GPUS")" \
          --master_port="$TEST_PORT" \
          test_ddp.py \
          --ckpt_path "$CUR_LCREC_CKPT" \
          --base_model "$BASE_MODEL" \
          --dataset "$DATASET" \
          --data_path "$DATA_ROOT" \
          --results_file "$CUR_LCREC_RESULTS" \
          --test_batch_size 1 \
          --num_beams 20 \
          --test_prompt_ids 0 \
          --index_file "$CUR_INDEX_ARG"
      )
      printf 'Completed [%s] LETTER-LC-Rec evaluation in %s.\n' "$TOK_LABEL" "$(format_duration "$((SECONDS - STEP_START))")"
      record_phase "[$TOK_LABEL] LETTER-LC-Rec evaluation" "$((SECONDS - STEP_START))"
      printf 'Stored LETTER-LC-Rec metrics: %s\n' "$CUR_LCREC_RESULTS"
    fi
  fi
done

print_phase_durations
if [[ "$TOKENIZER_ONLY" == true ]]; then
  printf '\nTokenizer and fixed-index pipeline completed in %s.\n' \
    "$(format_duration "$((SECONDS - PIPELINE_START))")"
else
  printf '\nFixed-length pipeline completed in %s.\n' \
    "$(format_duration "$((SECONDS - PIPELINE_START))")"
fi
