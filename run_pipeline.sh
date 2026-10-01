#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash run_pipeline.sh --dataset DATASET [options]

Unified runner for fixed-length and variable-length item tokenizer and
recommendation pipelines. Supports LETTER and vanilla RQ-VAE tokenizers,
fixed and variable-length index generation (with multiple truncation strategies),
and end-to-end downstream training and evaluation (TIGER and LC-Rec).

Required:
  --dataset NAME               Dataset name (e.g. Instruments)

Pipeline & Mode Options:
  --mode MODE                  Pipeline mode: fixed or varlen (default: fixed)
  --tokenizer TYPE             Tokenizer type: rqvae or letter (default: rqvae)
  --data-root PATH             Dataset directory parent (default: <repo>/data)
  --models LIST                Comma-separated: tiger,lcrec (default: tiger)
  --base-model PATH            Base model for LC-Rec (default: huggyllama/llama-7b)
  --tokenizer-only             Stop after index generation (skip downstream recommenders)
  --skip-training, --eval-only Skip training downstream models and run evaluation only
  --retrain-model              Force training downstream models even if checkpoint exists
  --skip-evaluation            Train recommenders without evaluating

RQ-VAE / Tokenizer Options:
  --embedding-file PATH        Item embedding .npy path
  --cf-embedding PATH          Collaborative-filtering embedding .pt path (required for letter)
  --rqvae-checkpoint PATH      Reuse an existing RQ-VAE/LETTER checkpoint (autodetected if omitted)
  --retrain-rqvae              Force training RQ-VAE even if a checkpoint exists
  --rqvae-epochs COUNT         RQ-VAE epochs (default: 10000)
  --rqvae-eval-step COUNT      RQ-VAE validation interval (default: 2000)
  --rqvae-device DEVICE        RQ-VAE device (default: cuda:0)
  --alpha VALUE                Collaborative-loss weight (default: 0.01 for letter, 0.0 for rqvae)
  --beta VALUE                 Diversity-loss weight (default: 0.0001 for letter, 0.0 for rqvae)
  --num-layers COUNT           Number of RQ-VAE codebook layers / SID length (default: 4 or max-length)
  --num-emb-list LIST          Explicit codebook sizes (e.g. "256 256 256 256")

Index Generation & Variable-Length Options:
  --fixed-index PATH           Intermediate fixed-length index to truncate (autodetected if omitted in varlen)
  --index-name NAME            Custom index filename or path
  --overwrite-index            Replace an existing generated index
  --min-length COUNT           Minimum SID length for varlen (default: 1)
  --max-length COUNT           Maximum SID length for varlen (default: 4)
  --strategy NAME              Truncation strategy: shortest_unique, popularity, collaborative, residual (default: shortest_unique)
                               (supports shorthand e.g. popularity:co_occurrence or popularity:user_entropy)
  --collab-signal SIGNAL       Collaborative signal: frequency, user_entropy, pagerank, co_occurrence, cf_density (default: frequency)
  --inter-file PATH            Interaction JSON for popularity/collaborative strategy
  --cf-emb-file PATH           Path to CF embeddings (.pt, .npy) for cf_density signal
  --residuals-file PATH        Residuals JSON for residual strategy
  --residual-threshold VALUE   Reconstruction error threshold for residual strategy (default: 0.2)

Execution & Device Options:
  --tiger-gpus IDS             CUDA devices for TIGER (default: autodetect, up to 2)
  --lcrec-gpus IDS             CUDA devices for LC-Rec (default: autodetect, up to 4)
  --results-file PATH          Override results JSON path
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

# Options and defaults
DATASET=""
DATA_ROOT="$REPO_ROOT/data"
MODE_ARG="fixed"
TOKENIZER_ARG="rqvae"
EMBEDDING_FILE=""
CF_EMBEDDING=""
RQ_CHECKPOINT=""
RETRAIN_RQVAE=false
RQ_EPOCHS="10000"
RQ_EVAL_STEP="2000"
RQ_DEVICE="cuda:0"
USER_ALPHA=""
USER_BETA=""
USER_NUM_LAYERS=""
USER_NUM_EMB_LIST=""
MIN_LENGTH="1"
MAX_LENGTH="4"
STRATEGY="shortest_unique"
COLLAB_SIGNAL="frequency"
INTER_FILE=""
CF_EMB_FILE=""
RESIDUALS_FILE=""
RESIDUAL_THRESHOLD="0.2"
FIXED_INDEX_PARAM=""
INDEX_NAME=""
MODELS="tiger"
BASE_MODEL="${BASE_MODEL:-huggyllama/llama-7b}"
TIGER_GPUS=""
LCREC_GPUS=""
RESULTS_FILE=""
SKIP_TRAINING=false
RETRAIN_MODEL=false
SKIP_EVALUATION=false
OVERWRITE_INDEX=false
TOKENIZER_ONLY=false
PYTHON_BIN="${PYTHON_BIN:-python3}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --mode|--index-type) MODE_ARG="$2"; shift 2 ;;
    --tokenizer|--tokenizer-type) TOKENIZER_ARG="$2"; shift 2 ;;
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
    --num-layers) USER_NUM_LAYERS="$2"; shift 2 ;;
    --num-emb-list) USER_NUM_EMB_LIST="$2"; shift 2 ;;
    --min-length) MIN_LENGTH="$2"; shift 2 ;;
    --max-length) MAX_LENGTH="$2"; shift 2 ;;
    --strategy) STRATEGY="$2"; shift 2 ;;
    --collab-signal|--popularity-signal) COLLAB_SIGNAL="$2"; shift 2 ;;
    --inter-file) INTER_FILE="$2"; shift 2 ;;
    --cf-emb-file) CF_EMB_FILE="$2"; shift 2 ;;
    --residuals-file) RESIDUALS_FILE="$2"; shift 2 ;;
    --residual-threshold) RESIDUAL_THRESHOLD="$2"; shift 2 ;;
    --fixed-index) FIXED_INDEX_PARAM="$2"; shift 2 ;;
    --index-name) INDEX_NAME="$2"; shift 2 ;;
    --models) MODELS="$2"; shift 2 ;;
    --base-model) BASE_MODEL="$2"; shift 2 ;;
    --tiger-gpus) TIGER_GPUS="$2"; shift 2 ;;
    --lcrec-gpus) LCREC_GPUS="$2"; shift 2 ;;
    --results-file) RESULTS_FILE="$2"; shift 2 ;;
    --skip-training|--eval-only) SKIP_TRAINING=true; shift ;;
    --retrain-model) RETRAIN_MODEL=true; shift ;;
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

# Normalize MODE (must be a single mode: fixed or varlen)
MODE="$(echo "$MODE_ARG" | tr -d ' ' | tr '[:upper:]' '[:lower:]')"
case "$MODE" in
  fixed|fix)
    MODE="fixed"
    ;;
  varlen|variable|variable-length|var)
    MODE="varlen"
    ;;
  *)
    printf 'Unknown mode: %s (choose fixed or varlen)\n' "$MODE_ARG" >&2
    exit 2
    ;;
esac

# Normalize TOKENIZER (must be a single tokenizer: letter or rqvae)
TOKENIZER="$(echo "$TOKENIZER_ARG" | tr -d ' ' | tr '[:upper:]' '[:lower:]')"
case "$TOKENIZER" in
  letter)
    TOKENIZER="letter"
    ;;
  rqvae|rq-vae|vanilla|vanilla_rqvae|vanilla-rqvae)
    TOKENIZER="rqvae"
    ;;
  *)
    printf 'Unknown tokenizer: %s (choose letter or rqvae)\n' "$TOKENIZER_ARG" >&2
    exit 2
    ;;
esac

# Validate lengths for variable-length mode
if [[ "$MODE" == "varlen" ]]; then
  if ! [[ "$MIN_LENGTH" =~ ^[1-9][0-9]*$ ]]; then
    printf '--min-length must be a positive integer: %s\n' "$MIN_LENGTH" >&2
    exit 2
  fi
  if ! [[ "$MAX_LENGTH" =~ ^[1-9][0-9]*$ ]]; then
    printf '--max-length must be a positive integer: %s\n' "$MAX_LENGTH" >&2
    exit 2
  fi
  if [[ "$MIN_LENGTH" -gt "$MAX_LENGTH" ]]; then
    printf '--min-length (%s) cannot exceed --max-length (%s).\n' "$MIN_LENGTH" "$MAX_LENGTH" >&2
    exit 2
  fi
fi

# Resolve NUM_LAYERS
if [[ -n "$USER_NUM_EMB_LIST" ]]; then
  IFS=', ' read -r -a NUM_EMB_LIST <<< "$USER_NUM_EMB_LIST"
  NUM_LAYERS="${#NUM_EMB_LIST[@]}"
elif [[ -n "$USER_NUM_LAYERS" ]]; then
  if ! [[ "$USER_NUM_LAYERS" =~ ^[1-9][0-9]*$ ]]; then
    printf '--num-layers must be a positive integer: %s\n' "$USER_NUM_LAYERS" >&2
    exit 2
  fi
  NUM_LAYERS="$USER_NUM_LAYERS"
  NUM_EMB_LIST=()
  for (( i=0; i<NUM_LAYERS; i++ )); do
    NUM_EMB_LIST+=(256)
  done
else
  NUM_LAYERS="$MAX_LENGTH"
  NUM_EMB_LIST=()
  for (( i=0; i<NUM_LAYERS; i++ )); do
    NUM_EMB_LIST+=(256)
  done
fi

if [[ "$MODE" == "varlen" && "$NUM_LAYERS" -lt "$MAX_LENGTH" ]]; then
  printf '--num-layers (%s) must be at least --max-length (%s).\n' "$NUM_LAYERS" "$MAX_LENGTH" >&2
  exit 2
fi

# Resolve varlen truncation strategy settings
STRAT_SUFFIX=""
STRAT_TAG=""
if [[ "$MODE" == "varlen" ]]; then
  if [[ "$STRATEGY" == *:* ]]; then
    COLLAB_SIGNAL="${STRATEGY#*:}"
    STRATEGY="${STRATEGY%%:*}"
  fi
  case "$STRATEGY" in
    shortest_unique)
      STRAT_SUFFIX=""
      STRAT_TAG=""
      ;;
    popularity|collaborative)
      case "$COLLAB_SIGNAL" in
        frequency|raw|pop)
          STRAT_SUFFIX=".pop"
          STRAT_TAG="-pop"
          ;;
        user_entropy|entropy)
          STRAT_SUFFIX=".pop-entropy"
          STRAT_TAG="-pop-entropy"
          ;;
        pagerank|pr)
          STRAT_SUFFIX=".pop-pagerank"
          STRAT_TAG="-pop-pagerank"
          ;;
        co_occurrence|cooccur|co_occur)
          STRAT_SUFFIX=".pop-cooccur"
          STRAT_TAG="-pop-cooccur"
          ;;
        cf_density|cf_isolation)
          STRAT_SUFFIX=".pop-cf"
          STRAT_TAG="-pop-cf"
          ;;
        *)
          printf 'Unknown collaborative signal: %s (choose frequency, user_entropy, pagerank, co_occurrence, cf_density)\n' "$COLLAB_SIGNAL" >&2
          exit 2
          ;;
      esac
      INTER_FILE="${INTER_FILE:-$DATA_ROOT/$DATASET/$DATASET.inter.json}"
      if [[ -z "$CF_EMB_FILE" && -f "$REPO_ROOT/RQ-VAE/ckpt/$DATASET-32d-sasrec.pt" ]]; then
        CF_EMB_FILE="$REPO_ROOT/RQ-VAE/ckpt/$DATASET-32d-sasrec.pt"
      fi
      if [[ ! -f "$INTER_FILE" && "$COLLAB_SIGNAL" != "cf_density" ]]; then
        printf 'Interaction file not found for %s strategy: %s\n' "$STRATEGY" "$INTER_FILE" >&2
        exit 1
      fi
      ;;
    residual)
      STRAT_SUFFIX=".res"
      STRAT_TAG="-res"
      if [[ -z "$RESIDUALS_FILE" ]]; then
        if [[ "$NUM_LAYERS" -ne 4 && -f "$DATA_ROOT/$DATASET/$DATASET.residuals.L${NUM_LAYERS}.json" ]]; then
          RESIDUALS_FILE="$DATA_ROOT/$DATASET/$DATASET.residuals.L${NUM_LAYERS}.json"
        elif [[ "$MAX_LENGTH" -ne 4 && -f "$DATA_ROOT/$DATASET/$DATASET.residuals.L${MAX_LENGTH}.json" ]]; then
          RESIDUALS_FILE="$DATA_ROOT/$DATASET/$DATASET.residuals.L${MAX_LENGTH}.json"
        elif [[ "$MAX_LENGTH" -ne 4 ]]; then
          RESIDUALS_FILE="$DATA_ROOT/$DATASET/$DATASET.residuals.L${MAX_LENGTH}.json"
        else
          RESIDUALS_FILE="$DATA_ROOT/$DATASET/$DATASET.residuals.json"
        fi
      fi
      if [[ ! -f "$RESIDUALS_FILE" ]]; then
        printf 'Residuals file not found for residual strategy: %s\n' "$RESIDUALS_FILE" >&2
        printf 'Generate it first using RQ-VAE/compute_residuals.py.\n' >&2
        exit 1
      fi
      ;;
    *)
      printf 'Unknown strategy: %s (choose shortest_unique, popularity, collaborative, or residual)\n' "$STRATEGY" >&2
      exit 2
      ;;
  esac
fi

EMBEDDING_FILE="${EMBEDDING_FILE:-$DATA_ROOT/$DATASET/$DATASET.emb-bert-base-uncased-td.npy}"
CF_EMBEDDING="${CF_EMBEDDING:-$REPO_ROOT/RQ-VAE/ckpt/$DATASET-32d-sasrec.pt}"

contains_model() {
  [[ ",$MODELS," == *",$1,"* ]]
}

if [[ "$TOKENIZER_ONLY" != true ]]; then
  if ! contains_model tiger && ! contains_model lcrec; then
    printf 'Select at least one supported model: tiger,lcrec\n' >&2
    exit 2
  fi
  if contains_model lcrec && [[ -z "$BASE_MODEL" ]]; then
    printf '%s\n' '--base-model is required when selecting lcrec.' >&2
    exit 2
  fi
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

# =========================================================================
# Tokenizer Configuration
# =========================================================================

if [[ "$TOKENIZER" == "letter" ]]; then
  TOK_NAME="letter"
  TOK_LABEL="LETTER"
  TOK_ALPHA="${USER_ALPHA:-0.01}"
  TOK_BETA="${USER_BETA:-0.0001}"
  TOK_CKPT_ROOT="$REPO_ROOT/checkpoint/$DATASET/$TOK_NAME"
  TOK_SK_ARGS=()
  TOK_NEEDS_CF=true
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
fi

TOK_INDEX_DIR="$DATA_ROOT/$DATASET/$TOK_NAME"

CUR_RQ_CHECKPOINT="$RQ_CHECKPOINT"

ensure_tokenizer_checkpoint() {
  if [[ -n "$CUR_RQ_CHECKPOINT" && -f "$CUR_RQ_CHECKPOINT" ]]; then
    return 0
  fi

  if [[ -n "$RQ_CHECKPOINT" && -f "$RQ_CHECKPOINT" ]]; then
    CHECK_LAYERS=$("$PYTHON_BIN" -c '
import sys, torch
try:
    ckpt = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
except TypeError:
    ckpt = torch.load(sys.argv[1], map_location="cpu")
args = ckpt.get("args")
if args and hasattr(args, "num_emb_list"):
    print(len(args.num_emb_list))
' "$RQ_CHECKPOINT" 2>/dev/null || true)
    if [[ "$CHECK_LAYERS" =~ ^[0-9]+$ && "$CHECK_LAYERS" -lt "$NUM_LAYERS" ]]; then
      printf "Specified RQ-VAE checkpoint has %s layers, fewer than requested %s.\n" "$CHECK_LAYERS" "$NUM_LAYERS" >&2
      exit 1
    fi
    CUR_RQ_CHECKPOINT="$RQ_CHECKPOINT"
    return 0
  fi

  local detected_ckpt=""
  if [[ "$RETRAIN_RQVAE" != true ]]; then
    detected_ckpt="$(find_latest_checkpoint "$TOK_CKPT_ROOT" "$NUM_LAYERS")"
    if [[ -n "$detected_ckpt" && -f "$detected_ckpt" ]]; then
      CUR_RQ_CHECKPOINT="$detected_ckpt"
      printf '\n[RQ-VAE] [%s] Autodetected existing %s-layer checkpoint: %s\n' "$TOK_LABEL" "$NUM_LAYERS" "$CUR_RQ_CHECKPOINT"
      printf '[RQ-VAE] Reusing existing checkpoint (pass --retrain-rqvae to force training).\n'
      return 0
    fi
  fi

  if [[ ! -f "$EMBEDDING_FILE" ]]; then
    printf 'Item embeddings not found: %s\nRun preprocess_item_embeddings.sh first.\n' "$EMBEDDING_FILE" >&2
    exit 1
  fi
  if [[ "$TOK_NEEDS_CF" == true && ! -f "$CF_EMBEDDING" ]]; then
    printf 'Collaborative-filtering embeddings not found: %s\n' "$CF_EMBEDDING" >&2
    exit 1
  fi

  printf '\n[RQ-VAE] [%s] Training tokenizer with %s layers...\n' "$TOK_LABEL" "$NUM_LAYERS"
  local step_start="$SECONDS"
  mkdir -p "$TOK_CKPT_ROOT"
  local rq_train_cmd=(
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
  if [[ -z "$CUR_RQ_CHECKPOINT" || ! -f "$CUR_RQ_CHECKPOINT" ]]; then
    printf 'RQ-VAE training completed without a best_collision_model.pth checkpoint.\n' >&2
    exit 1
  fi
  printf 'Completed %s training in %s.\n' "$TOK_LABEL" "$(format_duration "$((SECONDS - step_start))")"
  record_phase "[$TOK_LABEL] Tokenizer training" "$((SECONDS - step_start))"
  printf 'Stored %s checkpoint: %s\n' "$TOK_LABEL" "$CUR_RQ_CHECKPOINT"
}

generate_fixed_index_file() {
  local out_file="$1"
  ensure_tokenizer_checkpoint
  printf '\n[Index] [%s] Generating fixed-length item index...\n' "$TOK_LABEL"
  local step_start="$SECONDS"
  mkdir -p "$(dirname "$out_file")"
  "$PYTHON_BIN" "$REPO_ROOT/RQ-VAE/generate_indices.py" \
    --dataset "$DATASET" \
    --checkpoint-path "$CUR_RQ_CHECKPOINT" \
    --output-file "$out_file" \
    --device "$RQ_DEVICE"

  if [[ ! -f "$out_file" ]]; then
    printf 'Index generation completed without creating: %s\n' "$out_file" >&2
    exit 1
  fi
  printf 'Completed %s fixed index generation in %s.\n' "$TOK_LABEL" "$(format_duration "$((SECONDS - step_start))")"
  record_phase "[$TOK_LABEL] Fixed index generation" "$((SECONDS - step_start))"
  printf 'Stored fixed-length item index: %s\n' "$out_file"
}

# =========================================================================
# Mode Execution (fixed or varlen)
# =========================================================================

printf '\n=================================================================\n'
printf ' Mode: %s | Tokenizer: %s | Dataset: %s\n' "$MODE" "$TOK_LABEL" "$DATASET"
printf '=================================================================\n'

TARGET_INDEX_FILE=""
TARGET_INDEX_ARG=""

if [[ "$MODE" == "fixed" ]]; then
  # --- Fixed-Length Mode ---
  if [[ "$NUM_LAYERS" -eq 4 ]]; then
    DEF_NAME="$DATASET.index.fixed.json"
  else
    DEF_NAME="$DATASET.index.fixed.L${NUM_LAYERS}.json"
  fi

  if [[ -n "$INDEX_NAME" ]]; then
    if [[ "$INDEX_NAME" == /* ]]; then
      TARGET_INDEX_FILE="$INDEX_NAME"
      TARGET_INDEX_ARG="$INDEX_NAME"
    else
      TARGET_INDEX_FILE="$TOK_INDEX_DIR/$INDEX_NAME"
      TARGET_INDEX_ARG="$TOK_NAME/$INDEX_NAME"
    fi
  else
    TARGET_INDEX_FILE="$TOK_INDEX_DIR/$DEF_NAME"
    TARGET_INDEX_ARG="$TOK_NAME/$DEF_NAME"
  fi

  if [[ -f "$TARGET_INDEX_FILE" && "$OVERWRITE_INDEX" != true && "$RETRAIN_RQVAE" != true ]]; then
    printf '\n[Index] [%s] [Fixed] Found existing index: %s\n' "$TOK_LABEL" "$TARGET_INDEX_FILE"
    printf '[Index] Reusing existing index (pass --overwrite-index to regenerate).\n'
  else
    generate_fixed_index_file "$TARGET_INDEX_FILE"
  fi

  if [[ "$NUM_LAYERS" -eq 4 ]]; then
    CUR_TIGER_CKPT="./ckpt/$DATASET/$TOK_NAME/fixed"
    CUR_LCREC_CKPT="./ckpt/$DATASET/$TOK_NAME/fixed"
    CUR_TIGER_DEFAULT_RES="./results/$DATASET/$TOK_NAME/fixed.json"
    CUR_LCREC_DEFAULT_RES="./results/$DATASET/$TOK_NAME/fixed.json"
    CUR_LCREC_WANDB="${DATASET}-${TOK_NAME}-fixed"
  else
    CUR_TIGER_CKPT="./ckpt/$DATASET/$TOK_NAME/fixed-L${NUM_LAYERS}"
    CUR_LCREC_CKPT="./ckpt/$DATASET/$TOK_NAME/fixed-L${NUM_LAYERS}"
    CUR_TIGER_DEFAULT_RES="./results/$DATASET/$TOK_NAME/fixed_L${NUM_LAYERS}.json"
    CUR_LCREC_DEFAULT_RES="./results/$DATASET/$TOK_NAME/fixed_L${NUM_LAYERS}.json"
    CUR_LCREC_WANDB="${DATASET}-${TOK_NAME}-fixed-L${NUM_LAYERS}"
  fi

else
  # --- Variable-Length Mode ---
  # 1. Resolve intermediate fixed index
  INTERMEDIATE_FIXED_FILE=""
  REGEN_INTERMEDIATE=false

  if [[ -n "$FIXED_INDEX_PARAM" ]]; then
    if [[ "$FIXED_INDEX_PARAM" == /* ]]; then
      INTERMEDIATE_FIXED_FILE="$FIXED_INDEX_PARAM"
    elif [[ -f "$CALLER_DIR/$FIXED_INDEX_PARAM" ]]; then
      INTERMEDIATE_FIXED_FILE="$CALLER_DIR/$FIXED_INDEX_PARAM"
    elif [[ -f "$TOK_INDEX_DIR/$FIXED_INDEX_PARAM" ]]; then
      INTERMEDIATE_FIXED_FILE="$TOK_INDEX_DIR/$FIXED_INDEX_PARAM"
    else
      INTERMEDIATE_FIXED_FILE="$TOK_INDEX_DIR/$FIXED_INDEX_PARAM"
    fi
    if [[ ! -f "$INTERMEDIATE_FIXED_FILE" ]]; then
      printf 'Specified --fixed-index not found: %s\n' "$INTERMEDIATE_FIXED_FILE" >&2
      exit 1
    fi
  else
    # Autodetect intermediate index in tokenizer directory
    if [[ "$NUM_LAYERS" -eq 4 ]]; then
      cand1="$TOK_INDEX_DIR/$DATASET.index.fixed.json"
      cand2="$TOK_INDEX_DIR/$DATASET.index.fixed-for-varlen.json"
    else
      cand1="$TOK_INDEX_DIR/$DATASET.index.fixed.L${NUM_LAYERS}.json"
      cand2="$TOK_INDEX_DIR/$DATASET.index.fixed-for-varlen.L${NUM_LAYERS}.json"
    fi

    if [[ -f "$cand1" ]]; then
      INTERMEDIATE_FIXED_FILE="$cand1"
    elif [[ -f "$cand2" ]]; then
      INTERMEDIATE_FIXED_FILE="$cand2"
    else
      INTERMEDIATE_FIXED_FILE="$cand2"
      REGEN_INTERMEDIATE=true
    fi
  fi

  # Check length of existing intermediate index
  if [[ -f "$INTERMEDIATE_FIXED_FILE" && "$REGEN_INTERMEDIATE" != true ]]; then
    CHECK_TOKENS=$("$PYTHON_BIN" -c '
import sys, json
try:
    with open(sys.argv[1]) as f:
        d = json.load(f)
    if not d:
        print("0"); sys.exit(0)
    first_val = next(iter(d.values()))
    print(len(first_val) if isinstance(first_val, list) else "0")
except Exception:
    print("-1")
' "$INTERMEDIATE_FIXED_FILE" 2>/dev/null || echo "-1")

    if [[ "$CHECK_TOKENS" =~ ^[0-9]+$ && "$CHECK_TOKENS" -lt "$MAX_LENGTH" ]]; then
      if [[ -n "$FIXED_INDEX_PARAM" ]]; then
        printf 'Specified --fixed-index (%s) has %s tokens, fewer than --max-length (%s).\n' \
          "$INTERMEDIATE_FIXED_FILE" "$CHECK_TOKENS" "$MAX_LENGTH" >&2
        exit 1
      else
        printf '\n[Fixed index] Warning: %s has %s tokens, fewer than --max-length (%s).\n' \
          "$INTERMEDIATE_FIXED_FILE" "$CHECK_TOKENS" "$MAX_LENGTH"
        printf '[Fixed index] Will generate an intermediate fixed index with %s layers.\n' "$NUM_LAYERS"
        if [[ "$NUM_LAYERS" -eq 4 ]]; then
          INTERMEDIATE_FIXED_FILE="$TOK_INDEX_DIR/$DATASET.index.fixed-for-varlen.json"
        else
          INTERMEDIATE_FIXED_FILE="$TOK_INDEX_DIR/$DATASET.index.fixed-for-varlen.L${NUM_LAYERS}.json"
        fi
        REGEN_INTERMEDIATE=true
      fi
    fi
  fi

  if [[ ! -f "$INTERMEDIATE_FIXED_FILE" || "$OVERWRITE_INDEX" == true || "$RETRAIN_RQVAE" == true || "$REGEN_INTERMEDIATE" == true ]]; then
    generate_fixed_index_file "$INTERMEDIATE_FIXED_FILE"
  else
    printf '\n[Fixed index] [%s] Reusing intermediate fixed index: %s\n' "$TOK_LABEL" "$INTERMEDIATE_FIXED_FILE"
  fi

  # 2. Resolve variable index filename and path
  if [[ "$MAX_LENGTH" -eq 4 && "$MIN_LENGTH" -eq 1 ]]; then
    VAR_TAG=""
  elif [[ "$MIN_LENGTH" -eq 1 ]]; then
    VAR_TAG=".max${MAX_LENGTH}"
  else
    VAR_TAG=".min${MIN_LENGTH}-max${MAX_LENGTH}"
  fi

  DEF_VAR_NAME="$DATASET.index.varlen${STRAT_SUFFIX}${VAR_TAG}.json"

  if [[ -n "$INDEX_NAME" ]]; then
    if [[ "$INDEX_NAME" == /* ]]; then
      TARGET_INDEX_FILE="$INDEX_NAME"
      TARGET_INDEX_ARG="$INDEX_NAME"
    else
      TARGET_INDEX_FILE="$TOK_INDEX_DIR/$INDEX_NAME"
      TARGET_INDEX_ARG="$TOK_NAME/$INDEX_NAME"
    fi
  else
    TARGET_INDEX_FILE="$TOK_INDEX_DIR/$DEF_VAR_NAME"
    TARGET_INDEX_ARG="$TOK_NAME/$DEF_VAR_NAME"
  fi

  # 3. Generate variable index via truncation if needed
  if [[ -f "$TARGET_INDEX_FILE" && "$OVERWRITE_INDEX" != true ]]; then
    printf '\n[Variable index] [%s] Found existing variable-length index: %s\n' "$TOK_LABEL" "$TARGET_INDEX_FILE"
    printf '[Variable index] Reusing existing index (pass --overwrite-index to regenerate).\n'
  else
    printf '\n[Variable index] [%s] Creating variable-length index (%s)...\n' "$TOK_LABEL" "$STRATEGY"
    STEP_START="$SECONDS"
    mkdir -p "$(dirname "$TARGET_INDEX_FILE")"
    truncate_args=(
      "$PYTHON_BIN" "$REPO_ROOT/RQ-VAE/truncate_indices.py"
      --input "$INTERMEDIATE_FIXED_FILE"
      --output "$TARGET_INDEX_FILE"
      --min-length "$MIN_LENGTH"
      --max-length "$MAX_LENGTH"
      --strategy "$STRATEGY"
    )
    if [[ "$STRATEGY" == "popularity" || "$STRATEGY" == "collaborative" ]]; then
      if [[ -n "$INTER_FILE" ]]; then
        truncate_args+=(--inter-file "$INTER_FILE")
      fi
      truncate_args+=(--collab-signal "$COLLAB_SIGNAL")
      CF_CANDIDATE="${CF_EMB_FILE:-${CF_EMBEDDING}}"
      if [[ -n "$CF_CANDIDATE" ]]; then
        truncate_args+=(--cf-emb-file "$CF_CANDIDATE")
      fi
    elif [[ "$STRATEGY" == "residual" ]]; then
      truncate_args+=(--residuals-file "$RESIDUALS_FILE" --residual-threshold "$RESIDUAL_THRESHOLD")
    fi
    "${truncate_args[@]}"

    if [[ ! -f "$TARGET_INDEX_FILE" ]]; then
      printf 'Variable-length index generation completed without creating: %s\n' "$TARGET_INDEX_FILE" >&2
      exit 1
    fi
    printf 'Completed %s variable-length index generation in %s.\n' \
      "$TOK_LABEL" "$(format_duration "$((SECONDS - STEP_START))")"
    record_phase "[$TOK_LABEL] Variable index generation" "$((SECONDS - STEP_START))"
    printf 'Stored variable-length item index: %s\n' "$TARGET_INDEX_FILE"
    printf 'Stored variable-length index summary: %s\n' "${TARGET_INDEX_FILE%.json}.summary.json"
  fi

  # 4. Downstream checkpoint and results paths for varlen
  if [[ "$MAX_LENGTH" -eq 4 && "$MIN_LENGTH" -eq 1 ]]; then
    CKPT_TAG="varlen${STRAT_TAG}"
    RES_TAG="varlen${STRAT_TAG}"
  elif [[ "$MIN_LENGTH" -eq 1 ]]; then
    CKPT_TAG="varlen${STRAT_TAG}-max${MAX_LENGTH}"
    RES_TAG="varlen${STRAT_TAG}_max${MAX_LENGTH}"
  else
    CKPT_TAG="varlen${STRAT_TAG}-min${MIN_LENGTH}-max${MAX_LENGTH}"
    RES_TAG="varlen${STRAT_TAG}_min${MIN_LENGTH}-max${MAX_LENGTH}"
  fi

  CUR_TIGER_CKPT="./ckpt/$DATASET/$TOK_NAME/$CKPT_TAG"
  CUR_LCREC_CKPT="./ckpt/$DATASET/$TOK_NAME/$CKPT_TAG"
  CUR_TIGER_DEFAULT_RES="./results/$DATASET/$TOK_NAME/${RES_TAG}.json"
  CUR_LCREC_DEFAULT_RES="./results/$DATASET/$TOK_NAME/${RES_TAG}.json"
  CUR_LCREC_WANDB="${DATASET}-${TOK_NAME}-${CKPT_TAG}"
fi

# Resolve results file override if given
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
  CUR_TIGER_RESULTS="$CUR_TIGER_DEFAULT_RES"
  CUR_LCREC_RESULTS="$CUR_LCREC_DEFAULT_RES"
fi

if [[ "$TOKENIZER_ONLY" == true ]]; then
  print_phase_durations
  printf '\nTokenizer and index generation pipeline completed in %s.\n' \
    "$(format_duration "$((SECONDS - PIPELINE_START))")"
  exit 0
fi

# =========================================================================
# Train and Evaluate Downstream Models (TIGER, LC-Rec)
# =========================================================================

if contains_model tiger; then
  tiger_ckpt_dir="$CUR_TIGER_CKPT"
  if [[ "$tiger_ckpt_dir" != /* ]]; then
    tiger_ckpt_dir="$REPO_ROOT/LETTER-TIGER/$tiger_ckpt_dir"
  fi
  tiger_trained=false
  if [[ -f "$tiger_ckpt_dir/pytorch_model.bin" || -f "$tiger_ckpt_dir/model.safetensors" || -f "$tiger_ckpt_dir/trainer_state.json" ]]; then
    tiger_trained=true
  fi

  if [[ "$tiger_trained" == true && "$RETRAIN_MODEL" != true ]]; then
    printf '\n[TIGER] [%s] [%s] Found existing trained model: %s\n' "$TOK_LABEL" "$MODE" "$CUR_TIGER_CKPT"
    printf '[TIGER] Skipping training and resuming straight to evaluation (pass --retrain-model to retrain).\n'
  elif [[ "$SKIP_TRAINING" == true ]]; then
    printf '\n[TIGER] [%s] [%s] Skipping training (--skip-training specified).\n' "$TOK_LABEL" "$MODE"
  else
    printf '\n[TIGER] [%s] [%s] Training...\n' "$TOK_LABEL" "$MODE"
    STEP_START="$SECONDS"
    mkdir -p "$REPO_ROOT/LETTER-TIGER/$(dirname "$CUR_TIGER_RESULTS")"
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
          --index_file "$TARGET_INDEX_ARG" \
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
          --index_file "$TARGET_INDEX_ARG" \
          --temperature 1.0
      fi
    )
    printf 'Completed [%s] [%s] LETTER-TIGER training in %s.\n' \
      "$TOK_LABEL" "$MODE" "$(format_duration "$((SECONDS - STEP_START))")"
    record_phase "[$TOK_LABEL] [$MODE] LETTER-TIGER training" "$((SECONDS - STEP_START))"
    printf 'Stored LETTER-TIGER checkpoint: %s\n' "$CUR_TIGER_CKPT"
  fi

  if [[ "$SKIP_EVALUATION" != true ]]; then
    printf '\n[TIGER] [%s] [%s] Evaluating...\n' "$TOK_LABEL" "$MODE"
    STEP_START="$SECONDS"
    (
      cd "$REPO_ROOT/LETTER-TIGER"
      mkdir -p "$(dirname "$CUR_TIGER_RESULTS")"
      "$PYTHON_BIN" test.py \
        --gpu_id 0 \
        --ckpt_path "$CUR_TIGER_CKPT" \
        --dataset "$DATASET" \
        --data_path "$DATA_ROOT" \
        --results_file "$CUR_TIGER_RESULTS" \
        --test_batch_size 32 \
        --num_beams 20 \
        --test_prompt_ids 0 \
        --index_file "$TARGET_INDEX_ARG"
    )
    printf 'Completed [%s] [%s] LETTER-TIGER evaluation in %s.\n' \
      "$TOK_LABEL" "$MODE" "$(format_duration "$((SECONDS - STEP_START))")"
    record_phase "[$TOK_LABEL] [$MODE] LETTER-TIGER evaluation" "$((SECONDS - STEP_START))"
    printf 'Stored LETTER-TIGER metrics: %s\n' "$CUR_TIGER_RESULTS"
  fi
fi

if contains_model lcrec; then
  lcrec_ckpt_dir="$CUR_LCREC_CKPT"
  if [[ "$lcrec_ckpt_dir" != /* ]]; then
    lcrec_ckpt_dir="$REPO_ROOT/LETTER-LC-Rec/$lcrec_ckpt_dir"
  fi
  lcrec_trained=false
  if [[ -f "$lcrec_ckpt_dir/adapter_model.bin" || -f "$lcrec_ckpt_dir/adapter_model.safetensors" || -f "$lcrec_ckpt_dir/pytorch_model.bin" || -f "$lcrec_ckpt_dir/model.safetensors" || -f "$lcrec_ckpt_dir/trainer_state.json" ]]; then
    lcrec_trained=true
  fi

  if [[ "$lcrec_trained" == true && "$RETRAIN_MODEL" != true ]]; then
    printf '\n[LC-Rec] [%s] [%s] Found existing trained model: %s\n' "$TOK_LABEL" "$MODE" "$CUR_LCREC_CKPT"
    printf '[LC-Rec] Skipping training and resuming straight to evaluation (pass --retrain-model to retrain).\n'
  elif [[ "$SKIP_TRAINING" == true ]]; then
    printf '\n[LC-Rec] [%s] [%s] Skipping training (--skip-training specified).\n' "$TOK_LABEL" "$MODE"
  else
    printf '\n[LC-Rec] [%s] [%s] Training...\n' "$TOK_LABEL" "$MODE"
    STEP_START="$SECONDS"
    mkdir -p "$REPO_ROOT/LETTER-LC-Rec/$(dirname "$CUR_LCREC_RESULTS")"
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
          --index_file "$TARGET_INDEX_ARG" \
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
          --index_file "$TARGET_INDEX_ARG" \
          --wandb_run_name "$CUR_LCREC_WANDB" \
          --temperature 1.0
      fi
    )
    printf 'Completed [%s] [%s] LETTER-LC-Rec training in %s.\n' \
      "$TOK_LABEL" "$MODE" "$(format_duration "$((SECONDS - STEP_START))")"
    record_phase "[$TOK_LABEL] [$MODE] LETTER-LC-Rec training" "$((SECONDS - STEP_START))"
    printf 'Stored LETTER-LC-Rec checkpoint: %s\n' "$CUR_LCREC_CKPT"
  fi

  if [[ "$SKIP_EVALUATION" != true ]]; then
    printf '\n[LC-Rec] [%s] [%s] Evaluating...\n' "$TOK_LABEL" "$MODE"
    STEP_START="$SECONDS"
    (
      cd "$REPO_ROOT/LETTER-LC-Rec"
      mkdir -p "$(dirname "$CUR_LCREC_RESULTS")"
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
        --index_file "$TARGET_INDEX_ARG"
    )
    printf 'Completed [%s] [%s] LETTER-LC-Rec evaluation in %s.\n' \
      "$TOK_LABEL" "$MODE" "$(format_duration "$((SECONDS - STEP_START))")"
    record_phase "[$TOK_LABEL] [$MODE] LETTER-LC-Rec evaluation" "$((SECONDS - STEP_START))"
    printf 'Stored LETTER-LC-Rec metrics: %s\n' "$CUR_LCREC_RESULTS"
  fi
fi

print_phase_durations
printf '\nEnd-to-end recommendation pipeline completed in %s.\n' \
  "$(format_duration "$((SECONDS - PIPELINE_START))")"
