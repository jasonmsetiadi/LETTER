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
  --phase PHASE                Training phase: 1 (fixed-length training, Phase 1) or 1.5 (length-aware item-dependent training, Phase 1.5, default: 1)
  --target-lengths PATH        Target lengths JSON or index file override for Phase 1.5
  --tokenizer-only             Stop after index generation (skip downstream recommenders)
  --resume                     Check and reuse existing checkpoints/indices at each stage (default: false, runs from scratch)

RQ-VAE / Tokenizer Options:
  --embedding-file PATH        Item embedding .npy path
  --cf-embedding PATH          Collaborative-filtering embedding .pt path (required for letter)
  --rqvae-epochs COUNT         RQ-VAE epochs (default: 10000)
  --rqvae-eval-step COUNT      RQ-VAE validation interval (default: 2000)
  --rqvae-device DEVICE        RQ-VAE device (default: cuda:0)
  --alpha VALUE                Collaborative-loss weight (default: 0.01 for letter, 0.0 for rqvae)
  --beta VALUE                 Diversity-loss weight (default: 0.0001 for letter, 0.0 for rqvae)
  --num-layers COUNT           Number of RQ-VAE codebook layers / SID length (default: 4 or max-length)
  --num-emb-list LIST          Explicit codebook sizes (e.g. "256 256 256 256")

Index Generation & Variable-Length Options:
  --min-length COUNT           Minimum SID length for varlen (default: 1)
  --max-length COUNT           Maximum SID length for varlen (default: 4)
  --strategy NAME              Truncation strategy: shortest_unique, popularity, collaborative, residual (default: shortest_unique)
                               (supports shorthand e.g. popularity:co_occurrence or popularity:user_entropy)
  --collab-signal SIGNAL       Collaborative signal: frequency, user_entropy, pagerank, co_occurrence, cf_density (default: frequency)
  --inter-file PATH            Interaction JSON for popularity/collaborative strategy
  --cf-emb-file PATH           Path to CF embeddings (.pt, .npy) for cf_density signal
  --residuals-file PATH        Residuals JSON for residual strategy (computed automatically if omitted)
  --residual-threshold VALUE   Reconstruction error threshold for residual strategy (default: 0.2)

Execution & Device Options:
  --tiger-gpus IDS             CUDA devices for TIGER (default: autodetect, up to 2)
  --lcrec-gpus IDS             CUDA devices for LC-Rec (default: autodetect, up to 4)
  --results-file PATH          Override results JSON path
  --keep-checkpoints           Keep model checkpoints after evaluation (default: false, automatically cleaned up)
  --clean-checkpoints          Delete model checkpoints after evaluation completes (default: true, retains results JSON)
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
RESUME=false
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
MODELS="tiger"
BASE_MODEL="${BASE_MODEL:-huggyllama/llama-7b}"
PHASE="1"
TARGET_LENGTHS=""
TIGER_GPUS=""
LCREC_GPUS=""
RESULTS_FILE=""
CLEAN_CHECKPOINTS="${CLEAN_CHECKPOINTS:-true}"
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
    --resume) RESUME=true; shift ;;
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
    --phase|--training-phase) PHASE="$2"; shift 2 ;;
    --target-lengths|--target_lengths) TARGET_LENGTHS="$2"; shift 2 ;;
    --models) MODELS="$2"; shift 2 ;;
    --base-model) BASE_MODEL="$2"; shift 2 ;;
    --tiger-gpus) TIGER_GPUS="$2"; shift 2 ;;
    --lcrec-gpus) LCREC_GPUS="$2"; shift 2 ;;
    --results-file) RESULTS_FILE="$2"; shift 2 ;;
    --clean-checkpoints|--delete-ckpt-after-eval) CLEAN_CHECKPOINTS=true; shift ;;
    --keep-checkpoints|--no-clean-checkpoints) CLEAN_CHECKPOINTS=false; shift ;;
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
if [[ -n "$BASE_MODEL" && -d "$CALLER_DIR/$BASE_MODEL" ]]; then
  BASE_MODEL="$CALLER_DIR/$BASE_MODEL"
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  printf 'Python executable not found: %s\n' "$PYTHON_BIN" >&2
  exit 1
fi

# Normalize PHASE
case "$PHASE" in
  1|1.0)
    PHASE="1"
    ;;
  1.5)
    PHASE="1.5"
    ;;
  *)
    printf 'Unknown phase: %s (choose 1 or 1.5)\n' "$PHASE" >&2
    exit 2
    ;;
esac

if [[ -n "$TARGET_LENGTHS" && "$TARGET_LENGTHS" != /* ]]; then
  TARGET_LENGTHS="$CALLER_DIR/$TARGET_LENGTHS"
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

if [[ "$PHASE" == "1.5" && "$MODE" != "varlen" ]]; then
  printf 'Phase 1.5 performs length-aware variable-length training; setting --mode to varlen.\n'
  MODE="varlen"
fi

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
      if [[ "$PHASE" == "1.5" && -z "$TARGET_LENGTHS" && -z "$FIXED_INDEX_PARAM" ]]; then
        cand1="$DATA_ROOT/$DATASET/$TOKENIZER/$DATASET.index.fixed.L${NUM_LAYERS}.json"
        cand2="$DATA_ROOT/$DATASET/$DATASET.index.fixed.L${NUM_LAYERS}.json"
        if [[ -f "$cand1" ]]; then
          FIXED_INDEX_PARAM="$cand1"
          printf '[Phase 1.5] Autodetected reference fixed index for shortest_unique: %s\n' "$FIXED_INDEX_PARAM"
        elif [[ -f "$cand2" ]]; then
          FIXED_INDEX_PARAM="$cand2"
          printf '[Phase 1.5] Autodetected reference fixed index for shortest_unique: %s\n' "$FIXED_INDEX_PARAM"
        else
          printf '[Phase 1.5] Strategy "shortest_unique" requires a reference fixed-length index to determine prefix uniqueness.\n' >&2
          printf 'Please provide --target-lengths <path>, or run Phase 1 first.\n' >&2
          exit 1
        fi
      fi
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
      if [[ "$PHASE" != "1.5" ]]; then
        if [[ -z "$RESIDUALS_FILE" ]]; then
          RESIDUALS_FILE="$DATA_ROOT/$DATASET/$DATASET.residuals.L${MAX_LENGTH}.json"
        fi
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
  local desired_phase="${3:-1.0}"
  if [[ ! -d "$root" ]]; then
    return 0
  fi
  "$PYTHON_BIN" -c '
import sys, glob, os, torch
root = sys.argv[1]
desired_layers = int(sys.argv[2])
desired_phase = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
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
        ckpt_phase = float(ckpt.get("phase", getattr(args, "phase", 1.0)))
        if abs(ckpt_phase - desired_phase) < 0.1:
            valid.append(c)
if valid:
    print(max(valid, key=os.path.getmtime))
' "$root" "$desired_layers" "$desired_phase" 2>/dev/null || true
}

# =========================================================================
# Tokenizer Configuration
# =========================================================================

if [[ "$TOKENIZER" == "letter" ]]; then
  TOK_NAME="letter"
  TOK_LABEL="LETTER"
  TOK_ALPHA="${USER_ALPHA:-0.01}"
  TOK_BETA="${USER_BETA:-0.0001}"
  if [[ "$PHASE" == "1.5" ]]; then
    TOK_CKPT_ROOT="$REPO_ROOT/checkpoint/$DATASET/$TOK_NAME/phase1.5${STRAT_TAG}"
  else
    TOK_CKPT_ROOT="$REPO_ROOT/checkpoint/$DATASET/$TOK_NAME"
  fi
  TOK_SK_ARGS=()
  TOK_NEEDS_CF=true
else
  TOK_NAME="rqvae"
  TOK_LABEL="Vanilla RQ-VAE"
  TOK_ALPHA="${USER_ALPHA:-0.0}"
  TOK_BETA="${USER_BETA:-0.0}"
  if [[ "$PHASE" == "1.5" ]]; then
    TOK_CKPT_ROOT="$REPO_ROOT/checkpoint/$DATASET/$TOK_NAME/phase1.5${STRAT_TAG}"
  else
    TOK_CKPT_ROOT="$REPO_ROOT/checkpoint/$DATASET/$TOK_NAME"
  fi
  TOK_SK_ARGS=(--sk_epsilons)
  for (( i=0; i<NUM_LAYERS; i++ )); do
    TOK_SK_ARGS+=(0.0)
  done
  TOK_NEEDS_CF=false
fi

TOK_INDEX_DIR="$DATA_ROOT/$DATASET/$TOK_NAME"

CUR_RQ_CHECKPOINT=""

ensure_tokenizer_checkpoint() {
  if [[ -n "$CUR_RQ_CHECKPOINT" && -f "$CUR_RQ_CHECKPOINT" ]]; then
    return 0
  fi

  local detected_ckpt=""
  if [[ "$RESUME" == true ]]; then
    detected_ckpt="$(find_latest_checkpoint "$TOK_CKPT_ROOT" "$NUM_LAYERS" "$PHASE")"
    if [[ -n "$detected_ckpt" && -f "$detected_ckpt" ]]; then
      CUR_RQ_CHECKPOINT="$detected_ckpt"
      printf '\n[RQ-VAE] [%s] Found existing %s-layer (Phase %s) checkpoint: %s\n' "$TOK_LABEL" "$NUM_LAYERS" "$PHASE" "$CUR_RQ_CHECKPOINT"
      printf '[RQ-VAE] Reusing existing checkpoint.\n'
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

  printf '\n[RQ-VAE] [%s] Training tokenizer (Phase %s) with %s layers...\n' "$TOK_LABEL" "$PHASE" "$NUM_LAYERS"
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
  if [[ "$PHASE" == "1.5" ]]; then
    rq_train_cmd+=(
      --phase 1.5
      --target_length_strategy "$STRATEGY"
      --min_length "$MIN_LENGTH"
      --max_length "$MAX_LENGTH"
    )
    if [[ -n "$TARGET_LENGTHS" ]]; then
      rq_train_cmd+=(--target_lengths "$TARGET_LENGTHS")
    fi
    if [[ -n "$INTER_FILE" ]]; then
      rq_train_cmd+=(--inter_file "$INTER_FILE")
    fi
    if [[ -n "$COLLAB_SIGNAL" ]]; then
      rq_train_cmd+=(--collab_signal "$COLLAB_SIGNAL")
    fi
    if [[ -n "$RESIDUALS_FILE" ]]; then
      rq_train_cmd+=(--residuals_file "$RESIDUALS_FILE")
    fi
    if [[ "$STRATEGY" == "residual" && -n "$RESIDUAL_THRESHOLD" ]]; then
      rq_train_cmd+=(--residual_threshold "$RESIDUAL_THRESHOLD")
    fi
    if [[ -n "$FIXED_INDEX_PARAM" ]]; then
      rq_train_cmd+=(--fixed_index_file "$FIXED_INDEX_PARAM")
    fi
  fi

  "${rq_train_cmd[@]}"

  CUR_RQ_CHECKPOINT="$(find_latest_checkpoint "$TOK_CKPT_ROOT" "$NUM_LAYERS" "$PHASE")"
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

generate_varlen_index_phase1_5() {
  local out_file="$1"
  ensure_tokenizer_checkpoint
  printf '\n[Index] [%s] Generating Phase 1.5 variable-length item index...\n' "$TOK_LABEL"
  local step_start="$SECONDS"
  mkdir -p "$(dirname "$out_file")"
  local gen_cmd=(
    "$PYTHON_BIN" "$REPO_ROOT/RQ-VAE/generate_indices.py"
    --dataset "$DATASET"
    --checkpoint-path "$CUR_RQ_CHECKPOINT"
    --output-file "$out_file"
    --device "$RQ_DEVICE"
    --phase 1.5
  )
  if [[ -n "$TARGET_LENGTHS" ]]; then
    gen_cmd+=(--target-lengths "$TARGET_LENGTHS")
  fi
  "${gen_cmd[@]}"

  if [[ ! -f "$out_file" ]]; then
    printf 'Phase 1.5 index generation completed without creating: %s\n' "$out_file" >&2
    exit 1
  fi
  printf 'Completed %s Phase 1.5 variable index generation in %s.\n' "$TOK_LABEL" "$(format_duration "$((SECONDS - step_start))")"
  record_phase "[$TOK_LABEL] Phase 1.5 variable index generation" "$((SECONDS - step_start))"
  printf 'Stored Phase 1.5 variable-length item index: %s\n' "$out_file"
  printf 'Stored Phase 1.5 variable-length index summary: %s\n' "${out_file%.json}.summary.json"
}

# =========================================================================
# Mode Execution (fixed or varlen)
# =========================================================================

printf '\n=================================================================\n'
if [[ "$MODE" == "varlen" ]]; then
  printf ' Mode: %s (Phase %s) | Tokenizer: %s | Dataset: %s\n' "$MODE" "$PHASE" "$TOK_LABEL" "$DATASET"
else
  printf ' Mode: %s | Tokenizer: %s | Dataset: %s\n' "$MODE" "$TOK_LABEL" "$DATASET"
fi
printf '=================================================================\n'

TARGET_INDEX_FILE=""
TARGET_INDEX_ARG=""

if [[ "$MODE" == "fixed" ]]; then
  # --- Fixed-Length Mode ---
  DEF_NAME="$DATASET.index.fixed.L${NUM_LAYERS}.json"
  TARGET_INDEX_FILE="$TOK_INDEX_DIR/$DEF_NAME"
  TARGET_INDEX_ARG="$TOK_NAME/$DEF_NAME"

  if [[ "$RESUME" == true && -f "$TARGET_INDEX_FILE" ]]; then
    printf '\n[Index] [%s] [Fixed] Found existing index: %s\n' "$TOK_LABEL" "$TARGET_INDEX_FILE"
    printf '[Index] Reusing existing index.\n'
  else
    generate_fixed_index_file "$TARGET_INDEX_FILE"
  fi

  CUR_TIGER_CKPT="./ckpt/$DATASET/$TOK_NAME/fixed-L${NUM_LAYERS}"
  CUR_LCREC_CKPT="./ckpt/$DATASET/$TOK_NAME/fixed-L${NUM_LAYERS}"
  CUR_TIGER_DEFAULT_RES="./results/$DATASET/$TOK_NAME/fixed_L${NUM_LAYERS}.json"
  CUR_LCREC_DEFAULT_RES="./results/$DATASET/$TOK_NAME/fixed_L${NUM_LAYERS}.json"
  CUR_LCREC_WANDB="${DATASET}-${TOK_NAME}-fixed-L${NUM_LAYERS}"

else
  # --- Variable-Length Mode ---
  if [[ "$MIN_LENGTH" -eq 1 ]]; then
    VAR_TAG=".max${MAX_LENGTH}"
  else
    VAR_TAG=".min${MIN_LENGTH}-max${MAX_LENGTH}"
  fi

  if [[ "$PHASE" == "1.5" ]]; then
    DEF_VAR_NAME="$DATASET.index.varlen${STRAT_SUFFIX}-phase1.5${VAR_TAG}.json"
  else
    DEF_VAR_NAME="$DATASET.index.varlen${STRAT_SUFFIX}${VAR_TAG}.json"
  fi

  TARGET_INDEX_FILE="$TOK_INDEX_DIR/$DEF_VAR_NAME"
  TARGET_INDEX_ARG="$TOK_NAME/$DEF_VAR_NAME"

  if [[ "$PHASE" == "1.5" ]]; then
    # Phase 1.5: Direct length-aware generation
    if [[ "$RESUME" == true && -f "$TARGET_INDEX_FILE" ]]; then
      printf '\n[Variable index] [%s] [Phase 1.5] Found existing index: %s\n' "$TOK_LABEL" "$TARGET_INDEX_FILE"
      printf '[Variable index] Reusing existing index.\n'
    else
      generate_varlen_index_phase1_5 "$TARGET_INDEX_FILE"
    fi
  else
    # Phase 1: Post-hoc truncation from intermediate fixed index
    if [[ "$RESUME" == true && -f "$TARGET_INDEX_FILE" ]]; then
      printf '\n[Variable index] [%s] Found existing variable-length index: %s\n' "$TOK_LABEL" "$TARGET_INDEX_FILE"
      printf '[Variable index] Reusing existing index.\n'
    else
      INTERMEDIATE_FIXED_FILE="$TOK_INDEX_DIR/$DATASET.index.fixed-for-varlen.L${NUM_LAYERS}.json"
      reuse_intermediate=false
      if [[ "$RESUME" == true ]]; then
        cand1="$TOK_INDEX_DIR/$DATASET.index.fixed.L${NUM_LAYERS}.json"
        cand2="$TOK_INDEX_DIR/$DATASET.index.fixed-for-varlen.L${NUM_LAYERS}.json"

        if [[ -f "$cand1" ]]; then
          INTERMEDIATE_FIXED_FILE="$cand1"
          reuse_intermediate=true
        elif [[ -f "$cand2" ]]; then
          INTERMEDIATE_FIXED_FILE="$cand2"
          reuse_intermediate=true
        fi
      fi

      # Check length of existing intermediate index if attempting reuse
      if [[ "$reuse_intermediate" == true && -f "$INTERMEDIATE_FIXED_FILE" ]]; then
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
          printf '\n[Fixed index] Warning: %s has %s tokens, fewer than --max-length (%s).\n' \
            "$INTERMEDIATE_FIXED_FILE" "$CHECK_TOKENS" "$MAX_LENGTH"
          INTERMEDIATE_FIXED_FILE="$TOK_INDEX_DIR/$DATASET.index.fixed-for-varlen.L${NUM_LAYERS}.json"
          reuse_intermediate=false
        fi
      fi

      if [[ "$reuse_intermediate" == true && -f "$INTERMEDIATE_FIXED_FILE" ]]; then
        printf '\n[Fixed index] [%s] Reusing intermediate fixed index: %s\n' "$TOK_LABEL" "$INTERMEDIATE_FIXED_FILE"
      else
        generate_fixed_index_file "$INTERMEDIATE_FIXED_FILE"
      fi

      # Generate variable index via truncation
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
        if [[ ! -f "$RESIDUALS_FILE" ]]; then
          ensure_tokenizer_checkpoint
          printf '\n[Residuals] [%s] Computing reconstruction residuals per item...\n' "$TOK_LABEL"
          RES_START="$SECONDS"
          mkdir -p "$(dirname "$RESIDUALS_FILE")"
          "$PYTHON_BIN" "$REPO_ROOT/RQ-VAE/compute_residuals.py" \
            --dataset "$DATASET" \
            --checkpoint-path "$CUR_RQ_CHECKPOINT" \
            --data-path "$EMBEDDING_FILE" \
            --output-file "$RESIDUALS_FILE" \
            --device "$RQ_DEVICE"
          if [[ ! -f "$RESIDUALS_FILE" ]]; then
            printf 'Residual computation failed; could not create: %s\n' "$RESIDUALS_FILE" >&2
            exit 1
          fi
          printf 'Completed %s residual computation in %s.\n' \
            "$TOK_LABEL" "$(format_duration "$((SECONDS - RES_START))")"
          record_phase "[$TOK_LABEL] Residual computation" "$((SECONDS - RES_START))"
          printf 'Stored residuals file: %s\n' "$RESIDUALS_FILE"
        fi
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
  fi

  # Downstream checkpoint and results paths for varlen
  PHASE_CKPT_TAG=""
  PHASE_RES_TAG=""
  if [[ "$PHASE" == "1.5" ]]; then
    PHASE_CKPT_TAG="-phase1.5"
    PHASE_RES_TAG="_phase1.5"
  fi

  if [[ "$MIN_LENGTH" -eq 1 ]]; then
    CKPT_TAG="varlen${STRAT_TAG}${PHASE_CKPT_TAG}-max${MAX_LENGTH}"
    RES_TAG="varlen${STRAT_TAG}${PHASE_RES_TAG}_max${MAX_LENGTH}"
  else
    CKPT_TAG="varlen${STRAT_TAG}${PHASE_CKPT_TAG}-min${MIN_LENGTH}-max${MAX_LENGTH}"
    RES_TAG="varlen${STRAT_TAG}${PHASE_RES_TAG}_min${MIN_LENGTH}-max${MAX_LENGTH}"
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

  if [[ "$RESUME" == true && "$tiger_trained" == true ]]; then
    printf '\n[TIGER] [%s] [%s] Found existing trained model: %s\n' "$TOK_LABEL" "$MODE" "$CUR_TIGER_CKPT"
    printf '[TIGER] Skipping training and resuming straight to evaluation.\n'
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

  # Clean intermediate checkpoint folders (checkpoint-*) if any remain
  if [[ -d "$tiger_ckpt_dir" ]]; then
    rm -rf "$tiger_ckpt_dir"/checkpoint-* 2>/dev/null || true
  fi

  # Post-evaluation cleanup: remove entire model checkpoint if requested and results exist
  tiger_results_file="$CUR_TIGER_RESULTS"
  if [[ "$tiger_results_file" != /* ]]; then
    tiger_results_file="$REPO_ROOT/LETTER-TIGER/$tiger_results_file"
  fi
  if [[ "$CLEAN_CHECKPOINTS" == true ]]; then
    if [[ -f "$tiger_results_file" && -s "$tiger_results_file" ]]; then
      printf '[Cleanup] Removing TIGER checkpoint directory after evaluation: %s\n' "$tiger_ckpt_dir"
      rm -rf "$tiger_ckpt_dir"
    else
      printf '[Cleanup] Warning: Results file (%s) missing or empty; keeping checkpoint: %s\n' \
        "$tiger_results_file" "$tiger_ckpt_dir" >&2
    fi
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

  if [[ "$RESUME" == true && "$lcrec_trained" == true ]]; then
    printf '\n[LC-Rec] [%s] [%s] Found existing trained model: %s\n' "$TOK_LABEL" "$MODE" "$CUR_LCREC_CKPT"
    printf '[LC-Rec] Skipping training and resuming straight to evaluation.\n'
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

  # Clean intermediate checkpoint folders (checkpoint-*) if any remain
  if [[ -d "$lcrec_ckpt_dir" ]]; then
    rm -rf "$lcrec_ckpt_dir"/checkpoint-* 2>/dev/null || true
  fi

  # Post-evaluation cleanup: remove entire model checkpoint if requested and results exist
  lcrec_results_file="$CUR_LCREC_RESULTS"
  if [[ "$lcrec_results_file" != /* ]]; then
    lcrec_results_file="$REPO_ROOT/LETTER-LC-Rec/$lcrec_results_file"
  fi
  if [[ "$CLEAN_CHECKPOINTS" == true ]]; then
    if [[ -f "$lcrec_results_file" && -s "$lcrec_results_file" ]]; then
      printf '[Cleanup] Removing LC-Rec checkpoint directory after evaluation: %s\n' "$lcrec_ckpt_dir"
      rm -rf "$lcrec_ckpt_dir"
    else
      printf '[Cleanup] Warning: Results file (%s) missing or empty; keeping checkpoint: %s\n' \
        "$lcrec_results_file" "$lcrec_ckpt_dir" >&2
    fi
  fi
fi

# Post-pipeline cleanup: remove the tokenizer checkpoint used by this run.
# Keep this scoped to the selected checkpoint file so other layers, phases,
# strategies, or datasets sharing the checkpoint root are not affected.
if [[ "$CLEAN_CHECKPOINTS" == true && -n "$CUR_RQ_CHECKPOINT" && -f "$CUR_RQ_CHECKPOINT" ]]; then
  printf '[Cleanup] Removing %s tokenizer checkpoint: %s\n' "$TOK_LABEL" "$CUR_RQ_CHECKPOINT"
  rm -f "$CUR_RQ_CHECKPOINT"
  if [[ -d "$TOK_CKPT_ROOT" ]]; then
    find "$TOK_CKPT_ROOT" -depth -type d -empty -delete 2>/dev/null || true
  fi
fi

print_phase_durations
printf '\nEnd-to-end recommendation pipeline completed in %s.\n' \
  "$(format_duration "$((SECONDS - PIPELINE_START))")"
