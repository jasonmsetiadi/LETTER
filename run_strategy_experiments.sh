#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash run_strategy_experiments.sh [options]

Runs a comparative experiment evaluating fixed-length IDs against different
variable-length truncation strategies (shortest_unique, popularity, residual)
for LETTER recommendation pipelines, then outputs a consolidated comparison
table and reports (Hit@K, NDCG@K, Catalog Mean Length, Weighted Sequence Length).

Options:
  --dataset NAME               Dataset name (default: Instruments)
  --strategies LIST            Comma- or space-separated strategies:
                               fixed, shortest_unique, popularity, residual
                               (default: "fixed,shortest_unique,popularity,residual")
  --max-length COUNT           Maximum SID length / fixed codebook layers (default: 4)
  --min-length COUNT           Minimum SID length for varlen strategies (default: 1)
  --models LIST                Models: tiger, lcrec, or "tiger,lcrec" (default: tiger)
  --base-model PATH            Base model for LC-Rec (default: huggyllama/llama-7b)
  --data-root PATH             Dataset parent directory (default: <repo>/data)
  --inter-file PATH            Interaction JSON for popularity strategy
                               (default: <data-root>/<dataset>/<dataset>.inter.json)
  --collab-signal SIGNAL       Collaborative signal for popularity/collaborative:
                               frequency, user_entropy, pagerank, target, cf_density (default: frequency)
  --cf-emb-file PATH           Path to CF embeddings (.pt, .npy) for cf_density signal
  --residuals-file PATH        Residuals JSON for residual strategy
                               (default: <data-root>/<dataset>/<dataset>.residuals.json)
  --residual-threshold VALUE   Reconstruction error threshold for residual strategy (default: 0.2)
  --auto-compute-residuals     Compute residuals with RQ-VAE/compute_residuals.py if missing
  --rqvae-epochs COUNT         RQ-VAE epochs (default: 10000)
  --rqvae-eval-step COUNT      RQ-VAE validation interval (default: 2000)
  --rqvae-device DEVICE        RQ-VAE device (default: cuda:0)
  --alpha VALUE                Collaborative-loss weight (default: 0.01)
  --beta VALUE                 Diversity-loss weight (default: 0.0001)
  --tiger-gpus IDS             CUDA devices for TIGER (default: autodetect, up to 2)
  --lcrec-gpus IDS             CUDA devices for LC-Rec (default: autodetect, up to 4)
  --skip-existing              Skip runs whose final results JSON already exists
  --retrain-rqvae              Force retraining RQ-VAE even if a matching checkpoint exists
  --overwrite-index            Force regenerating indices even if they exist
  --tokenizer-only             Only generate tokenizers & indices, skip downstream training
  --skip-evaluation            Train recommenders without evaluating
  --summary-only               Only generate and print the comparison summary table from existing results
  --reset-comparison           Overwrite strategy comparison report without merging existing results
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

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CALLER_DIR="$(pwd)"
EXPERIMENT_START="$SECONDS"

DATASET="Instruments"
STRATEGIES_STR="fixed,shortest_unique,popularity,residual"
MAX_LENGTH="4"
MIN_LENGTH="1"
MODELS="tiger"
BASE_MODEL="${BASE_MODEL:-huggyllama/llama-7b}"
DATA_ROOT="$REPO_ROOT/data"
INTER_FILE=""
RESIDUALS_FILE=""
RESIDUAL_THRESHOLD="0.2"
AUTO_COMPUTE_RESIDUALS=false
RQ_EPOCHS="10000"
RQ_EVAL_STEP="2000"
RQ_DEVICE="cuda:0"
ALPHA="0.01"
BETA="0.0001"
TIGER_GPUS=""
LCREC_GPUS=""
COLLAB_SIGNAL="frequency"
COLLAB_SIGNALS_STR=""
CF_EMB_FILE=""
SKIP_EXISTING=false
RETRAIN_RQVAE=false
OVERWRITE_INDEX=false
TOKENIZER_ONLY=false
SKIP_EVALUATION=false
SUMMARY_ONLY=false
RESET_COMPARISON=false
PYTHON_BIN="${PYTHON_BIN:-python3}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --strategies) STRATEGIES_STR="$2"; shift 2 ;;
    --max-length) MAX_LENGTH="$2"; shift 2 ;;
    --min-length) MIN_LENGTH="$2"; shift 2 ;;
    --models) MODELS="$2"; shift 2 ;;
    --base-model) BASE_MODEL="$2"; shift 2 ;;
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --inter-file) INTER_FILE="$2"; shift 2 ;;
    --collab-signal|--popularity-signal) COLLAB_SIGNAL="$2"; shift 2 ;;
    --collab-signals) COLLAB_SIGNALS_STR="$2"; shift 2 ;;
    --cf-emb-file) CF_EMB_FILE="$2"; shift 2 ;;
    --residuals-file) RESIDUALS_FILE="$2"; shift 2 ;;
    --residual-threshold) RESIDUAL_THRESHOLD="$2"; shift 2 ;;
    --auto-compute-residuals) AUTO_COMPUTE_RESIDUALS=true; shift ;;
    --rqvae-epochs) RQ_EPOCHS="$2"; shift 2 ;;
    --rqvae-eval-step) RQ_EVAL_STEP="$2"; shift 2 ;;
    --rqvae-device) RQ_DEVICE="$2"; shift 2 ;;
    --alpha) ALPHA="$2"; shift 2 ;;
    --beta) BETA="$2"; shift 2 ;;
    --tiger-gpus) TIGER_GPUS="$2"; shift 2 ;;
    --lcrec-gpus) LCREC_GPUS="$2"; shift 2 ;;
    --skip-existing) SKIP_EXISTING=true; shift ;;
    --retrain-rqvae) RETRAIN_RQVAE=true; shift ;;
    --overwrite-index) OVERWRITE_INDEX=true; shift ;;
    --tokenizer-only) TOKENIZER_ONLY=true; shift ;;
    --skip-evaluation) SKIP_EVALUATION=true; shift ;;
    --summary-only) SUMMARY_ONLY=true; shift ;;
    --reset-comparison|--no-merge) RESET_COMPARISON=true; shift ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "$DATA_ROOT" != /* ]]; then
  DATA_ROOT="$CALLER_DIR/$DATA_ROOT"
fi
if [[ -n "$BASE_MODEL" && -d "$CALLER_DIR/$BASE_MODEL" ]]; then
  BASE_MODEL="$CALLER_DIR/$BASE_MODEL"
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  printf 'Python executable not found: %s\n' "$PYTHON_BIN" >&2
  exit 1
fi

INTER_FILE="${INTER_FILE:-$DATA_ROOT/$DATASET/$DATASET.inter.json}"
RESIDUALS_FILE="${RESIDUALS_FILE:-$DATA_ROOT/$DATASET/$DATASET.residuals.json}"
if [[ -z "$CF_EMB_FILE" && -f "$REPO_ROOT/RQ-VAE/ckpt/$DATASET-32d-sasrec.pt" ]]; then
  CF_EMB_FILE="$REPO_ROOT/RQ-VAE/ckpt/$DATASET-32d-sasrec.pt"
fi

# Parse and expand strategies
EXPANDED_STRATEGIES=()
IFS=', ' read -r -a RAW_STRATEGIES <<< "$STRATEGIES_STR"
for S in "${RAW_STRATEGIES[@]}"; do
  if [[ "$S" == "popularity" || "$S" == "collaborative" ]] && [[ -n "$COLLAB_SIGNALS_STR" ]]; then
    if [[ "$COLLAB_SIGNALS_STR" == "all" ]]; then
      SIG_LIST=("frequency" "user_entropy" "pagerank" "target")
      if [[ -n "$CF_EMB_FILE" ]]; then SIG_LIST+=("cf_density"); fi
    else
      IFS=', ' read -r -a SIG_LIST <<< "$COLLAB_SIGNALS_STR"
    fi
    for sig_item in "${SIG_LIST[@]}"; do
      EXPANDED_STRATEGIES+=("popularity:${sig_item}")
    done
  else
    EXPANDED_STRATEGIES+=("$S")
  fi
done
STRATEGIES=("${EXPANDED_STRATEGIES[@]}")

for S in "${STRATEGIES[@]}"; do
  case "$S" in
    fixed|shortest_unique|popularity*|collaborative*|residual) ;;
    *) printf 'Unsupported strategy: %s (choose from: fixed, shortest_unique, popularity, collaborative, residual)\n' "$S" >&2; exit 2 ;;
  esac
done

contains_word() {
  local list="$1"
  local item="$2"
  [[ ",$list," == *",$item,"* || " $list " == *" $item "* ]]
}

get_result_path() {
  local model_dir="$1"
  local strat="$2"
  local tag=""
  if [[ "$MAX_LENGTH" -ne 4 || "$MIN_LENGTH" -ne 1 ]]; then
    tag="_max${MAX_LENGTH}"
  fi

  case "$strat" in
    fixed)
      if [[ "$MAX_LENGTH" -eq 4 ]]; then
        printf '%s/%s/results/%s/fixed.json' "$REPO_ROOT" "$model_dir" "$DATASET"
      else
        printf '%s/%s/results/%s/fixed_L%s.json' "$REPO_ROOT" "$model_dir" "$DATASET" "$MAX_LENGTH"
      fi
      ;;
    shortest_unique)
      printf '%s/%s/results/%s/varlen%s.json' "$REPO_ROOT" "$model_dir" "$DATASET" "$tag"
      ;;
    popularity*|collaborative*)
      local sig="$COLLAB_SIGNAL"
      if [[ "$strat" == *:* ]]; then
        sig="${strat#*:}"
      fi
      local pop_tag="-pop"
      case "$sig" in
        frequency|raw) pop_tag="-pop" ;;
        user_entropy|entropy) pop_tag="-pop-entropy" ;;
        pagerank|pr) pop_tag="-pop-pagerank" ;;
        target|target_frequency) pop_tag="-pop-target" ;;
        cf_density) pop_tag="-pop-cf" ;;
        *) pop_tag="-pop-$sig" ;;
      esac
      if [[ -z "$tag" ]]; then
        printf '%s/%s/results/%s/varlen%s.json' "$REPO_ROOT" "$model_dir" "$DATASET" "$pop_tag"
      else
        printf '%s/%s/results/%s/varlen%s%s.json' "$REPO_ROOT" "$model_dir" "$DATASET" "$pop_tag" "$tag"
      fi
      ;;
    residual)
      if [[ -z "$tag" ]]; then
        printf '%s/%s/results/%s/varlen-res.json' "$REPO_ROOT" "$model_dir" "$DATASET"
      else
        printf '%s/%s/results/%s/varlen-res%s.json' "$REPO_ROOT" "$model_dir" "$DATASET" "$tag"
      fi
      ;;
  esac
}

check_strategy_results_exist() {
  local strat="$1"
  local model_item
  for model_item in tiger lcrec; do
    if contains_word "$MODELS" "$model_item"; then
      local mdir="LETTER-TIGER"
      if [[ "$model_item" == "lcrec" ]]; then mdir="LETTER-LC-Rec"; fi
      local rpath
      rpath="$(get_result_path "$mdir" "$strat")"
      if [[ ! -f "$rpath" ]]; then
        return 1
      fi
    fi
  done
  return 0
}

printf '\n=================================================================\n'
printf ' LETTER Strategy Comparison Experiment\n'
printf ' Dataset:     %s\n' "$DATASET"
printf ' Strategies:  %s\n' "${STRATEGIES[*]}"
printf ' Models:      %s\n' "$MODELS"
printf ' Max Length:  %s\n' "$MAX_LENGTH"
printf ' Min Length:  %s\n' "$MIN_LENGTH"
printf '=================================================================\n'

if [[ "$SUMMARY_ONLY" != true ]]; then
  for STRAT in "${STRATEGIES[@]}"; do
    if [[ "$SKIP_EXISTING" == true && "$TOKENIZER_ONLY" != true && "$SKIP_EVALUATION" != true ]] && check_strategy_results_exist "$STRAT"; then
      printf '\n[Experiment] Skipping strategy "%s" (results already exist).\n' "$STRAT"
      continue
    fi

    printf '\n=================================================================\n'
    printf ' [Experiment] Running Strategy: %s\n' "$STRAT"
    printf '=================================================================\n'

    if [[ "$STRAT" == "fixed" ]]; then
      fixed_cmd=(
        bash "$REPO_ROOT/run_fixed_length_pipeline.sh"
        --dataset "$DATASET"
        --data-root "$DATA_ROOT"
        --num-layers "$MAX_LENGTH"
        --models "$MODELS"
        --base-model "$BASE_MODEL"
        --rqvae-epochs "$RQ_EPOCHS"
        --rqvae-eval-step "$RQ_EVAL_STEP"
        --rqvae-device "$RQ_DEVICE"
        --alpha "$ALPHA"
        --beta "$BETA"
        --python "$PYTHON_BIN"
      )
      if [[ -n "$TIGER_GPUS" ]]; then fixed_cmd+=(--tiger-gpus "$TIGER_GPUS"); fi
      if [[ -n "$LCREC_GPUS" ]]; then fixed_cmd+=(--lcrec-gpus "$LCREC_GPUS"); fi
      if [[ "$RETRAIN_RQVAE" == true ]]; then fixed_cmd+=(--retrain-rqvae); fi
      if [[ "$OVERWRITE_INDEX" == true ]]; then fixed_cmd+=(--overwrite-index); fi
      if [[ "$TOKENIZER_ONLY" == true ]]; then fixed_cmd+=(--tokenizer-only); fi
      if [[ "$SKIP_EVALUATION" == true ]]; then fixed_cmd+=(--skip-evaluation); fi

      "${fixed_cmd[@]}"

    elif [[ "$STRAT" == "residual" ]]; then
      if [[ ! -f "$RESIDUALS_FILE" ]]; then
        if [[ "$AUTO_COMPUTE_RESIDUALS" == true ]]; then
          printf '\n[Residuals] Computing reconstruction residuals per item...\n'
          # Find checkpoint
          RQ_CKPT="$("$PYTHON_BIN" -c '
import sys, glob, os
root = sys.argv[1]
candidates = glob.glob(os.path.join(root, "**", "best_collision_model.pth"), recursive=True)
if not candidates:
    candidates = glob.glob(os.path.join(root, "**", "best_loss_model.pth"), recursive=True)
if candidates:
    print(max(candidates, key=os.path.getmtime))
' "$REPO_ROOT/checkpoint/$DATASET" 2>/dev/null || true)"

          if [[ -z "$RQ_CKPT" || ! -f "$RQ_CKPT" ]]; then
            printf 'Error: Cannot auto-compute residuals; no RQ-VAE checkpoint found in checkpoint/%s\n' "$DATASET" >&2
            exit 1
          fi

          "$PYTHON_BIN" "$REPO_ROOT/RQ-VAE/compute_residuals.py" \
            --dataset "$DATASET" \
            --checkpoint-path "$RQ_CKPT" \
            --output-file "$RESIDUALS_FILE" \
            --device "$RQ_DEVICE"
        else
          printf 'Error: Residuals file not found: %s\n' "$RESIDUALS_FILE" >&2
          printf 'Pass --auto-compute-residuals or generate it first with RQ-VAE/compute_residuals.py\n' >&2
          exit 1
        fi
      fi

      varlen_cmd=(
        bash "$REPO_ROOT/run_variable_length_pipeline.sh"
        --dataset "$DATASET"
        --data-root "$DATA_ROOT"
        --min-length "$MIN_LENGTH"
        --max-length "$MAX_LENGTH"
        --num-layers "$MAX_LENGTH"
        --strategy "residual"
        --residuals-file "$RESIDUALS_FILE"
        --residual-threshold "$RESIDUAL_THRESHOLD"
        --models "$MODELS"
        --base-model "$BASE_MODEL"
        --rqvae-epochs "$RQ_EPOCHS"
        --rqvae-eval-step "$RQ_EVAL_STEP"
        --rqvae-device "$RQ_DEVICE"
        --alpha "$ALPHA"
        --beta "$BETA"
        --python "$PYTHON_BIN"
      )
      if [[ -n "$TIGER_GPUS" ]]; then varlen_cmd+=(--tiger-gpus "$TIGER_GPUS"); fi
      if [[ -n "$LCREC_GPUS" ]]; then varlen_cmd+=(--lcrec-gpus "$LCREC_GPUS"); fi
      if [[ "$RETRAIN_RQVAE" == true ]]; then varlen_cmd+=(--retrain-rqvae); fi
      if [[ "$OVERWRITE_INDEX" == true ]]; then varlen_cmd+=(--overwrite-index); fi
      if [[ "$TOKENIZER_ONLY" == true ]]; then varlen_cmd+=(--tokenizer-only); fi
      if [[ "$SKIP_EVALUATION" == true ]]; then varlen_cmd+=(--skip-evaluation); fi

      "${varlen_cmd[@]}"

    elif [[ "$STRAT" == popularity* || "$STRAT" == collaborative* ]]; then
      CURRENT_STRAT="popularity"
      CURRENT_SIGNAL="$COLLAB_SIGNAL"
      if [[ "$STRAT" == *:* ]]; then
        CURRENT_SIGNAL="${STRAT#*:}"
      fi
      if [[ ! -f "$INTER_FILE" && "$CURRENT_SIGNAL" != "cf_density" ]]; then
        printf 'Error: Interaction file not found for %s strategy: %s\n' "$STRAT" "$INTER_FILE" >&2
        exit 1
      fi

      varlen_cmd=(
        bash "$REPO_ROOT/run_variable_length_pipeline.sh"
        --dataset "$DATASET"
        --data-root "$DATA_ROOT"
        --min-length "$MIN_LENGTH"
        --max-length "$MAX_LENGTH"
        --num-layers "$MAX_LENGTH"
        --strategy "$CURRENT_STRAT"
        --collab-signal "$CURRENT_SIGNAL"
        --inter-file "$INTER_FILE"
        --models "$MODELS"
        --base-model "$BASE_MODEL"
        --rqvae-epochs "$RQ_EPOCHS"
        --rqvae-eval-step "$RQ_EVAL_STEP"
        --rqvae-device "$RQ_DEVICE"
        --alpha "$ALPHA"
        --beta "$BETA"
        --python "$PYTHON_BIN"
      )
      if [[ -n "$CF_EMB_FILE" ]]; then varlen_cmd+=(--cf-emb-file "$CF_EMB_FILE"); fi
      if [[ -n "$TIGER_GPUS" ]]; then varlen_cmd+=(--tiger-gpus "$TIGER_GPUS"); fi
      if [[ -n "$LCREC_GPUS" ]]; then varlen_cmd+=(--lcrec-gpus "$LCREC_GPUS"); fi
      if [[ "$RETRAIN_RQVAE" == true ]]; then varlen_cmd+=(--retrain-rqvae); fi
      if [[ "$OVERWRITE_INDEX" == true ]]; then varlen_cmd+=(--overwrite-index); fi
      if [[ "$TOKENIZER_ONLY" == true ]]; then varlen_cmd+=(--tokenizer-only); fi
      if [[ "$SKIP_EVALUATION" == true ]]; then varlen_cmd+=(--skip-evaluation); fi

      "${varlen_cmd[@]}"

    elif [[ "$STRAT" == "shortest_unique" ]]; then
      varlen_cmd=(
        bash "$REPO_ROOT/run_variable_length_pipeline.sh"
        --dataset "$DATASET"
        --data-root "$DATA_ROOT"
        --min-length "$MIN_LENGTH"
        --max-length "$MAX_LENGTH"
        --num-layers "$MAX_LENGTH"
        --strategy "shortest_unique"
        --models "$MODELS"
        --base-model "$BASE_MODEL"
        --rqvae-epochs "$RQ_EPOCHS"
        --rqvae-eval-step "$RQ_EVAL_STEP"
        --rqvae-device "$RQ_DEVICE"
        --alpha "$ALPHA"
        --beta "$BETA"
        --python "$PYTHON_BIN"
      )
      if [[ -n "$TIGER_GPUS" ]]; then varlen_cmd+=(--tiger-gpus "$TIGER_GPUS"); fi
      if [[ -n "$LCREC_GPUS" ]]; then varlen_cmd+=(--lcrec-gpus "$LCREC_GPUS"); fi
      if [[ "$RETRAIN_RQVAE" == true ]]; then varlen_cmd+=(--retrain-rqvae); fi
      if [[ "$OVERWRITE_INDEX" == true ]]; then varlen_cmd+=(--overwrite-index); fi
      if [[ "$TOKENIZER_ONLY" == true ]]; then varlen_cmd+=(--tokenizer-only); fi
      if [[ "$SKIP_EVALUATION" == true ]]; then varlen_cmd+=(--skip-evaluation); fi

      "${varlen_cmd[@]}"
    fi
  done
fi

# --- Consolidated Comparison Table Generation ---
report_cmd=(
  "$PYTHON_BIN" "$REPO_ROOT/RQ-VAE/generate_comparison_report.py"
  --dataset "$DATASET"
  --repo-root "$REPO_ROOT"
  --data-root "$DATA_ROOT"
  --max-length "$MAX_LENGTH"
  --min-length "$MIN_LENGTH"
  --strategies "${STRATEGIES[*]}"
  --models "$MODELS"
)
if [[ "$RESET_COMPARISON" == true ]]; then
  report_cmd+=(--no-merge)
fi
"${report_cmd[@]}"

printf '\nExperiment finished in %s.\n' \
  "$(format_duration "$((SECONDS - EXPERIMENT_START))")"
