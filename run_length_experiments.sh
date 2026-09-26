#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash run_length_experiments.sh [options]

Runs a comparative experiment across multiple codebook lengths / layer depths
for both fixed-length and variable-length LETTER pipelines, then outputs a
consolidated comparison table of recommendation metrics (Hit@K, NDCG@K).

Options:
  --dataset NAME               Dataset name (default: Instruments)
  --lengths LIST               Lengths to test, e.g. "4 5" or "5 6" (default: "4 5")
  --modes LIST                 Pipelines: fixed, varlen, or "fixed,varlen" (default: "fixed,varlen")
  --models LIST                Models: tiger, lcrec, or "tiger,lcrec" (default: tiger)
  --base-model PATH            Base model for LC-Rec (default: huggyllama/llama-7b)
  --data-root PATH             Dataset parent directory (default: <repo>/data)
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
LENGTHS_STR="4 5"
MODES_STR="fixed,varlen"
MODELS="tiger"
BASE_MODEL="${BASE_MODEL:-huggyllama/llama-7b}"
DATA_ROOT="$REPO_ROOT/data"
RQ_EPOCHS="10000"
RQ_EVAL_STEP="2000"
RQ_DEVICE="cuda:0"
ALPHA="0.01"
BETA="0.0001"
TIGER_GPUS=""
LCREC_GPUS=""
SKIP_EXISTING=false
RETRAIN_RQVAE=false
OVERWRITE_INDEX=false
TOKENIZER_ONLY=false
SKIP_EVALUATION=false
PYTHON_BIN="${PYTHON_BIN:-python3}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --lengths) LENGTHS_STR="$2"; shift 2 ;;
    --modes) MODES_STR="$2"; shift 2 ;;
    --models) MODELS="$2"; shift 2 ;;
    --base-model) BASE_MODEL="$2"; shift 2 ;;
    --data-root) DATA_ROOT="$2"; shift 2 ;;
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

# Parse lengths into array
IFS=', ' read -r -a LENGTHS <<< "$LENGTHS_STR"
for L in "${LENGTHS[@]}"; do
  if ! [[ "$L" =~ ^[1-9][0-9]*$ ]]; then
    printf 'Invalid length (must be positive integer): %s\n' "$L" >&2
    exit 2
  fi
done

contains_word() {
  local list="$1"
  local item="$2"
  [[ ",$list," == *",$item,"* || " $list " == *" $item "* ]]
}

check_all_results_exist() {
  local mode="$1"
  local length="$2"
  local model_item
  local res_file

  for model_item in tiger lcrec; do
    if contains_word "$MODELS" "$model_item"; then
      local model_dir="LETTER-TIGER"
      if [[ "$model_item" == "lcrec" ]]; then
        model_dir="LETTER-LC-Rec"
      fi

      if [[ "$mode" == "fixed" ]]; then
        if [[ "$length" -eq 4 ]]; then
          res_file="$REPO_ROOT/$model_dir/results/$DATASET/fixed.json"
        else
          res_file="$REPO_ROOT/$model_dir/results/$DATASET/fixed_L${length}.json"
        fi
      else
        if [[ "$length" -eq 4 ]]; then
          res_file="$REPO_ROOT/$model_dir/results/$DATASET/varlen.json"
        else
          res_file="$REPO_ROOT/$model_dir/results/$DATASET/varlen_max${length}.json"
        fi
      fi

      if [[ ! -f "$res_file" ]]; then
        return 1
      fi
    fi
  done
  return 0
}

printf '\n=================================================================\n'
printf ' Starting LETTER Layer Depth Experiment\n'
printf ' Dataset:  %s\n' "$DATASET"
printf ' Lengths:  %s\n' "${LENGTHS[*]}"
printf ' Modes:    %s\n' "$MODES_STR"
printf ' Models:   %s\n' "$MODELS"
printf '=================================================================\n'

for L in "${LENGTHS[@]}"; do
  # --- Fixed-Length Pipeline ---
  if contains_word "$MODES_STR" "fixed"; then
    if [[ "$SKIP_EXISTING" == true && "$TOKENIZER_ONLY" != true && "$SKIP_EVALUATION" != true ]] && check_all_results_exist "fixed" "$L"; then
      printf '\n[Experiment] Skipping FIXED-LENGTH length=%s (results already exist).\n' "$L"
    else
      printf '\n=================================================================\n'
      printf ' [Experiment] Running FIXED-LENGTH pipeline (length = %s)\n' "$L"
      printf '=================================================================\n'
      fixed_cmd=(
        bash "$REPO_ROOT/run_fixed_length_pipeline.sh"
        --dataset "$DATASET"
        --data-root "$DATA_ROOT"
        --num-layers "$L"
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
    fi
  fi

  # --- Variable-Length Pipeline ---
  if contains_word "$MODES_STR" "varlen"; then
    if [[ "$SKIP_EXISTING" == true && "$TOKENIZER_ONLY" != true && "$SKIP_EVALUATION" != true ]] && check_all_results_exist "varlen" "$L"; then
      printf '\n[Experiment] Skipping VARIABLE-LENGTH max_length=%s (results already exist).\n' "$L"
    else
      printf '\n=================================================================\n'
      printf ' [Experiment] Running VARIABLE-LENGTH pipeline (max_length = %s)\n' "$L"
      printf '=================================================================\n'
      varlen_cmd=(
        bash "$REPO_ROOT/run_variable_length_pipeline.sh"
        --dataset "$DATASET"
        --data-root "$DATA_ROOT"
        --min-length 1
        --max-length "$L"
        --num-layers "$L"
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
  fi
done

# --- Display Consolidated Results Table ---
if [[ "$TOKENIZER_ONLY" != true && "$SKIP_EVALUATION" != true ]]; then
  "$PYTHON_BIN" -c '
import sys, json, os

dataset = sys.argv[1]
repo_root = sys.argv[2]
data_root = sys.argv[3]
lengths_input = sys.argv[4].split()
models_input = [m.strip() for m in sys.argv[5].split(",") if m.strip()]

tested_lengths = sorted(list(set([int(x) for x in lengths_input] + ([4] if os.path.exists(os.path.join(repo_root, "LETTER-TIGER/results", dataset, "fixed.json")) else []))))

row_fmt = "| {:<8} | {:<8} | {:<6} | {:<7} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8} |"
sep = "+----------+----------+--------+---------+----------+----------+----------+----------+----------+"
header = row_fmt.format("Model", "Mode", "Max L", "Mean L", "Hit@1", "Hit@5", "Hit@10", "NDCG@5", "NDCG@10")

rows = [
    "\n" + "=" * 90,
    f" LETTER Experiment Results Summary ({dataset})",
    "=" * 90,
    header,
    sep,
]

for model in models_input:
    model_name = "TIGER" if model == "tiger" else "LC-Rec"
    model_dir = "LETTER-TIGER" if model == "tiger" else "LETTER-LC-Rec"

    for l in tested_lengths:
        for mode in ["fixed", "varlen"]:
            if mode == "fixed":
                fname = "fixed.json" if l == 4 else f"fixed_L{l}.json"
                mean_l = f"{l}.00"
            else:
                fname = "varlen.json" if l == 4 else f"varlen_max{l}.json"
                summary_fname = f"{dataset}.index.varlen.summary.json" if l == 4 else f"{dataset}.index.varlen.max{l}.summary.json"
                summary_path = os.path.join(data_root, dataset, summary_fname)
                if os.path.isfile(summary_path):
                    try:
                        with open(summary_path) as sf:
                            sdata = json.load(sf)
                        ml_val = sdata.get("mean_length", float(l))
                        mean_l = f"{ml_val:.2f}"
                    except Exception:
                        mean_l = f"~{l}"
                else:
                    mean_l = f"~{l}"

            fpath = os.path.join(repo_root, model_dir, "results", dataset, fname)
            if os.path.isfile(fpath):
                try:
                    with open(fpath) as f:
                        data = json.load(f)
                    res = data.get("mean_results", {})
                    v1 = res.get("hit@1")
                    v5 = res.get("hit@5")
                    v10 = res.get("hit@10")
                    vn5 = res.get("ndcg@5")
                    vn10 = res.get("ndcg@10")
                    h1 = f"{v1 * 100:.2f}%" if v1 is not None else "-"
                    h5 = f"{v5 * 100:.2f}%" if v5 is not None else "-"
                    h10 = f"{v10 * 100:.2f}%" if v10 is not None else "-"
                    n5 = f"{vn5 * 100:.2f}%" if vn5 is not None else "-"
                    n10 = f"{vn10 * 100:.2f}%" if vn10 is not None else "-"
                    rows.append(row_fmt.format(model_name, mode, str(l), mean_l, h1, h5, h10, n5, n10))
                except Exception:
                    rows.append(row_fmt.format(model_name, mode, str(l), mean_l, "(err)", "-", "-", "-", "-"))
            else:
                rows.append(row_fmt.format(model_name, mode, str(l), mean_l, "(pending)", "-", "-", "-", "-"))

rows.append(sep)
summary_text = "\n".join(rows)
print(summary_text)

summary_file = os.path.join(repo_root, f"experiment_summary_{dataset}.txt")
with open(summary_file, "w") as sf:
    sf.write(summary_text + "\n")
print(f"Summary table saved to: {summary_file}\n")
' "$DATASET" "$REPO_ROOT" "$DATA_ROOT" "${LENGTHS[*]}" "$MODELS"
fi

printf '\nExperiment finished in %s.\n' \
  "$(format_duration "$((SECONDS - EXPERIMENT_START))")"
