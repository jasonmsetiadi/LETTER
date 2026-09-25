#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash data_process/preprocess_item_embeddings.sh \
    --dataset DATASET \
    [--plm-checkpoint MODEL_OR_PATH] \
    [--data-root PATH] [--gpu-id ID] [--plm-name NAME] \
    [--max-sent-len TOKENS] [--python PATH] [--overwrite]

Generates item-text embeddings from <data-root>/<dataset>/<dataset>.item.json.
The output is stored as:
  <data-root>/<dataset>/<dataset>.emb-<plm-name>-td.npy
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CALLER_DIR="$(pwd)"
DATA_ROOT="$REPO_ROOT/data"
DATASET=""
PLM_CHECKPOINT="google/flan-t5-xl"
PLM_NAME="flan-t5-xl"
GPU_ID="0"
MAX_SENT_LEN="2048"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OVERWRITE=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --plm-checkpoint)
      PLM_CHECKPOINT="$2"
      shift 2
      ;;
    --data-root)
      DATA_ROOT="$2"
      shift 2
      ;;
    --gpu-id)
      GPU_ID="$2"
      shift 2
      ;;
    --plm-name)
      PLM_NAME="$2"
      shift 2
      ;;
    --max-sent-len)
      MAX_SENT_LEN="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --overwrite)
      OVERWRITE=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown argument: %s\n\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$DATASET" ]]; then
  printf '%s\n\n' '--dataset is required.' >&2
  usage >&2
  exit 2
fi

if [[ "$PLM_CHECKPOINT" == /* || "$PLM_CHECKPOINT" == ./* || "$PLM_CHECKPOINT" == ../* ]] && [[ ! -d "$PLM_CHECKPOINT" ]]; then
  printf 'Local model checkpoint directory not found: %s\n' "$PLM_CHECKPOINT" >&2
  printf 'Provide an existing directory or a Hugging Face model ID such as google/flan-t5-xl.\n' >&2
  exit 1
fi

if [[ "$DATA_ROOT" != /* ]]; then
  DATA_ROOT="$CALLER_DIR/$DATA_ROOT"
fi

ITEM_FILE="$DATA_ROOT/$DATASET/$DATASET.item.json"
OUTPUT_FILE="$DATA_ROOT/$DATASET/$DATASET.emb-$PLM_NAME-td.npy"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  printf 'Python executable not found: %s\n' "$PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -f "$ITEM_FILE" ]]; then
  printf 'Processed item metadata not found: %s\n' "$ITEM_FILE" >&2
  exit 1
fi

if [[ -e "$OUTPUT_FILE" && "$OVERWRITE" != true ]]; then
  printf 'Output already exists: %s\nUse --overwrite to replace it.\n' "$OUTPUT_FILE" >&2
  exit 1
fi

cd "$REPO_ROOT"
"$PYTHON_BIN" data_process/amazon_text_emb.py \
  --dataset "$DATASET" \
  --root "$DATA_ROOT" \
  --gpu_id "$GPU_ID" \
  --plm_name "$PLM_NAME" \
  --plm_checkpoint "$PLM_CHECKPOINT" \
  --max_sent_len "$MAX_SENT_LEN"

if [[ ! -f "$OUTPUT_FILE" ]]; then
  printf 'Embedding generation completed without creating: %s\n' "$OUTPUT_FILE" >&2
  exit 1
fi

printf 'Stored item embeddings: %s\n' "$OUTPUT_FILE"
