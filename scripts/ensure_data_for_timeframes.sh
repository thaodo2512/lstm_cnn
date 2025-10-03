#!/usr/bin/env bash
set -euo pipefail

# Usage: ensure_data_for_timeframes.sh <config_path> <timeframes_csv> <timerange> [blocks] [block_days]
# Example: ensure_data_for_timeframes.sh /freqtrade/user_data/config.json "1h,5m" "20240101-20250930" 3 30

CONFIG_PATH=${1:-/freqtrade/user_data/config.json}
TF_CSV=${2:-"1h"}
TR=${3:-"20240101-20250930"}
BLOCKS=${4:-3}
BLOCK_DAYS=${5:-30}

IFS=',' read -r -a TFS <<< "$TF_CSV"

echo "Ensuring OHLCV for timeframes: ${TFS[*]} timerange=${TR} blocks=${BLOCKS} block_days=${BLOCK_DAYS}"

for TF in "${TFS[@]}"; do
  TF_TRIMMED=$(echo "$TF" | xargs)
  if [[ -z "$TF_TRIMMED" ]]; then
    continue
  fi
  echo "Downloading data for timeframe: $TF_TRIMMED"
  freqtrade download-data --config "$CONFIG_PATH" -t "$TF_TRIMMED" --timerange "$TR" || true
  echo "Prepending pre-start history for FreqAI training windows: $TF_TRIMMED"
  bash /workspace/scripts/freqtrade_download_prestart_blocks.sh "$CONFIG_PATH" "$TF_TRIMMED" "$TR" "$BLOCKS" "$BLOCK_DAYS" || true
done

echo "Data ensure step complete."

