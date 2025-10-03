#!/usr/bin/env bash
set -euo pipefail

# Download OHLCV data in fixed-size blocks to ensure enough history
# Usage:
#   freqtrade_download_blocks.sh <config_path> <timeframes> <start_yyyymmdd> <end_yyyymmdd> <block_days>
# Defaults (when args are omitted) are aligned with docker-compose env vars.

CONFIG_PATH="${1:-/freqtrade/user_data/config.json}"
TIMEFRAMES="${2:-1h}"
START_DATE="${3:-20230101}"
END_DATE="${4:-20250930}"
BLOCK_DAYS="${5:-30}"

echo "Downloading data in ${BLOCK_DAYS}-day blocks from ${START_DATE} to ${END_DATE} for timeframes: ${TIMEFRAMES}" >&2

current="${START_DATE}"
while [[ "${current}" < "${END_DATE}" ]]; do
  next=$(date -u -d "${current} +${BLOCK_DAYS} day" +%Y%m%d)
  if [[ "${next}" > "${END_DATE}" ]]; then
    next="${END_DATE}"
  fi
  echo "-> Block: ${current}-${next}" >&2
  freqtrade download-data \
    --config "${CONFIG_PATH}" \
    --timeframes ${TIMEFRAMES} \
    --timerange "${current}-${next}"

  current="${next}"
done

echo "Download blocks complete." >&2

