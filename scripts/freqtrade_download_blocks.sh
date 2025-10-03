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

ymd_to_iso() { echo "${1:0:4}-${1:4:2}-${1:6:2}"; }
iso_to_ymd() { echo "${1//-/}"; }

current_iso="$(ymd_to_iso "${START_DATE}")"
while [[ "${current}" < "${END_DATE}" ]]; do
  next_iso=$(date -u -d "${current_iso} +${BLOCK_DAYS} days" +%Y-%m-%d)
  next="$(iso_to_ymd "${next_iso}")"
  if [[ "${next}" > "${END_DATE}" ]]; then
    next="${END_DATE}"
    next_iso="$(ymd_to_iso "${next}")"
  fi
  cur_ymd="$(iso_to_ymd "${current_iso}")"
  echo "-> Block: ${cur_ymd}-${next}" >&2
  freqtrade download-data \
    --config "${CONFIG_PATH}" \
    -t ${TIMEFRAMES} \
    --timerange "${cur_ymd}-${next}"

  current_iso="${next_iso}"
done

echo "Download blocks complete." >&2
