#!/usr/bin/env bash
set -euo pipefail

# Download OHLCV data in N blocks of 30 days each, counting back from today (UTC).
# Usage:
#   freqtrade_download_recent_blocks.sh <config_path> <timeframes> <blocks> [block_days]
# Example: blocks=2 => downloads last 60 days in two 30-day chunks.

CONFIG_PATH="${1:-/freqtrade/user_data/config.json}"
TIMEFRAMES="${2:-1h}"
BLOCKS="${3:-2}"
BLOCK_DAYS="${4:-30}"

end_date=$(date -u +%Y%m%d)
echo "Downloading ${BLOCKS} blocks of ${BLOCK_DAYS} days from ${end_date} backwards for timeframes: ${TIMEFRAMES}" >&2

for i in $(seq 1 "${BLOCKS}"); do
  start_date=$(date -u -d "${end_date} - ${BLOCK_DAYS} day" +%Y%m%d)
  echo "-> Block ${i}: ${start_date}-${end_date}" >&2
  freqtrade download-data \
    --config "${CONFIG_PATH}" \
    --timeframes ${TIMEFRAMES} \
    --timerange "${start_date}-${end_date}"
  end_date="${start_date}"
done

echo "Recent block downloads complete." >&2

