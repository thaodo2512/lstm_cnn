#!/usr/bin/env bash
set -euo pipefail

# Download N blocks of <block_days> ending at the BACKTEST START date (UTC),
# so FreqAI has enough pre-history to train before backtesting begins.
# Usage:
#   freqtrade_download_prestart_blocks.sh <config_path> <timeframes> <timerange> <blocks> [block_days]
# Example: timerange=20240101-20250930, blocks=2, block_days=30 ->
#   downloads 20231102-20240101 and 20231003-20231102 (approx, UTC math).

CONFIG_PATH="${1:-/freqtrade/user_data/config.json}"
TIMEFRAMES="${2:-1h}"
TIMERANGE="${3:-20240101-20250930}"
BLOCKS="${4:-2}"
BLOCK_DAYS="${5:-30}"

START_DATE="${TIMERANGE%%-*}"
if [[ -z "${START_DATE}" || "${START_DATE}" == "${TIMERANGE}" ]]; then
  echo "Invalid TIMERANGE '${TIMERANGE}'. Expected format YYYYMMDD-YYYYMMDD" >&2
  exit 1
fi

ymd_to_iso() { echo "${1:0:4}-${1:4:2}-${1:6:2}"; }
iso_to_ymd() { echo "${1//-/}"; }

end_date_iso="$(ymd_to_iso "${START_DATE}")"
echo "Downloading ${BLOCKS} pre-start blocks of ${BLOCK_DAYS} days ending at ${START_DATE} for ${TIMEFRAMES}" >&2

for i in $(seq 1 "${BLOCKS}"); do
  start_date_iso=$(date -u -d "${end_date_iso} - ${BLOCK_DAYS} days" +%Y-%m-%d)
  start_date="$(iso_to_ymd "${start_date_iso}")"
  end_date="$(iso_to_ymd "${end_date_iso}")"
  echo "-> Pre-block ${i}: ${start_date}-${end_date}" >&2
  freqtrade download-data \
    --config "${CONFIG_PATH}" \
    -t ${TIMEFRAMES} \
    --timerange "${start_date}-${end_date}"
  end_date_iso="${start_date_iso}"
done

echo "Pre-start block downloads complete." >&2
