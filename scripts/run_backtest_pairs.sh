#!/usr/bin/env bash
set -euo pipefail

# Runs a FreqAI backtest for a specific set of pairs without permanently
# modifying the main config. Accepts PAIRS (comma-separated), TIMERANGE, etc.

PAIRS_CSV=${PAIRS:-BTC/USDT:USDT}
TIMERANGE=${TIMERANGE:-20240101-20240201}
TIMEFRAMES=${TIMEFRAMES:-1h}
STRATEGY=${STRATEGY:-FreqAIHybridExample}
FREQAIMODEL=${FREQAIMODEL:-HybridTimeseriesFreqAIModel_tinhn}
CONFIG_IN=${CONFIG_IN:-/freqtrade/user_data/config.json}
CONFIG_TMP=${CONFIG_TMP:-/tmp/config_pairs.json}
BLOCKS=${BLOCKS:-1}
BLOCK_DAYS=${BLOCK_DAYS:-30}

echo "Pairs: $PAIRS_CSV"
echo "Timerange: $TIMERANGE  Timeframes: $TIMEFRAMES  Strategy: $STRATEGY  Model: $FREQAIMODEL"

# Build temporary config with custom whitelist via Python (avoids jq dependency)
python - <<'PY'
import json, os, sys
src = os.environ.get('CONFIG_IN')
dst = os.environ.get('CONFIG_TMP')
pairs = [p.strip() for p in os.environ.get('PAIRS_CSV','').split(',') if p.strip()]
if not pairs:
    print('No pairs provided via PAIRS env.', file=sys.stderr)
    sys.exit(2)
conf = json.load(open(src))
conf.setdefault('exchange', {}).setdefault('pair_whitelist', [])
conf['exchange']['pair_whitelist'] = pairs
with open(dst,'w') as f:
    json.dump(conf, f, indent=2)
print(f'Wrote temp config to {dst} with {len(pairs)} pairs')
PY

echo "Listing available data (pre-download)..."
freqtrade list-data --config "$CONFIG_TMP" --show-timerange || true

echo "Downloading backtest data for $PAIRS_CSV ..."
freqtrade download-data --config "$CONFIG_TMP" -t "$TIMEFRAMES" --timerange "$TIMERANGE"

echo "Ensuring pre-start history for FreqAI (optional)..."
bash /workspace/scripts/freqtrade_download_prestart_blocks.sh "$CONFIG_TMP" "$TIMEFRAMES" "$TIMERANGE" "$BLOCKS" "$BLOCK_DAYS" || true

echo "Running backtest..."
mkdir -p /freqtrade/user_data/backtest_results
freqtrade backtesting \
  --config "$CONFIG_TMP" \
  --strategy "$STRATEGY" \
  --freqaimodel "$FREQAIMODEL" \
  --timerange "$TIMERANGE" \
  --cache day \
  --export trades \
  --export-filename /freqtrade/user_data/backtest_results/freqai_trades.json

echo "Backtest finished. Results: /freqtrade/user_data/backtest_results/freqai_trades.json"

