#!/usr/bin/env bash
set -euo pipefail

# Runs a Freqtrade dry-run bot that sends Telegram trade notifications
# for your strategy's signals. Accepts PAIRS (comma-separated), STRATEGY, etc.

PAIRS_CSV=${PAIRS:-BTC/USDT:USDT}
TIMEFRAMES=${TIMEFRAMES:-1h}
TIMERANGE=${TIMERANGE:-20240101-20240201}
STRATEGY=${STRATEGY:-FreqAIHybridExample}
FREQAIMODEL=${FREQAIMODEL:-HybridTimeseriesFreqAIModel_tinhn}
CONFIG_IN=${CONFIG_IN:-/freqtrade/user_data/config.json}
CONFIG_TMP=${CONFIG_TMP:-/tmp/config_pairs_trade.json}

echo "Pairs: $PAIRS_CSV  Strategy: $STRATEGY  Model: $FREQAIMODEL"

# Build temporary config with custom whitelist (avoid editing main config)
CONFIG_IN="$CONFIG_IN" CONFIG_TMP="$CONFIG_TMP" PAIRS="$PAIRS_CSV" python - <<'PY'
import json, os, sys
src = os.environ.get('CONFIG_IN')
dst = os.environ.get('CONFIG_TMP')
pairs_raw = os.environ.get('PAIRS') or os.environ.get('PAIRS_CSV') or ''
pairs = [p.strip() for p in pairs_raw.split(',') if p.strip()]
if not pairs:
    print('No pairs provided via PAIRS env.', file=sys.stderr)
    sys.exit(2)
conf = json.load(open(src))
conf.setdefault('exchange', {}).setdefault('pair_whitelist', [])
conf['exchange']['pair_whitelist'] = pairs
# Force dry-run to avoid real orders
conf['dry_run'] = True
# Keep dry-run wallet sane
conf['stake_amount'] = 100
with open(dst,'w') as f:
    json.dump(conf, f, indent=2)
print(f'Wrote temp trade config to {dst} with {len(pairs)} pairs')
PY

echo "Pre-downloading data for ${TIMEFRAMES} ${TIMERANGE} ..."
freqtrade list-data --config "$CONFIG_TMP" --show-timerange || true
# Split TIMEFRAMES (comma or space separated) and download each individually
TF_INPUT="${TIMEFRAMES:-1h}"
for tf in $(echo "$TF_INPUT" | tr ',' ' '); do
  tf=$(echo "$tf" | xargs)
  [[ -z "$tf" ]] && continue
  echo "-> Downloading timeframe: $tf"
  freqtrade download-data --config "$CONFIG_TMP" -t "$tf" --timerange "$TIMERANGE" || true
done

echo "Starting Freqtrade dry-run with Telegram (if configured) ..."
exec freqtrade trade \
  --config "$CONFIG_TMP" \
  --strategy "$STRATEGY" \
  --freqaimodel "$FREQAIMODEL"
