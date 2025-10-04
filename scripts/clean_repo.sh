#!/usr/bin/env bash
set -euo pipefail

# Cleanup utility for this repo. Removes generated artifacts, Freqtrade data/models,
# backtest results, caches, and (optionally) docker state. Defaults to dry-run.
#
# Usage examples:
#   scripts/clean_repo.sh --all --yes
#   scripts/clean_repo.sh --models --data --backtests --yes
#   scripts/clean_repo.sh --docker --yes   # stop/remove compose services + builder cache hint
#
# Flags
#   --artifacts    Remove ./artifacts* folders
#   --data         Remove ./freqtrade_user_data/data
#   --models       Remove ./freqtrade_user_data/models
#   --backtests    Remove ./freqtrade_user_data/backtest_results
#   --db           Remove ./tradesv3.dryrun.sqlite*
#   --logs         Remove ./freqtrade_user_data/logs (if exists)
#   --caches       Remove __pycache__ and .mypy_cache/.pytest_cache
#   --docker       Run 'docker compose down' and suggest/prune builder cache; remove compose volume(s) with --docker-volumes
#   --docker-volumes  Use with --docker to also remove named volumes (e.g., freqtrade_ui_data)
#   --all          Shorthand: artifacts,data,models,backtests,db,logs,caches
#   --yes          Perform deletion (otherwise dry-run)
#   --force        Skip repo root safety check

SHOW=1
DO_ARTIFACTS=0; DO_DATA=0; DO_MODELS=0; DO_BTEST=0; DO_DB=0; DO_LOGS=0; DO_CACHES=0; DO_DOCKER=0; DO_DOCKER_VOL=0
CONFIRM=0; FORCE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --artifacts) DO_ARTIFACTS=1 ;;
    --data)      DO_DATA=1 ;;
    --models)    DO_MODELS=1 ;;
    --backtests) DO_BTEST=1 ;;
    --db)        DO_DB=1 ;;
    --logs)      DO_LOGS=1 ;;
    --caches)    DO_CACHES=1 ;;
    --docker)    DO_DOCKER=1 ;;
    --docker-volumes) DO_DOCKER_VOL=1 ;;
    --all)       DO_ARTIFACTS=1; DO_DATA=1; DO_MODELS=1; DO_BTEST=1; DO_DB=1; DO_LOGS=1; DO_CACHES=1 ;;
    --yes)       CONFIRM=1 ;;
    --force)     FORCE=1 ;;
    -h|--help)
      sed -n '1,80p' "$0" | sed -n '2,60p'; exit 0 ;;
    *) echo "Unknown flag: $1"; exit 2 ;;
  esac; shift
done

# Safety: ensure we're at repo root unless --force
if [[ $FORCE -eq 0 ]]; then
  if [[ ! -f "docker-compose.yml" ]] || [[ ! -f "hybrid_lstm_transformer_crypto.py" ]]; then
    echo "Refusing to run outside repo root. Use --force to override." >&2
    exit 3
  fi
fi

targets=()
add() { targets+=("$1"); }

[[ $DO_ARTIFACTS -eq 1 ]] && add "artifacts*"
[[ $DO_DATA -eq 1      ]] && add "freqtrade_user_data/data"
[[ $DO_MODELS -eq 1    ]] && add "freqtrade_user_data/models"
[[ $DO_BTEST -eq 1     ]] && add "freqtrade_user_data/backtest_results"
[[ $DO_DB -eq 1        ]] && add "tradesv3.dryrun.sqlite*"
[[ $DO_LOGS -eq 1      ]] && add "freqtrade_user_data/logs"
[[ $DO_CACHES -eq 1    ]] && add "**/__pycache__" "**/.pytest_cache" "**/.mypy_cache"

if [[ ${#targets[@]} -eq 0 && $DO_DOCKER -eq 0 ]]; then
  echo "Nothing to do. Pass flags (e.g., --all, --models, --data, --docker)." >&2
  exit 1
fi

echo "Cleanup plan (dry-run=${CONFIRM:-0}):"
for t in "${targets[@]}"; do echo "  rm -rf $t"; done
if [[ $DO_DOCKER -eq 1 ]]; then
  echo "  docker compose down${DO_DOCKER_VOL:+ -v}"
  echo "  docker builder prune -f   # optional (not run automatically)"
fi

if [[ $CONFIRM -ne 1 ]]; then
  echo "Dry-run. Re-run with --yes to apply."
  exit 0
fi

# Apply deletions
for t in "${targets[@]}"; do
  shopt -s nullglob globstar
  for p in $t; do
    if [[ -e "$p" ]]; then
      echo "Removing: $p"
      rm -rf --one-file-system -- "$p"
    fi
  done
done

if [[ $DO_DOCKER -eq 1 ]]; then
  echo "Stopping compose services..."
  docker compose down ${DO_DOCKER_VOL:+ -v} || true
  echo "You can also prune build cache with: docker builder prune -f"
fi

echo "Cleanup complete."

