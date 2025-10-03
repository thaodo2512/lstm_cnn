#!/usr/bin/env python3
"""
Fetch Binance USDT‑perpetual (USDⓈ‑M) pairs via ccxt and optionally update
freqtrade_user_data/config.json's pair_whitelist.

Usage examples:
  - Print pairs:
      python scripts/fetch_binance_perp_pairs.py --quote USDT
  - Update config whitelist:
      python scripts/fetch_binance_perp_pairs.py --quote USDT \
        --update-config freqtrade_user_data/config.json

Options:
  --quote USDT (default): Only include pairs with the given quote.
  --include-inactive: Also include inactive markets (default: only active).
  --limit N: Only keep top N alphabetically (for testing).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List


def fetch_pairs(quote: str = "USDT", include_inactive: bool = False) -> List[str]:
    try:
        import ccxt  # type: ignore
    except Exception as e:
        print(f"ccxt not installed: {e}\nInstall with: pip install ccxt", file=sys.stderr)
        sys.exit(1)

    ex = ccxt.binance({
        "enableRateLimit": True,
        "options": {"defaultType": "future"},  # USDⓈ‑M (linear) futures
    })
    markets = ex.load_markets()
    pairs: List[str] = []
    for m in markets.values():
        # Keep USDⓈ‑M perpetual swaps with the requested quote
        if not m.get("swap"):
            continue
        if m.get("quote") != quote:
            continue
        if not include_inactive and not m.get("active", False):
            continue
        sym = m.get("symbol")
        if sym and sym not in pairs:
            pairs.append(sym)
    pairs.sort()
    return pairs


def update_config(config_path: Path, pairs: List[str]) -> None:
    data = json.loads(config_path.read_text())
    if "exchange" not in data:
        raise RuntimeError("Config missing 'exchange' section")
    ex = data["exchange"]
    ex["pair_whitelist"] = pairs
    config_path.write_text(json.dumps(data, indent=2))
    print(f"Updated pair_whitelist with {len(pairs)} pairs in {config_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quote", default="USDT")
    ap.add_argument("--include-inactive", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--update-config", type=str, default="")
    args = ap.parse_args()

    pairs = fetch_pairs(args.quote, include_inactive=args.include_inactive)
    if args.limit and args.limit > 0:
        pairs = pairs[: args.limit]
    if args.update_config:
        update_config(Path(args.update_config), pairs)
    else:
        print(json.dumps(pairs, indent=2))


if __name__ == "__main__":
    main()

