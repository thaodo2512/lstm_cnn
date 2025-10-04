#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=str, help="Path to config.json")
    ap.add_argument("pairs", type=str, nargs="+", help="Pairs to set as whitelist (space-separated)")
    args = ap.parse_args()

    p = Path(args.config)
    data = json.loads(p.read_text())
    if "exchange" not in data:
        raise SystemExit("Config missing 'exchange' section")
    data["exchange"].setdefault("pair_whitelist", [])
    data["exchange"]["pair_whitelist"] = args.pairs
    p.write_text(json.dumps(data, indent=2))
    print(f"Updated {p} with {len(args.pairs)} pair(s)")


if __name__ == "__main__":
    main()

