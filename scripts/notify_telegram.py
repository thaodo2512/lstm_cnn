#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request


def send_message(token: str, chat_id: str, text: str) -> None:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = json.dumps({"chat_id": chat_id, "text": text}).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=20) as resp:  # nosec B310
        resp.read()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="Path to exported JSON (trades or signals)")
    ap.add_argument("--pairs", default="", help="CSV of pairs used")
    ap.add_argument("--timerange", default="")
    ap.add_argument("--timeframes", default="")
    ap.add_argument("--export-type", default="trades", choices=["trades", "signals"]) 
    ap.add_argument("--token", default=os.environ.get("TELEGRAM_BOT_TOKEN", ""))
    ap.add_argument("--chat-id", default=os.environ.get("TELEGRAM_CHAT_ID", ""))
    args = ap.parse_args()

    if not args.token or not args["chat_id"] if isinstance(args, dict) else not args.chat_id:
        print("Missing Telegram token/chat-id. Skipping notification.")
        return

    path = args.file
    try:
        with open(path, "r") as f:
            data = json.load(f)
        count = len(data) if isinstance(data, list) else 0
    except Exception as e:
        text = (
            f"Backtest finished. Failed to read export file: {path}\n"
            f"Error: {e}"
        )
        send_message(args.token, args.chat_id, text)
        return

    pairs_short = args.pairs.replace(":USDT", "").split(",") if args.pairs else []
    if len(pairs_short) > 6:
        pairs_disp = ", ".join(pairs_short[:6]) + f" (+{len(pairs_short)-6} more)"
    else:
        pairs_disp = ", ".join(pairs_short)

    text = (
        f"FreqAI Backtest completed\n"
        f"Export: {args.export_type} ({count} rows)\n"
        f"Pairs: {pairs_disp}\n"
        f"Timeframes: {args.timeframes}\n"
        f"Timerange: {args.timerange}\n"
        f"File: {os.path.basename(path)}"
    )
    send_message(args.token, args.chat_id, text)


if __name__ == "__main__":
    main()

