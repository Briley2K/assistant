"""
Append-only conversation log (JSONL). The assistant writes turns here; the
control panel tails the file to show the chat live. One JSON object per line:
{"ts": <unix seconds>, "role": "user"|"assistant", "text": "..."}
"""
import os
import json
import time

import config


def log(role: str, text: str) -> None:
    """Append one turn. No-ops on empty text or if logging is disabled."""
    if not config.CHATLOG_ENABLED:
        return
    text = (text or "").strip()
    if not text:
        return
    try:
        os.makedirs(os.path.dirname(config.CHATLOG_PATH), exist_ok=True)
        with open(config.CHATLOG_PATH, "a") as f:
            f.write(json.dumps({"ts": time.time(), "role": role, "text": text}) + "\n")
    except OSError:
        pass


def tail(limit: int = 300) -> list[dict]:
    """Return the most recent `limit` turns (oldest first)."""
    try:
        with open(config.CHATLOG_PATH) as f:
            lines = f.readlines()[-limit:]
    except FileNotFoundError:
        return []
    turns = []
    for line in lines:
        try:
            turns.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return turns


def clear() -> None:
    try:
        open(config.CHATLOG_PATH, "w").close()
    except OSError:
        pass
